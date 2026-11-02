const std = @import("std");
const mlx = @import("mlx");
const Context = @import("../context.zig").Context;
const Tokenizer = @import("../tokenizer.zig").Tokenizer;
const template = @import("chat_template");
const ChatTemplate = template.ChatTemplate;
const Queue = std.Io.Queue;
const models = @import("root.zig");

const log = std.log.scoped(.qwen);

pub const Config = struct {
    architectures: []const []const u8,
    attention_dropout: f64,
    bos_token_id: u32,
    eos_token_id: u32,
    hidden_act: []const u8,
    hidden_size: u32,
    initializer_range: f64,
    intermediate_size: u32,
    max_position_embeddings: u32,
    max_window_layers: u32,
    model_type: []const u8,
    num_attention_heads: u32,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    rms_norm_eps: f64,
    rope_theta: f64,
    sliding_window: u32,
    tie_word_embeddings: bool,
    torch_dtype: []const u8,
    transformers_version: []const u8,
    use_cache: bool,
    use_sliding_window: bool,
    vocab_size: u32,
};

const TokenizerConfig = struct {
    chat_template: []const u8,
};

pub const Pipeline = struct {
    const Self = @This();

    gpa: std.mem.Allocator,
    io: std.Io,
    strm: mlx.Stream,
    tokenizer: Tokenizer,
    transformer_layers: std.ArrayList(mlx.nn.TransformerBlock),
    norm: mlx.Tensor,
    embedding_weights: mlx.Tensor,
    chat_template: ChatTemplate,

    fn preprocess(self: Self, input: []const u8) !mlx.Tensor {
        const messages: []const template.Message = &.{
            .{ .role = "user", .content = input },
        };

        const initial_prompt = try self.chat_template.render(self.gpa, messages, .{});
        defer self.gpa.free(initial_prompt);

        return self.tokenizer.encode(initial_prompt);
    }

    pub fn callStreaming(self: Self, input: []const u8, output: *Queue([]const u8)) !void {
        log.debug("generating response for: \"{s}...\"", .{input[0..@min(25, input.len)]});
        defer output.close(self.io);

        var encoded_input = try self.preprocess(input);
        defer encoded_input.deinit();

        log.debug("encoded prompt into token IDs (dimensions={any})", .{encoded_input.shape()});

        var first_pass = true;
        while (true) {
            defer first_pass = false;
            var ref = try self.embedding_weights.take(encoded_input, 0, self.strm);
            defer ref.deinit();

            if (first_pass)
                log.debug("looked up token embeddings (dimensions={any})", .{ref.shape()});

            for (self.transformer_layers.items) |*l| {
                const next_ref = try mlx.nn.Layer.forward(l, ref, self.strm);
                ref.deinit();
                ref = next_ref;
            }

            if (first_pass)
                log.debug("computed hidden states through {d} transformer layers (dimensions={any})", .{ self.transformer_layers.items.len, ref.shape() });

            try ref.rmsMut(self.norm, self.strm);

            if (first_pass)
                log.debug("applied final RMS normalization to hidden states (dimensions={any})", .{ref.shape()});

            const ref_shape = try ref.shape();
            const tokens_size = ref_shape[1];
            const logits_size = ref_shape[2];
            var logits = try ref.slice(&[_]c_int{ 0, tokens_size - 1, 0 }, &[_]c_int{ 1, tokens_size, logits_size }, &[_]c_int{ 1, 1, 1 }, self.strm);
            defer logits.deinit();

            if (first_pass)
                log.debug("selected the last token position's hidden state (dimensions={any})", .{logits.shape()});

            var projection = try self.embedding_weights.transpose(self.strm);
            defer projection.deinit();

            try logits.matMulMut(projection, self.strm);

            if (first_pass)
                log.debug("projected the last hidden state into vocabulary logits (dimensions={any})", .{logits.shape()});

            var id_tensor = try logits.argmax(-1, self.strm);
            defer id_tensor.deinit();

            const id = try id_tensor.itemU32();

            const token = try self.tokenizer.decode(self.gpa, &[1]u32{id});
            errdefer self.gpa.free(token);

            if (first_pass)
                log.debug("selected the next token ID with argmax (dimensions={any}, id={d}, text={s})", .{ id_tensor.shape(), id, token });

            if (id == 151645 or id == 151643) break;

            try encoded_input.concatMut(id_tensor, -1, self.strm);

            if (first_pass)
                log.debug("appended the generated token ID to the input sequence (dimensions={any})", .{encoded_input.shape()});

            _ = try output.put(self.io, &[1][]const u8{token}, 1);
        }
    }

    pub fn call(self: Self, input: []const u8) !std.ArrayList(u8) {
        var encoded_input = try self.preprocess(input);
        defer encoded_input.deinit();

        var buf: [32][]const u8 = undefined;
        var queue: Queue([]const u8) = .init(&buf);
        errdefer queue.close(self.io);

        var writer = std.Io.Writer.Allocating.init(self.gpa);
        defer writer.deinit();

        {
            const handle = try std.Thread.spawn(
                .{},
                models.collectStreamingResponse,
                .{ self.gpa, self.io, &queue, &writer.writer, models.CollectOptions{} },
            );
            defer {
                queue.close(self.io);
                handle.join();
            }
            try self.callStreaming(input, &queue);
        }

        return writer.toArrayList();
    }

    pub fn deinit(self: *Self) void {
        for (self.transformer_layers.items) |*layer| {
            layer.deinit();
        }
        self.transformer_layers.deinit(self.gpa);
        self.norm.deinit();
        self.embedding_weights.deinit();
        self.tokenizer.deinit();
        self.chat_template.deinit();
    }
};

pub const Handler = struct {
    pub const WEIGHTS_FILE = "model.safetensors";
    pub const TOKENIZER_FILE = "tokenizer.json";
    pub const TOKENIZER_CONFIG_FILE = "tokenizer_config.json";
    pub const GENERATION_CONFIG_FILE = "generation_config.json";

    pub fn inspect(ctx: Context, info: *models.ModelInfo) !void {
        const config = switch (info.config) {
            .qwen_config => |parsed| parsed.value,
            else => return error.WrongModelConfig,
        };
        std.debug.print("Model Config (config.json)\n", .{});
        std.debug.print("{s: <20}{s}\n", .{ "Model:", config.model_type });
        std.debug.print("{s: <20}{s}\n", .{ "Model dir:", info.model_dir });
        std.debug.print("\n", .{});
        if (config.architectures.len > 0) {
            std.debug.print("{s: <20}", .{"Architectures:"});
            std.debug.print("[", .{});
            for (config.architectures[0 .. config.architectures.len - 1]) |arch| {
                std.debug.print("{s}, ", .{arch});
            }
            std.debug.print("{s}]\n", .{config.architectures[config.architectures.len - 1]});
        }
        std.debug.print("{s: <20}{d}\n", .{ "Layers:", config.num_hidden_layers });
        std.debug.print("{s: <20}{d}\n", .{ "Hidden Size:", config.hidden_size });
        std.debug.print("{s: <20}{d}\n", .{ "MLP size:", config.intermediate_size });
        std.debug.print("{s: <20}{d}\n", .{ "Attention Heads:", config.num_attention_heads });
        std.debug.print("{s: <20}{d}\n", .{ "KV heads:", config.num_key_value_heads });
        std.debug.print("{s: <20}{d}\n", .{ "Head dimension:", config.hidden_size / config.num_attention_heads });
        std.debug.print("{s: <20}{d}\n", .{ "Vocab Size:", config.vocab_size });
        std.debug.print("{s: <20}{d}\n", .{ "RoPE theta:", config.rope_theta });
        std.debug.print("{s: <20}{e}\n", .{ "RMSNorm epsilon:", config.rms_norm_eps });
        std.debug.print("{s: <20}{s}\n", .{ "Tied embeddings:", if (config.tie_word_embeddings) "yes" else "no" });
        std.debug.print("{s: <20}{s}\n", .{ "Weight dtype:", config.torch_dtype });

        const safetensor_file_stat = try std.Io.Dir.cwd().statFile(ctx.io, info.safetensor_file_path, .{ .follow_symlinks = false });

        std.debug.print("\n", .{});
        std.debug.print("{s: <20}{s}\n", .{ "Checkpoint:", WEIGHTS_FILE });
        std.debug.print("{s: <20}{d}\n", .{ "Checkpoint bytes:", safetensor_file_stat.size });

        var embeddings = try info.safetensor_weights.get("model.embed_tokens.weight");
        defer embeddings.deinit();

        std.debug.print("{s: <20}{d}\n", .{ "Embeddings ndim:", try embeddings.ndim() });
        std.debug.print("{s: <20}{d}\n", .{ "Embeddings dim(0):", try embeddings.dim(0) });
        std.debug.print("{s: <20}{s}\n", .{ "Dtype:", @tagName(try embeddings.dtype()) });

        std.debug.print("\n", .{});
        std.debug.print("Safetensor Metadata:\n", .{});
        var metadata_it = try info.safetensor_metadata.iterator();
        defer metadata_it.deinit();

        while (metadata_it.next()) |entry| {
            const key = entry.key_ptr;
            const val = entry.val_ptr;
            std.debug.print("{s: <20}{s}\n", .{ key, val });
        }

        std.debug.print("\n", .{});
        std.debug.print("Safetensor tensors:\n", .{});
        var weights_it = try info.safetensor_weights.iterator();
        defer weights_it.deinit();

        while (weights_it.next()) |entry| {
            const key = entry.key_ptr;
            var val = entry.val_ptr;
            defer val.deinit();
            std.debug.print("> {s:<50} {any} {s}\n", .{ key, try val.shape(), @tagName(try val.dtype()) });
        }
    }

    fn getChatTemplate(ctx: Context, tokenizer_config_file_path: []const u8) !ChatTemplate {
        const json = try std.Io.Dir.cwd().readFileAlloc(ctx.io, tokenizer_config_file_path, ctx.gpa, .unlimited);
        defer ctx.gpa.free(json);

        const tokenizer_config = try std.json.parseFromSlice(TokenizerConfig, ctx.gpa, json, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        defer tokenizer_config.deinit();

        return .init(ctx.gpa, tokenizer_config.value.chat_template);
    }

    pub fn buildPipeline(ctx: Context, info: *const models.ModelInfo) !Pipeline {
        const config = switch (info.config) {
            .qwen_config => |parsed| parsed.value,
            .empty => return error.MissingModelConfig,
        };
        var layers: std.ArrayList(mlx.nn.TransformerBlock) = .empty;
        errdefer {
            for (layers.items) |*layer| {
                layer.deinit();
            }
            layers.deinit(ctx.gpa);
        }

        for (0..config.num_hidden_layers) |i| {
            var block = try mlx.nn.TransformerBlock.init(info.safetensor_weights, i, .{ .num_attention_heads = config.num_attention_heads, .num_key_value_heads = config.num_key_value_heads, .rope_theta = @floatCast(config.rope_theta) });
            errdefer block.deinit();
            try layers.append(ctx.gpa, block);
        }

        var norm = try info.safetensor_weights.get("model.norm.weight");
        errdefer norm.deinit();
        var embedding_weights = try info.safetensor_weights.get("model.embed_tokens.weight");
        errdefer embedding_weights.deinit();

        var tokenizer = try Tokenizer.fromFile(ctx, info.tokenizer_file_path);
        errdefer tokenizer.deinit();

        return .{
            .gpa = ctx.gpa,
            .io = ctx.io,
            .strm = ctx.stream,
            .tokenizer = tokenizer,
            .transformer_layers = layers,
            .norm = norm,
            .embedding_weights = embedding_weights,
            .chat_template = try getChatTemplate(ctx, info.tokenizer_config_file_path),
        };
    }
};

test "Qwen checkpoint generates matching owned buffered and streamed responses" {
    const build_options = @import("build_options");
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    var strm = try mlx.init(.{ .require_gpu = build_options.metal });
    defer strm.deinit();
    const ctx: Context = .{ .gpa = gpa, .io = io, .stream = strm };

    var buffered, var streamed = blk: {
        var info = models.getModelInfo(ctx, "qwen") catch |err| switch (err) {
            error.FileNotFound => return error.SkipZigTest,
            else => return err,
        };
        defer info.deinit(gpa);
        const config = switch (info.config) {
            .qwen_config => |parsed| parsed.value,
            .empty => return error.MissingModelConfig,
        };
        var pipeline = try Handler.buildPipeline(ctx, &info);
        defer pipeline.deinit();
        try std.testing.expect(config.num_hidden_layers > 0);
        try std.testing.expectEqual(@as(usize, config.num_hidden_layers), pipeline.transformer_layers.items.len);

        var response = try pipeline.call("hi");
        errdefer response.deinit(gpa);

        var writer = std.Io.Writer.Allocating.init(gpa);
        defer writer.deinit();
        var buffer: [2][]const u8 = undefined;
        var queue: Queue([]const u8) = .init(&buffer);
        {
            const handle = try std.Thread.spawn(
                .{},
                models.collectStreamingResponse,
                .{ gpa, io, &queue, &writer.writer, models.CollectOptions{} },
            );
            defer {
                queue.close(io);
                handle.join();
            }
            try pipeline.callStreaming("hi", &queue);
        }
        try std.testing.expectError(error.Closed, queue.getOne(io));
        break :blk .{ response, writer.toArrayList() };
    };
    defer buffered.deinit(gpa);
    defer streamed.deinit(gpa);

    // The returned text must remain valid after the pipeline and model are freed.
    try std.testing.expect(buffered.items.len > 0);
    try std.testing.expect(std.unicode.utf8ValidateSlice(buffered.items));
    try std.testing.expectEqualStrings(buffered.items, streamed.items);
}
