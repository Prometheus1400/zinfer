const std = @import("std");
const mlx = @import("mlx");
const Context = @import("context").Context;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const template = @import("chat_template");
const ChatTemplate = @import("chat_template").ChatTemplate;
const Queue = @import("std").Io.Queue;
const log = std.log.scoped(.models);
const model_utils = @import("model_utils.zig");

pub const ModelInfo = struct {
    model_dir: []const u8,
    safetensor_file_path: [:0]const u8,
    tokenizer_file_path: []const u8,
    tokenizer_config_file_path: []const u8,
    generation_config_file_path: []const u8,
    config: ModelConfig,
    safetensor_metadata: mlx.Map(.string),
    safetensor_weights: mlx.Map(.tensor),

    pub fn deinit(self: *@This(), gpa: std.mem.Allocator) void {
        self.safetensor_metadata.deinit();
        self.safetensor_weights.deinit();
        gpa.free(self.model_dir);
        gpa.free(self.safetensor_file_path);
        gpa.free(self.tokenizer_file_path);
        gpa.free(self.tokenizer_config_file_path);
        gpa.free(self.generation_config_file_path);
        self.config.deinit();
    }
};

pub const ModelConfig = union(enum) {
    const Self = @This();

    qwen_config: std.json.Parsed(QwenConfig),
    empty,

    pub fn deinit(self: *Self) void {
        switch (self.*) {
            .qwen_config => |config| {
                config.deinit();
            },
            .empty => {},
        }

        self.* = .empty;
    }
};

pub const CommonConfig = struct {
    model_type: []const u8,
};

pub const QwenConfig = struct {
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

pub const TokenizerConfig = struct {
    chat_template: []const u8,
};

pub const QwenPipeline = struct {
    const Self = @This();

    gpa: std.mem.Allocator,
    io: std.Io,
    strm: mlx.Stream,
    tokenizer: Tokenizer,
    transformer_layers: std.ArrayList(mlx.nn.TransformerBlock),
    norm: mlx.Tensor,
    embedding_weights: mlx.Tensor,
    chat_template: ChatTemplate,

    pub fn preprocess(self: Self, input: []const u8) !mlx.Tensor {
        const messages: []const template.Message = &.{
            .{ .role = "user", .content = input },
        };

        const initial_prompt = try self.chat_template.render(self.gpa, messages, .{});
        defer self.gpa.free(initial_prompt);

        return self.tokenizer.encode(initial_prompt);
    }

    pub fn callStreaming(self: Self, input: []const u8, output: *Queue([]const u8)) !void {
        defer output.close(self.io);

        var encoded_input = try self.preprocess(input);
        defer encoded_input.deinit();

        while (true) {
            var ref = try self.embedding_weights.take(encoded_input, 0, self.strm);
            defer ref.deinit();

            for (self.transformer_layers.items) |*l| {
                const next_ref = try mlx.nn.Layer.forward(l, ref, self.strm);
                ref.deinit();
                ref = next_ref;
            }

            try ref.rmsMut(self.norm, self.strm);

            const ref_shape = try ref.shape();
            const tokens = ref_shape[1];
            const logits = ref_shape[2];
            var final_logits = try ref.slice(&[_]c_int{ 0, tokens - 1, 0 }, &[_]c_int{ 1, tokens, logits }, &[_]c_int{ 1, 1, 1 }, self.strm);
            defer final_logits.deinit();

            var projection = try self.embedding_weights.transpose(self.strm);
            defer projection.deinit();

            try final_logits.matMulMut(projection, self.strm);

            var id_tensor = try final_logits.argmax(-1, self.strm);
            defer id_tensor.deinit();

            const id = try id_tensor.itemU32();
            if (id == 151645 or id == 151643) break;

            try encoded_input.concatMut(id_tensor, -1, self.strm);

            const token = try self.tokenizer.decode(self.gpa, &[1]u32{id});
            errdefer self.gpa.free(token);

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
                model_utils.collectStreamingResponse,
                .{ self.gpa, self.io, &queue, &writer.writer, model_utils.CollectOptions{} },
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

pub const QwenHandler = struct {
    pub const TransformerBlock = @import("models/qwen/transformer.zig").TransformerBlock;

    pub const WEIGHTS_FILE = "model.safetensors";
    pub const TOKENIZER_FILE = "tokenizer.json";
    pub const TOKENIZER_CONFIG_FILE = "tokenizer_config.json";
    pub const GENERATION_CONFIG_FILE = "generation_config.json";

    pub fn inspect(ctx: Context, info: *ModelInfo) !void {
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

    pub fn buildPipeline(ctx: Context, info: *const ModelInfo) !QwenPipeline {
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

pub fn getModelInfo(ctx: Context, model: []const u8) !ModelInfo {
    const model_dir_path = try ctx.gpa.print("{s}/{s}", .{ "models", model });
    errdefer ctx.gpa.free(model_dir_path);

    const model_dir = try std.Io.Dir.cwd().openDir(ctx.io, model_dir_path, .{});
    defer model_dir.close(ctx.io);

    const text = try model_dir.readFileAlloc(ctx.io, "config.json", ctx.gpa, .unlimited);
    defer ctx.gpa.free(text);

    const common_config = try std.json.parseFromSlice(CommonConfig, ctx.gpa, text, .{ .ignore_unknown_fields = true });
    defer common_config.deinit();

    if (std.mem.eql(u8, common_config.value.model_type, "qwen2")) {
        var qwen_config = try std.json.parseFromSlice(QwenConfig, ctx.gpa, text, .{
            .ignore_unknown_fields = true,
            .allocate = .alloc_always,
        });
        errdefer qwen_config.deinit();
        const weights_file_path = try std.mem.concatWithSentinel(ctx.gpa, u8, &.{ model_dir_path, "/", QwenHandler.WEIGHTS_FILE }, 0);
        errdefer ctx.gpa.free(weights_file_path);
        const tokenizer_file_path = try ctx.gpa.print("{s}/{s}", .{ model_dir_path, QwenHandler.TOKENIZER_FILE });
        errdefer ctx.gpa.free(tokenizer_file_path);
        const tokenizer_config_file_path = try ctx.gpa.print("{s}/{s}", .{ model_dir_path, QwenHandler.TOKENIZER_CONFIG_FILE });
        errdefer ctx.gpa.free(tokenizer_config_file_path);
        const generation_config_file_path = try ctx.gpa.print("{s}/{s}", .{ model_dir_path, QwenHandler.GENERATION_CONFIG_FILE });
        errdefer ctx.gpa.free(generation_config_file_path);
        var metadata, var weights = try mlx.io.loadSafetensors(ctx.stream, weights_file_path);
        errdefer metadata.deinit();
        errdefer weights.deinit();
        return .{
            .model_dir = model_dir_path,
            .safetensor_file_path = weights_file_path,
            .tokenizer_file_path = tokenizer_file_path,
            .tokenizer_config_file_path = tokenizer_config_file_path,
            .generation_config_file_path = generation_config_file_path,
            .config = .{ .qwen_config = qwen_config },
            .safetensor_metadata = metadata,
            .safetensor_weights = weights,
        };
    }

    return error.UnknownModel;
}

test "ModelInfo owns its paths parsed config and safetensors" {
    var strm = try mlx.init(.{});
    defer strm.deinit();
    const ctx: Context = .{ .gpa = std.testing.allocator, .io = std.testing.io, .stream = strm };
    _ = std.Io.Dir.cwd().statFile(ctx.io, "models/qwen/model.safetensors", .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    var info = getModelInfo(ctx, "qwen") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer info.deinit(ctx.gpa);

    try std.testing.expectEqualStrings("models/qwen", info.model_dir);
    try std.testing.expectEqualStrings("models/qwen/model.safetensors", info.safetensor_file_path);
    try std.testing.expectEqualStrings("models/qwen/tokenizer.json", info.tokenizer_file_path);
    try std.testing.expectEqualStrings("models/qwen/tokenizer_config.json", info.tokenizer_config_file_path);
    try std.testing.expectEqualStrings("models/qwen/generation_config.json", info.generation_config_file_path);
    try std.testing.expectEqual(@as(u8, 0), info.safetensor_file_path[info.safetensor_file_path.len]);
    try info.safetensor_metadata.ensure();
    var embeddings = try info.safetensor_weights.get("model.embed_tokens.weight");
    defer embeddings.deinit();
    try std.testing.expectEqualSlices(c_int, &.{ 151936, 896 }, try embeddings.shape());
    switch (info.config) {
        .qwen_config => |config| {
            try std.testing.expectEqualStrings("qwen2", config.value.model_type);
            try std.testing.expectEqualStrings("Qwen2ForCausalLM", config.value.architectures[0]);
        },
        .empty => return error.MissingModelConfig,
    }
}

test "Qwen checkpoint pipeline produces causal logits through all layers" {
    const c = @import("mlx_c");
    const build_options = @import("build_options");
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const model_dir_path = "models/qwen";
    _ = std.Io.Dir.cwd().statFile(io, model_dir_path ++ "/model.safetensors", .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    const json = try std.Io.Dir.cwd().readFileAlloc(io, model_dir_path ++ "/config.json", gpa, .unlimited);
    defer gpa.free(json);
    const parsed = try std.json.parseFromSlice(QwenConfig, gpa, json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const config = parsed.value;

    var strm = try mlx.init(.{ .require_gpu = build_options.metal });
    defer strm.deinit();
    const ctx: Context = .{ .gpa = gpa, .io = io, .stream = strm };

    const tokens = 3;
    const hidden_size: usize = config.hidden_size;
    const shape = [_]c_int{ 1, tokens, @intCast(hidden_size) };
    const data = try gpa.alloc(f32, tokens * hidden_size);
    defer gpa.free(data);
    for (data, 0..) |*value, i| {
        value.* = (@as(f32, @floatFromInt(i % 29)) - 14.0) * 0.01;
    }
    const changed_data = try gpa.dupe(f32, data);
    defer gpa.free(changed_data);
    for (changed_data[2 * hidden_size ..], 0..) |*value, i| {
        value.* += if (i % 2 == 0) @as(f32, 0.2) else -0.15;
    }

    // The pipeline accepts hidden states and returns vocabulary logits.
    var output, var changed_output = blk: {
        var info = try getModelInfo(ctx, "qwen");
        defer info.deinit(gpa);
        var pipeline = try QwenHandler.buildPipeline(ctx, &info);
        defer pipeline.deinit();
        try std.testing.expect(config.num_hidden_layers > 0);
        try std.testing.expectEqual(@as(usize, config.num_hidden_layers), pipeline.transformer_layers.items.len);

        var input = try mlx.Tensor.fromSlice(f32, data, &shape);
        defer input.deinit();
        var changed_input = try mlx.Tensor.fromSlice(f32, changed_data, &shape);
        defer changed_input.deinit();
        var result = try pipeline.call(input);
        errdefer result.deinit();
        const changed_result = try pipeline.call(changed_input);
        break :blk .{ result, changed_result };
    };
    defer output.deinit();
    defer changed_output.deinit();

    // Evaluate after releasing ModelInfo, the pipeline, inputs, and
    // intermediate handles to check that the graph retains its dependencies.
    const output_shape = [_]c_int{ 1, tokens, @intCast(config.vocab_size) };
    try std.testing.expectEqualSlices(c_int, &output_shape, try output.shape());
    try std.testing.expectEqualSlices(c_int, &output_shape, try changed_output.shape());
    try std.testing.expectEqual(mlx.DType.float32, try output.dtype());
    try std.testing.expectEqual(mlx.DType.float32, try changed_output.dtype());
    if (c.mlx_array_eval(output.p) != 0 or c.mlx_array_eval(changed_output.p) != 0)
        return error.MlxEvaluationFailed;
    const actual = c.mlx_array_data_float32(output.p);
    const changed = c.mlx_array_data_float32(changed_output.p);
    const strides = c.mlx_array_strides(output.p);
    const changed_strides = c.mlx_array_strides(changed_output.p);
    var final_token_changed = false;
    for (0..tokens) |token| {
        for (0..config.vocab_size) |feature| {
            const value = actual[token * strides[1] + feature * strides[2]];
            const changed_value = changed[token * changed_strides[1] + feature * changed_strides[2]];
            try std.testing.expect(std.math.isFinite(value));
            try std.testing.expect(std.math.isFinite(changed_value));
            if (token < 2) {
                try std.testing.expectApproxEqAbs(value, changed_value, 1e-4);
            } else if (@abs(value - changed_value) > 1e-4) {
                final_token_changed = true;
            }
        }
    }
    try std.testing.expect(final_token_changed);
}
