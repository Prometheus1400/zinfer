const mlx = @import("mlx");
const models = @import("root.zig");
const std = @import("std");
const template = @import("chat_template");

const ChatTemplate = template.ChatTemplate;
const Context = @import("../context.zig").Context;
const Queue = std.Io.Queue;
const Stream = mlx.Stream;
const Tensor = mlx.Tensor;
const Tokenizer = @import("../tokenizer.zig").Tokenizer;

const log = std.log.scoped(.qwen);

/// Maps the the config.json for qwen2
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

pub const Handler = struct {
    pub const WEIGHTS_FILE = "model.safetensors";
    pub const TOKENIZER_FILE = "tokenizer.json";
    pub const TOKENIZER_CONFIG_FILE = "tokenizer_config.json";
    pub const GENERATION_CONFIG_FILE = "generation_config.json";

    pub fn inspect(ctx: Context, info: *models.ModelInfo, writer: *std.Io.Writer) !void {
        const config = switch (info.config) {
            .qwen_config => |parsed| parsed.value,
            else => return error.WrongModelConfig,
        };
        try writer.print("Model Config (config.json)\n", .{});
        try writer.print("{s: <20}{s}\n", .{ "Model:", config.model_type });
        try writer.print("{s: <20}{s}\n", .{ "Model dir:", info.model_dir });
        try writer.print("\n", .{});
        if (config.architectures.len > 0) {
            try writer.print("{s: <20}", .{"Architectures:"});
            try writer.print("[", .{});
            for (config.architectures[0 .. config.architectures.len - 1]) |arch| {
                try writer.print("{s}, ", .{arch});
            }
            try writer.print("{s}]\n", .{config.architectures[config.architectures.len - 1]});
        }
        try writer.print("{s: <20}{d}\n", .{ "Layers:", config.num_hidden_layers });
        try writer.print("{s: <20}{d}\n", .{ "Hidden Size:", config.hidden_size });
        try writer.print("{s: <20}{d}\n", .{ "MLP size:", config.intermediate_size });
        try writer.print("{s: <20}{d}\n", .{ "Attention Heads:", config.num_attention_heads });
        try writer.print("{s: <20}{d}\n", .{ "KV heads:", config.num_key_value_heads });
        try writer.print("{s: <20}{d}\n", .{ "Head dimension:", config.hidden_size / config.num_attention_heads });
        try writer.print("{s: <20}{d}\n", .{ "Vocab Size:", config.vocab_size });
        try writer.print("{s: <20}{d}\n", .{ "RoPE theta:", config.rope_theta });
        try writer.print("{s: <20}{e}\n", .{ "RMSNorm epsilon:", config.rms_norm_eps });
        try writer.print("{s: <20}{s}\n", .{ "Tied embeddings:", if (config.tie_word_embeddings) "yes" else "no" });
        try writer.print("{s: <20}{s}\n", .{ "Weight dtype:", config.torch_dtype });

        const safetensor_file_stat = try std.Io.Dir.cwd().statFile(ctx.io, info.safetensor_file_path, .{ .follow_symlinks = false });

        try writer.print("\n", .{});
        try writer.print("{s: <20}{s}\n", .{ "Checkpoint:", WEIGHTS_FILE });
        try writer.print("{s: <20}{d}\n", .{ "Checkpoint bytes:", safetensor_file_stat.size });

        var embeddings = try info.safetensor_weights.get("model.embed_tokens.weight");
        defer embeddings.deinit();

        try writer.print("{s: <20}{d}\n", .{ "Embeddings ndim:", try embeddings.ndim() });
        try writer.print("{s: <20}{d}\n", .{ "Embeddings dim(0):", try embeddings.dim(0) });
        try writer.print("{s: <20}{s}\n", .{ "Dtype:", @tagName(try embeddings.dtype()) });

        try writer.print("\n", .{});
        try writer.print("Safetensor Metadata:\n", .{});
        var metadata_it = try info.safetensor_metadata.iterator();
        defer metadata_it.deinit();

        while (metadata_it.next()) |entry| {
            const key = entry.key_ptr;
            const val = entry.val_ptr;
            try writer.print("{s: <20}{s}\n", .{ key, val });
        }

        try writer.print("\n", .{});
        try writer.print("Safetensor tensors:\n", .{});
        var weights_it = try info.safetensor_weights.iterator();
        defer weights_it.deinit();

        while (weights_it.next()) |entry| {
            const key = entry.key_ptr;
            var val = entry.val_ptr;
            defer val.deinit();
            try writer.print("> {s:<50} {any} {s}\n", .{ key, try val.shape(), @tagName(try val.dtype()) });
        }
    }

    fn getChatTemplate(ctx: Context, tokenizer_config_file_path: []const u8) !ChatTemplate {
        const json = try std.Io.Dir.cwd().readFileAlloc(ctx.io, tokenizer_config_file_path, ctx.gpa, .unlimited);
        defer ctx.gpa.free(json);

        const tokenizer_config = try std.json.parseFromSlice(TokenizerConfig, ctx.gpa, json, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        defer tokenizer_config.deinit();

        return .init(ctx.gpa, tokenizer_config.value.chat_template);
    }

    // TODO: Apply max_new_tokens and kv_caching_strategy from pipeline_args to the returned pipeline.
    // zig fmt: off
    pub fn buildPipeline(
        ctx: Context, 
        info: *const models.ModelInfo,
        pipeline_args: struct {
            /// Zero selects greedy decoding; positive values scale logits before sampling.
            temperature: f32 = 0.7,
            repetition_penalty: f32 = 1.2,
            max_new_tokens: u32 = 2048,
            kv_caching_strategy: enum {
                none,
                simple
            } = .simple,
        }
    ) !Pipeline {
    // zig fmt: on
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

        var kv_cache: std.ArrayList(KVCachePair) = try .initCapacity(ctx.gpa, config.num_hidden_layers);
        for (0..config.num_hidden_layers) |_| {
            try kv_cache.append(ctx.gpa, .{});
        }

        return .{
            .gpa = ctx.gpa,
            .io = ctx.io,
            .strm = ctx.stream,
            .tokenizer = tokenizer,
            .transformer_layers = layers,
            .norm = norm,
            .embedding_weights = embedding_weights,
            .chat_template = try getChatTemplate(ctx, info.tokenizer_config_file_path),
            .stop_tokens = try ctx.gpa.dupe(u32, info.generation_config.value.eos_token_id),
            // .pipeline_generation_config = .{ .generation_config = info.generation_config.value },
            .kv_cache = kv_cache,
            .temperature = pipeline_args.temperature,
            .repetition_penalty = pipeline_args.repetition_penalty,
        };
    }
};

// zig fmt: off
const TokenizerConfig = struct { 
    chat_template: []const u8,
    additional_special_tokens: [][]const u8,
    model_max_length: u32,
    eos_token: []const u8,
    pad_token: []const u8 
};
// zig fmt: on

const KVCachePair = struct {
    k: ?Tensor = null,
    v: ?Tensor = null,
};

pub const Pipeline = struct {
    const Self = @This();

    gpa: std.mem.Allocator,
    io: std.Io,
    strm: Stream,
    tokenizer: Tokenizer,
    transformer_layers: std.ArrayList(mlx.nn.TransformerBlock),
    norm: Tensor,
    embedding_weights: Tensor,
    chat_template: ChatTemplate,

    kv_cache: std.ArrayList(KVCachePair),
    stop_tokens: []const u32,
    temperature: f32,
    max_new_tokens: u32 = 512,
    repetition_penalty: f32,
    needs_prefill: bool = true,

    pub fn deinit(self: *Self) void {
        for (self.transformer_layers.items) |*layer| {
            layer.deinit();
        }
        self.transformer_layers.deinit(self.gpa);
        self.norm.deinit();
        self.embedding_weights.deinit();
        self.tokenizer.deinit();
        self.chat_template.deinit();
        for (self.kv_cache.items) |*tensor| {
            if (tensor.k) |*k| k.deinit();
            if (tensor.v) |*v| v.deinit();
        }
        self.kv_cache.deinit(self.gpa);
        self.gpa.free(self.stop_tokens);
    }

    /// Applies the chat template and tokenizes
    fn preprocess(self: Self, input: []const u8) !Tensor {
        const messages: []const template.Message = &.{
            .{ .role = "user", .content = input },
        };

        const initial_prompt = try self.chat_template.render(self.gpa, messages, .{});
        defer self.gpa.free(initial_prompt);

        return self.tokenizer.encode(initial_prompt);
    }

    fn forward(self: Self, input: Tensor, options: struct { generated_tokens: *Tensor, prefill: bool }) !?Tensor {
        // Look up an embedding vector for each input token ID.
        var x = try self.embedding_weights.take(input, 0, self.strm);
        defer x.deinit();

        // Pass the hidden vectors through each transformer layer, updating its KV cache.
        for (self.transformer_layers.items, 0..) |*l, i| {
            const layer_cache = &self.kv_cache.items[i];
            const transformer_layer_output = try l.forward(x, self.strm, .{ .prefill = options.prefill, .cached_k = &layer_cache.k, .cached_v = &layer_cache.v });
            x.deinit();
            x = transformer_layer_output;
        }

        // Apply the final RMSNorm.
        try x.rmsMut(self.norm, self.strm);

        // Shape: [batch, token count, hidden size].
        const ref_shape = try x.shape();
        const tokens_size = ref_shape[1];
        const logits_size = ref_shape[2];

        // Select the last input position's hidden vector to predict the next token.
        var last_hidden_state = try x.slice(&[_]c_int{ 0, tokens_size - 1, 0 }, &[_]c_int{ 1, tokens_size, logits_size }, &[_]c_int{ 1, 1, 1 }, self.strm);
        defer last_hidden_state.deinit();

        var projection = try self.embedding_weights.transpose(self.strm);
        defer projection.deinit();

        // Project the hidden vector to vocabulary logits using the shared embedding weights.
        var next_token_logits = try last_hidden_state.matMul(projection, self.strm);
        defer next_token_logits.deinit();

        // repetition penalty
        // zig fmt: off
        var prev_gen_tokens_probs = try next_token_logits.gather(
            &[_]Tensor{options.generated_tokens.*},
            &[_]c_int{2},
            &[_]c_int{1, 1, 1},
            self.strm
        );
        // zig fmt: on
        defer prev_gen_tokens_probs.deinit();

        var penalty = Tensor.scalar(self.repetition_penalty);
        defer penalty.deinit();
        var positive_penalty_results = try prev_gen_tokens_probs.div(penalty, self.strm);
        defer positive_penalty_results.deinit();
        var negative_penalty_results = try prev_gen_tokens_probs.mul(penalty, self.strm);
        defer negative_penalty_results.deinit();

        const zeros = Tensor.scalar(0);
        var mask = try prev_gen_tokens_probs.greaterEqual(zeros, self.strm);
        defer mask.deinit();

        var updated_prev_gen_tokens_probs = try Tensor.where(mask, positive_penalty_results, negative_penalty_results, self.strm);
        defer updated_prev_gen_tokens_probs.deinit();

        var penalty_applied_next_token_logits = try next_token_logits.scatter(&[_]Tensor{options.generated_tokens.*}, updated_prev_gen_tokens_probs, &[_]c_int{2}, self.strm);
        defer penalty_applied_next_token_logits.deinit();

        var id_tensor = blk: {
            if (self.temperature == 0) {
                // Greedy decoding selects the highest-scoring token deterministically.
                break :blk try penalty_applied_next_token_logits.argmax(-1, self.strm);
            } else {
                var temp = Tensor.scalar(self.temperature);
                defer temp.deinit();
                try penalty_applied_next_token_logits.divMut(temp, self.strm);
                // TODO: top-k and top-p sampling
                break :blk try penalty_applied_next_token_logits.categorical(-1, self.strm);
            }
        };

        try options.generated_tokens.concatMut(id_tensor, 1, self.strm);

        if (std.mem.findScalar(u32, self.stop_tokens, try id_tensor.itemU32()) != null) {
            defer id_tensor.deinit();
            return null;
        }

        return id_tensor;
    }

    fn emitToken(self: Self, token_tensor: Tensor, output: *Queue([]const u8)) !void {
        const id = try token_tensor.itemU32();
        const token = try self.tokenizer.decode(self.gpa, &[1]u32{id});
        errdefer self.gpa.free(token);
        _ = try output.put(self.io, &[1][]const u8{token}, 1);
    }

    pub fn callStreaming(self: Self, input: []const u8, output: *Queue([]const u8)) !void {
        log.debug("generating response for: \"{s}...\"", .{input[0..@min(25, input.len)]});
        defer output.close(self.io);

        var encoded_input = try self.preprocess(input);
        defer encoded_input.deinit();

        log.debug("encoded prompt into token IDs (dimensions={any})", .{encoded_input.shape()});

        var generated_tokens = try Tensor.zeros(&[_]c_int{ 1, 0 }, .int32, self.strm);
        defer generated_tokens.deinit();

        var token_tensor = try self.forward(encoded_input, .{ .generated_tokens = &generated_tokens, .prefill = true }) orelse return;
        defer token_tensor.deinit();
        try self.emitToken(token_tensor, output);

        for (1..self.max_new_tokens) |_| {
            const output_token_tensor = try self.forward(token_tensor, .{ .generated_tokens = &generated_tokens, .prefill = false }) orelse return;
            try self.emitToken(output_token_tensor, output);

            token_tensor.deinit();
            token_tensor = output_token_tensor;
        }

        log.debug("max_new_tokens = {d} limit reached", .{self.max_new_tokens});
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
            const collect_streaming_response_args: std.meta.ArgsTuple(@TypeOf(models.collectStreamingResponse)) = .{ self.gpa, self.io, &queue, &writer.writer, .{} };
            const handle = try std.Thread.spawn(.{}, models.collectStreamingResponse, collect_streaming_response_args);
            defer {
                queue.close(self.io);
                handle.join();
            }
            try self.callStreaming(input, &queue);
        }

        return writer.toArrayList();
    }
};
