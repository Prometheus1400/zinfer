const mlx = @import("mlx");
const models = @import("root.zig");
const std = @import("std");
const template = @import("chat_template");
const common = @import("common.zig");

const cache = mlx.nn.cache;

const ChatTemplate = template.ChatTemplate;
const Context = @import("../context.zig").Context;
const ConversationState = @import("conversation_state.zig").ConversationState;
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

const TokenizerConfig = struct {
    chat_template: []const u8,
    additional_special_tokens: [][]const u8,
    model_max_length: u32,
    eos_token: []const u8,
    pad_token: []const u8,
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
                simple,
            } = .simple,
        },
    ) !Pipeline {
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
            .stop_tokens = try ctx.gpa.dupe(u32, info.generation_config.value.eos_token_id),
            .conversation_states = .init(ctx.gpa),
            .temperature = pipeline_args.temperature,
            .repetition_penalty = pipeline_args.repetition_penalty,
            .pad_token = 69, // TODO
        };
    }
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

    stop_tokens: []const u32,
    pad_token: u32,
    temperature: f32,
    max_new_tokens: u32 = 512,
    repetition_penalty: f32,
    conversation_states: std.StringHashMap(ConversationState),

    pub fn deinit(self: *Self) void {
        for (self.transformer_layers.items) |*layer| {
            layer.deinit();
        }
        self.transformer_layers.deinit(self.gpa);
        self.norm.deinit();
        self.embedding_weights.deinit();
        self.tokenizer.deinit();
        self.chat_template.deinit();
        self.gpa.free(self.stop_tokens);

        var conversation_states_iterator = self.conversation_states.iterator();
        while (conversation_states_iterator.next()) |entry| {
            // self.gpa.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.gpa);
        }
        self.conversation_states.deinit();
    }

    fn padBatch(self: Self, batch: []const []const c_int) ![][]const c_int {
        var max_seq_length: usize = 0;
        for (batch) |row| {
            max_seq_length = @max(max_seq_length, row.len);
        }

        var padded_batch = try std.ArrayList([]const c_int).initCapacity(self.gpa, batch.len);
        errdefer {
            for (padded_batch.items) |row| {
                self.gpa.free(row);
            }
            padded_batch.deinit(self.gpa);
        }

        for (batch) |row| {
            const offset = max_seq_length - row.len;
            const new_row_buf = try self.gpa.alloc(c_int, max_seq_length);
            @memset(new_row_buf[0..offset], @intCast(self.pad_token));
            @memcpy(new_row_buf[offset..], row);
            padded_batch.appendAssumeCapacity(new_row_buf);
        }

        return padded_batch.toOwnedSlice(self.gpa);
    }

    /// Copies a rectangular batch of token IDs into a tensor.
    fn batchToTensor(self: Self, batch: []const []const c_int) !Tensor {
        if (batch.len == 0) return error.EmptyBatch;
        const max_seq_len = batch[0].len;
        const flattened = try self.gpa.alloc(c_int, batch.len * max_seq_len);
        defer self.gpa.free(flattened);
        for (batch, 0..) |row, i| {
            if (row.len != max_seq_len) return error.InconsistentMaxSeqLen;
            @memcpy(flattened[i * max_seq_len .. i * max_seq_len + max_seq_len], row);
        }
        return try mlx.Tensor.fromSlice(c_int, flattened, &[_]c_int{ @intCast(batch.len), @intCast(max_seq_len) });
    }

    /// Renders the chat template and tokenizes
    fn preprocess(self: Self, batch: []const common.Conversation) ![][]const c_int {
        var encodings = std.ArrayList([]const c_int).empty;
        errdefer {
            for (encodings.items) |encoding| {
                self.gpa.free(encoding);
            }
            encodings.deinit(self.gpa);
        }
        var arena = std.heap.ArenaAllocator.init(self.gpa);
        defer arena.deinit();
        for (batch) |conversation| {
            const prompt = try self.chat_template.render(arena.allocator(), conversation.messages, .{});
            try encodings.append(self.gpa, try self.tokenizer.encode(self.gpa, prompt));
        }
        return try encodings.toOwnedSlice(self.gpa);
    }

    /// Returns newly sampled token IDs owned by the caller; free with self.gpa.
    /// Borrows token rows and generated histories in conversation_ids order.
    fn forward(
        self: *Self,
        batch: []const []const c_int,
        conversation_ids: []const []const u8,
        generated_token_ids: []const []const c_int,
        opts: struct {
            prefill: bool,
        },
    ) ![]const c_int {
        if (batch.len != conversation_ids.len or batch.len != generated_token_ids.len) {
            return error.InconsistentForwardOptions;
        }

        const uncached_token_ids = try self.uncached_tokens_batch(batch, conversation_ids);
        defer self.gpa.free(uncached_token_ids);

        const real_token_counts = try self.gpa.alloc(usize, batch.len);
        defer self.gpa.free(real_token_counts);
        for (uncached_token_ids, 0..) |row, i| {
            real_token_counts[i] = row.len;
        }

        const padded_uncached_token_ids = try self.padBatch(uncached_token_ids);
        defer {
            for (padded_uncached_token_ids) |row| {
                self.gpa.free(row);
            }
            self.gpa.free(padded_uncached_token_ids);
        }
        var input = try self.batchToTensor(padded_uncached_token_ids);
        defer input.deinit();

        if (opts.prefill) {
            log.debug("encoded prompt into token IDs (dimensions={any})", .{input.shape()});
        }

        var sampled = try self.forwardTensor(input, .{
            .generated_token_ids = generated_token_ids,
            .prefill = opts.prefill,
            .conversation_ids = conversation_ids,
            .real_token_counts = real_token_counts,
        });
        defer sampled.deinit();

        const token_ids = try self.gpa.alloc(c_int, try sampled.dim(0));
        errdefer self.gpa.free(token_ids);
        for (token_ids, 0..) |*id, r| {
            const row: c_int = @intCast(r);
            var sampled_row = try sampled.slice(&.{ row, 0 }, &.{ row + 1, 1 }, &.{ 1, 1 }, self.strm);
            defer sampled_row.deinit();
            id.* = @intCast(try sampled_row.itemU32());
        }

        // Only processed inputs are represented in the KV cache.
        for (uncached_token_ids, conversation_ids) |row, id| {
            const state = self.conversation_states.getPtr(id) orelse return error.NoConversationState;
            try state.cached_token_ids.appendSlice(self.gpa, row);
        }
        return token_ids;
    }

    fn forwardTensor(
        self: Self,
        input: Tensor,
        ctx: struct {
            generated_token_ids: []const []const c_int,
            prefill: bool,
            conversation_ids: []const []const u8,
            real_token_counts: []const usize,
        },
    ) !Tensor {
        // Look up an embedding vector for each input token ID.
        var x = try self.embedding_weights.take(input, 0, self.strm);
        defer x.deinit();

        // Pass the hidden vectors through each transformer layer, updating its KV cache.
        const layer_caches = try self.gpa.alloc(*cache.ContiguousKVCache, ctx.conversation_ids.len);
        defer self.gpa.free(layer_caches);
        for (self.transformer_layers.items, 0..) |*l, i| {
            for (ctx.conversation_ids, 0..) |id, b| {
                const state = self.conversation_states.getPtr(id) orelse return error.NoConversationState;
                layer_caches[b] = &state.kv_cache.items[i];
            }
            const transformer_layer_output = try l.forward(
                x,
                self.strm,
                .{
                    .gpa = self.gpa,
                    .prefill = ctx.prefill,
                    .kv_caches = layer_caches,
                    .real_token_counts = ctx.real_token_counts,
                },
            );
            x.deinit();
            x = transformer_layer_output;
        }

        // Apply the final RMSNorm.
        try x.rmsMut(self.norm, self.strm);

        // Shape: [batch, token count, hidden size].
        const ref_shape = try x.shape();
        const batch_size = ref_shape[0];
        const tokens_size = ref_shape[1];
        const logits_size = ref_shape[2];

        // Select the last input position's hidden vector to predict the next token.
        var last_hidden_state = try x.slice(&[_]c_int{ 0, tokens_size - 1, 0 }, &[_]c_int{ batch_size, tokens_size, logits_size }, &[_]c_int{ 1, 1, 1 }, self.strm);
        defer last_hidden_state.deinit();

        var projection = try self.embedding_weights.transpose(self.strm);
        defer projection.deinit();

        // Project the hidden vector to vocabulary logits using the shared embedding weights.
        var next_token_logits = try last_hidden_state.matMul(projection, self.strm);
        defer next_token_logits.deinit();

        // Apply each row's repetition penalty using its borrowed generation history.
        var penalty = Tensor.scalar(self.repetition_penalty);
        defer penalty.deinit();
        var zeros = Tensor.scalar(0);
        defer zeros.deinit();
        const vocab_size: c_int = @intCast(try next_token_logits.dim(2));
        for (ctx.generated_token_ids, 0..) |history, r| {
            if (history.len == 0) continue;

            const row: c_int = @intCast(r);
            const start: []const c_int = &.{ row, 0, 0 };
            const stop: []const c_int = &.{ row + 1, 1, vocab_size };
            const strides: []const c_int = &.{ 1, 1, 1 };
            var row_logits = try next_token_logits.slice(start, stop, strides, self.strm);
            defer row_logits.deinit();
            var history_tensor = try Tensor.fromSlice(c_int, history, &.{ 1, @intCast(history.len) });
            defer history_tensor.deinit();

            var prev_gen_tokens_probs = try row_logits.gather(
                &.{history_tensor},
                &.{2},
                &.{ 1, 1, 1 },
                self.strm,
            );
            defer prev_gen_tokens_probs.deinit();
            var positive_penalty_results = try prev_gen_tokens_probs.div(penalty, self.strm);
            defer positive_penalty_results.deinit();
            var negative_penalty_results = try prev_gen_tokens_probs.mul(penalty, self.strm);
            defer negative_penalty_results.deinit();
            var mask = try prev_gen_tokens_probs.greaterEqual(zeros, self.strm);
            defer mask.deinit();
            var updated_prev_gen_tokens_probs = try Tensor.where(mask, positive_penalty_results, negative_penalty_results, self.strm);
            defer updated_prev_gen_tokens_probs.deinit();
            var penalized_row = try row_logits.scatter(&.{history_tensor}, updated_prev_gen_tokens_probs, &.{2}, self.strm);
            defer penalized_row.deinit();

            try next_token_logits.sliceUpdateMut(penalized_row, start, stop, strides, self.strm);
        }

        const id_tensor = blk: {
            if (self.temperature == 0) {
                // Greedy decoding selects the highest-scoring token deterministically.
                break :blk try next_token_logits.argmax(-1, self.strm);
            } else {
                var temp = Tensor.scalar(self.temperature);
                defer temp.deinit();
                try next_token_logits.divMut(temp, self.strm);
                // TODO: top-k and top-p sampling
                break :blk try next_token_logits.categorical(-1, self.strm);
            }
        };

        return id_tensor;
    }

    fn emitToken(self: Self, id: c_int, output: *Queue([]const u8)) !void {
        const token = try self.tokenizer.decode(self.gpa, &[1]u32{@intCast(id)});
        errdefer self.gpa.free(token);
        _ = try output.put(self.io, &[1][]const u8{token}, 1);
    }

    /// returns a view into the uncached tokens we will need to process - do not call free on it
    fn uncached_tokens(self: Self, token_ids: []const c_int, conversation_id: []const u8) ![]const c_int {
        const conversation = self.conversation_states.getPtr(conversation_id) orelse return error.NoConversationState;

        if (std.mem.cutPrefix(c_int, token_ids, conversation.cached_token_ids.items)) |new_tokens| {
            return new_tokens;
        }

        conversation.reset();
        return token_ids;
    }

    fn uncached_tokens_batch(self: Pipeline, token_ids_batch: []const []const c_int, conversation_ids: []const []const u8) ![][]const c_int {
        var uncached_tokens_batch_list = std.ArrayList([]const c_int).empty;
        errdefer uncached_tokens_batch_list.deinit(self.gpa);

        for (token_ids_batch, conversation_ids) |token_ids, conversation_id| {
            try uncached_tokens_batch_list.append(self.gpa, try self.uncached_tokens(token_ids, conversation_id));
        }

        return try uncached_tokens_batch_list.toOwnedSlice(self.gpa);
    }

    pub fn callStreaming(self: *Self, conversation: common.Conversation, output: *Queue([]const u8)) !void {
        const conversations: []const common.Conversation = &.{conversation};
        log.debug("generating response for batch of size: {d}", .{conversations.len});

        // if any error happens we can reset the conversation states for conversations involved
        errdefer {
            for (conversations) |row| {
                const state = self.conversation_states.getPtr(row.id) orelse continue;
                state.reset();
            }
        }

        if (!self.conversation_states.contains(conversation.id)) {
            try self.conversation_states.put(conversation.id, try .init(self.gpa, self.transformer_layers.items.len, self.strm));
        }

        // close output channel regardless of what happens
        defer output.close(self.io);

        const token_ids_batch = try self.preprocess(conversations);
        defer {
            for (token_ids_batch) |row| {
                self.gpa.free(row);
            }
            self.gpa.free(token_ids_batch);
        }

        var tokens: std.ArrayList(c_int) = .empty;
        defer tokens.deinit(self.gpa);
        try tokens.appendSlice(self.gpa, token_ids_batch[0]);
        const generation_start = tokens.items.len;

        var batch = [_][]const c_int{tokens.items};
        var generated_token_ids = [_][]const c_int{tokens.items[generation_start..]};
        const conversation_ids: []const []const u8 = &.{conversation.id};

        var token_ids = try self.forward(&batch, conversation_ids, &generated_token_ids, .{ .prefill = true });
        defer self.gpa.free(token_ids);

        try tokens.append(self.gpa, token_ids[0]);
        batch[0] = tokens.items;
        generated_token_ids[0] = tokens.items[generation_start..];

        if (std.mem.findScalar(u32, self.stop_tokens, @intCast(token_ids[0])) != null) {
            // forward pass to fill the kv cache
            const discarded = try self.forward(&batch, conversation_ids, &generated_token_ids, .{ .prefill = false });
            defer self.gpa.free(discarded);
            return;
        }
        try self.emitToken(token_ids[0], output);

        // start at 1 to account for token generated during the prefill
        for (1..self.max_new_tokens) |_| {
            const new_token_ids = try self.forward(&batch, conversation_ids, &generated_token_ids, .{ .prefill = false });
            errdefer self.gpa.free(new_token_ids);

            try tokens.append(self.gpa, new_token_ids[0]);
            batch[0] = tokens.items;
            generated_token_ids[0] = tokens.items[generation_start..];

            if (std.mem.findScalar(u32, self.stop_tokens, @intCast(new_token_ids[0])) != null) {
                // forward pass to fill the kv cache
                const discarded = try self.forward(&batch, conversation_ids, &generated_token_ids, .{ .prefill = false });
                defer self.gpa.free(discarded);
                self.gpa.free(new_token_ids);
                return;
            }
            try self.emitToken(new_token_ids[0], output);

            self.gpa.free(token_ids);
            token_ids = new_token_ids;
        }

        log.debug("max_new_tokens = {d} limit reached", .{self.max_new_tokens});
    }

    pub fn call(self: *Self, conversation: common.Conversation) !std.ArrayList(u8) {
        var buf: [32][]const u8 = undefined;
        var queue: Queue([]const u8) = .init(&buf);
        errdefer queue.close(self.io);

        var writer = std.Io.Writer.Allocating.init(self.gpa);
        defer writer.deinit();

        {
            const collect_streaming_response_args: std.meta.ArgsTuple(@TypeOf(models.collectStreamingResponse)) = .{ self.gpa, self.io, &queue, &.{&writer.writer}, .{} };
            const handle = try std.Thread.spawn(.{}, models.collectStreamingResponse, collect_streaming_response_args);
            defer {
                queue.close(self.io);
                handle.join();
            }
            try self.callStreaming(conversation, &queue);
        }

        return writer.toArrayList();
    }
};
