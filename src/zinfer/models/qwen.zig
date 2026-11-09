const mlx = @import("mlx");
const models = @import("root.zig");
const std = @import("std");
const template = @import("chat_template.zig");
const common = @import("common.zig");
const Clock = std.Io.Clock.awake;

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
};

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

pub const Batch = struct {
    const Self = @This();

    list: std.ArrayList(std.ArrayList(c_int)), //owned
    conversation_ids: std.ArrayList([]const u8), //view
    max_size: usize = 32,

    pub fn init(max_size: ?usize) Self {
        return .{
            .list = .empty,
            .conversation_ids = .empty,
            .max_size = max_size orelse 32,
        };
    }

    pub fn deinit(self: *Self, gpa: std.mem.Allocator) void {
        for (self.list.items) |*b| b.deinit(gpa);
        self.list.deinit(gpa);
        self.conversation_ids.deinit(gpa);
    }

    pub fn permit(self: *Self, gpa: std.mem.Allocator, item: []c_int, conversation_id: []const u8) !void {
        if (self.list.items.len >= self.max_size) return error.BatchFull;
        try self.list.append(gpa, .fromOwnedSlice(item));
        try self.conversation_ids.append(gpa, conversation_id);
    }

    // Increases the seq-length of the batches by adding the new tokens. Never increases the number of items
    // in the batch. Prunes the completed items from the batch too. Returns the indices that were removed.
    pub fn update(self: *Self, gpa: std.mem.Allocator, new_tokens: []const ?c_int) ![]usize {
        var to_remove: std.ArrayList(usize) = .empty;
        for (new_tokens, 0..) |opt_token, i| {
            if (opt_token) |token| {
                try self.list.items[i].append(gpa, token);
            } else {
                try to_remove.append(gpa, i);
            }
        }

        var it = std.mem.reverseIterator(to_remove.items);
        while (it.next()) |i| {
            var removed_item = self.list.orderedRemove(i);
            removed_item.deinit(gpa);
            _ = self.conversation_ids.orderedRemove(i);
        }

        return to_remove.toOwnedSlice(gpa);
    }

    pub fn drain(self: *Self, gpa: std.mem.Allocator) !struct { list: [][]c_int, conversation_ids: [][]const u8 } {
        var list = try gpa.alloc([]c_int, self.list.items.len);
        for (self.list.items, 0..) |*item, i| {
            list[i] = try item.toOwnedSlice(gpa);
        }
        self.list.clearRetainingCapacity();
        const conversation_ids = try self.conversation_ids.toOwnedSlice(gpa);

        return .{ .list = list, .conversation_ids = conversation_ids };
    }

    pub fn full(self: Self) bool {
        return self.list.items.len >= self.max_size;
    }
    pub fn empty(self: Self) bool {
        return self.list.items.len == 0;
    }
};

pub const Pipeline = struct {
    const Self = @This();

    strm: Stream,

    tokenizer: Tokenizer,
    transformer_layers: std.ArrayList(mlx.nn.TransformerBlock),
    norm: Tensor,
    embedding_weights: Tensor,
    chat_template: ChatTemplate,

    stop_tokens: []const c_int,
    temperature: f32,
    max_new_tokens: u32 = 512,
    repetition_penalty: f32,
    conversation_states: std.StringHashMap(ConversationState),

    pub fn deinit(self: *Self, gpa: std.mem.Allocator) void {
        for (self.transformer_layers.items) |*layer| {
            layer.deinit();
        }
        self.transformer_layers.deinit(gpa);
        self.norm.deinit();
        self.embedding_weights.deinit();
        self.tokenizer.deinit();
        self.chat_template.deinit();
        gpa.free(self.stop_tokens);

        var conversation_states_iterator = self.conversation_states.iterator();
        while (conversation_states_iterator.next()) |entry| {
            // self.gpa.free(entry.key_ptr.*);
            entry.value_ptr.deinit(gpa);
        }
        self.conversation_states.deinit();
    }

    // TODO: Apply max_new_tokens and kv_caching_strategy from pipeline_args to the returned pipeline.
    pub fn init(ctx: Context, info: *const models.ModelInfo, pipeline_args: struct {
        /// Zero selects greedy decoding; positive values scale logits before sampling.
        temperature: f32 = 0.7,
        repetition_penalty: f32 = 1.2,
        max_new_tokens: u32 = 2048,
        kv_caching_strategy: enum {
            none,
            simple,
        } = .simple,
    }) !Self {
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

        const stop_tokens = try ctx.gpa.alloc(c_int, info.generation_config.value.eos_token_id.len);
        errdefer ctx.gpa.free(stop_tokens);
        for (info.generation_config.value.eos_token_id, stop_tokens) |id, *stop_token| {
            stop_token.* = std.math.cast(c_int, id) orelse return error.InvalidTokenId;
        }

        return .{
            .strm = ctx.stream,
            .tokenizer = tokenizer,
            .transformer_layers = layers,
            .norm = norm,
            .embedding_weights = embedding_weights,
            .chat_template = try getChatTemplate(ctx, info.tokenizer_config_file_path),
            .stop_tokens = stop_tokens,
            .conversation_states = .init(ctx.gpa),
            .temperature = pipeline_args.temperature,
            .repetition_penalty = pipeline_args.repetition_penalty,
        };
    }

    pub fn forward(
        self: *Self,
        gpa: std.mem.Allocator,
        batch: Batch,
        // batch: []const []const c_int,
        opts: struct {
            prefill: bool,
            timings: ?*mlx.ForwardTimings = null,
            // cached_token_ids: []const []const c_int,
            // conversation_ids: []const []const u8,
            // real_token_counts: []const usize,
        },
    ) ![]const ?c_int {
        // used for anything that only stays alive for the duration of this function call
        // be careful about using this in place of 'self.gpa' (for things that modify Self)
        var arena_impl: std.heap.ArenaAllocator = .init(gpa);
        const arena = arena_impl.allocator();
        defer arena_impl.deinit();

        var batch_list = try arena.alloc([]const c_int, batch.list.items.len);
        for (batch.list.items, 0..) |item, i| {
            batch_list[i] = item.items;
        }
        const conversation_ids = batch.conversation_ids.items;

        var cached_token_ids: [][]const c_int = try arena.alloc([]const c_int, batch.list.items.len);
        for (batch.conversation_ids.items, 0..) |conversation_id, i| {
            // initialize the conversation state like in the case of new conversations
            if (!self.conversation_states.contains(conversation_id)) {
                try self.conversation_states.put(conversation_id, try .init(gpa, self.transformer_layers.items.len, self.strm));
            }
            const state = self.conversation_states.get(conversation_id) orelse unreachable;
            cached_token_ids[i] = state.cached_token_ids.items;
        }

        const uncached_token_ids_batch = try common.uncached_tokens_batch(arena, batch_list, conversation_ids, &self.conversation_states);
        var real_token_counts = try arena.alloc(usize, uncached_token_ids_batch.len);
        for (uncached_token_ids_batch, 0..) |row, i| {
            real_token_counts[i] = row.len;
        }
        const padded_uncached_token_ids = try common.padBatch(arena, uncached_token_ids_batch);
        const input = try common.batchToTensor(arena, padded_uncached_token_ids);
        const graph_started = if (opts.timings) |t| Clock.now(t.io) else null;
        var sampled = try self.forwardTensor(
            gpa,
            input,
            .{
                .cached_token_ids = cached_token_ids,
                .conversation_ids = batch.conversation_ids.items,
                .real_token_counts = real_token_counts,
                .prefill = opts.prefill,
                .timings = opts.timings,
            },
        );
        if (opts.timings) |t| {
            t.graph_build_ns += graph_started.?.durationTo(Clock.now(t.io)).toNanoseconds();
        }

        for (uncached_token_ids_batch, batch.conversation_ids.items) |uncached_token_id, conversation_id| {
            try self.conversation_states.getPtr(conversation_id).?.cached_token_ids.appendSlice(gpa, uncached_token_id);
        }

        // using self.gpa since it's owned
        const token_ids = try gpa.alloc(?c_int, try sampled.dim(0));
        errdefer gpa.free(token_ids);
        const evaluation_started = if (opts.timings) |t| Clock.now(t.io) else null;
        for (token_ids, 0..) |*id, r| {
            const row: c_int = @intCast(r);
            var sampled_row = try sampled.slice(&.{ row, 0 }, &.{ row + 1, 1 }, &.{ 1, 1 }, self.strm);
            defer sampled_row.deinit();
            id.* = @intCast(try sampled_row.itemU32());
        }
        if (opts.timings) |t| {
            t.evaluation_ns += evaluation_started.?.durationTo(Clock.now(t.io)).toNanoseconds();
        }

        for (token_ids) |*opt_token_id| {
            if (opt_token_id.*) |token_id| {
                if (std.mem.findScalar(c_int, self.stop_tokens, token_id) != null) {
                    opt_token_id.* = null;
                }
            }
        }

        return token_ids;
    }

    fn forwardTensor(
        self: Self,
        gpa: std.mem.Allocator,
        input: Tensor,
        ctx: struct {
            cached_token_ids: []const []const c_int,
            conversation_ids: []const []const u8,
            real_token_counts: []const usize,
            prefill: bool,
            timings: ?*mlx.ForwardTimings = null,
        },
    ) !Tensor {
        // Look up an embedding vector for each input token ID.
        var x = try self.embedding_weights.take(input, 0, self.strm);
        defer x.deinit();

        // Pass the hidden vectors through each transformer layer, updating its KV cache.
        const layer_caches = try gpa.alloc(*cache.ContiguousKVCache, ctx.conversation_ids.len);
        defer gpa.free(layer_caches);
        for (self.transformer_layers.items, 0..) |*l, i| {
            for (ctx.conversation_ids, 0..) |id, b| {
                const state = self.conversation_states.getPtr(id) orelse return error.NoConversationState;
                layer_caches[b] = &state.kv_cache.items[i];
            }
            const transformer_layer_output = try l.forward(
                x,
                self.strm,
                .{
                    .gpa = gpa,
                    .prefill = ctx.prefill,
                    .kv_caches = layer_caches,
                    .real_token_counts = ctx.real_token_counts,
                    .timings = ctx.timings,
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
        const vocab_size = (try next_token_logits.shape())[2];
        for (ctx.cached_token_ids, 0..) |history, r| {
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

    pub fn callStreamingBatch(self: *Self, gpa: std.mem.Allocator, io: std.Io, conversations: []const common.Conversation, outputs: []const *Queue([]const u8)) !void {
        var arena_impl: std.heap.ArenaAllocator = .init(gpa);
        const arena = arena_impl.allocator();
        defer arena_impl.deinit();

        // if any error happens we can reset the conversation states for conversations involved
        errdefer {
            for (conversations) |row| {
                const state = self.conversation_states.getPtr(row.id) orelse continue;
                state.reset();
            }
            for (outputs) |output| {
                output.close(io);
            }
        }

        var token_ids_batch: std.ArrayList(std.ArrayList(c_int)) = try .initCapacity(arena, conversations.len);
        for (try common.applyChatTemplateAndTokenize(arena, conversations, self.chat_template, self.tokenizer)) |encoded_row| {
            token_ids_batch.appendAssumeCapacity(.fromOwnedSlice(encoded_row));
        }
        var conversation_ids: std.ArrayList([]const u8) = try .initCapacity(arena, conversations.len);
        for (conversations) |conversation| conversation_ids.appendAssumeCapacity(conversation.id);
        var queue_view: std.ArrayList(*Queue([]const u8)) = .fromOwnedSlice(try arena.dupe(*Queue([]const u8), outputs));
        var batch: Batch = .{ .list = token_ids_batch, .conversation_ids = conversation_ids };

        var prefill = true;
        while (batch.list.items.len > 0) {
            defer prefill = false;

            const generated_token_ids = try self.forward(gpa, batch, .{ .prefill = prefill });
            defer gpa.free(generated_token_ids);

            const removed = try batch.update(arena, generated_token_ids);
            var it = std.mem.reverseIterator(removed);
            while (it.next()) |i| {
                queue_view.orderedRemove(i).close(io);
            }

            for (batch.list.items, queue_view.items) |row, queue| {
                const new_token_id = row.last() orelse unreachable;
                try common.decodeAndEmitToken(gpa, io, new_token_id, self.tokenizer, queue);
            }
        }
    }

    pub fn call(self: *Self, gpa: std.mem.Allocator, conversation: common.Conversation) !std.ArrayList(u8) {
        var buf: [32][]const u8 = undefined;
        var queue: Queue([]const u8) = .init(&buf);
        errdefer queue.close(self.io);

        var writer = std.Io.Writer.Allocating.init(gpa);
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

    pub fn is_conversation_cached(self: Self, conversation_id: []const u8) bool {
        const state = self.conversation_states.get(conversation_id) orelse return false;
        return state.cached_token_ids.items.len > 0;
    }
};

// Requires the local models/qwen checkpoint and the native MLX libraries.
test "Qwen batch matches individual greedy responses" {
    const Collector = struct {
        fn collect(gpa: std.mem.Allocator, io: std.Io, queue: *Queue([]const u8), output: *std.ArrayList(u8), failure: *?anyerror) void {
            while (true) {
                const token = queue.getOne(io) catch |err| {
                    if (err != error.Closed) failure.* = err;
                    return;
                };
                defer gpa.free(token);
                output.appendSlice(gpa, token) catch |err| {
                    failure.* = err;
                    queue.close(io);
                    return;
                };
            }
        }

        fn run(pipeline: *Pipeline, ctx: Context, conversations: []const common.Conversation, outputs: []std.ArrayList(u8)) !void {
            // Unbuffered queues ensure consumers drain the streams during generation.
            const queues = try ctx.gpa.alloc(Queue([]const u8), conversations.len);
            defer ctx.gpa.free(queues);
            const queue_ptrs = try ctx.gpa.alloc(*Queue([]const u8), conversations.len);
            defer ctx.gpa.free(queue_ptrs);
            const threads = try ctx.gpa.alloc(std.Thread, conversations.len);
            defer ctx.gpa.free(threads);
            const failures = try ctx.gpa.alloc(?anyerror, conversations.len);
            defer ctx.gpa.free(failures);
            @memset(failures, null);
            for (queues, queue_ptrs) |*queue, *ptr| {
                queue.* = .init(&.{});
                ptr.* = queue;
            }
            var started: usize = 0;
            {
                defer {
                    for (queue_ptrs) |queue| queue.close(ctx.io);
                    for (threads[0..started]) |thread| thread.join();
                }
                for (outputs, 0..) |*output, i| {
                    threads[i] = try std.Thread.spawn(.{}, collect, .{ ctx.gpa, ctx.io, queue_ptrs[i], output, &failures[i] });
                    started += 1;
                }
                try pipeline.callStreamingBatch(ctx.gpa, ctx.io, conversations, queue_ptrs);
            }
            for (failures) |failure| {
                if (failure) |err| return err;
            }
        }
    };

    const gpa = std.testing.allocator;
    var stream = try mlx.init(.{});
    defer stream.deinit();
    const ctx: Context = .{ .gpa = gpa, .io = std.testing.io, .stream = stream };
    var info = try models.getModelInfo(ctx, "qwen");
    defer info.deinit(gpa);
    const batch_size = 32;
    // IDs are borrowed by the pipeline's conversation-state map.
    var ids: [batch_size][32]u8 = undefined;
    var pipeline = try Pipeline.init(ctx, &info, .{ .temperature = 0 });
    defer pipeline.deinit(gpa);

    // Different prompt and response lengths exercise padding and a shrinking batch.
    const prompts = [_]common.Conversation{
        .{ .id = "individual-sum", .messages = &.{.{ .role = "user", .content = "Reply with only the number: 2 + 2 = ?" }} },
        .{ .id = "individual-count", .messages = &.{.{ .role = "user", .content = "Count from one to five, separated by commas. Output only the numbers." }} },
    };
    var individual: [2]std.ArrayList(u8) = .{ .empty, .empty };
    defer for (&individual) |*output| output.deinit(gpa);
    var batched: [batch_size]std.ArrayList(u8) = @splat(.empty);
    defer for (&batched) |*output| output.deinit(gpa);

    for (prompts, 0..) |conversation, i| {
        std.debug.print("\nIndividual {s}:\n", .{conversation.id});
        try Collector.run(&pipeline, ctx, prompts[i .. i + 1], individual[i .. i + 1]);
        std.debug.print("{s}\n", .{individual[i].items});
        try std.testing.expect(individual[i].items.len > 0);
    }
    try std.testing.expect(!std.mem.eql(u8, individual[0].items, individual[1].items));

    // Repeat the two prompts with distinct IDs, so each cache is independent.
    var conversations: [batch_size]common.Conversation = undefined;
    for (&conversations, &ids, 0..) |*conversation, *id, i| {
        conversation.* = .{
            .id = try std.fmt.bufPrint(id, "batch-{d}", .{i}),
            .messages = prompts[i % prompts.len].messages,
        };
    }
    std.debug.print("\nBatch of {d} conversations:\n", .{batch_size});
    try Collector.run(&pipeline, ctx, &conversations, &batched);
    for (batched, 0..) |actual, i| {
        try std.testing.expectEqualStrings(individual[i % prompts.len].items, actual.items);
    }
    std.debug.print("All {d} responses match their individual baselines.\n", .{batch_size});
}
