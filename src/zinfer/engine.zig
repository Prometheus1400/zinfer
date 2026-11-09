const std = @import("std");
const Pipeline = @import("models/qwen.zig").Pipeline;
const Conversation = @import("models/common.zig").Conversation;
const Queue = std.Io.Queue;
const Batch = @import("models/qwen.zig").Batch;
const common = @import("models/common.zig");
const ForwardTimings = @import("mlx").ForwardTimings;
pub const Metrics = @import("engine_metrics.zig").Metrics;
const Clock = std.Io.Clock.awake;

pub const Engine = struct {
    const Self = @This();

    const AdmittedItem = struct {
        conversation: Conversation,
        queue: *Queue([]const u8),
    };

    const EncodedItem = struct {
        encoded: []c_int,
        conversation_id: []const u8,
        queue: *Queue([]const u8),
    };

    pipeline: Pipeline,
    encoding_buffer: []AdmittedItem,
    encoding_queue: Queue(AdmittedItem),
    batching_buffer: []EncodedItem,
    batching_queue: Queue(EncodedItem),
    metrics: Metrics = .init(),

    prefill_max_batch_size: usize = 8,
    decode_max_batch_size: usize = 32,

    encoder_handle: ?std.Thread = null,
    batcher_handle: ?std.Thread = null,

    pub fn deinit(self: *Self, gpa: std.mem.Allocator, io: std.Io) void {
        self.encoding_queue.close(io);
        self.batching_queue.close(io);
        if (self.encoder_handle) |handle| {
            handle.join();
        }
        if (self.batcher_handle) |handle| {
            handle.join();
        }

        gpa.free(self.encoding_buffer);
        gpa.free(self.batching_buffer);

        self.pipeline.deinit(gpa);
    }

    pub fn init(gpa: std.mem.Allocator, pipeline: Pipeline, internal_queue_size: usize) !Self {
        const encode_buffer = try gpa.alloc(AdmittedItem, internal_queue_size);
        errdefer gpa.free(encode_buffer);
        const prefill_buffer = try gpa.alloc(EncodedItem, internal_queue_size);
        errdefer gpa.free(prefill_buffer);
        return .{
            .pipeline = pipeline,
            .encoding_buffer = encode_buffer,
            .encoding_queue = .init(encode_buffer),
            .batching_buffer = prefill_buffer,
            .batching_queue = .init(prefill_buffer),
        };
    }

    fn batcher(
        gpa: std.mem.Allocator,
        io: std.Io,
        pipeline: *Pipeline,
        batching_queue: *Queue(EncodedItem),
        metrics: *Metrics,
        prefill_max_batch_size: usize,
        decode_max_batch_size: usize,
    ) !void {
        var prefill_batch = Batch.init(prefill_max_batch_size);
        defer prefill_batch.deinit(gpa);

        var decode_batch = Batch.init(decode_max_batch_size);
        defer decode_batch.deinit(gpa);

        defer metrics.setSequences(0, 0, 0);

        var prefill_queues = std.ArrayList(*Queue([]const u8)).empty;
        defer prefill_queues.deinit(gpa);

        var prefilled_items = std.ArrayList(EncodedItem).empty;
        defer prefilled_items.deinit(gpa);

        var queues = std.ArrayList(*Queue([]const u8)).empty;
        defer queues.deinit(gpa);

        var time = Clock.now(io);

        var buf: [1]EncodedItem = undefined;
        while (true) {
            // greedily fill the decode batch if possible
            while (!decode_batch.full() and prefilled_items.items.len > 0) {
                const removed = prefilled_items.orderedRemove(0);
                try decode_batch.permit(gpa, removed.encoded, removed.conversation_id);
                try queues.append(gpa, removed.queue);
                metrics.setSequences(prefill_batch.list.items.len, decode_batch.list.items.len, prefilled_items.items.len);
            }

            // Can fit any new item into one of the batches. This is important because
            // we don't know what kind of item we will get.
            while (!decode_batch.full() and !prefill_batch.full()) {
                const read = (if (decode_batch.empty() and prefill_batch.empty())
                    // if empty we should just block - no useful work to do
                    batching_queue.get(io, &buf, 1)
                else
                    // batch is not empty we should do a forward pass if
                    // there's nothing in the queue
                    batching_queue.get(io, &buf, 0)) catch |e| switch (e) {
                    error.Closed => return,
                    else => return e,
                };
                if (read == 0) break;

                const item = buf[0];
                if (pipeline.is_conversation_cached(item.conversation_id)) {
                    try decode_batch.permit(gpa, item.encoded, item.conversation_id);
                    try queues.append(gpa, item.queue);
                } else {
                    if (prefill_batch.empty()) {
                        time = Clock.now(io);
                    }
                    try prefill_batch.permit(gpa, item.encoded, item.conversation_id);
                    try prefill_queues.append(gpa, item.queue);
                }
                metrics.setSequences(prefill_batch.list.items.len, decode_batch.list.items.len, prefilled_items.items.len);
            }

            // At this point we KNOW that at least one of the batches is not empty

            // try to process prefill
            if (prefill_batch.full() or (decode_batch.empty() and !prefill_batch.empty()) or (!prefill_batch.empty() and time.durationTo(Clock.now(io)).toMilliseconds() > 200)) {
                defer prefill_queues.clearRetainingCapacity();

                metrics.prefill_batch_size.observe(@intCast(prefill_batch.list.items.len));
                const new_tokens = blk: {
                    const started = Clock.now(io);
                    defer metrics.forward_prefill_duration.observe(
                        @as(f64, @floatFromInt(started.durationTo(Clock.now(io)).toNanoseconds())) / std.time.ns_per_s,
                    );
                    break :blk try pipeline.forward(gpa, prefill_batch, .{ .prefill = true });
                };
                defer gpa.free(new_tokens);
                const removed = try prefill_batch.update(gpa, new_tokens);
                defer gpa.free(removed);
                metrics.completed.incrBy(@intCast(removed.len));
                metrics.setSequences(prefill_batch.list.items.len, decode_batch.list.items.len, prefilled_items.items.len);
                var it = std.mem.reverseIterator(removed);
                while (it.next()) |i| {
                    prefill_queues.orderedRemove(i).close(io);
                }

                for (prefill_batch.list.items, prefill_queues.items) |row, queue| {
                    const new_token_id = row.last() orelse unreachable;
                    try common.decodeAndEmitToken(gpa, io, new_token_id, pipeline.tokenizer, queue);
                    metrics.prefill_output_tokens.incr();
                    metrics.output_tokens.incr();
                }
                const prefilled = try prefill_batch.drain(gpa);
                defer {
                    gpa.free(prefilled.list);
                    gpa.free(prefilled.conversation_ids);
                }
                for (prefilled.list, prefilled.conversation_ids, prefill_queues.items) |encoded, cid, q| {
                    try prefilled_items.append(gpa, .{ .encoded = encoded, .conversation_id = cid, .queue = q });
                }
                metrics.setSequences(prefill_batch.list.items.len, decode_batch.list.items.len, prefilled_items.items.len);
            }
            if (!decode_batch.empty()) {
                // Every row contains the complete history, including the input
                // token for this step and excluding the next sampled token.
                var total_context: u64 = 0;
                var max_context: u64 = 0;
                for (decode_batch.list.items) |row| {
                    const length: u64 = @intCast(row.items.len);
                    metrics.decode_context_tokens.observe(length);
                    total_context += length;
                    max_context = @max(max_context, length);
                }
                metrics.decode_context_mean_tokens.set(@as(f64, @floatFromInt(total_context)) / @as(f64, @floatFromInt(decode_batch.list.items.len)));
                metrics.decode_context_max_tokens.set(max_context);

                // Observe the actual model batch, not the size of the queue read.
                // forward() reads sampled token values, so this includes MLX work
                // being evaluated rather than just constructing a lazy graph.
                metrics.decode_batch_size.observe(@intCast(decode_batch.list.items.len));
                var timings: ForwardTimings = .{ .io = io };
                const new_tokens = blk: {
                    const started = Clock.now(io);
                    defer metrics.forward_decode_duration.observe(
                        @as(f64, @floatFromInt(started.durationTo(Clock.now(io)).toNanoseconds())) / std.time.ns_per_s,
                    );
                    break :blk try pipeline.forward(gpa, decode_batch, .{ .prefill = false, .timings = &timings });
                };
                metrics.observeDecodeTimings(timings);
                defer gpa.free(new_tokens);
                const removed = try decode_batch.update(gpa, new_tokens);
                defer gpa.free(removed);
                metrics.completed.incrBy(@intCast(removed.len));
                metrics.setSequences(prefill_batch.list.items.len, decode_batch.list.items.len, prefilled_items.items.len);
                var it = std.mem.reverseIterator(removed);
                while (it.next()) |i| {
                    queues.orderedRemove(i).close(io);
                }

                for (decode_batch.list.items, queues.items) |row, queue| {
                    const new_token_id = row.last() orelse unreachable;
                    try common.decodeAndEmitToken(gpa, io, new_token_id, pipeline.tokenizer, queue);
                    metrics.decode_output_tokens.incr();
                    metrics.output_tokens.incr();
                }
            }
        }
    }

    fn encoder(
        gpa: std.mem.Allocator,
        io: std.Io,
        pipeline: *Pipeline,
        encoding_queue: *Queue(AdmittedItem),
        batching_queue: *Queue(EncodedItem),
    ) !void {
        var buf: [32]AdmittedItem = undefined;
        var conversations: [32]Conversation = undefined;
        while (true) {
            const read = encoding_queue.get(io, &buf, 1) catch |e| switch (e) {
                error.Closed => return,
                else => return e,
            };
            for (buf[0..read], 0..) |item, i| {
                conversations[i] = item.conversation;
            }

            const encoded = try common.applyChatTemplateAndTokenize(
                gpa,
                conversations[0..read],
                pipeline.chat_template,
                pipeline.tokenizer,
            );
            defer gpa.free(encoded);

            for (encoded, buf[0..read]) |e, item| {
                try batching_queue.putOne(io, .{
                    .encoded = e,
                    .conversation_id = item.conversation.id,
                    .queue = item.queue,
                });
            }
        }
    }

    pub fn start(self: *Self, gpa: std.mem.Allocator, io: std.Io) !void {
        const encoder_handle = try std.Thread.spawn(
            .{},
            encoder,
            .{
                gpa,
                io,
                &self.pipeline,
                &self.encoding_queue,
                &self.batching_queue,
            },
        );
        self.encoder_handle = encoder_handle;
        const batcher_handle = try std.Thread.spawn(
            .{},
            batcher,
            .{
                gpa,
                io,
                &self.pipeline,
                &self.batching_queue,
                &self.metrics,
                self.prefill_max_batch_size,
                self.decode_max_batch_size,
            },
        );
        self.batcher_handle = batcher_handle;
    }

    pub fn submitStreaming(self: *Self, io: std.Io, conversation: Conversation, queue: *Queue([]const u8)) !void {
        const queue_item: AdmittedItem = .{ .conversation = conversation, .queue = queue };
        try self.encoding_queue.putOne(io, queue_item);
    }
};

// Requires models/qwen and the native libraries. Run with:
// zig build test-engine --test-timeout 60s
test "Engine streams requests end to end" {
    const mlx = @import("mlx");
    const models = @import("models/root.zig");
    const Context = @import("context.zig").Context;

    const Collector = struct {
        queue: Queue([]const u8) = .init(&.{}),
        output: std.ArrayList(u8) = .empty,
        chunks: usize = 0,
        first_chunk: ?*std.Io.Event = null,
        resume_reader: ?*std.Io.Event = null,

        fn collect(self: *@This(), gpa: std.mem.Allocator, io: std.Io) anyerror!void {
            while (true) {
                const chunk = self.queue.getOne(io) catch |err| switch (err) {
                    error.Closed => return,
                    else => return err,
                };
                defer gpa.free(chunk);
                try self.output.appendSlice(gpa, chunk);
                self.chunks += 1;
                if (self.chunks == 1) {
                    if (self.first_chunk) |event| event.set(io);
                    if (self.resume_reader) |event| try event.wait(io);
                }
            }
        }
    };

    const Client = struct {
        fn run(
            gpa: std.mem.Allocator,
            io: std.Io,
            engine: *Engine,
            conversation: Conversation,
            collector: *Collector,
            start: *std.Io.Event,
            submitted: *std.Io.Event,
        ) anyerror!void {
            errdefer collector.queue.close(io);
            try start.wait(io);
            try engine.submitStreaming(io, conversation, &collector.queue);
            submitted.set(io);
            try collector.collect(gpa, io);
        }
    };

    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var stream = try mlx.init(.{ .cross_thread = true });
    defer stream.deinit();
    const ctx: Context = .{ .gpa = gpa, .io = io, .stream = stream };
    var info = try models.getModelInfo(ctx, "qwen");
    defer info.deinit(gpa);

    const conversations = [_]Conversation{
        .{ .id = "engine-count", .messages = &.{.{ .role = "user", .content = "Count from one to twenty, separated by commas. Output only the numbers." }} },
        .{ .id = "engine-sum", .messages = &.{.{ .role = "user", .content = "Reply with only the number: 2 + 2 = ?" }} },
        .{ .id = "engine-capital", .messages = &.{.{ .role = "user", .content = "What is the capital of France? Reply with only the city name." }} },
        .{ .id = "engine-colors", .messages = &.{.{ .role = "user", .content = "List three primary colors, separated by commas. Output only the colors." }} },
        .{ .id = "engine-product", .messages = &.{.{ .role = "user", .content = "Reply with only the number: 7 * 8 = ?" }} },
        .{ .id = "engine-days", .messages = &.{.{ .role = "user", .content = "List the days of the week in order, separated by commas. Output only the days." }} },
        .{ .id = "engine-opposite", .messages = &.{.{ .role = "user", .content = "What is the opposite of hot? Reply with one word." }} },
        .{ .id = "engine-months", .messages = &.{.{ .role = "user", .content = "List the first six months of the year, separated by commas. Output only the months." }} },
    };
    var expected: [conversations.len]Collector = @splat(.{});
    defer for (&expected) |*collector| collector.output.deinit(gpa);
    var actual: [conversations.len]Collector = @splat(.{});
    defer for (&actual) |*collector| collector.output.deinit(gpa);

    var engine = blk: {
        var pipeline = try Pipeline.init(ctx, &info, .{ .temperature = 0 });
        defer pipeline.deinit(gpa);

        // Establish each response independently. The comparison checks routing
        // and complete output, not just whether either worker produced some text.
        for (&expected, 0..) |*collector, i| {
            var reader = try std.Io.concurrent(io, Collector.collect, .{ collector, gpa, io });
            defer reader.cancel(io) catch {};
            errdefer collector.queue.close(io);
            try pipeline.callStreamingBatch(gpa, io, conversations[i .. i + 1], &.{&collector.queue});
            try reader.await(io);
            try std.testing.expect(collector.chunks > 0);
            try std.testing.expect(collector.output.items.len > 0);
        }
        try std.testing.expect(!std.mem.eql(u8, expected[0].output.items, expected[1].output.items));
        // Load fresh weights so the worker also exercises lazy CPU loading.
        // Reusing the baseline pipeline would hide stream-affinity bugs because
        // its weights have already been evaluated on this thread.
        var engine_info = try models.getModelInfo(ctx, "qwen");
        defer engine_info.deinit(gpa);
        var engine_pipeline = try Pipeline.init(ctx, &engine_info, .{ .temperature = 0 });
        errdefer engine_pipeline.deinit(gpa);
        // Smaller than the client count so submissions also exercise backpressure.
        break :blk try Engine.init(gpa, engine_pipeline, 2);
    };
    defer engine.deinit(gpa, io);

    var readers: [conversations.len]std.Io.Future(anyerror!void) = undefined;
    var started_readers: usize = 0;
    var first_chunk: std.Io.Event = .unset;
    var start_clients: std.Io.Event = .unset;
    var submitted: std.Io.Event = .unset;
    var resume_reader: std.Io.Event = .unset;
    defer {
        // Wake blocked producers/consumers before joining engine workers if an
        // assertion or submission fails. Successful streams must close themselves.
        start_clients.set(io);
        resume_reader.set(io);
        engine.encoding_queue.close(io);
        engine.batching_queue.close(io);
        for (&actual) |*collector| collector.queue.close(io);
        for (readers[0..started_readers]) |*reader| reader.cancel(io) catch {};
    }

    // Pause the first reader after its first token. Its unbuffered output queue
    // keeps that request active while the other clients begin submitting.
    actual[0].first_chunk = &first_chunk;
    actual[0].resume_reader = &resume_reader;
    readers[0] = try std.Io.concurrent(io, Collector.collect, .{ &actual[0], gpa, io });
    started_readers = 1;
    for (actual[1..], conversations[1..], 1..) |*collector, conversation, i| {
        readers[i] = try std.Io.concurrent(io, Client.run, .{
            gpa, io, &engine, conversation, collector, &start_clients, &submitted,
        });
        started_readers += 1;
    }
    try engine.start(gpa, io);
    try engine.submitStreaming(io, conversations[0], &actual[0].queue);
    try first_chunk.wait(io);
    start_clients.set(io);
    // Ensure a later request is admitted before allowing the first to finish.
    try submitted.wait(io);
    resume_reader.set(io);

    for (&readers) |*reader| {
        // Completion requires the engine to close the output queue, not merely
        // emit its first token. The test runner timeout also covers shutdown.
        try reader.await(io);
    }
    for (&actual, &expected, conversations) |*collector, *baseline, conversation| {
        try std.testing.expect(collector.chunks > 0);
        std.testing.expectEqualStrings(baseline.output.items, collector.output.items) catch |err| {
            std.debug.print("Request {s} differed from its individual baseline.\n", .{conversation.id});
            return err;
        };
    }
}
