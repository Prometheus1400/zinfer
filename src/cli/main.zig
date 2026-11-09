const std = @import("std");
const args = @import("args.zig");
const server = @import("server/main.zig");
const build_options = @import("build_options");
const zinfer = @import("zinfer");
const mlx = zinfer.mlx;
const models = zinfer.models;

const Context = zinfer.Context;
const Queue = std.Io.Queue;

const log = std.log.scoped(.main);

pub const std_options: std.Options = .{
    .log_level = @field(std.log.Level, @tagName(build_options.log_level)),
};

fn inspectModel(ctx: Context, a: args.InspectArgs) !void {
    log.debug("inspecting model: {s}", .{a.model});

    var info = try models.getModelInfo(ctx, a.model);
    defer info.deinit(ctx.gpa);

    var stdout_buf: [128]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(ctx.io, &stdout_buf);

    switch (info.config) {
        .qwen_config => try models.qwen.inspect(ctx, &info, &stdout.interface),
        .empty => {},
    }
}

fn generateResponse(ctx: Context, a: args.PromptArgs) !void {
    var info = try models.getModelInfo(ctx, a.model);
    defer info.deinit(ctx.gpa);

    var stdout_buf: [128]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(ctx.io, &stdout_buf);
    var stdout = &stdout_writer.interface;

    switch (info.config) {
        .qwen_config => {
            var pipeline = try models.qwen.Pipeline.init(ctx, &info, .{});
            defer pipeline.deinit(ctx.gpa);

            const conversation: zinfer.models.common.Conversation = .{
                .id = "1",
                .messages = &.{
                    .{
                        .role = "user",
                        .content = a.prompt,
                    },
                },
            };
            if (a.stream) {
                var queue_buf: [32][]const u8 = undefined;
                var queue: Queue([]const u8) = .init(&queue_buf);
                errdefer queue.close(ctx.io);

                {
                    const collect_streaming_response_args: std.meta.ArgsTuple(@TypeOf(models.utils.collectStreamingResponse)) = .{ ctx.gpa, ctx.io, &queue, &.{stdout}, .{} };
                    const handle = try std.Thread.spawn(.{}, models.utils.collectStreamingResponse, collect_streaming_response_args);
                    defer {
                        queue.close(ctx.io);
                        std.Thread.join(handle);
                    }
                    try pipeline.callStreamingBatch(ctx.gpa, ctx.io, &.{conversation}, &.{&queue});
                }
                try stdout.print("\n", .{});
            } else {
                // var res = try pipeline.call(conversation);
                // defer res.deinit(ctx.gpa);
                // try stdout.print("{s}\n", .{res.items});
            }
        },
        .empty => {},
    }
}

fn handleChat(ctx: Context, a: args.ChatArgs) !void {
    var info = try models.getModelInfo(ctx, a.model);
    defer info.deinit(ctx.gpa);

    var stdout_buf: [128]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(ctx.io, &stdout_buf);
    var stdout = &stdout_writer.interface;

    var stdin_buf: [128]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(ctx.io, &stdin_buf);
    var stdin_collector = std.Io.Writer.Allocating.init(ctx.gpa);
    defer stdin_collector.deinit();

    var model_output_collector = std.Io.Writer.Allocating.init(ctx.gpa);
    defer model_output_collector.deinit();

    switch (info.config) {
        .qwen_config => {
            var pipeline = try models.qwen.Pipeline.init(ctx, &info, .{});
            defer pipeline.deinit(ctx.gpa);

            try stdout.print("> ", .{});
            try stdout.flush();
            var messages: std.ArrayList(models.chat_template.Message) = .empty;
            defer {
                for (messages.items) |m| ctx.gpa.free(m.content);
                messages.deinit(ctx.gpa);
            }
            while (try stdin_reader.interface.streamDelimiterEnding(&stdin_collector.writer, '\n') != 0) {
                defer stdin_reader.interface.toss(1);
                try messages.append(
                    ctx.gpa,
                    .{ .role = "user", .content = try stdin_collector.toOwnedSlice() },
                );

                var queue_buf: [32][]const u8 = undefined;
                var queue: Queue([]const u8) = .init(&queue_buf);
                errdefer queue.close(ctx.io);

                try stdout.print("🤖 ", .{});
                try stdout.flush();
                {
                    const collect_streaming_response_args: std.meta.ArgsTuple(@TypeOf(models.utils.collectStreamingResponse)) = .{
                        ctx.gpa,
                        ctx.io,
                        &queue,
                        &.{ stdout, &model_output_collector.writer },
                        .{},
                    };
                    const handle = try std.Thread.spawn(.{}, models.utils.collectStreamingResponse, collect_streaming_response_args);
                    defer {
                        queue.close(ctx.io);
                        std.Thread.join(handle);
                    }
                    try pipeline.callStreamingBatch(ctx.gpa, ctx.io, &.{.{ .id = "1", .messages = messages.items }}, &.{&queue});
                }
                try stdout.print("\n> ", .{});
                try stdout.flush();

                try messages.append(
                    ctx.gpa,
                    .{ .role = "assistant", .content = try model_output_collector.toOwnedSlice() },
                );
            }
        },
        .empty => {},
    }
}

pub fn main(init: std.process.Init) !void {
    const parsed_args = try args.parseArgs(init.minimal.args);
    if (parsed_args.command) |c| {
        if (c == .serve) return server.run(init, c.serve);
    }
    var stream = try mlx.init(.{});
    defer stream.deinit();
    const ctx: Context = .{ .gpa = init.gpa, .io = init.io, .stream = stream };
    if (parsed_args.command) |c| {
        switch (c) {
            .inspect => {
                try inspectModel(ctx, c.inspect);
            },
            .prompt => {
                try generateResponse(ctx, c.prompt);
            },
            .chat => {
                try handleChat(ctx, c.chat);
            },
            .connect => {},
            .serve => unreachable,
        }
    }
}
