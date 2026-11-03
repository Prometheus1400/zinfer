const args = @import("args.zig");
const build_options = @import("build_options");
const mlx = zinfer.mlx;
const models = zinfer.models;
const std = @import("std");
const zinfer = @import("zinfer");

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
        .qwen_config => try models.qwen.Handler.inspect(ctx, &info, &stdout.interface),
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
            var pipeline = try models.qwen.Handler.buildPipeline(ctx, &info, .{});
            defer pipeline.deinit();

            if (a.stream) {
                var queue_buf: [32][]const u8 = undefined;
                var queue: Queue([]const u8) = .init(&queue_buf);
                errdefer queue.close(ctx.io);

                {
                    const collect_streaming_response_args: std.meta.ArgsTuple(@TypeOf(models.collectStreamingResponse)) = .{ ctx.gpa, ctx.io, &queue, stdout, .{} };
                    const handle = try std.Thread.spawn(.{}, models.collectStreamingResponse, collect_streaming_response_args);
                    defer {
                        queue.close(ctx.io);
                        std.Thread.join(handle);
                    }
                    try pipeline.callStreaming(a.prompt, &queue);
                }
                try stdout.print("\n", .{});
            } else {
                var res = try pipeline.call(a.prompt);
                defer res.deinit(ctx.gpa);
                try stdout.print("{s}\n", .{res.items});
            }
        },
        .empty => {},
    }
}

pub fn main(init: std.process.Init) !void {
    var stream = try mlx.init(.{});
    defer stream.deinit();

    const ctx: Context = .{ .gpa = init.gpa, .io = init.io, .stream = stream };

    const parsed_args = try args.parseArgs(init.minimal.args);
    if (parsed_args.command) |c| {
        switch (c) {
            .inspect => {
                try inspectModel(ctx, c.inspect);
            },
            .prompt => {
                try generateResponse(ctx, c.prompt);
            },
        }
    }
}
