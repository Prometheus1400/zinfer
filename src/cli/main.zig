const std = @import("std");
const zinfer = @import("zinfer");
const mlx = zinfer.mlx;
const args = @import("args.zig");
const models = zinfer.models;
pub const Context = zinfer.Context;
const log = std.log.scoped(.main);
const Queue = @import("std").Io.Queue;

fn inspectModel(ctx: Context, a: args.InspectArgs) !void {
    log.debug("inspecting model: {s}", .{a.model});

    var info = try models.getModelInfo(ctx, a.model);
    defer info.deinit(ctx.gpa);

    switch (info.config) {
        .qwen_config => try models.qwen.Handler.inspect(ctx, &info),
        .empty => {},
    }
}

fn generateResponse(ctx: Context, a: args.PromptArgs) !void {
    log.debug("generating response from model: {s} with prompt: {s}", .{ a.model, a.prompt });

    var info = try models.getModelInfo(ctx, a.model);
    defer info.deinit(ctx.gpa);

    switch (info.config) {
        .qwen_config => {
            var pipeline = try models.qwen.Handler.buildPipeline(ctx, &info);
            defer pipeline.deinit();

            if (a.stream) {
                var queue_buf: [32][]const u8 = undefined;
                var queue: Queue([]const u8) = .init(&queue_buf);
                errdefer queue.close(ctx.io);

                var stdout_buf: [128]u8 = undefined;
                var stdout = std.Io.File.stdout().writerStreaming(ctx.io, &stdout_buf);
                {
                    const handle = try std.Thread.spawn(.{}, models.collectStreamingResponse, .{ ctx.gpa, ctx.io, &queue, &stdout.interface, models.CollectOptions{} });
                    defer {
                        queue.close(ctx.io);
                        std.Thread.join(handle);
                    }
                    try pipeline.callStreaming(a.prompt, &queue);
                }
                std.debug.print("\n", .{});
            } else {
                var res = try pipeline.call(a.prompt);
                defer res.deinit(ctx.gpa);
                std.debug.print("{s}\n", .{res.items});
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
