const std = @import("std");
const mlx = @import("mlx");
const Context = @import("context").Context;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const template = @import("chat_template");
const ChatTemplate = @import("chat_template").ChatTemplate;
const Queue = @import("std").Io.Queue;
const log = std.log.scoped(.models);

pub const CollectOptions = struct {
    flush: bool = true,
};

pub fn collectStreamingResponse(gpa: std.mem.Allocator, io: std.Io, queue: *Queue([]const u8), writer: *std.Io.Writer, options: CollectOptions) !void {
    while (true) {
        const token = queue.getOne(io) catch |e| {
            switch (e) {
                error.Closed => break,
                error.Canceled => return e,
            }
        };
        defer gpa.free(token);

        try writer.writeAll(token);
        if (options.flush) {
            try writer.flush();
        }
    }
}
