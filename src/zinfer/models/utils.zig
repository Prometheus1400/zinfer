const std = @import("std");
const Queue = std.Io.Queue;

pub fn collectStreamingResponse(gpa: std.mem.Allocator, io: std.Io, queue: *Queue([]const u8), writers: []const *std.Io.Writer, options: struct { flush: bool = true }) !void {
    while (true) {
        const token = queue.getOne(io) catch |e| {
            switch (e) {
                error.Closed => break,
                error.Canceled => return e,
            }
        };
        defer gpa.free(token);

        for (writers) |writer| {
            try writer.writeAll(token);
            if (options.flush) {
                try writer.flush();
            }
        }
    }
}
