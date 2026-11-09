const c = @import("mlx_c");
const common = @import("common.zig");
const std = @import("std");

const log = std.log.scoped(.mlx);

pub const InitOptions = struct {
    require_gpu: bool = false,
    /// Allow sequential use from different threads. The caller must serialize
    /// access; MLX does not synchronize this stream internally.
    cross_thread: bool = false,
};

pub fn init(opts: InitOptions) !Stream {
    var gpu_available = false;
    try common.check(c.mlx_metal_is_available(&gpu_available));

    if (!gpu_available and opts.require_gpu) {
        return error.GpuUnavailable;
    }

    log.debug("Using {s} backend", .{if (gpu_available) "GPU" else "CPU"});

    const strm = if (opts.cross_thread) blk: {
        const device = c.mlx_device_new_type(if (gpu_available) c.MLX_GPU else c.MLX_CPU, 0);
        defer _ = c.mlx_device_free(device);
        break :blk c.mlx_stream_new_thread_unsafe(device);
    } else if (gpu_available)
        c.mlx_default_gpu_stream_new()
    else
        c.mlx_default_cpu_stream_new();

    const stream: Stream = .{ .p = strm, .cross_thread = opts.cross_thread };
    try stream.ensure();
    return stream;
}

pub const Stream = struct {
    const Self = @This();

    p: c.mlx_stream,
    /// Borrowed copies retain this policy, including streams held by KV caches.
    cross_thread: bool = false,

    pub fn ensure(self: Self) !void {
        if (self.p.ctx == null) {
            return error.NullStream;
        }
    }

    /// Releases the owned handle. Empty and already released handles are allowed.
    pub fn deinit(self: *Self) void {
        common.check(c.mlx_stream_free(self.p)) catch @panic("Failed to free MLX stream");
        self.p.ctx = null;
    }
};
