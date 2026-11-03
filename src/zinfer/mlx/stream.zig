const c = @import("mlx_c");
const common = @import("common.zig");
const std = @import("std");

const log = std.log.scoped(.mlx);

pub const InitOptions = struct {
    require_gpu: bool = false,
};

pub fn init(opts: InitOptions) !Stream {
    var gpu_available = false;
    try common.check(c.mlx_metal_is_available(&gpu_available));

    if (!gpu_available and opts.require_gpu) {
        return error.GpuUnavailable;
    }

    log.debug("Using {s} backend", .{if (gpu_available) "GPU" else "CPU"});

    const strm = if (gpu_available)
        c.mlx_default_gpu_stream_new()
    else
        c.mlx_default_cpu_stream_new();

    return .{ .p = strm };
}

pub const Stream = struct {
    const Self = @This();

    p: c.mlx_stream,

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
