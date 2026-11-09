const c = @import("mlx_c");
const common = @import("common.zig");

const Map = @import("map.zig").Map;
const Stream = @import("stream.zig").Stream;

pub fn loadSafetensors(strm: Stream, path: []const u8) !struct { Map(.string), Map(.tensor) } {
    try common.ensure(.{strm});
    // File loading is a CPU operation on Metal. Lazy loads must also be usable
    // on the worker thread when the execution stream permits a thread handoff.
    const load_stream = if (strm.cross_thread) blk: {
        const device = c.mlx_device_new_type(c.MLX_CPU, 0);
        defer _ = c.mlx_device_free(device);
        break :blk c.mlx_stream_new_thread_unsafe(device);
    } else c.mlx_default_cpu_stream_new();
    defer _ = c.mlx_stream_free(load_stream);
    if (load_stream.ctx == null) return error.NullStream;

    var weights = c.mlx_map_string_to_array_new();
    errdefer _ = c.mlx_map_string_to_array_free(weights);

    var metadata = c.mlx_map_string_to_string_new();
    errdefer _ = c.mlx_map_string_to_string_free(metadata);

    try common.check(c.mlx_load_safetensors(
        &weights,
        &metadata,
        path.ptr,
        load_stream,
    ));

    return .{
        .{ .p = metadata },
        .{ .p = weights },
    };
}
