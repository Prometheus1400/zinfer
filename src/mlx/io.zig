const c = @import("mlx_c");
const common = @import("common.zig");
const Stream = @import("stream.zig").MlxStream;
const Map = @import("map.zig").Map;

pub fn loadSafetensors(strm: Stream, path: []const u8) !struct { Map(.string), Map(.tensor) } {
    try common.ensure(.{strm});
    // File loading is a CPU operation on Metal. Later tensor operations can
    // consume these lazy arrays on the caller's GPU stream.
    const load_stream = c.mlx_default_cpu_stream_new();
    defer _ = c.mlx_stream_free(load_stream);

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
