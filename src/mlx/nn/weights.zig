const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Map = @import("../map.zig").Map;

pub fn load(weights: Map(.tensor), layer: usize, name: []const u8) !Tensor {
    var buf: [1024]u8 = undefined;
    const key = try std.mem.printSentinel(&buf, "model.layers.{d}.{s}", .{ layer, name }, 0);
    return weights.get(key);
}
