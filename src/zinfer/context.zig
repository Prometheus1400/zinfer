const std = @import("std");
const mlx = @import("mlx");

pub const Context = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    stream: mlx.Stream,
};
