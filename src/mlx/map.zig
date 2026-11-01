const std = @import("std");
const c = @import("mlx_c");
const common = @import("common.zig");
const Tensor = @import("tensor.zig").Tensor;

pub const MapValueType = enum {
    string,
    tensor,
};

/// holds a reference to the map owned string
/// shared ownership of the Tensor (you must call deinit on it)
pub fn Entry(comptime T: MapValueType) type {
    const EntryType = switch (T) {
        .string => []const u8,
        .tensor => Tensor,
    };
    return struct {
        key_ptr: []const u8,
        val_ptr: EntryType,
    };
}

pub fn Iterator(comptime T: MapValueType) type {
    const IteratorType = switch (T) {
        .tensor => c.mlx_map_string_to_array_iterator,
        .string => c.mlx_map_string_to_string_iterator,
    };
    return struct {
        const Self = @This();

        p: IteratorType,

        pub fn next(self: Self) ?Entry(T) {
            common.ensure(.{self}) catch return null;

            switch (T) {
                .tensor => {
                    var key: [*c]const u8 = null;
                    var value: Tensor = .new();
                    common.check(c.mlx_map_string_to_array_iterator_next(&key, &value.p, self.p)) catch return null;
                    return .{ .key_ptr = std.mem.span(key), .val_ptr = value };
                },
                .string => {
                    var key: [*c]const u8 = null;
                    var value: [*c]const u8 = null;
                    common.check(c.mlx_map_string_to_string_iterator_next(&key, &value, self.p)) catch return null;
                    return .{ .key_ptr = std.mem.span(key), .val_ptr = std.mem.span(value) };
                },
            }
        }

        /// Releases the owned handle. Empty and already released handles are allowed.
        pub fn deinit(self: *Self) void {
            switch (T) {
                .tensor => {
                    common.check(c.mlx_map_string_to_array_iterator_free(self.p)) catch @panic("Failed to free MLX tensor map iterator");
                },
                .string => {
                    common.check(c.mlx_map_string_to_string_iterator_free(self.p)) catch @panic("Failed to free MLX string map iterator");
                },
            }
            self.p.ctx = null;
            self.p.map_ctx = null;
        }

        pub fn ensure(self: Self) !void {
            if (self.p.ctx == null) {
                return error.NullMap;
            }
        }
    };
}

/// Map of string -> type (T)
pub fn Map(comptime T: MapValueType) type {
    const MapType = switch (T) {
        .tensor => c.mlx_map_string_to_array,
        .string => c.mlx_map_string_to_string,
    };
    const ValueType = switch (T) {
        .tensor => Tensor,
        .string => []const u8,
    };
    return struct {
        const Self = @This();

        p: MapType,

        pub fn get(self: Self, key: []const u8) !ValueType {
            try common.ensure(.{self});
            switch (T) {
                .tensor => {
                    var val = Tensor.new();
                    try common.check(c.mlx_map_string_to_array_get(&val.p, self.p, key.ptr));
                    return val;
                },
                .string => {
                    const key_arr = [_][*]const u8{key.ptr};
                    try common.check(c.mlx_map_string_to_string_get(&key_arr, self.p, key.ptr));
                    return std.mem.span(key_arr[0]);
                },
            }
        }

        pub fn iterator(self: *Self) !Iterator(T) {
            try common.ensure(.{self});
            return .{ .p = switch (T) {
                .tensor => c.mlx_map_string_to_array_iterator_new(self.p),
                .string => c.mlx_map_string_to_string_iterator_new(self.p),
            } };
        }

        /// Releases the owned handle. Empty and already released handles are allowed.
        pub fn deinit(self: *Self) void {
            switch (T) {
                .tensor => {
                    common.check(c.mlx_map_string_to_array_free(self.p)) catch @panic("Failed to free MLX tensor map");
                },
                .string => {
                    common.check(c.mlx_map_string_to_string_free(self.p)) catch @panic("Failed to free MLX string map");
                },
            }
            self.p.ctx = null;
        }

        pub fn ensure(self: Self) !void {
            if (self.p.ctx == null) {
                return error.NullMap;
            }
        }
    };
}
