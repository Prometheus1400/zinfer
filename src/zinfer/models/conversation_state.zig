const std = @import("std");
const mlx = @import("mlx");

pub const ConversationState = struct {
    const Self = @This();
    ttl_seconds: usize = 60,
    cached_token_ids: std.ArrayList(c_int),
    kv_cache: std.ArrayList(mlx.nn.cache.ContiguousKVCache),

    pub fn init(gpa: std.mem.Allocator, num_layers: usize, strm: mlx.Stream) !Self {
        var kv_cache: std.ArrayList(mlx.nn.cache.ContiguousKVCache) = try .initCapacity(gpa, num_layers);
        for (0..num_layers) |_| {
            try kv_cache.append(gpa, .{ .strm = strm });
        }

        return .{
            .cached_token_ids = .empty,
            .kv_cache = kv_cache,
        };
    }

    pub fn deinit(self: *Self, gpa: std.mem.Allocator) void {
        self.cached_token_ids.deinit(gpa);
        for (self.kv_cache.items) |*c| c.deinit();
        self.kv_cache.deinit(gpa);
    }

    pub fn reset(self: *Self) void {
        for (self.kv_cache.items) |*c| {
            c.clear();
        }
        self.cached_token_ids.clearRetainingCapacity();
    }

    // TODO:
    // pub fn is_expired(self: Self, ) bool {}
};
