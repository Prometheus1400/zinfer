const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const Stream = @import("../stream.zig").Stream;
const Dtype = @import("../dtype.zig").DType;

const log = std.log.scoped(.cache);

pub const ContiguousKVCache = struct {
    const Self = @This();

    // gpa: std.mem.Allocator,
    strm: Stream,
    len: usize = 0,
    cap: usize = 0,
    block_size: usize = 512,
    k_cache: ?Tensor = null,
    v_cache: ?Tensor = null,

    pub fn deinit(self: *Self) void {
        self.clear();
    }

    pub fn clear(self: *Self) void {
        if (self.k_cache) |*k_cache| k_cache.deinit();
        if (self.v_cache) |*v_cache| v_cache.deinit();
        self.k_cache = null;
        self.v_cache = null;
        self.len = 0;
        self.cap = 0;
    }

    fn remaining_capacity(self: Self) usize {
        return self.cap - self.len;
    }

    fn grow(self: *Self, amount: usize) !void {
        const k_cache = &(self.k_cache orelse return error.CacheNull);
        const v_cache = &(self.v_cache orelse return error.CacheNull);
        const new_cap = try std.math.add(usize, self.cap, amount);
        const token_capacity = std.math.cast(c_int, new_cap) orelse return error.CacheCapacityTooLarge;

        const k_shape = try k_cache.shape();
        var new_k_cache = try Tensor.zeros(&[_]c_int{ k_shape[0], k_shape[1], token_capacity, k_shape[3] }, try k_cache.dtype(), self.strm);
        errdefer new_k_cache.deinit();
        const v_shape = try v_cache.shape();
        var new_v_cache = try Tensor.zeros(&[_]c_int{ v_shape[0], v_shape[1], token_capacity, v_shape[3] }, try v_cache.dtype(), self.strm);
        errdefer new_v_cache.deinit();

        try new_k_cache.sliceUpdateMut(k_cache.*, &[_]c_int{ 0, 0, 0, 0 }, &[_]c_int{ k_shape[0], k_shape[1], k_shape[2], k_shape[3] }, &[_]c_int{ 1, 1, 1, 1 }, self.strm);
        try new_v_cache.sliceUpdateMut(v_cache.*, &[_]c_int{ 0, 0, 0, 0 }, &[_]c_int{ v_shape[0], v_shape[1], v_shape[2], v_shape[3] }, &[_]c_int{ 1, 1, 1, 1 }, self.strm);
        // Publish the replacement pair only after both copies succeed.
        k_cache.deinit();
        k_cache.* = new_k_cache;
        v_cache.deinit();
        v_cache.* = new_v_cache;
        self.cap = new_cap;
    }

    pub fn append(self: *Self, options: struct { k: Tensor, v: Tensor, token_index: usize = 2 }) !void {
        const k = options.k;
        const v = options.v;
        const token_index = options.token_index;
        const new_tokens = try k.dim(token_index);

        if (self.k_cache == null or self.v_cache == null) {
            var new_k_cache = try k.clone();
            errdefer new_k_cache.deinit();
            const new_v_cache = try v.clone();
            self.k_cache = new_k_cache;
            self.v_cache = new_v_cache;
            self.len = new_tokens;
            self.cap = new_tokens;
            return;
        }

        if (self.remaining_capacity() < new_tokens) {
            // needs new allocations
            const num_new_blocks = try std.math.divCeil(usize, new_tokens, self.block_size);
            const amount = try std.math.mul(usize, num_new_blocks, self.block_size);
            try self.grow(amount);
        }

        const k_shape = try k.shape();
        const v_shape = try v.shape();
        const start: c_int = @intCast(self.len);
        const end: c_int = @intCast(self.len + new_tokens);
        try self.k_cache.?.sliceUpdateMut(k, &[_]c_int{ 0, 0, start, 0 }, &[_]c_int{ k_shape[0], k_shape[1], end, k_shape[3] }, &[_]c_int{ 1, 1, 1, 1 }, self.strm);
        try self.v_cache.?.sliceUpdateMut(v, &[_]c_int{ 0, 0, start, 0 }, &[_]c_int{ v_shape[0], v_shape[1], end, v_shape[3] }, &[_]c_int{ 1, 1, 1, 1 }, self.strm);
        self.len += new_tokens;
    }

    /// Returns owned views of the populated [batch, heads, tokens, dim] prefix.
    /// The caller must deinit both returned tensors.
    pub fn get(self: Self) !struct { k: Tensor, v: Tensor } {
        const k_cache = self.k_cache orelse return error.CacheNull;
        const v_cache = self.v_cache orelse return error.CacheNull;
        const k_shape = try k_cache.shape();
        const v_shape = try v_cache.shape();

        const len: c_int = @intCast(self.len);
        const start = [_]c_int{ 0, 0, 0, 0 };
        const strides = [_]c_int{ 1, 1, 1, 1 };
        var k = try k_cache.slice(&start, &.{ k_shape[0], k_shape[1], len, k_shape[3] }, &strides, self.strm);
        errdefer k.deinit();
        const v = try v_cache.slice(&start, &.{ v_shape[0], v_shape[1], len, v_shape[3] }, &strides, self.strm);
        return .{ .k = k, .v = v };
    }
};

// pub const KVCache = struct {
//     const Self = @This();
//
//     gpa: std.mem.Allocator,
//     strm: Stream,
//     len: usize = 0,
//     block_size: usize = 512,
//     k_cache: std.ArrayList(Tensor) = .empty,
//     v_cache: std.ArrayList(Tensor) = .empty,
//
//     pub fn deinit(self: *Self) void {
//         for (self.k_cache.items) |*t| {
//             t.deinit();
//         }
//         self.k_cache.deinit(self.gpa);
//         for (self.v_cache.items) |*t| {
//             t.deinit();
//         }
//         self.v_cache.deinit(self.gpa);
//     }
//
//     fn new_blocks(self: Self, shape: []const c_int, dtype: Dtype, token_index: usize) !struct { k_block: Tensor, v_block: Tensor } {
//         if (token_index >= shape.len) return error.InvalidTokenAxis;
//         const shape_copy = try self.gpa.dupe(c_int, shape);
//         defer self.gpa.free(shape_copy);
//
//         shape_copy[token_index] = @intCast(self.block_size);
//         return .{
//             .k_block = try Tensor.zeros(shape_copy, dtype, self.strm),
//             .v_block = try Tensor.zeros(shape_copy, dtype, self.strm),
//         };
//     }
//
//     const SliceBounds = struct {
//         start: [4]c_int = .{ 0, 0, 0, 0 },
//         stop: [4]c_int,
//         strides: [4]c_int = .{ 1, 1, 1, 1 },
//
//         fn init(shape: []const c_int, token_index: usize, offset: usize, count: usize) !SliceBounds {
//             if (shape.len != 4) return error.UnsupportedRank;
//             if (token_index >= shape.len) return error.InvalidTokenAxis;
//
//             var bounds: SliceBounds = .{ .stop = shape[0..4].* };
//             bounds.start[token_index] = @intCast(offset);
//             bounds.stop[token_index] = @intCast(offset + count);
//             return bounds;
//         }
//     };
//
//     fn remaining_capacity(self: Self) usize {
//         return self.block_size * self.k_cache.items.len - self.len;
//     }
//
//     fn block_offset(self: Self) usize {
//         return @mod(self.len, self.block_size);
//     }
//
//     /// Appends 4D K/V tensors. The default layout is [batch, heads, tokens, dim].
//     pub fn append(self: *Self, options: struct { k: Tensor, v: Tensor, token_index: usize = 2 }) !void {
//         const k = options.k;
//         const v = options.v;
//         const token_index = options.token_index;
//         const k_shape = try k.shape();
//         const v_shape = try v.shape();
//         // Validate both layouts before allocating or changing the cache.
//         _ = try SliceBounds.init(k_shape, token_index, 0, 0);
//         _ = try SliceBounds.init(v_shape, token_index, 0, 0);
//
//         // put first block if it)'s missing
//         if (self.len == 0) {
//             const blocks = try self.new_blocks(k_shape, try k.dtype(), token_index);
//             try self.k_cache.append(self.gpa, blocks.k_block);
//             try self.v_cache.append(self.gpa, blocks.v_block);
//         }
//
//         const new_tokens = try k.dim(token_index);
//
//         if (self.remaining_capacity() < new_tokens) {
//             // needs new allocations
//             const num_new_blocks = try std.math.divCeil(usize, new_tokens, self.block_size);
//             var offset: usize = 0;
//             for (0..num_new_blocks - 1) |_| {
//                 const k_bounds = try SliceBounds.init(k_shape, token_index, offset, self.block_size);
//                 const v_bounds = try SliceBounds.init(v_shape, token_index, offset, self.block_size);
//                 const new_k_block = try k.slice(&k_bounds.start, &k_bounds.stop, &k_bounds.strides, self.strm);
//                 const new_v_block = try v.slice(&v_bounds.start, &v_bounds.stop, &v_bounds.strides, self.strm);
//                 try self.k_cache.append(self.gpa, new_k_block);
//                 try self.v_cache.append(self.gpa, new_v_block);
//                 offset += self.block_size;
//             }
//             self.len += offset;
//
//             const blocks = try self.new_blocks(k_shape, try k.dtype(), token_index);
//             try self.k_cache.append(self.gpa, blocks.k_block);
//             try self.v_cache.append(self.gpa, blocks.v_block);
//
//             const last_k_block = self.k_cache.lastPtr() orelse unreachable;
//             const last_v_block = self.v_cache.lastPtr() orelse unreachable;
//             const last_block_offset = self.block_offset();
//             const remaining_new_tokens = new_tokens - offset;
//             const k_bounds = try SliceBounds.init(k_shape, token_index, last_block_offset, remaining_new_tokens);
//             const v_bounds = try SliceBounds.init(v_shape, token_index, last_block_offset, remaining_new_tokens);
//             try last_k_block.sliceUpdateMut(k, &k_bounds.start, &k_bounds.stop, &k_bounds.strides, self.strm);
//             try last_v_block.sliceUpdateMut(v, &v_bounds.start, &v_bounds.stop, &v_bounds.strides, self.strm);
//
//             self.len += remaining_new_tokens;
//         } else {
//             const last_k_block = self.k_cache.lastPtr() orelse unreachable;
//             const last_v_block = self.v_cache.lastPtr() orelse unreachable;
//             const offset = self.block_offset();
//             const k_bounds = try SliceBounds.init(k_shape, token_index, offset, new_tokens);
//             const v_bounds = try SliceBounds.init(v_shape, token_index, offset, new_tokens);
//             try last_k_block.sliceUpdateMut(k, &k_bounds.start, &k_bounds.stop, &k_bounds.strides, self.strm);
//             try last_v_block.sliceUpdateMut(v, &v_bounds.start, &v_bounds.stop, &v_bounds.strides, self.strm);
//
//             self.len += new_tokens;
//         }
//     }
//
//     pub fn get(self: Self) !struct { k: Tensor, v: Tensor } {
//     }
//
// };
