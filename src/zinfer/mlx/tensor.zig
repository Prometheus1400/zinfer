const c = @import("mlx_c");
const common = @import("common.zig");
const std = @import("std");
const stream = @import("stream.zig");

const DType = @import("dtype.zig").DType;

pub const Tensor = struct {
    const Self = @This();

    p: c.mlx_array,

    pub fn new() Self {
        return .{ .p = c.mlx_array_new() };
    }

    pub fn scalar(value: f32) Self {
        return .{ .p = c.mlx_array_new_float32(value) };
    }

    /// Creates an independently owned handle sharing the same data and graph.
    pub fn clone(self: Self) !Self {
        try common.ensure(.{self});

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);

        try common.check(c.mlx_array_set(&res, self.p));
        return .{ .p = res };
    }

    pub fn fromSlice(comptime T: type, data: []const T, shape_: []const c_int) !Self {
        const arr = switch (T) {
            f32 => c.mlx_array_new_data(data.ptr, shape_.ptr, @intCast(shape_.len), c.MLX_FLOAT32),
            c_int => c.mlx_array_new_data(data.ptr, shape_.ptr, @intCast(shape_.len), c.MLX_INT32),
            else => return error.UnsupportedType,
        };

        return .{ .p = arr };
    }

    /// Evaluates this tensor's lazy graph and waits for the result.
    pub fn eval(self: Self) !void {
        try common.ensure(.{self});
        try common.check(c.mlx_array_eval(self.p));
    }

    /// Evaluates a uint32 tensor containing one element and returns its value.
    pub fn itemU32(self: Self) !u32 {
        try common.ensure(.{self});
        try self.eval();

        var value: u32 = undefined;
        try common.check(c.mlx_array_item_uint32(&value, self.p));
        return value;
    }

    pub fn ndim(self: Self) !usize {
        try common.ensure(.{self});
        return @intCast(c.mlx_array_ndim(self.p));
    }

    pub fn dim(self: Self, d: usize) !usize {
        try common.ensure(.{self});
        return @intCast(c.mlx_array_dim(self.p, @intCast(d)));
    }

    pub fn shape(self: Self) ![]const c_int {
        try common.ensure(.{self});
        return c.mlx_array_shape(self.p)[0..c.mlx_array_ndim(self.p)];
    }

    pub fn dtype(self: Self) !DType {
        try common.ensure(.{self});
        const d = c.mlx_array_dtype(self.p);
        return @fromBackingInt(d);
    }

    pub fn slice(self: Self, start: []const c_int, stop: []const c_int, strides: []const c_int, strm: stream.Stream) !Tensor {
        try common.ensure(.{ self, strm });
        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);

        try common.check(c.mlx_slice(&res, self.p, start.ptr, start.len, stop.ptr, stop.len, strides.ptr, strides.len, strm.p));

        return .{ .p = res };
    }

    /// Returns an owned, lazy tensor with the selected slice replaced by update.
    /// Bounds and strides follow slice(); update must broadcast to that slice.
    pub fn sliceUpdate(src: Tensor, update: Tensor, start: []const c_int, stop: []const c_int, strides: []const c_int, strm: stream.Stream) !Tensor {
        try common.ensure(.{ src, update, strm });

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);

        try common.check(c.mlx_slice_update(
            &res,
            src.p,
            update.p,
            start.ptr,
            start.len,
            stop.ptr,
            stop.len,
            strides.ptr,
            strides.len,
            strm.p,
        ));
        return .{ .p = res };
    }

    pub fn formatInto(self: Self, buf: []u8) ![]const u8 {
        try common.ensure(.{self});
        var text = c.mlx_string_new();
        defer _ = c.mlx_string_free(text);
        try common.check(c.mlx_array_tostring(&text, self.p));
        const data = std.mem.span(c.mlx_string_data(text));
        if (data.len > buf.len) return error.BufferTooSmall;

        const result = buf[0..data.len];
        @memcpy(result, data);
        return result;
    }

    /// Releases the owned handle. Empty and already released handles are allowed.
    pub fn deinit(self: *Self) void {
        common.check(c.mlx_array_free(self.p)) catch @panic("Failed to free MLX array");

        self.p.ctx = null;
    }

    // The following operations create new tensors.
    // (either refcount bumps to same underlying data or new data, both need "deinit")
    //
    // Implemented as static methods to make that relationship more clear.

    /// Selects entries along axis using an integer tensor of indices.
    pub fn take(a: Tensor, indices: Tensor, axis: c_int, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, indices, strm });

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);

        try common.check(c.mlx_take_axis(&res, a.p, indices.p, axis, strm.p));
        return .{ .p = res };
    }

    pub fn matMul(a: Tensor, b: Tensor, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, b, strm });
        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);
        try common.check(c.mlx_matmul(&res, a.p, b.p, strm.p));
        return .{ .p = res };
    }

    pub fn mul(a: Tensor, b: Tensor, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, b, strm });
        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);
        try common.check(c.mlx_multiply(&res, a.p, b.p, strm.p));
        return .{ .p = res };
    }

    pub fn add(a: Tensor, b: Tensor, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, b, strm });

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);
        try common.check(c.mlx_add(&res, a.p, b.p, strm.p));
        return .{ .p = res };
    }

    pub fn div(a: Tensor, b: Tensor, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, b, strm });

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);
        try common.check(c.mlx_divide(&res, a.p, b.p, strm.p));
        return .{ .p = res };
    }

    pub fn concat(a: Tensor, b: Tensor, axis: c_int, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, b, strm });

        const handles = [_]c.mlx_array{ a.p, b.p };
        var arrays = c.mlx_vector_array_new();
        defer _ = c.mlx_vector_array_free(arrays);
        try common.check(c.mlx_vector_array_set_data(&arrays, &handles, handles.len));

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);
        try common.check(c.mlx_concatenate_axis(&res, arrays, axis, strm.p));
        return .{ .p = res };
    }

    pub fn transpose(a: Tensor, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, strm });

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);

        try common.check(c.mlx_transpose(&res, a.p, strm.p));
        return .{ .p = res };
    }

    pub fn silu(a: Tensor, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, strm });

        var sig = c.mlx_array_new();
        defer _ = c.mlx_array_free(sig);

        try common.check(c.mlx_sigmoid(&sig, a.p, strm.p));

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);

        try common.check(c.mlx_multiply(&res, a.p, sig, strm.p));

        return .{ .p = res };
    }

    /// Returns the index of the maximum along axis, removing that dimension.
    pub fn argmax(a: Tensor, axis: c_int, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, strm });

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);

        try common.check(c.mlx_argmax_axis(&res, a.p, axis, false, strm.p));
        return .{ .p = res };
    }

    /// Samples uint32 indices from logits along axis, removing that dimension.
    /// Uses MLX's default random state. Pass logits, not probabilities.
    pub fn categorical(a: Tensor, axis: c_int, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, strm });

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);

        try common.check(c.mlx_random_categorical(&res, a.p, axis, .{ .ctx = null }, strm.p));
        return .{ .p = res };
    }

    pub fn softmax(a: Tensor, axis: c_int, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, strm });

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);

        try common.check(c.mlx_softmax_axis(&res, a.p, axis, true, strm.p));
        return .{ .p = res };
    }

    pub const AttentionMask = union(enum) {
        none,
        /// Aligns the last query with the last key, including during decoding.
        causal,
        /// Boolean or additive mask broadcastable to [batch, query_heads, queries, keys].
        tensor: Tensor,
    };

    /// Computes softmax(scale * QK^T + mask) V, returning an owned, lazy tensor.
    /// Q: [batch, query_heads, queries, head_dim].
    /// K/V: [batch, kv_heads, keys, head_dim/value_dim].
    /// Query heads must be a multiple of KV heads; do not repeat K/V for GQA.
    /// The usual scale is 1 / sqrt(head_dim).
    pub fn scaledDotProductAttention(q: Tensor, k: Tensor, v: Tensor, scale: f32, mask: AttentionMask, strm: stream.Stream) !Tensor {
        try common.ensure(.{ q, k, v, strm });
        if (mask == .tensor) try common.ensure(.{mask.tensor});

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);

        try common.check(c.mlx_fast_scaled_dot_product_attention(
            &res,
            q.p,
            k.p,
            v.p,
            scale,
            switch (mask) {
                .none => "",
                .causal => "causal",
                .tensor => "array",
            },
            switch (mask) {
                .tensor => |t| t.p,
                else => .{ .ctx = null },
            },
            .{ .ctx = null }, // No attention sinks.
            false, // Let MLX choose the appropriate implementation.
            strm.p,
        ));
        return .{ .p = res };
    }

    /// Uses split-half rotation with unscaled token positions.
    pub fn rope(a: Tensor, dims: c_int, base: f32, offset: c_int, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, strm });

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);

        try common.check(c.mlx_fast_rope(
            &res,
            a.p,
            dims,
            false,
            .{ .has_value = true, .value = base },
            1.0,
            offset,
            .{ .ctx = null },
            strm.p,
        ));
        return .{ .p = res };
    }

    pub fn rms(a: Tensor, w: Tensor, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, w, strm });

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);

        try common.check(c.mlx_fast_rms_norm(&res, a.p, w.p, 1e-6, strm.p));
        return .{ .p = res };
    }

    pub fn reshape(a: Tensor, shape_: []const c_int, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, strm });

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);

        try common.check(c.mlx_reshape(&res, a.p, shape_.ptr, shape_.len, strm.p));
        return .{ .p = res };
    }

    pub fn permute(a: Tensor, axes: []const c_int, strm: stream.Stream) !Tensor {
        try common.ensure(.{ a, strm });

        var res = c.mlx_array_new();
        errdefer _ = c.mlx_array_free(res);

        try common.check(c.mlx_transpose_axes(&res, a.p, axes.ptr, axes.len, strm.p));
        return .{ .p = res };
    }

    // Mut methods replace this wrapper's owned handle, not necessarily its
    // underlying buffer. Creating a result must succeed before releasing the
    // old handle. The caller still calls deinit once on this wrapper.

    pub fn sliceUpdateMut(self: *Self, update: Tensor, start: []const c_int, stop: []const c_int, strides: []const c_int, strm: stream.Stream) !void {
        self.replace(try sliceUpdate(self.*, update, start, stop, strides, strm));
    }

    pub fn matMulMut(self: *Self, b: Tensor, strm: stream.Stream) !void {
        self.replace(try matMul(self.*, b, strm));
    }

    pub fn mulMut(self: *Self, b: Tensor, strm: stream.Stream) !void {
        self.replace(try mul(self.*, b, strm));
    }

    pub fn addMut(self: *Self, b: Tensor, strm: stream.Stream) !void {
        self.replace(try add(self.*, b, strm));
    }

    pub fn divMut(self: *Self, b: Tensor, strm: stream.Stream) !void {
        self.replace(try div(self.*, b, strm));
    }

    pub fn concatMut(self: *Self, b: Tensor, axis: c_int, strm: stream.Stream) !void {
        self.replace(try concat(self.*, b, axis, strm));
    }

    pub fn transposeMut(self: *Self, strm: stream.Stream) !void {
        self.replace(try transpose(self.*, strm));
    }

    pub fn siluMut(self: *Self, strm: stream.Stream) !void {
        self.replace(try silu(self.*, strm));
    }

    pub fn softmaxMut(self: *Self, axis: c_int, strm: stream.Stream) !void {
        self.replace(try softmax(self.*, axis, strm));
    }

    pub fn ropeMut(self: *Self, dims: c_int, base: f32, offset: c_int, strm: stream.Stream) !void {
        self.replace(try rope(self.*, dims, base, offset, strm));
    }

    pub fn rmsMut(self: *Self, w: Tensor, strm: stream.Stream) !void {
        self.replace(try rms(self.*, w, strm));
    }

    pub fn reshapeMut(self: *Self, shape_: []const c_int, strm: stream.Stream) !void {
        self.replace(try reshape(self.*, shape_, strm));
    }

    pub fn permuteMut(self: *Self, axes: []const c_int, strm: stream.Stream) !void {
        self.replace(try permute(self.*, axes, strm));
    }

    fn replace(self: *Self, replacement: Self) void {
        self.deinit();
        self.* = replacement;
    }

    pub fn ensure(self: Self) !void {
        if (self.p.ctx == null) {
            return error.NullArray;
        }
    }
};
