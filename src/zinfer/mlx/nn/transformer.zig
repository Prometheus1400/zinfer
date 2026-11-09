const std = @import("std");
const Clock = std.Io.Clock.awake;

const loadWeight = @import("weights.zig").load;

const Map = @import("../map.zig").Map;
const Stream = @import("../stream.zig").Stream;
const SwiGluBlock = @import("swiglu.zig").SwiGluBlock;
const Tensor = @import("../tensor.zig").Tensor;
const ContiguousKVCache = @import("cache.zig").ContiguousKVCache;
const ForwardTimings = @import("../forward_timings.zig").ForwardTimings;

pub const Config = struct {
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rope_theta: f32,
};

pub const AttentionContext = struct {
    gpa: std.mem.Allocator,
    prefill: bool,
    /// borrowed caches in batch order, for this transformer layer only.
    kv_caches: []const *ContiguousKVCache,
    /// mapping to look up the actual non-padded tokens for a row in the batch
    real_token_counts: []const usize,
    timings: ?*ForwardTimings = null,
};

pub const TransformerBlock = struct {
    const Self = @This();

    attention: AttentionBlock,
    mlp: SwiGluBlock,
    input_layernorm_w: Tensor,
    post_attention_layernorm_w: Tensor,

    pub fn init(weights: Map(.tensor), layer: usize, config: Config) !Self {
        var attention = try AttentionBlock.init(weights, layer, config);
        errdefer attention.deinit();
        var mlp = try SwiGluBlock.init(weights, layer);
        errdefer mlp.deinit();
        var input_layernorm_w = try loadWeight(weights, layer, "input_layernorm.weight");
        errdefer input_layernorm_w.deinit();
        var post_attention_layernorm_w = try loadWeight(weights, layer, "post_attention_layernorm.weight");
        errdefer post_attention_layernorm_w.deinit();

        return .{
            .attention = attention,
            .mlp = mlp,
            .input_layernorm_w = input_layernorm_w,
            .post_attention_layernorm_w = post_attention_layernorm_w,
        };
    }

    pub fn deinit(self: *Self) void {
        self.attention.deinit();
        self.mlp.deinit();
        self.input_layernorm_w.deinit();
        self.post_attention_layernorm_w.deinit();
    }

    pub fn forward(self: *Self, x: Tensor, strm: Stream, opts: AttentionContext) !Tensor {
        if (try x.dim(0) != opts.kv_caches.len) {
            return error.InconsistentForwardOptions;
        }

        var rms_1 = try Tensor.rms(x, self.input_layernorm_w, strm);
        defer rms_1.deinit();

        var attention_output = try self.attention.forward(rms_1, strm, opts);
        defer attention_output.deinit();

        var h = try Tensor.add(attention_output, x, strm);
        defer h.deinit();

        var rms_2 = try Tensor.rms(h, self.post_attention_layernorm_w, strm);
        defer rms_2.deinit();

        var mlp_output = try self.mlp.forward(rms_2, strm);
        defer mlp_output.deinit();

        const output = try Tensor.add(mlp_output, h, strm);
        return output;
    }
};

pub const AttentionBlock = struct {
    const Self = @This();

    q_w: Tensor,
    q_bias: Tensor,
    k_w: Tensor,
    k_bias: Tensor,
    v_w: Tensor,
    v_bias: Tensor,
    o_proj_w: Tensor,
    num_attention_heads: c_int,
    num_key_value_heads: c_int,
    rope_theta: f32,

    pub fn init(weights: Map(.tensor), layer: usize, config: Config) !Self {
        const num_attention_heads = std.math.cast(c_int, config.num_attention_heads) orelse return error.InvalidAttentionHeadCount;
        const num_key_value_heads = std.math.cast(c_int, config.num_key_value_heads) orelse return error.InvalidAttentionHeadCount;

        var q_w = try loadWeight(weights, layer, "self_attn.q_proj.weight");
        errdefer q_w.deinit();

        var q_bias = try loadWeight(weights, layer, "self_attn.q_proj.bias");
        errdefer q_bias.deinit();

        var k_w = try loadWeight(weights, layer, "self_attn.k_proj.weight");
        errdefer k_w.deinit();

        var k_bias = try loadWeight(weights, layer, "self_attn.k_proj.bias");
        errdefer k_bias.deinit();

        var v_w = try loadWeight(weights, layer, "self_attn.v_proj.weight");
        errdefer v_w.deinit();

        var v_bias = try loadWeight(weights, layer, "self_attn.v_proj.bias");
        errdefer v_bias.deinit();

        var o_proj_w = try loadWeight(weights, layer, "self_attn.o_proj.weight");
        errdefer o_proj_w.deinit();

        return .{
            .q_w = q_w,
            .q_bias = q_bias,
            .k_w = k_w,
            .k_bias = k_bias,
            .v_w = v_w,
            .v_bias = v_bias,
            .o_proj_w = o_proj_w,
            .num_attention_heads = num_attention_heads,
            .num_key_value_heads = num_key_value_heads,
            .rope_theta = config.rope_theta,
        };
    }

    pub fn deinit(self: *Self) void {
        self.q_w.deinit();
        self.q_bias.deinit();
        self.k_w.deinit();
        self.k_bias.deinit();
        self.v_w.deinit();
        self.v_bias.deinit();
        self.o_proj_w.deinit();
    }

    pub fn forward(self: *Self, x: Tensor, strm: Stream, opts: AttentionContext) !Tensor {
        const input_shape = try x.shape();

        // TODO: do this once on init and resuse the transposed handles (unsure if this will improve perf)
        var q_w_tran = try self.q_w.transpose(strm);
        defer q_w_tran.deinit();
        var k_w_tran = try self.k_w.transpose(strm);
        defer k_w_tran.deinit();
        var v_w_tran = try self.v_w.transpose(strm);
        defer v_w_tran.deinit();

        var q = try Tensor.matMul(x, q_w_tran, strm);
        defer q.deinit();
        try q.addMut(self.q_bias, strm);
        var k = try Tensor.matMul(x, k_w_tran, strm);
        defer k.deinit();
        try k.addMut(self.k_bias, strm);
        var v = try Tensor.matMul(x, v_w_tran, strm);
        defer v.deinit();
        try v.addMut(self.v_bias, strm);

        var q_heads = try Tensor.reshape(q, &[_]c_int{
            input_shape[0], // batch
            input_shape[1], // tokens
            self.num_attention_heads, // heads
            -1, // infer head dimension: 64
        }, strm);
        defer q_heads.deinit();
        try q_heads.permuteMut(&.{ 0, 2, 1, 3 }, strm);

        const head_size = (try q_heads.shape())[3];

        for (0..try q_heads.dim(0)) |r| {
            const batch_index: c_int = @intCast(r);
            const row_width = input_shape[1];
            const first_token = row_width - @as(c_int, @intCast(opts.real_token_counts[r]));

            const start: []const c_int = &.{
                batch_index,
                0,
                first_token,
                0,
            };
            const stop: []const c_int = &.{
                batch_index + 1,
                self.num_attention_heads,
                row_width,
                head_size,
            };
            const strides: []const c_int = &.{ 1, 1, 1, 1 };

            var q_row = try q_heads.slice(
                start,
                stop,
                strides,
                strm,
            );
            defer q_row.deinit();
            const offset: c_int = @intCast(opts.kv_caches[r].len);

            try q_row.ropeMut(head_size, self.rope_theta, offset, strm);

            try q_heads.sliceUpdateMut(
                q_row,
                start,
                stop,
                strides,
                strm,
            );
        }

        const cache_mask_started = if (opts.timings) |t| Clock.now(t.io) else null;
        const kv_shape = try k.shape();
        const new_kv_shape = [_]c_int{
            kv_shape[0], // batch
            kv_shape[1], // tokens
            self.num_key_value_heads, // heads
            -1, // infer head dimension: 64
        };

        var v_heads = try Tensor.reshape(v, &new_kv_shape, strm);
        defer v_heads.deinit();
        try v_heads.permuteMut(&.{ 0, 2, 1, 3 }, strm);

        var k_heads = try Tensor.reshape(k, &new_kv_shape, strm);
        defer k_heads.deinit();
        try k_heads.permuteMut(&.{ 0, 2, 1, 3 }, strm);
        for (0..try k_heads.dim(0)) |r| {
            const batch_index: c_int = @intCast(r);
            const row_width = input_shape[1];
            const first_token = row_width - @as(c_int, @intCast(opts.real_token_counts[r]));

            const start: []const c_int = &.{
                batch_index,
                0,
                first_token,
                0,
            };
            const stop: []const c_int = &.{
                batch_index + 1,
                self.num_key_value_heads,
                row_width,
                head_size,
            };
            const strides: []const c_int = &.{ 1, 1, 1, 1 };

            var k_row = try k_heads.slice(start, stop, strides, strm);
            defer k_row.deinit();
            const offset: c_int = @intCast(opts.kv_caches[r].len);

            try k_row.ropeMut(head_size, self.rope_theta, offset, strm);
            var v_row = try v_heads.slice(start, stop, strides, strm);
            defer v_row.deinit();

            try opts.kv_caches[r].append(.{ .k = k_row, .v = v_row });
        }

        var max_cache_len: usize = 0;
        for (opts.kv_caches) |row_cache| {
            max_cache_len = @max(max_cache_len, row_cache.len);
        }

        const cache_shape = [_]c_int{
            input_shape[0],
            self.num_key_value_heads,
            @intCast(max_cache_len),
            head_size,
        };

        var resolved_k = try Tensor.zeros(&cache_shape, try k_heads.dtype(), strm);
        defer resolved_k.deinit();

        var resolved_v = try Tensor.zeros(&cache_shape, try v_heads.dtype(), strm);
        defer resolved_v.deinit();

        for (opts.kv_caches, 0..) |row_cache, r| {
            const kv = try row_cache.get();
            var cached_k = kv.k;
            defer cached_k.deinit();
            var cached_v = kv.v;
            defer cached_v.deinit();

            const batch_index: c_int = @intCast(r);
            const padding: c_int = @intCast(max_cache_len - row_cache.len);

            const start: []const c_int = &.{ batch_index, 0, padding, 0 };
            const stop: []const c_int = &.{
                batch_index + 1,
                self.num_key_value_heads,
                cache_shape[2],
                head_size,
            };
            const strides: []const c_int = &.{ 1, 1, 1, 1 };

            try resolved_k.sliceUpdateMut(cached_k, start, stop, strides, strm);
            try resolved_v.sliceUpdateMut(cached_v, start, stop, strides, strm);
        }

        // TODO: understand this better start ---------------------------------------
        const batch_size = opts.kv_caches.len;
        const query_width: usize = @intCast(input_shape[1]);
        const values = try opts.gpa.alloc(
            f16,
            batch_size * query_width * max_cache_len,
        );
        defer opts.gpa.free(values);

        @memset(values, -std.math.inf(f16));
        for (opts.kv_caches, 0..) |row_cache, b| {
            const incoming = opts.real_token_counts[b];
            const query_padding = query_width - incoming;
            const key_padding = max_cache_len - row_cache.len;
            const old_tokens = row_cache.len - incoming;

            for (0..query_width) |i| {
                // Find the beginning of this query's mask row.
                const row_start = (b * query_width + i) * max_cache_len;

                const allowed_count = if (i < query_padding)
                    1 // Dummy attention for a padded query.
                else
                    old_tokens + (i - query_padding) + 1;

                const start = row_start + key_padding;
                @memset(values[start .. start + allowed_count], 0);
            }
        }

        var attention_mask = try Tensor.fromSlice(f16, values, &.{
            input_shape[0],
            1,
            input_shape[1],
            cache_shape[2],
        });
        defer attention_mask.deinit();
        // TODO: understand this better stop ---------------------------------------

        if (opts.timings) |t| {
            t.cache_mask_build_ns += cache_mask_started.?.durationTo(Clock.now(t.io)).toNanoseconds();
        }

        var attention_output = try Tensor.scaledDotProductAttention(
            q_heads,
            resolved_k,
            resolved_v,
            1 / @sqrt(@as(f32, @floatFromInt(head_size))),
            .{ .tensor = attention_mask },
            strm,
        );
        errdefer attention_output.deinit();
        try attention_output.permuteMut(&.{ 0, 2, 1, 3 }, strm);
        try attention_output.reshapeMut(&.{ input_shape[0], input_shape[1], -1 }, strm);

        var o_proj_w_tran = try self.o_proj_w.transpose(strm);
        defer o_proj_w_tran.deinit();
        try attention_output.matMulMut(o_proj_w_tran, strm);
        return attention_output;
    }
};
