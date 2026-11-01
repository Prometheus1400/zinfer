const std = @import("std");
const c = @import("mlx_c");
const common = @import("../common.zig");
const Tensor = @import("../tensor.zig").Tensor;
const Map = @import("../map.zig").Map;
const MlxStream = @import("../stream.zig").MlxStream;
const Layer = @import("layer.zig");
const SwiGluBlock = @import("swiglu.zig").SwiGluBlock;
const loadWeight = @import("weights.zig").load;

pub const TransformerBlock = struct {
    const Self = @This();

    attention: AttentionBlock,
    mlp: SwiGluBlock,
    input_layernorm_w: Tensor,
    post_attention_layernorm_w: Tensor,

    pub const Config = AttentionBlock.Config;

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
        Layer.deinit(&self.attention);
        Layer.deinit(&self.mlp);
        self.input_layernorm_w.deinit();
        self.post_attention_layernorm_w.deinit();
    }

    pub fn forward(self: *Self, x: Tensor, strm: MlxStream) !Tensor {
        var rms_1 = try Tensor.rms(x, self.input_layernorm_w, strm);
        defer rms_1.deinit();
        var attention_output = try Layer.forward(&self.attention, rms_1, strm);
        defer attention_output.deinit();
        var h = try Tensor.add(attention_output, x, strm);
        defer h.deinit();

        var rms_2 = try Tensor.rms(h, self.post_attention_layernorm_w, strm);
        defer rms_2.deinit();
        var mlp_output = try Layer.forward(&self.mlp, rms_2, strm);
        defer mlp_output.deinit();
        return Tensor.add(mlp_output, h, strm);
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
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rope_theta: f32,

    pub const Config = struct {
        num_attention_heads: usize,
        num_key_value_heads: usize,
        rope_theta: f32,
    };

    pub fn init(weights: Map(.tensor), layer: usize, config: Config) !Self {
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
            .num_attention_heads = config.num_attention_heads,
            .num_key_value_heads = config.num_key_value_heads,
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

    pub fn forward(self: *Self, x: Tensor, strm: MlxStream) !Tensor {
        const input_shape = try x.shape();

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

        const new_q_shape = [_]c_int{
            input_shape[0], // batch
            input_shape[1], // tokens
            @intCast(self.num_attention_heads), // heads
            -1, // infer head dimension: 64
        };
        var q_heads = try Tensor.reshape(q, &new_q_shape, strm);
        defer q_heads.deinit();
        try q_heads.permuteMut(&.{ 0, 2, 1, 3 }, strm);
        const head_size: c_int = @intCast(try q_heads.dim(3));
        try q_heads.ropeMut(head_size, self.rope_theta, 0, strm);
        const q_heads_shape = try q_heads.shape();

        const new_kv_shape = [_]c_int{
            input_shape[0], // batch
            input_shape[1], // tokens
            @intCast(self.num_key_value_heads), // heads
            -1, // infer head dimension: 64
        };
        var k_heads = try Tensor.reshape(k, &new_kv_shape, strm);
        defer k_heads.deinit();
        try k_heads.permuteMut(&.{ 0, 2, 1, 3 }, strm);
        try k_heads.ropeMut(head_size, self.rope_theta, 0, strm);

        var v_heads = try Tensor.reshape(v, &new_kv_shape, strm);
        defer v_heads.deinit();
        try v_heads.permuteMut(&.{ 0, 2, 1, 3 }, strm);

        const v_heads_shape = try v_heads.shape();
        var v_head_1 = try v_heads.slice(&.{ 0, 0, 0, 0 }, &.{ 1, 1, v_heads_shape[2], v_heads_shape[3] }, &.{ 1, 1, 1, 1 }, strm);
        defer v_head_1.deinit();
        try v_head_1.reshapeMut(&.{ v_heads_shape[2], v_heads_shape[3] }, strm);

        var v_head_2 = try v_heads.slice(&.{ 0, 1, 0, 0 }, &.{ 1, 2, v_heads_shape[2], v_heads_shape[3] }, &.{ 1, 1, 1, 1 }, strm);
        defer v_head_2.deinit();
        try v_head_2.reshapeMut(&.{ v_heads_shape[2], v_heads_shape[3] }, strm);

        const k_heads_shape = try k_heads.shape();
        var k_head_1 = try k_heads.slice(&.{ 0, 0, 0, 0 }, &.{ 1, 1, k_heads_shape[2], k_heads_shape[3] }, &.{ 1, 1, 1, 1 }, strm);
        defer k_head_1.deinit();
        try k_head_1.reshapeMut(&.{ k_heads_shape[2], k_heads_shape[3] }, strm);

        var k_head_2 = try k_heads.slice(&.{ 0, 1, 0, 0 }, &.{ 1, 2, k_heads_shape[2], k_heads_shape[3] }, &.{ 1, 1, 1, 1 }, strm);
        defer k_head_2.deinit();
        try k_head_2.reshapeMut(&.{ k_heads_shape[2], k_heads_shape[3] }, strm);

        const token_count = q_heads_shape[2];
        const num_heads: usize = @intCast(q_heads_shape[1]);
        var scale = Tensor.scalar(@sqrt(@as(f32, @floatFromInt(head_size))));
        defer scale.deinit();
        var mask = try causalMask(token_count, strm);
        defer mask.deinit();

        var attention_output_opt: ?Tensor = null;
        defer if (attention_output_opt) |*a| a.deinit();

        for (0..num_heads) |head| {
            const h: c_int = @intCast(head);

            var q_head = try q_heads.slice(
                &.{ 0, h, 0, 0 },
                &.{ 1, h + 1, token_count, head_size },
                &.{ 1, 1, 1, 1 },
                strm,
            );
            defer q_head.deinit();

            try q_head.reshapeMut(&.{ token_count, head_size }, strm);

            const k_head = if (head < num_heads / 2) k_head_1 else k_head_2;
            const v_head = if (head < num_heads / 2) v_head_1 else v_head_2;

            var k_head_t = try k_head.transpose(strm);
            defer k_head_t.deinit();

            try q_head.matMulMut(k_head_t, strm);
            try q_head.divMut(scale, strm);
            try q_head.addMut(mask, strm);

            try q_head.softmaxMut(-1, strm);
            try q_head.matMulMut(v_head, strm);

            if (attention_output_opt) |*attention_output| {
                try attention_output.concatMut(q_head, -1, strm);
            } else {
                attention_output_opt = try q_head.clone();
            }
        }

        var o_proj_w_tran = try self.o_proj_w.transpose(strm);
        defer o_proj_w_tran.deinit();
        try attention_output_opt.?.matMulMut(o_proj_w_tran, strm);

        const output = attention_output_opt.?;
        attention_output_opt = null;
        return output;
    }
};

// Zeros on and below the diagonal; negative infinity above it.
fn causalMask(token_count: c_int, strm: MlxStream) !Tensor {
    try common.ensure(.{strm});

    var blocked = Tensor.scalar(-std.math.inf(f32));
    defer blocked.deinit();

    const shape = [_]c_int{ token_count, token_count };
    var full = c.mlx_array_new();
    defer _ = c.mlx_array_free(full);
    try common.check(c.mlx_full(&full, &shape, shape.len, blocked.p, c.MLX_FLOAT32, strm.p));

    var res = c.mlx_array_new();
    errdefer _ = c.mlx_array_free(res);
    // Keep only entries strictly above the diagonal; set the rest to zero.
    try common.check(c.mlx_triu(&res, full, 1, strm.p));
    return .{ .p = res };
}

test "init loads the requested layer and retains weights after map cleanup" {
    var weights: Map(.tensor) = .{ .p = c.mlx_map_string_to_array_new() };
    defer weights.deinit();
    const config: TransformerBlock.Config = .{
        .num_attention_heads = 14,
        .num_key_value_heads = 2,
        .rope_theta = 1_000_000,
    };
    try std.testing.expectError(error.MlxGenericError, TransformerBlock.init(weights, 3, config));

    const fields = .{
        .{ "q_w", "self_attn.q_proj.weight" },
        .{ "q_bias", "self_attn.q_proj.bias" },
        .{ "k_w", "self_attn.k_proj.weight" },
        .{ "k_bias", "self_attn.k_proj.bias" },
        .{ "v_w", "self_attn.v_proj.weight" },
        .{ "v_bias", "self_attn.v_proj.bias" },
        .{ "input_layernorm_w", "input_layernorm.weight" },
        .{ "post_attention_layernorm_w", "post_attention_layernorm.weight" },
        .{ "gate_proj_w", "mlp.gate_proj.weight" },
        .{ "up_proj_w", "mlp.up_proj.weight" },
        .{ "down_proj_w", "mlp.down_proj.weight" },
        .{ "o_proj_w", "self_attn.o_proj.weight" },
    };
    inline for (fields, 0..) |field, index| {
        var buf: [160]u8 = undefined;
        var other_layer = Tensor.scalar(-1);
        defer other_layer.deinit();
        const other_key = try std.mem.printSentinel(&buf, "model.layers.0.{s}", .{field[1]}, 0);
        try common.check(c.mlx_map_string_to_array_insert(weights.p, other_key.ptr, other_layer.p));

        var value = Tensor.scalar(@floatFromInt(index + 1));
        defer value.deinit();
        const key = try std.mem.printSentinel(&buf, "model.layers.3.{s}", .{field[1]}, 0);
        try common.check(c.mlx_map_string_to_array_insert(weights.p, key.ptr, value.p));
    }

    var block = try TransformerBlock.init(weights, 3, config);
    defer block.deinit();
    weights.deinit();
    try std.testing.expectEqual(config.num_attention_heads, block.attention.num_attention_heads);
    try std.testing.expectEqual(config.num_key_value_heads, block.attention.num_key_value_heads);
    try std.testing.expectEqual(config.rope_theta, block.attention.rope_theta);
    const loaded = [_]Tensor{
        block.attention.q_w,
        block.attention.q_bias,
        block.attention.k_w,
        block.attention.k_bias,
        block.attention.v_w,
        block.attention.v_bias,
        block.input_layernorm_w,
        block.post_attention_layernorm_w,
        block.mlp.gate_proj_w,
        block.mlp.up_proj_w,
        block.mlp.down_proj_w,
        block.attention.o_proj_w,
    };
    for (loaded, 0..) |tensor, index| {
        try common.check(c.mlx_array_eval(tensor.p));
        try std.testing.expectEqual(@as(f32, @floatFromInt(index + 1)), c.mlx_array_data_float32(tensor.p)[0]);
    }
}

test "causal mask blocks future tokens before softmax" {
    const build_options = @import("build_options");
    var strm = try @import("../stream.zig").init(.{ .require_gpu = build_options.metal });
    defer strm.deinit();

    var mask = try causalMask(3, strm);
    defer mask.deinit();
    try std.testing.expectEqualSlices(c_int, &.{ 3, 3 }, try mask.shape());
    try common.check(c.mlx_array_eval(mask.p));
    const inf = std.math.inf(f32);
    const mask_data = c.mlx_array_data_float32(mask.p);
    try std.testing.expectEqualSlices(f32, &.{ 0, -inf, -inf, 0, 0, -inf, 0, 0, 0 }, mask_data[0..9]);

    // Future scores are deliberately much larger than the allowed scores.
    var scores = try Tensor.fromSlice(f32, &.{ 0, 1000, 1000, 0, 0, 1000, 0, 0, 0 }, &.{ 3, 3 });
    defer scores.deinit();
    try scores.addMut(mask, strm);
    try scores.softmaxMut(-1, strm);
    try common.check(c.mlx_array_eval(scores.p));
    const actual = c.mlx_array_data_float32(scores.p);
    const expected = [_]f32{ 1, 0, 0, 0.5, 0.5, 0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
    for (expected, actual[0..expected.len]) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 1e-5);
    }
}
