const loadWeight = @import("weights.zig").load;

const Map = @import("../map.zig").Map;
const Stream = @import("../stream.zig").Stream;
const SwiGluBlock = @import("swiglu.zig").SwiGluBlock;
const Tensor = @import("../tensor.zig").Tensor;

pub const Config = struct {
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rope_theta: f32,
};

pub const AttentionContext = struct {
    prefill: bool,
    cached_k: *?Tensor,
    cached_v: *?Tensor,
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

    pub fn forward(self: *Self, x: Tensor, strm: Stream, options: AttentionContext) !Tensor {
        var rms_1 = try Tensor.rms(x, self.input_layernorm_w, strm);
        defer rms_1.deinit();

        var attention_output = try self.attention.forward(rms_1, strm, options);
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
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rope_theta: f32,

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

    pub fn forward(self: *Self, x: Tensor, strm: Stream, options: AttentionContext) !Tensor {
        if ((options.cached_k.* == null and options.cached_v.* != null) or (options.cached_k.* != null and options.cached_v.* == null)) {
            return error.InconsistentForwardOptions;
        }

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

        // zig fmt: off
        var q_heads = try Tensor.reshape(q, &[_]c_int{
            input_shape[0],                     // batch
            input_shape[1],                     // tokens
            @intCast(self.num_attention_heads), // heads
            -1,                                 // infer head dimension: 64
        }, strm);
        // zig fmt: on
        defer q_heads.deinit();
        try q_heads.permuteMut(&.{ 0, 2, 1, 3 }, strm);

        const head_size: c_int = @intCast(try q_heads.dim(3));
        const offset: c_int = if (options.cached_k.*) |*cached_k|
            @intCast(try cached_k.dim(2))
        else
            0;
        try q_heads.ropeMut(head_size, self.rope_theta, offset, strm);

        const kv_shape = try k.shape();
        // zig fmt: off
        const new_kv_shape = [_]c_int{
            kv_shape[0],                        // batch
            kv_shape[1],                        // tokens
            @intCast(self.num_key_value_heads), // heads
            -1,                                 // infer head dimension: 64
        };
        // zig fmt: on
        var k_heads = try Tensor.reshape(k, &new_kv_shape, strm);
        defer k_heads.deinit();
        try k_heads.permuteMut(&.{ 0, 2, 1, 3 }, strm);
        try k_heads.ropeMut(head_size, self.rope_theta, offset, strm);

        var v_heads = try Tensor.reshape(v, &new_kv_shape, strm);
        defer v_heads.deinit();
        try v_heads.permuteMut(&.{ 0, 2, 1, 3 }, strm);

        var resolved_k: Tensor = undefined;
        var resolved_v: Tensor = undefined;
        if (options.prefill and options.cached_k.* == null and options.cached_v.* == null) {
            options.cached_k.* = try k_heads.clone();
            options.cached_v.* = try v_heads.clone();
            resolved_k = k_heads;
            resolved_v = v_heads;
        } else {
            try options.cached_k.*.?.concatMut(k_heads, 2, strm);
            try options.cached_v.*.?.concatMut(v_heads, 2, strm);
            resolved_k = options.cached_k.*.?;
            resolved_v = options.cached_v.*.?;
        }

        var attention_output = try Tensor.scaledDotProductAttention(q_heads, resolved_k, resolved_v, 1 / @sqrt(@as(f32, @floatFromInt(head_size))), .causal, strm);
        errdefer attention_output.deinit();
        try attention_output.permuteMut(&.{ 0, 2, 1, 3 }, strm);
        try attention_output.reshapeMut(&.{ input_shape[0], input_shape[1], -1 }, strm);

        var o_proj_w_tran = try self.o_proj_w.transpose(strm);
        defer o_proj_w_tran.deinit();
        try attention_output.matMulMut(o_proj_w_tran, strm);
        return attention_output;
    }
};
