const Tensor = @import("../tensor.zig").Tensor;
const Map = @import("../map.zig").Map;
const Stream = @import("../stream.zig").Stream;
const loadWeight = @import("weights.zig").load;

pub const SwiGluBlock = struct {
    const Self = @This();

    gate_proj_w: Tensor,
    up_proj_w: Tensor,
    down_proj_w: Tensor,

    pub fn init(weights: Map(.tensor), layer: usize) !Self {
        var gate_proj_w = try loadWeight(weights, layer, "mlp.gate_proj.weight");
        errdefer gate_proj_w.deinit();

        var up_proj_w = try loadWeight(weights, layer, "mlp.up_proj.weight");
        errdefer up_proj_w.deinit();

        var down_proj_w = try loadWeight(weights, layer, "mlp.down_proj.weight");
        errdefer down_proj_w.deinit();

        return .{
            .gate_proj_w = gate_proj_w,
            .up_proj_w = up_proj_w,
            .down_proj_w = down_proj_w,
        };
    }

    pub fn deinit(self: *Self) void {
        self.gate_proj_w.deinit();
        self.up_proj_w.deinit();
        self.down_proj_w.deinit();
    }

    pub fn forward(self: *Self, x: Tensor, strm: Stream) !Tensor {
        var gate_proj_w_tran = try self.gate_proj_w.transpose(strm);
        defer gate_proj_w_tran.deinit();
        var mlp_output = try Tensor.matMul(x, gate_proj_w_tran, strm);
        errdefer mlp_output.deinit();
        try mlp_output.siluMut(strm);

        var up_proj_w_tran = try self.up_proj_w.transpose(strm);
        defer up_proj_w_tran.deinit();
        var up_proj_output = try Tensor.matMul(x, up_proj_w_tran, strm);
        defer up_proj_output.deinit();

        try mlp_output.mulMut(up_proj_output, strm);

        var down_proj_w_tran = try self.down_proj_w.transpose(strm);
        defer down_proj_w_tran.deinit();
        try mlp_output.matMulMut(down_proj_w_tran, strm);

        return mlp_output;
    }
};
