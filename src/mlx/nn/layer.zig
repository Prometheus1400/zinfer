const Tensor = @import("../tensor.zig").Tensor;
const Stream = @import("../stream.zig").MlxStream;

/// The concrete pointer type specializes this call at compile time.
/// Each layer borrows its input and returns an owned Tensor.
pub fn forward(layer: anytype, x: Tensor, strm: Stream) !Tensor {
    return layer.forward(x, strm);
}

pub fn deinit(layer: anytype) void {
    layer.deinit();
}
