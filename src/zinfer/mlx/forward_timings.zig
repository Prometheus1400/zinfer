const std = @import("std");

/// Optional wall-clock observations for one forward call. These add no MLX
/// evaluation barriers: GPU cache work remains part of evaluation/readback.
pub const ForwardTimings = struct {
    io: std.Io,
    /// Host construction of cache updates, packed KV tensors, and masks,
    /// accumulated across transformer layers. A subset of graph_build_ns.
    cache_mask_build_ns: i96 = 0,
    /// Host construction of the complete model/sampling graph.
    graph_build_ns: i96 = 0,
    /// Existing sampled-token evaluation and host readback loop.
    evaluation_ns: i96 = 0,
};
