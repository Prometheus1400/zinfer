pub const Tensor = @import("tensor.zig").Tensor;
pub const DType = @import("dtype.zig").DType;
pub const Stream = @import("stream.zig").MlxStream;
pub const InitOptions = @import("stream.zig").InitOptions;
pub const init = @import("stream.zig").init;
pub const Map = @import("map.zig").Map;

pub const io = @import("io.zig");
pub const nn = @import("nn/root.zig");

test "clone survives replacing and freeing the original handle" {
    const std = @import("std");
    const c = @import("mlx_c");
    const build_options = @import("build_options");
    var strm = try init(.{ .require_gpu = build_options.metal });
    defer strm.deinit();

    var cloned = blk: {
        var input = try Tensor.fromSlice(f32, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
        defer input.deinit();
        var original = try Tensor.add(input, input, strm);
        defer original.deinit();
        var result = try original.clone();
        errdefer result.deinit();
        try std.testing.expect(result.p.ctx != original.p.ctx);
        try original.reshapeMut(&.{4}, strm);
        try std.testing.expectEqualSlices(c_int, &.{ 2, 2 }, try result.shape());
        break :blk result;
    };
    defer cloned.deinit();

    try cloned.eval();
    const actual = c.mlx_array_data_float32(cloned.p);
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6, 8 }, actual[0..4]);
    cloned.deinit();
    try std.testing.expectError(error.NullArray, cloned.clone());
}

test "concat joins columns and concatMut joins rows while retaining inputs" {
    const std = @import("std");
    const c = @import("mlx_c");
    const common = @import("common.zig");
    const build_options = @import("build_options");
    var strm = try init(.{ .require_gpu = build_options.metal });
    defer strm.deinit();

    var a = try Tensor.fromSlice(f32, &.{ 1, 2, 3, 4 }, &.{ 2, 2 });
    defer a.deinit();
    var b = try Tensor.fromSlice(f32, &.{ 5, 6 }, &.{ 2, 1 });
    defer b.deinit();
    var columns = try a.concat(b, -1, strm);
    defer columns.deinit();
    try a.concatMut(a, 0, strm);
    var rows = try a.clone();
    defer rows.deinit();
    a.deinit();
    b.deinit();

    try std.testing.expectEqualSlices(c_int, &.{ 2, 3 }, try columns.shape());
    try common.check(c.mlx_array_eval(columns.p));
    const column_data = c.mlx_array_data_float32(columns.p);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 5, 3, 4, 6 }, column_data[0..6]);

    try std.testing.expectEqualSlices(c_int, &.{ 4, 2 }, try rows.shape());
    try common.check(c.mlx_array_eval(rows.p));
    const row_data = c.mlx_array_data_float32(rows.p);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 1, 2, 3, 4 }, row_data[0..8]);
}

test "slice selects a head and retains its input" {
    const std = @import("std");
    const c = @import("mlx_c");
    const common = @import("common.zig");
    const build_options = @import("build_options");
    var strm = try init(.{ .require_gpu = build_options.metal });
    defer strm.deinit();

    var selected = blk: {
        var input = try Tensor.fromSlice(f32, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, &.{ 1, 2, 2, 2 });
        defer input.deinit();
        break :blk try input.slice(&.{ 0, 1, 0, 0 }, &.{ 1, 2, 2, 2 }, &.{ 1, 1, 1, 1 }, strm);
    };
    defer selected.deinit();

    try std.testing.expectEqualSlices(c_int, &.{ 1, 1, 2, 2 }, try selected.shape());
    try common.check(c.mlx_array_eval(selected.p));
    const actual = c.mlx_array_data_float32(selected.p);
    try std.testing.expectEqualSlices(f32, &.{ 4, 5, 6, 7 }, actual[0..4]);
}

test "softmax normalizes each row with large scores" {
    const std = @import("std");
    const c = @import("mlx_c");
    const common = @import("common.zig");
    const build_options = @import("build_options");
    var strm = try init(.{ .require_gpu = build_options.metal });
    defer strm.deinit();

    var scores = try Tensor.fromSlice(f32, &.{ 1000, 1001, 1002, 0, 0, 0 }, &.{ 2, 3 });
    defer scores.deinit();
    var probabilities = try scores.softmax(-1, strm);
    defer probabilities.deinit();
    try scores.softmaxMut(1, strm);

    const expected = [_]f32{ 0.09003057, 0.24472847, 0.66524096, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
    for ([_]Tensor{ probabilities, scores }) |result| {
        try std.testing.expectEqualSlices(c_int, &.{ 2, 3 }, try result.shape());
        try common.check(c.mlx_array_eval(result.p));
        const actual = c.mlx_array_data_float32(result.p);
        for (expected, actual[0..expected.len]) |want, got| {
            try std.testing.expectApproxEqAbs(want, got, 1e-5);
        }
    }
}

test "RoPE matches split-half rotations at zero and nonzero offsets" {
    const std = @import("std");
    const c = @import("mlx_c");
    const common = @import("common.zig");
    const build_options = @import("build_options");
    var strm = try init(.{ .require_gpu = build_options.metal });
    defer strm.deinit();

    const data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const base: f32 = 1_000_000;
    var input = try Tensor.fromSlice(f32, &data, &.{ 1, 2, 4 });
    defer input.deinit();
    var rotated = try input.rope(4, base, 0, strm);
    defer rotated.deinit();
    try input.ropeMut(4, base, 3, strm);

    const Case = struct { tensor: Tensor, offset: f32 };
    for ([_]Case{ .{ .tensor = rotated, .offset = 0 }, .{ .tensor = input, .offset = 3 } }) |case| {
        try std.testing.expectEqualSlices(c_int, &.{ 1, 2, 4 }, try case.tensor.shape());
        try common.check(c.mlx_array_eval(case.tensor.p));
        const actual = c.mlx_array_data_float32(case.tensor.p);
        for (0..2) |token| {
            for (0..2) |pair| {
                const position = case.offset + @as(f32, @floatFromInt(token));
                const exponent = @as(f32, @floatFromInt(pair)) / 2.0;
                const angle = position / std.math.pow(f32, base, exponent);
                const i = token * 4 + pair;
                const x = data[i];
                const y = data[i + 2];
                try std.testing.expectApproxEqAbs(x * @cos(angle) - y * @sin(angle), actual[i], 1e-5);
                try std.testing.expectApproxEqAbs(x * @sin(angle) + y * @cos(angle), actual[i + 2], 1e-5);
            }
        }
    }
}

test "probe" {
    const std = @import("std");
    const c = @import("mlx_c");
    const common = @import("common.zig");
    const build_options = @import("build_options");
    var strm = try init(.{ .require_gpu = build_options.metal });
    defer strm.deinit();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const b_data = [_]f32{ 5, 6, 7, 8 };
    const shape = [_]c_int{ 2, 2 };

    var a = try Tensor.fromSlice(f32, &a_data, &shape);
    defer a.deinit();
    try std.testing.expectEqualSlices(c_int, &shape, try a.shape());
    var b = try Tensor.fromSlice(f32, &b_data, &shape);
    defer b.deinit();

    var res = try a.matMul(b, strm);
    defer res.deinit();

    const expected = [_]f32{ 19, 22, 43, 50 };
    try std.testing.expectEqual(@as(usize, 2), c.mlx_array_ndim(res.p));
    try std.testing.expectEqual(@as(c_int, 2), c.mlx_array_dim(res.p, 0));
    try std.testing.expectEqual(@as(c_int, 2), c.mlx_array_dim(res.p, 1));
    try common.check(c.mlx_array_eval(res.p));
    const values = c.mlx_array_data_float32(res.p);
    try std.testing.expectEqualSlices(f32, &expected, values[0..expected.len]);
}

test "transformer blocks compile with generic Layer helpers" {
    const std = @import("std");
    const TransformerBlock = @import("nn/transformer.zig").TransformerBlock;
    std.mem.doNotOptimizeAway(&TransformerBlock.forward);
}

test "tensor and stream cleanup accepts empty handles while operations reject them" {
    const std = @import("std");
    const build_options = @import("build_options");

    var empty_tensor = Tensor.new();
    try std.testing.expectError(error.NullArray, empty_tensor.ensure());
    try std.testing.expectError(error.NullArray, empty_tensor.eval());
    empty_tensor.deinit();
    try std.testing.expect(empty_tensor.p.ctx == null);

    var empty_stream: Stream = .{ .p = .{ .ctx = null } };
    empty_stream.deinit();
    try std.testing.expectError(error.NullStream, empty_stream.ensure());

    var tensor = try Tensor.fromSlice(f32, &.{1}, &.{1});
    defer tensor.deinit();
    try tensor.ensure();
    tensor.deinit();
    try std.testing.expect(tensor.p.ctx == null);
    try std.testing.expectError(error.NullArray, tensor.eval());
    try std.testing.expectError(error.NullArray, tensor.ndim());
    tensor.deinit();

    var strm = try init(.{ .require_gpu = build_options.metal });
    defer strm.deinit();
    try strm.ensure();
    strm.deinit();
    try std.testing.expect(strm.p.ctx == null);
    try std.testing.expectError(error.NullStream, strm.ensure());
    strm.deinit();
}

test "map and iterator cleanup can be repeated while operations reject freed handles" {
    const std = @import("std");
    const c = @import("mlx_c");
    const MapValueType = @import("map.zig").MapValueType;

    inline for ([_]MapValueType{ .tensor, .string }) |value_type| {
        var empty_map: Map(value_type) = .{ .p = .{ .ctx = null } };
        empty_map.deinit();
        try std.testing.expectError(error.NullMap, empty_map.ensure());

        var empty_iterator: @import("map.zig").Iterator(value_type) = .{ .p = .{ .ctx = null, .map_ctx = null } };
        empty_iterator.deinit();
        try std.testing.expectError(error.NullMap, empty_iterator.ensure());

        var map: Map(value_type) = .{ .p = switch (value_type) {
            .tensor => c.mlx_map_string_to_array_new(),
            .string => c.mlx_map_string_to_string_new(),
        } };
        defer map.deinit();
        try map.ensure();
        var iterator = try map.iterator();
        defer iterator.deinit();
        try iterator.ensure();
        try std.testing.expect(iterator.next() == null);
        iterator.deinit();
        try std.testing.expect(iterator.p.ctx == null);
        try std.testing.expect(iterator.p.map_ctx == null);
        try std.testing.expectError(error.NullMap, iterator.ensure());
        iterator.deinit();
        try std.testing.expect(iterator.next() == null);
        map.deinit();
        try std.testing.expect(map.p.ctx == null);
        try std.testing.expectError(error.NullMap, map.ensure());
        try std.testing.expectError(error.NullMap, map.iterator());
        map.deinit();
    }
}

test "normalized projections and gated outputs retain lazy dependencies" {
    const std = @import("std");
    const c = @import("mlx_c");
    const common = @import("common.zig");
    const build_options = @import("build_options");
    var strm = try init(.{ .require_gpu = build_options.metal });
    defer strm.deinit();

    const input_data = [_]f32{ 1, 2, 3, -2, 1, -0.5 };
    const norm_data = [_]f32{ 1, 0.5, 2 };
    const weight_data = [_]f32{ 1, 0, -1, 0, 2, 1 };
    const bias_data = [_]f32{ 0.25, -0.5 };
    const down_data = [_]f32{ 1, 0, 0, 1, 1, -1 };

    // Release all intermediate handles before evaluating the final output.
    var output = blk: {
        var x = try Tensor.fromSlice(f32, &input_data, &.{ 1, 2, 3 });
        defer x.deinit();
        var norm_w = try Tensor.fromSlice(f32, &norm_data, &.{3});
        defer norm_w.deinit();
        var weight = try Tensor.fromSlice(f32, &weight_data, &.{ 2, 3 });
        defer weight.deinit();
        var bias = try Tensor.fromSlice(f32, &bias_data, &.{2});
        defer bias.deinit();
        var down_w = try Tensor.fromSlice(f32, &down_data, &.{ 3, 2 });
        defer down_w.deinit();

        var normalized = try Tensor.rms(x, norm_w, strm);
        defer normalized.deinit();
        var wt = try Tensor.transpose(weight, strm);
        defer wt.deinit();
        var projected = try Tensor.matMul(normalized, wt, strm);
        defer projected.deinit();
        var biased = try Tensor.add(projected, bias, strm);
        defer biased.deinit();
        var activated = try Tensor.silu(biased, strm);
        defer activated.deinit();
        var gated = try Tensor.mul(activated, biased, strm);
        defer gated.deinit();
        var down_wt = try Tensor.transpose(down_w, strm);
        defer down_wt.deinit();
        var down = try Tensor.matMul(gated, down_wt, strm);
        defer down.deinit();
        break :blk try Tensor.add(down, x, strm);
    };
    defer output.deinit();

    try std.testing.expectEqualSlices(c_int, &.{ 1, 2, 3 }, try output.shape());
    try common.check(c.mlx_array_eval(output.p));
    const actual = c.mlx_array_data_float32(output.p);

    // Compute the reference with scalar arithmetic, independently of MLX.
    for (0..2) |token| {
        const x = input_data[token * 3 ..][0..3];
        var sum_squares: f32 = 0;
        for (x) |value| sum_squares += value * value;
        const denominator = @sqrt(sum_squares / 3 + 1e-6);
        var gated: [2]f32 = undefined;
        for (0..2) |feature| {
            var projected = bias_data[feature];
            for (0..3) |i| {
                projected += x[i] * norm_data[i] / denominator * weight_data[feature * 3 + i];
            }
            gated[feature] = projected * projected / (1 + @exp(-projected));
        }
        for (0..3) |feature| {
            const expected = x[feature] + gated[0] * down_data[feature * 2] + gated[1] * down_data[feature * 2 + 1];
            try std.testing.expectApproxEqAbs(expected, actual[token * 3 + feature], 1e-4);
        }
    }
}

test "Mut operations preserve earlier graph values and replace only on success" {
    const std = @import("std");
    const c = @import("mlx_c");
    const common = @import("common.zig");
    const build_options = @import("build_options");
    var strm = try init(.{ .require_gpu = build_options.metal });
    defer strm.deinit();

    var a = try Tensor.fromSlice(f32, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 });
    defer a.deinit();
    var earlier = try Tensor.transpose(a, strm);
    defer earlier.deinit();

    const original_ctx = a.p.ctx;
    const invalid_stream: Stream = .{ .p = .{ .ctx = null } };
    try std.testing.expectError(error.NullStream, a.reshapeMut(&.{6}, invalid_stream));
    try std.testing.expect(a.p.ctx == original_ctx);
    try std.testing.expectEqualSlices(c_int, &.{ 2, 3 }, try a.shape());

    try a.reshapeMut(&.{ 1, 2, 3 }, strm);
    try a.permuteMut(&.{ 0, 2, 1 }, strm);
    try std.testing.expectEqualSlices(c_int, &.{ 1, 3, 2 }, try a.shape());
    try a.permuteMut(&.{ 0, 2, 1 }, strm);
    try a.reshapeMut(&.{ 2, 3 }, strm);
    try a.transposeMut(strm);
    try a.transposeMut(strm);

    var weight = try Tensor.fromSlice(f32, &.{ 1, 0, 0, 1, 1, 1 }, &.{ 3, 2 });
    defer weight.deinit();
    var bias = try Tensor.fromSlice(f32, &.{ 1, -1 }, &.{2});
    defer bias.deinit();
    var norm = try Tensor.fromSlice(f32, &.{ 1, 2 }, &.{2});
    defer norm.deinit();
    try a.matMulMut(weight, strm);
    try a.addMut(bias, strm);
    try a.mulMut(bias, strm);
    try a.divMut(bias, strm);
    try a.rmsMut(norm, strm);
    try a.siluMut(strm);

    try std.testing.expectEqualSlices(c_int, &.{ 2, 2 }, try a.shape());
    try common.check(c.mlx_array_eval(a.p));
    const actual = c.mlx_array_data_float32(a.p);
    const projected = [_]f32{ 5, 4, 11, 10 };
    for (0..2) |row| {
        const x = projected[row * 2 ..][0..2];
        const denominator = @sqrt((x[0] * x[0] + x[1] * x[1]) / 2 + 1e-6);
        for (0..2) |feature| {
            const normalized = x[feature] / denominator * @as(f32, @floatFromInt(feature + 1));
            const expected = normalized / (1 + @exp(-normalized));
            try std.testing.expectApproxEqAbs(expected, actual[row * 2 + feature], 1e-5);
        }
    }

    // A branch made before mutation must still evaluate to the original values.
    try common.check(c.mlx_array_eval(earlier.p));
    const previous = c.mlx_array_data_float32(earlier.p);
    const strides = c.mlx_array_strides(earlier.p);
    const expected_previous = [_]f32{ 1, 4, 2, 5, 3, 6 };
    for (0..3) |row| {
        for (0..2) |column| {
            try std.testing.expectEqual(
                expected_previous[row * 2 + column],
                previous[row * strides[0] + column * strides[1]],
            );
        }
    }
}
