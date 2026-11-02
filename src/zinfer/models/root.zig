const std = @import("std");
const mlx = @import("mlx");
const Context = @import("../context.zig").Context;
const Queue = std.Io.Queue;

pub const qwen = @import("qwen.zig");

const CommonConfig = struct {
    model_type: []const u8,
};

pub const ModelConfig = union(enum) {
    const Self = @This();

    qwen_config: std.json.Parsed(qwen.Config),
    empty,

    pub fn deinit(self: *Self) void {
        switch (self.*) {
            .qwen_config => |config| {
                config.deinit();
            },
            .empty => {},
        }

        self.* = .empty;
    }
};

pub const ModelInfo = struct {
    model_dir: []const u8,
    safetensor_file_path: [:0]const u8,
    tokenizer_file_path: []const u8,
    tokenizer_config_file_path: []const u8,
    generation_config_file_path: []const u8,
    config: ModelConfig,
    safetensor_metadata: mlx.Map(.string),
    safetensor_weights: mlx.Map(.tensor),

    pub fn deinit(self: *@This(), gpa: std.mem.Allocator) void {
        self.safetensor_metadata.deinit();
        self.safetensor_weights.deinit();
        gpa.free(self.model_dir);
        gpa.free(self.safetensor_file_path);
        gpa.free(self.tokenizer_file_path);
        gpa.free(self.tokenizer_config_file_path);
        gpa.free(self.generation_config_file_path);
        self.config.deinit();
    }
};

pub fn getModelInfo(ctx: Context, model: []const u8) !ModelInfo {
    const model_dir_path = try ctx.gpa.print("{s}/{s}", .{ "models", model });
    errdefer ctx.gpa.free(model_dir_path);

    const model_dir = try std.Io.Dir.cwd().openDir(ctx.io, model_dir_path, .{});
    defer model_dir.close(ctx.io);

    const text = try model_dir.readFileAlloc(ctx.io, "config.json", ctx.gpa, .unlimited);
    defer ctx.gpa.free(text);

    const common_config = try std.json.parseFromSlice(CommonConfig, ctx.gpa, text, .{ .ignore_unknown_fields = true });
    defer common_config.deinit();

    if (std.mem.eql(u8, common_config.value.model_type, "qwen2")) {
        var qwen_config = try std.json.parseFromSlice(qwen.Config, ctx.gpa, text, .{
            .ignore_unknown_fields = true,
            .allocate = .alloc_always,
        });
        errdefer qwen_config.deinit();
        const weights_file_path = try std.mem.concatWithSentinel(ctx.gpa, u8, &.{ model_dir_path, "/", qwen.Handler.WEIGHTS_FILE }, 0);
        errdefer ctx.gpa.free(weights_file_path);
        const tokenizer_file_path = try ctx.gpa.print("{s}/{s}", .{ model_dir_path, qwen.Handler.TOKENIZER_FILE });
        errdefer ctx.gpa.free(tokenizer_file_path);
        const tokenizer_config_file_path = try ctx.gpa.print("{s}/{s}", .{ model_dir_path, qwen.Handler.TOKENIZER_CONFIG_FILE });
        errdefer ctx.gpa.free(tokenizer_config_file_path);
        const generation_config_file_path = try ctx.gpa.print("{s}/{s}", .{ model_dir_path, qwen.Handler.GENERATION_CONFIG_FILE });
        errdefer ctx.gpa.free(generation_config_file_path);
        var metadata, var weights = try mlx.io.loadSafetensors(ctx.stream, weights_file_path);
        errdefer metadata.deinit();
        errdefer weights.deinit();
        return .{
            .model_dir = model_dir_path,
            .safetensor_file_path = weights_file_path,
            .tokenizer_file_path = tokenizer_file_path,
            .tokenizer_config_file_path = tokenizer_config_file_path,
            .generation_config_file_path = generation_config_file_path,
            .config = .{ .qwen_config = qwen_config },
            .safetensor_metadata = metadata,
            .safetensor_weights = weights,
        };
    }

    return error.UnknownModel;
}

pub const CollectOptions = struct {
    flush: bool = true,
};

pub fn collectStreamingResponse(gpa: std.mem.Allocator, io: std.Io, queue: *Queue([]const u8), writer: *std.Io.Writer, options: CollectOptions) !void {
    while (true) {
        const token = queue.getOne(io) catch |e| {
            switch (e) {
                error.Closed => break,
                error.Canceled => return e,
            }
        };
        defer gpa.free(token);

        try writer.writeAll(token);
        if (options.flush) {
            try writer.flush();
        }
    }
}

test "ModelInfo owns its paths parsed config and safetensors" {
    var strm = try mlx.init(.{});
    defer strm.deinit();
    const ctx: Context = .{ .gpa = std.testing.allocator, .io = std.testing.io, .stream = strm };
    _ = std.Io.Dir.cwd().statFile(ctx.io, "models/qwen/model.safetensors", .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    var info = getModelInfo(ctx, "qwen") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer info.deinit(ctx.gpa);

    try std.testing.expectEqualStrings("models/qwen", info.model_dir);
    try std.testing.expectEqualStrings("models/qwen/model.safetensors", info.safetensor_file_path);
    try std.testing.expectEqualStrings("models/qwen/tokenizer.json", info.tokenizer_file_path);
    try std.testing.expectEqualStrings("models/qwen/tokenizer_config.json", info.tokenizer_config_file_path);
    try std.testing.expectEqualStrings("models/qwen/generation_config.json", info.generation_config_file_path);
    try std.testing.expectEqual(@as(u8, 0), info.safetensor_file_path[info.safetensor_file_path.len]);
    try info.safetensor_metadata.ensure();
    var embeddings = try info.safetensor_weights.get("model.embed_tokens.weight");
    defer embeddings.deinit();
    try std.testing.expectEqualSlices(c_int, &.{ 151936, 896 }, try embeddings.shape());
    switch (info.config) {
        .qwen_config => |config| {
            try std.testing.expectEqualStrings("qwen2", config.value.model_type);
            try std.testing.expectEqualStrings("Qwen2ForCausalLM", config.value.architectures[0]);
        },
        .empty => return error.MissingModelConfig,
    }
}
