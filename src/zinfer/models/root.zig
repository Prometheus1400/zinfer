const mlx = @import("mlx");
const std = @import("std");

const Context = @import("../context.zig").Context;
const Queue = std.Io.Queue;

pub const qwen = @import("qwen.zig");

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

pub const GenerationConfig = struct {
    bos_token_id: ?u32 = null,
    pad_token_id: ?u32 = null,
    do_sample: bool = false,
    eos_token_id: []const u32 = &.{},
    repetition_penalty: f32 = 1.0,
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    top_k: u32 = 50,
    transformers_version: ?[]const u8 = null,
};

pub const ModelInfo = struct {
    model_dir: []const u8,
    safetensor_file_path: [:0]const u8,
    tokenizer_file_path: []const u8,
    tokenizer_config_file_path: []const u8,
    generation_config_file_path: []const u8,
    config: ModelConfig,
    generation_config: std.json.Parsed(GenerationConfig),
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
        self.generation_config.deinit();
    }
};

pub fn getModelInfo(ctx: Context, model: []const u8) !ModelInfo {
    const model_dir_path = try ctx.gpa.print("{s}/{s}", .{ "models", model });
    errdefer ctx.gpa.free(model_dir_path);

    const model_dir = try std.Io.Dir.cwd().openDir(ctx.io, model_dir_path, .{});
    defer model_dir.close(ctx.io);

    const text = try model_dir.readFileAlloc(ctx.io, "config.json", ctx.gpa, .unlimited);
    defer ctx.gpa.free(text);

    const common_config = try std.json.parseFromSlice(struct { model_type: []const u8 }, ctx.gpa, text, .{ .ignore_unknown_fields = true });
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

        const generation_json = try model_dir.readFileAlloc(ctx.io, qwen.Handler.GENERATION_CONFIG_FILE, ctx.gpa, .unlimited);
        defer ctx.gpa.free(generation_json);

        const generation_config = try std.json.parseFromSlice(GenerationConfig, ctx.gpa, generation_json, .{
            .ignore_unknown_fields = true,
            .allocate = .alloc_always,
        });
        errdefer generation_config.deinit();

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
            .generation_config = generation_config,
            .safetensor_metadata = metadata,
            .safetensor_weights = weights,
        };
    }

    return error.UnknownModel;
}

pub fn collectStreamingResponse(gpa: std.mem.Allocator, io: std.Io, queue: *Queue([]const u8), writer: *std.Io.Writer, options: struct { flush: bool = true }) !void {
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
