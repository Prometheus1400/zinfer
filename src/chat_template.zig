const std = @import("std");
const c = @import("minijinja_c");

pub const Message = struct {
    role: []const u8,
    content: []const u8,
};

pub const RenderOptions = struct {
    add_generation_prompt: bool = true,
};

/// Owns a compiled Jinja chat template. Copies of this struct share its handle;
/// call deinit only on the owning instance.
pub const ChatTemplate = struct {
    const Self = @This();

    p: ?*c.mj_env,

    /// The source is copied by MiniJinja and can be freed after this returns.
    pub fn init(gpa: std.mem.Allocator, source: []const u8) !Self {
        const source_z = try terminated(gpa, source);
        defer gpa.free(source_z);

        const p = c.mj_env_new() orelse return error.JinjaError;
        errdefer c.mj_env_free(p);
        if (!c.mj_env_add_template(p, "chat", source_z.ptr))
            return error.InvalidTemplate;

        return .{ .p = p };
    }

    /// Returns an owned prompt string; the caller frees it with gpa.free.
    /// Supports text messages. Tool schemas and multimodal content are not supplied.
    pub fn render(self: Self, gpa: std.mem.Allocator, messages: []const Message, options: RenderOptions) ![]u8 {
        const p = self.p orelse return error.ClosedTemplate;
        var context = c.mj_value_new_object();
        defer c.mj_value_decref(&context);
        var list = c.mj_value_new_list();
        defer c.mj_value_decref(&list);

        for (messages) |message| {
            var value = c.mj_value_new_object();
            defer c.mj_value_decref(&value);
            try check(c.mj_value_set_string_key(&value, "role", try stringValue(gpa, message.role)));
            try check(c.mj_value_set_string_key(&value, "content", try stringValue(gpa, message.content)));
            try check(c.mj_value_append(&list, moveValue(&value)));
        }

        try check(c.mj_value_set_string_key(&context, "messages", moveValue(&list)));
        try check(c.mj_value_set_string_key(&context, "add_generation_prompt", c.mj_value_new_bool(options.add_generation_prompt)));

        // Rendering consumes the context even when it fails.
        const rendered = c.mj_env_render_template(p, "chat", moveValue(&context));
        if (rendered == null) return error.TemplateRenderFailed;
        defer c.mj_str_free(rendered);
        return gpa.dupe(u8, std.mem.span(rendered));
    }

    pub fn deinit(self: *Self) void {
        if (self.p) |p| c.mj_env_free(p);
        self.p = null;
    }
};

fn check(ok: bool) !void {
    if (!ok) return error.JinjaError;
}

fn terminated(gpa: std.mem.Allocator, text: []const u8) ![:0]u8 {
    if (std.mem.findScalar(u8, text, 0) != null) return error.EmbeddedNull;
    return gpa.dupeSentinel(u8, text, 0);
}

fn stringValue(gpa: std.mem.Allocator, text: []const u8) !c.mj_value {
    const text_z = try terminated(gpa, text);
    defer gpa.free(text_z);
    var value = c.mj_value_new_string(text_z.ptr);
    if (c.mj_err_is_set()) {
        c.mj_value_decref(&value);
        return error.JinjaError;
    }
    return value;
}

// C calls consume values by value. Leave an undefined value in the old slot so
// its deferred decref cannot free the transferred object a second time.
fn moveValue(value: *c.mj_value) c.mj_value {
    const moved = value.*;
    value.* = c.mj_value_new_undefined();
    return moved;
}

test "Qwen template renders default system message and assistant prefix" {
    const gpa = std.testing.allocator;
    const json = std.Io.Dir.cwd().readFileAlloc(std.testing.io, "models/qwen/tokenizer_config.json", gpa, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer gpa.free(json);
    const config = try std.json.parseFromSlice(struct { chat_template: []const u8 }, gpa, json, .{ .ignore_unknown_fields = true });
    defer config.deinit();

    var template = try ChatTemplate.init(gpa, config.value.chat_template);
    defer template.deinit();
    const rendered = try template.render(gpa, &.{.{ .role = "user", .content = "hi" }}, .{});
    defer gpa.free(rendered);
    try std.testing.expectEqualStrings(
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n" ++
            "<|im_start|>user\nhi<|im_end|>\n" ++
            "<|im_start|>assistant\n",
        rendered,
    );

    const without_prefix = try template.render(gpa, &.{
        .{ .role = "system", .content = "Be brief." },
        .{ .role = "user", .content = "hello\nworld" },
        .{ .role = "assistant", .content = "Hi!" },
    }, .{ .add_generation_prompt = false });
    defer gpa.free(without_prefix);
    try std.testing.expectEqualStrings(
        "<|im_start|>system\nBe brief.<|im_end|>\n" ++
            "<|im_start|>user\nhello\nworld<|im_end|>\n" ++
            "<|im_start|>assistant\nHi!<|im_end|>\n",
        without_prefix,
    );
}

test "rendered text owns its bytes after rerender and template cleanup" {
    const gpa = std.testing.allocator;
    const source = try gpa.dupe(u8, "{{ messages[0].content }}");
    var template = try ChatTemplate.init(gpa, source);
    defer template.deinit();
    gpa.free(source);

    const first = try template.render(gpa, &.{.{ .role = "user", .content = "<hello> & 世界" }}, .{});
    defer gpa.free(first);
    const second = try template.render(gpa, &.{.{ .role = "user", .content = "second" }}, .{});
    defer gpa.free(second);
    template.deinit();
    try std.testing.expectEqualStrings("<hello> & 世界", first);
    try std.testing.expectEqualStrings("second", second);
    try std.testing.expectError(error.ClosedTemplate, template.render(gpa, &.{}, .{}));
}

test "malformed templates rendering errors and null bytes return errors" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.InvalidTemplate, ChatTemplate.init(gpa, "{% if %}"));
    try std.testing.expectError(error.EmbeddedNull, ChatTemplate.init(gpa, "hi\x00there"));

    var template = try ChatTemplate.init(gpa, "{{ messages[0].content | missing_filter }}");
    defer template.deinit();
    try std.testing.expectError(error.TemplateRenderFailed, template.render(gpa, &.{.{ .role = "user", .content = "hi" }}, .{}));
    try std.testing.expectError(error.EmbeddedNull, template.render(gpa, &.{.{ .role = "user", .content = "hi\x00there" }}, .{}));
}
