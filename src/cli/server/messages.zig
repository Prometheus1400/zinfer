const std = @import("std");
const zinfer = @import("zinfer");
const Request = std.http.Server.Request;
const Engine = zinfer.engine.Engine;
const TokenQueue = std.Io.Queue([]const u8);

const Input = struct {
    messages: []const zinfer.models.chat_template.Message,
};

const max_body_bytes = 1024 * 1024;

pub fn handle(ctx: zinfer.Context, engine: *Engine, request: *Request, id: []const u8) !void {
    if (request.head.content_length) |length| {
        if (length > max_body_bytes)
            return respondError(ctx.gpa, request, .payload_too_large, "Request body exceeds 1 MiB.");
    }
    var body_buffer: [4096]u8 = undefined;
    const reader = try request.readerExpectContinue(&body_buffer);
    const body = reader.allocRemaining(ctx.gpa, .limited(max_body_bytes)) catch |err| switch (err) {
        error.OutOfMemory => return err,
        error.StreamTooLong => return respondError(ctx.gpa, request, .payload_too_large, "Request body exceeds 1 MiB."),
        else => return respondError(ctx.gpa, request, .bad_request, "Could not read request body."),
    };
    defer ctx.gpa.free(body);

    const parsed = std.json.parseFromSlice(Input, ctx.gpa, body, .{ .allocate = .alloc_always }) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return respondError(ctx.gpa, request, .bad_request, "Expected a messages array containing role/content strings."),
    };
    defer parsed.deinit();

    if (parsed.value.messages.len == 0)
        return respondError(ctx.gpa, request, .bad_request, "Messages must not be empty.");
    for (parsed.value.messages) |message| {
        if (!std.mem.eql(u8, message.role, "system") and
            !std.mem.eql(u8, message.role, "user") and
            !std.mem.eql(u8, message.role, "assistant"))
            return respondError(ctx.gpa, request, .bad_request, "Message role must be system, user, or assistant.");
        if (std.mem.findScalar(u8, message.content, 0) != null)
            return respondError(ctx.gpa, request, .bad_request, "Message content must not contain a null byte.");
    }

    var token_buffer: [32][]const u8 = undefined;
    var tokens: TokenQueue = .init(&token_buffer);
    try engine.submitStreaming(ctx.io, .{ .id = id, .messages = parsed.value.messages }, &tokens);
    // The engine borrows the parsed messages and queue. On every exit after
    // submission, drain through closure before releasing either one. Closing
    // this queue early would make the engine's producer fail.
    defer drain(ctx.gpa, ctx.io, &tokens);

    var output: std.ArrayList(u8) = .empty;
    defer output.deinit(ctx.gpa);
    while (tokens.getOneUncancelable(ctx.io)) |token| {
        defer ctx.gpa.free(token);
        try output.appendSlice(ctx.gpa, token);
    } else |err| switch (err) {
        error.Closed => {},
    }

    try respondJson(ctx.gpa, request, .ok, .{
        .id = id,
        .role = "assistant",
        .content = output.items,
    });
}

fn drain(gpa: std.mem.Allocator, io: std.Io, tokens: *TokenQueue) void {
    while (tokens.getOneUncancelable(io)) |token| {
        gpa.free(token);
    } else |err| switch (err) {
        error.Closed => {},
    }
}

fn respondError(gpa: std.mem.Allocator, request: *Request, status: std.http.Status, message: []const u8) !void {
    try respondJson(gpa, request, status, .{ .@"error" = message });
}

fn respondJson(gpa: std.mem.Allocator, request: *Request, status: std.http.Status, value: anytype) !void {
    const body = try std.json.Stringify.valueAlloc(gpa, value, .{});
    defer gpa.free(body);
    try request.respond(body, .{
        .status = status,
        .keep_alive = false,
        .extra_headers = &.{
            .{ .name = "content-type", .value = "application/json" },
        },
    });
}
