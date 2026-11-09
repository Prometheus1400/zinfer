const std = @import("std");
const net = std.Io.net;
const Request = std.http.Server.Request;
const Metrics = @import("metrics.zig").Metrics;
const messages = @import("messages.zig");
const zinfer = @import("zinfer");
const Engine = zinfer.engine.Engine;
const ServeArgs = @import("../args.zig").ServeArgs;

const Server = struct {
    ctx: zinfer.Context,
    engine: *Engine,
    metrics: Metrics = .{},
    request_ids: std.mem.Allocator,
    id_mutex: std.Io.Mutex = .init,
    next_id: u64 = 0,

    fn nextConversationId(self: *Server) ![]const u8 {
        try self.id_mutex.lock(self.ctx.io);
        defer self.id_mutex.unlock(self.ctx.io);
        const id = self.next_id;
        self.next_id += 1;
        return self.request_ids.print("http-{d}", .{id});
    }
};

pub fn run(init: std.process.Init, options: ServeArgs) !void {
    const io = init.io;
    const address = try net.IpAddress.parseIp4("127.0.0.1", options.port);

    var listener = try address.listen(io, .{ .reuse_address = true });
    defer listener.deinit(io);

    // The pipeline retains borrowed conversation IDs in its cache map. Keep
    // these allocations alive until after the engine has been deinitialized.
    var request_ids: std.heap.ArenaAllocator = .init(init.gpa);
    defer request_ids.deinit();

    var model_stream = try zinfer.mlx.init(.{ .cross_thread = true });
    defer model_stream.deinit();
    const ctx: zinfer.Context = .{ .gpa = init.gpa, .io = io, .stream = model_stream };
    var info = try zinfer.models.getModelInfo(ctx, options.model);
    defer info.deinit(init.gpa);
    var engine = blk: {
        var pipeline = try zinfer.models.qwen.Pipeline.init(ctx, &info, .{});
        errdefer pipeline.deinit(init.gpa);
        break :blk try Engine.init(init.gpa, pipeline, 32);
    };
    defer engine.deinit(init.gpa, io);
    engine.prefill_max_batch_size = options.prefill_max_batch_size;
    engine.decode_max_batch_size = options.decode_max_batch_size;
    try engine.start(init.gpa, io);

    var app: Server = .{
        .ctx = ctx,
        .engine = &engine,
        .request_ids = request_ids.allocator(),
    };
    var connections: std.Io.Group = .init;
    // Finish handlers before destroying the engine or their borrowed inputs.
    defer connections.await(io) catch {};

    std.log.info("http://127.0.0.1:{d} (models/{s})", .{ options.port, options.model });

    while (true) {
        const stream = try listener.accept(io);
        connections.concurrent(io, serveConnection, .{ &app, stream }) catch |err| {
            stream.close(io);
            std.log.warn("could not start connection handler: {t}", .{err});
        };
    }
}

fn serveConnection(app: *Server, stream: net.Stream) void {
    defer stream.close(app.ctx.io);
    handleConnection(app, stream) catch |err| {
        std.log.warn("request failed: {t}", .{err});
    };
}

fn handleConnection(app: *Server, stream: net.Stream) !void {
    var read_buf: [8192]u8 = undefined;
    var write_buf: [4096]u8 = undefined;
    var reader = stream.reader(app.ctx.io, &read_buf);
    var writer = stream.writer(app.ctx.io, &write_buf);

    var server = std.http.Server.init(
        &reader.interface,
        &writer.interface,
    );

    var request = try server.receiveHead();
    try route(app, &request);
}

fn route(app: *Server, request: *Request) !void {
    // Match the path independently of query parameters.
    const path = std.mem.sliceTo(request.head.target, '?');

    if (std.mem.eql(u8, path, "/metrics")) {
        return switch (request.head.method) {
            .GET, .HEAD => serveMetrics(app, request),
            else => request.respond("Method not allowed\n", .{
                .status = .method_not_allowed,
                .keep_alive = false,
                .extra_headers = &.{
                    .{ .name = "allow", .value = "GET, HEAD" },
                },
            }),
        };
    }

    if (std.mem.eql(u8, path, "/health")) {
        std.log.info("{f} handling request", .{std.json.fmt(.{ .path = path }, .{})});
        return switch (request.head.method) {
            .GET, .HEAD => health(request),
            else => request.respond("Method not allowed\n", .{
                .status = .method_not_allowed,
                .keep_alive = false,
                .extra_headers = &.{
                    .{ .name = "allow", .value = "GET, HEAD" },
                },
            }),
        };
    }
    if (std.mem.eql(u8, path, "/api/v1/messages")) {
        app.metrics.requests.incr();
        app.metrics.active.incr();
        defer app.metrics.active.incrBy(-1);
        return switch (request.head.method) {
            .POST => messages.handle(app.ctx, app.engine, request, try app.nextConversationId()),
            else => request.respond("Method not allowed\n", .{
                .status = .method_not_allowed,
                .keep_alive = false,
                .extra_headers = &.{
                    .{ .name = "allow", .value = "POST" },
                },
            }),
        };
    }

    try request.respond("Not found\n", .{
        .status = .not_found,
        .keep_alive = false,
    });
}

fn serveMetrics(app: *Server, request: *Request) !void {
    var output: std.Io.Writer.Allocating = .init(app.ctx.gpa);
    defer output.deinit();
    try app.metrics.write(&output.writer);
    try app.engine.metrics.write(&output.writer);
    try request.respond(output.written(), .{
        .keep_alive = false,
        .extra_headers = &.{
            .{ .name = "content-type", .value = "text/plain; version=0.0.4; charset=utf-8" },
        },
    });
}

fn health(request: *Request) !void {
    try request.respond("{\"status\":\"ok\"}\n", .{
        .keep_alive = false,
        .extra_headers = &.{
            .{ .name = "content-type", .value = "application/json" },
        },
    });
}
