const std = @import("std");
const m = @import("metrics");

pub const Metrics = struct {
    requests: m.Counter(u64) = .init(
        "zinfer_http_requests_total",
        .{ .help = "Total requests received for /api/v1/messages." },
        .{},
    ),
    active: m.Gauge(i64) = .init(
        "zinfer_http_requests_active",
        .{ .help = "Requests to /api/v1/messages currently being handled." },
        .{},
    ),

    pub fn write(self: *Metrics, writer: *std.Io.Writer) !void {
        try m.write(self, writer);
    }
};
