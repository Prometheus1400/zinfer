const std = @import("std");
const m = @import("metrics");
const mlx = @import("mlx");

/// Each engine owns its metrics. Workers update them; an application can
/// serialize them concurrently without reading mutable pipeline state.
pub const Metrics = struct {
    prefill_batch_size: m.Histogram(u64, &.{ 1, 2, 4, 8, 16, 32 }),
    decode_batch_size: m.Histogram(u64, &.{ 1, 2, 4, 8, 16, 32 }),
    forward_prefill_duration: m.Histogram(f64, &.{ 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5 }),
    forward_decode_duration: m.Histogram(f64, &.{ 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5 }),
    active_sequences: m.Gauge(u64),
    prefill_sequences: m.Gauge(u64),
    decode_sequences: m.Gauge(u64),
    decode_ready_sequences: m.Gauge(u64),
    completed: m.Counter(u64),
    output_tokens: m.Counter(u64),
    prefill_output_tokens: m.Counter(u64),
    decode_output_tokens: m.Counter(u64),
    decode_context_tokens: m.Histogram(u64, &.{ 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 }),
    decode_context_mean_tokens: m.Gauge(f64),
    decode_context_max_tokens: m.Gauge(u64),
    decode_cache_mask_build_duration: m.Histogram(f64, &.{ 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1 }),
    decode_graph_build_duration: m.Histogram(f64, &.{ 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1 }),
    decode_evaluation_duration: m.Histogram(f64, &.{ 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5 }),

    pub fn init() Metrics {
        // Generate all Prometheus headers once, with enough compile-time budget
        // for the histogram families and their bucket labels.
        return comptime blk: {
            @setEvalBranchQuota(200_000);
            break :blk Metrics{
                .prefill_batch_size = .init(
                    "zinfer_engine_prefill_batch_size",
                    .{ .help = "Number of sequences passed to each prefill forward call." },
                    .{},
                ),
                .decode_batch_size = .init(
                    "zinfer_engine_decode_batch_size",
                    .{ .help = "Number of sequences passed to each decode forward call." },
                    .{},
                ),
                .forward_prefill_duration = .init(
                    "zinfer_engine_forward_prefill_duration_seconds",
                    .{ .help = "Wall time per prefill forward call, including evaluated token results." },
                    .{},
                ),
                .forward_decode_duration = .init(
                    "zinfer_engine_forward_decode_duration_seconds",
                    .{ .help = "Wall time per decode forward call, including evaluated token results." },
                    .{},
                ),
                .active_sequences = .init(
                    "zinfer_engine_active_sequences",
                    .{ .help = "Total sequences owned by the batcher across prefill, decode, and ready-for-decode collections." },
                    .{},
                ),
                .prefill_sequences = .init(
                    "zinfer_engine_prefill_sequences",
                    .{ .help = "Sequences in the prefill batch, including those waiting for it to run." },
                    .{},
                ),
                .decode_sequences = .init(
                    "zinfer_engine_decode_sequences",
                    .{ .help = "Sequences admitted to the decode batch." },
                    .{},
                ),
                .decode_ready_sequences = .init(
                    "zinfer_engine_decode_ready_sequences",
                    .{ .help = "Prefilled sequences waiting for a decode batch slot." },
                    .{},
                ),
                .completed = .init(
                    "zinfer_engine_requests_completed_total",
                    .{ .help = "Requests completed by the batcher after a stop token." },
                    .{},
                ),
                .output_tokens = .init(
                    "zinfer_engine_output_tokens_total",
                    .{ .help = "Generated tokens successfully emitted, excluding stop tokens." },
                    .{},
                ),
                .prefill_output_tokens = .init(
                    "zinfer_engine_prefill_output_tokens_total",
                    .{ .help = "First response tokens successfully emitted by prefill, excluding stop tokens." },
                    .{},
                ),
                .decode_output_tokens = .init(
                    "zinfer_engine_decode_output_tokens_total",
                    .{ .help = "Response tokens successfully emitted by decode, excluding stop tokens." },
                    .{},
                ),
                .decode_context_tokens = .init(
                    "zinfer_engine_decode_context_tokens",
                    .{ .help = "Input history length per sequence per decode forward call, including the token being consumed." },
                    .{},
                ),
                .decode_context_mean_tokens = .init(
                    "zinfer_engine_decode_context_mean_tokens",
                    .{ .help = "Mean input history length in the most recent decode call; zero while decode is empty." },
                    .{},
                ),
                .decode_context_max_tokens = .init(
                    "zinfer_engine_decode_context_max_tokens",
                    .{ .help = "Maximum input history length in the most recent decode call; zero while decode is empty." },
                    .{},
                ),
                .decode_cache_mask_build_duration = .init(
                    "zinfer_engine_decode_cache_mask_build_duration_seconds",
                    .{ .help = "Host wall time constructing cache updates, packed KV graphs, and masks across all layers per decode call; excludes deferred GPU execution." },
                    .{},
                ),
                .decode_graph_build_duration = .init(
                    "zinfer_engine_decode_graph_build_duration_seconds",
                    .{ .help = "Other model and sampling graph construction wall time per decode call, excluding cache/mask construction and deferred GPU execution." },
                    .{},
                ),
                .decode_evaluation_duration = .init(
                    "zinfer_engine_decode_evaluation_duration_seconds",
                    .{ .help = "Wall time evaluating sampled tokens and reading them back per decode call; includes deferred GPU cache and model work." },
                    .{},
                ),
            };
        };
    }

    /// Called by the batcher after admissions, removals, and transfers. Scrapes
    /// read each gauge atomically, rather than a transactional snapshot.
    pub fn setSequences(self: *Metrics, prefill: usize, decode: usize, ready: usize) void {
        self.prefill_sequences.set(@intCast(prefill));
        self.decode_sequences.set(@intCast(decode));
        self.decode_ready_sequences.set(@intCast(ready));
        self.active_sequences.set(@intCast(prefill + decode + ready));
        if (decode == 0) {
            self.decode_context_mean_tokens.set(0);
            self.decode_context_max_tokens.set(0);
        }
    }

    pub fn observeDecodeTimings(self: *Metrics, timings: mlx.ForwardTimings) void {
        self.decode_cache_mask_build_duration.observe(seconds(timings.cache_mask_build_ns));
        self.decode_graph_build_duration.observe(seconds(timings.graph_build_ns - timings.cache_mask_build_ns));
        self.decode_evaluation_duration.observe(seconds(timings.evaluation_ns));
    }

    fn seconds(ns: i96) f64 {
        return @as(f64, @floatFromInt(ns)) / std.time.ns_per_s;
    }

    pub fn write(self: *Metrics, writer: *std.Io.Writer) !void {
        try m.write(self, writer);
    }
};
