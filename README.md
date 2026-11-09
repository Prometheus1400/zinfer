# zinfer

Zig project with MLX-C, a Hugging Face tokenizer, and MiniJinja chat templates.

## Prerequisites

- Apple Silicon macOS (native builds only).
- Zig **0.17.0**.
- CMake **3.25+** and Ninja (`brew install cmake ninja`).
- Rust and Cargo for compiling the tokenizer and MiniJinja C libraries. Rust is needed at build
  time; running the installed executable does not require it.
- Xcode with the Metal compiler available through `xcrun --find metal` for
  the default GPU build. Use `-Dmetal=false` for a CPU-only native build.

MLX-C is pinned as a Git submodule at
`57731b00e8bed710e0750e7356da9f18f8eb3e20` (0.7.0), selecting MLX **v0.32.4**
through upstream CMake. The first native build downloads and compiles dependencies;
subsequent builds reuse CMake/Ninja's cached objects.

The tokenizer C API comes from `mlc-ai/tokenizers-cpp`, pinned at
`c586c52f93f7b060753bd2388eb96a105cb7374d`. Only its Rust-backed C library is
built. `cmake/tokenizers.Cargo.lock` pins its Rust dependencies, including
Hugging Face Tokenizers **0.21.4**. Cargo builds from a cache-local source copy
with `--locked`, leaving the upstream submodule unchanged.

MiniJinja is pinned at `d3a824ed267fca9f96c3882b569979560b24f282` (3.0.5).
Its C API is built as a static library using upstream's committed Cargo lockfile
and the `json` feature for Jinja's `tojson` filter. The submodule stays unchanged.

## Commands

```sh
git submodule update --init --recursive
zig build             # build and install the CLI and static library into zig-out/
zig build cli         # install zig-out/bin/zinfer
zig build zinfer      # install zig-out/lib/libzinfer.a
zig build check       # check the CLI and all library tests without running them
zig build run-cli -- serve  # run the HTTP server at http://127.0.0.1:8080
zig build run-cli -- inspect --model qwen
zig build run-cli -- prompt --model qwen -p "Hello!" --stream
zig build run-cli -- chat --model qwen
zig build native      # build all native dependencies
zig build tokenizers  # build only the tokenizer C library
zig build minijinja   # build only the MiniJinja C library
zig build bindings    # translate only the C headers
zig build test-qwen   # compare batched and individual greedy responses using models/qwen
zig build test-engine --test-timeout 60s  # compare engine streams with individual baselines
```

ZLS uses the `check` step for build-on-save compiler diagnostics. It checks the
CLI (including the server) and library tests without emitting their binaries or
loading model checkpoints. Native prerequisites may still be built or refreshed
as needed.

The `prompt` command generates a single response; `--stream` prints it as it is
generated. The `chat` command starts an interactive session: type a message at
the `>` prompt and press Enter. Replies stream automatically, and the model
stays loaded with its KV cache retained between turns. Press Ctrl+D at an empty
prompt to exit, or Ctrl+C to interrupt.

Options: `-Dmetal=false`, `-Dnative-jobs=2` (default: 4), and
`-Doptimize=fast` (or `--release=fast`). Zig code defaults to Debug; native
dependencies always build in Release mode.

Use `-Dlog-level=err|warn|info|debug` to set the CLI's compile-time log level,
including runtime library logs. The default is `debug` in Debug builds and
`info` in release builds. For example:

```sh
zig build run-cli -Dlog-level=info -- chat --model qwen
```

## HTTP messages

The server loads the selected model from `models/<model>` once and submits
requests to a shared engine. Start it with `zig build run-cli -- serve`.
Optional flags are `--model` / `-m` (default `qwen`), `--port` (default 8080),
`--prefill-max-batch-size` (default 8), and `--decode-max-batch-size` (default 32).
Batch sizes must be greater than zero. For example:

```sh
zig build run-cli -- serve --model qwen --port 8081 --prefill-max-batch-size 4 --decode-max-batch-size 16
```

Connections are handled concurrently, so health checks and metrics remain
available during generation.

```sh
curl http://127.0.0.1:8080/api/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with only the number: 2 + 2 = ?"}]}'
```

`POST /api/v1/messages` returns one JSON response after generation finishes:

```json
{"id":"http-0","role":"assistant","content":"4"}
```

Send the full message history on each request, using `system`, `user`, and
`assistant` roles with string content. Each request gets a fresh conversation
ID. The request body limit is 1 MiB; malformed input returns 400, oversized
bodies return 413, and unsupported methods return 405 with `Allow: POST`.

`GET /metrics` exposes request totals and active requests for the messages route,
plus metrics owned by the shared engine:

| Metric | Meaning |
| --- | --- |
| `zinfer_engine_prefill_batch_size` | Histogram of sequences passed to each prefill call |
| `zinfer_engine_decode_batch_size` | Histogram of sequences passed to each decode call |
| `zinfer_engine_forward_prefill_duration_seconds` | Histogram of prefill time, including evaluation of sampled tokens |
| `zinfer_engine_forward_decode_duration_seconds` | Histogram of decode time, including evaluation of sampled tokens |
| `zinfer_engine_prefill_sequences` | Sequences waiting in or running through the prefill batch |
| `zinfer_engine_decode_sequences` | Sequences admitted to the decode batch |
| `zinfer_engine_decode_ready_sequences` | Prefilled sequences waiting for a decode slot |
| `zinfer_engine_active_sequences` | Sum of the three batcher-owned collections; excludes upstream queues |
| `zinfer_engine_requests_completed_total` | Requests finished by the engine |
| `zinfer_engine_output_tokens_total` | Tokens emitted, excluding stop tokens |
| `zinfer_engine_prefill_output_tokens_total` | First response tokens emitted by prefill |
| `zinfer_engine_decode_output_tokens_total` | Subsequent response tokens emitted by decode |
| `zinfer_engine_decode_context_tokens` | Histogram of input history lengths, observed once per sequence per decode call |
| `zinfer_engine_decode_context_mean_tokens` | Mean input history length in the last decode batch; zero while decode is empty |
| `zinfer_engine_decode_context_max_tokens` | Maximum input history length in the last decode batch; zero while decode is empty |
| `zinfer_engine_decode_cache_mask_build_duration_seconds` | Histogram of host cache-update, packed-KV graph, and mask construction time per decode call |
| `zinfer_engine_decode_graph_build_duration_seconds` | Histogram of other model/sampling graph construction time per decode call |
| `zinfer_engine_decode_evaluation_duration_seconds` | Histogram of existing MLX evaluation and token readback time per decode call |

Inspect these with `curl -s http://127.0.0.1:8080/metrics`. A histogram exposes
`_bucket`, `_sum`, and `_count` samples. After a load test, divide
`zinfer_engine_decode_batch_size_sum` by `zinfer_engine_decode_batch_size_count`
for the mean sequences per decode pass. Use the corresponding `prefill` metrics
for prompt passes. If a phase's `_count` exceeds its bucket with `le="1"`, some
passes contained multiple sequences. Decode counts successive token steps,
rather than counting each HTTP request once. Prefill output-token counts measure
the first response tokens, not prompt-input tokens. Values are
cumulative since server startup, so use differences between scrapes to measure
one load test. A scrape reads atomic metric values rather than pipeline state;
values sampled during an update may reflect slightly different instants.

Decode context lengths include the input token consumed by that call and exclude
the next sampled token. The timing breakdown adds no evaluation barriers: GPU
cache work remains part of MLX evaluation. The three sub-timers exclude input
preparation, history bookkeeping, and cleanup, so they are not an exact partition
of total forward time. Grafana shows mean forward times from `_sum / _count`
alongside the bucket-based median and p95 estimates for both phases.

`GET /health` remains a health check. If a client disconnects, its handler
finishes draining generation before releasing the engine's borrowed request
data and queue. The prototype retains conversation caches and their IDs until
shutdown; cache eviction is still needed for long-running use.

For graphs, run `./observability/run.sh` alongside the server and open
<http://localhost:3000/d/zinfer-engine/zinfer-engine>. It starts local Prometheus
and Grafana with a provisioned engine dashboard. Ctrl+C stops monitoring and
deletes its temporary history. See [observability/README.md](observability/README.md)
for installation, port overrides, and panel descriptions.

## Chat templates

The Qwen pipeline loads the `chat_template` from `tokenizer_config.json` and
renders it for each input before encoding, including the assistant prefix.
The tokenizer and chat-template wrappers are internal to the runtime; the public
`zinfer` module exports `mlx`, `Context`, and `models`.

Chat templates support text messages. Tool schemas and multimodal content are
not supplied to the template.
