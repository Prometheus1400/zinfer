# Local engine dashboard

Prometheus scrapes zinfer once per second. Grafana opens a provisioned **Zinfer
Engine** dashboard with phase-specific sequence counts, mean batch sizes, output
token throughput, forward-duration percentiles, and batch-size distributions.
It also shows decode context lengths and a breakdown of host graph construction
versus MLX evaluation/readback time.

## Start

Install the native macOS tools once:

```sh
brew install prometheus grafana
```

Start zinfer from the project root in one terminal:

```sh
zig build run-cli -- serve
```

In another terminal:

```sh
./observability/run.sh
```

Open <http://localhost:3000/d/zinfer-engine/zinfer-engine>. The dashboard is
readable without signing in, and both monitoring tools bind to `127.0.0.1`.
Prometheus's query UI is at <http://localhost:9090>.

Wait for a few scrapes before running requests. The dashboard refreshes every
second, initially showing the last five minutes. Drag over a load-test interval
to zoom in. Idle rate/percentile panels may be blank when there are no forward
passes to measure. A DOWN target means the Zig server is not reachable.

The script starts its own foreground session. It prints the temporary directory
containing the logs, watches both processes, and exits if either one stops.
Grafana's Prometheus plugin is pinned to 13.2.5 and downloaded into the temporary
directory on startup, so starting a session requires network access. This setup
was checked with Prometheus 3.15.1 and Grafana 13.2.4.

## Temporary storage

Every invocation creates a fresh directory under the system temporary directory
for Prometheus history, Grafana's database, plugin files, and logs. These tools
write to disk during the session. **Ctrl+C stops both tools and deletes the
directory.** Normal exit, SIGTERM, and startup failure use the same cleanup.
An uncatchable termination such as SIGKILL can leave that temporary directory
behind.

The checked-in configs and dashboard remain in this folder. Edit
`dashboards/zinfer.json` to retain dashboard changes across runs.

Zinfer runs independently: stopping monitoring does not stop inference or reset
the engine's counters. After restarting monitoring, historical samples are gone,
but absolute engine counters still reflect the current zinfer process's lifetime.

## Ports and paths

Defaults are zinfer `8080`, Prometheus `9090`, and Grafana `3000`. Override them
for another server or to avoid a port conflict:

```sh
ZINFER_PORT=8081 ZINFER_PROMETHEUS_PORT=9091 ZINFER_GRAFANA_PORT=3001 \
  ./observability/run.sh
```

The launcher locates Grafana's frontend assets through `brew --prefix grafana`.
For a standalone Grafana install, set `GRAFANA_HOMEPATH` to its directory
containing `conf/defaults.ini` and the `public/` assets.

## Reading the dashboard

- **Active requests and sequences:** HTTP activity includes parsing, queueing,
  generation, and response writing. The three engine lines count distinct
  collections: pending/running prefill, admitted decode, and prefilled requests
  waiting for a decode slot. These are phase membership gauges, not simultaneous
  GPU execution counts. Their sum is exported as `zinfer_engine_active_sequences`;
  it excludes requests still in the upstream queues. Scrapes can miss brief peaks.
- **Average sequences per forward:** separate prefill and decode lines, each
  dividing its recent batch-size sum by its forward-call count. Decode includes
  every token step, rather than counting each HTTP request once.
- **Forward calls with batching:** separate prefill and decode percentages for
  calls containing multiple sequences. The two batch-size distribution charts
  show which sizes each phase actually executed.
- **Prefill/decode forward duration:** separate mean/median/p95 charts, including
  MLX evaluation. The mean uses observed duration sum/count, independently of
  histogram bucket interpolation. These exclude queueing and token delivery;
  they measure model calls, rather than end-to-end HTTP latency.
- **Generated tokens per second:** prefill emits the first response token; decode
  emits subsequent tokens. The total counter is their sum. Prefill's output-token
  metric does not count prompt-input tokens. Stop tokens are excluded.
- **Completed requests per second:** completions across both phases.
- **Decode context length:** mean and maximum full input-history lengths from
  the latest decode batch, including its consumed input token and excluding the
  next sampled token. Gauges reset to zero when decode is empty. The exported
  `zinfer_engine_decode_context_tokens` histogram observes every sequence on every
  decode call; long-running sequences therefore contribute multiple observations.
- **Decode forward timing breakdown:** mean host cache/mask construction time
  accumulated across layers, other model/sampling graph construction time, and
  existing MLX evaluation plus token readback time. The total forward mean is
  overlaid for comparison. Input preparation, history bookkeeping, and cleanup
  are outside those sub-timers, so their sum need not equal total forward time.

MLX is lazy. The cache/mask timer measures host construction of cache updates,
packed-KV graphs, and attention masks. Deferred GPU cache packing still executes
inside the evaluation timer alongside the model. No extra evaluation barriers
are introduced. This breakdown is not GPU kernel profiling, and cannot isolate
GPU cache-copy time from GPU attention/MLP execution.

The target, engine counters, and active gauges are raw values. Rate panels use
Grafana's scrape-aware `$__rate_interval`; the window grows when you zoom out.

## Files

- `run.sh`: process lifecycle, port overrides, and temporary storage.
- `prometheus.yml`: scrape configuration, copied into each session.
- `grafana.ini`: Grafana configuration with session-specific paths.
- `provisioning/`: registers Prometheus and loads the dashboard automatically.
- `dashboards/zinfer.json`: version-controlled panels and PromQL expressions.
