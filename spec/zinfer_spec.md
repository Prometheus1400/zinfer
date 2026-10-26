---
title: "ZINFER — A Minimal LLM Inference Engine for Apple Silicon"
subtitle: "Self-directed systems programming project · Specification v2.1"
date: "October 25, 2026"
---

# Project brief

**Project:** ZINFER — a compact, single-machine LLM inference engine.  
**Language:** Zig for the engine and control flow; MLX-C for tensor primitives and Apple GPU execution.  
**Hardware:** One Apple Silicon MacBook Air, including an 8 GB unified-memory configuration.  
**Pacing:** Self-directed. No deadlines, grades, mandatory presentation, or fixed number of work hours.  
**Completion:** Two milestones across ten sequential phases (0–9): a verified single-request engine with KV caching and a focused benchmark, followed by batching, serving, and broader benchmarks. Each phase has individually demonstrable acceptance checks.

> **Mission.** Implement the forward pass of a real, pretrained decoder-only transformer, then develop it into a small engine supporting greedy generation, explicit per-request KV caches, true batched execution, and dynamic request scheduling. MLX performs tensor arithmetic and GPU work; your Zig code owns the model architecture, inference state, and runtime behavior.

The deliverable is a learning-oriented engineering artifact, **not** a production inference service. Optimize for understanding, correctness, reproducibility, and a comprehensible codebase over features or raw throughput. You may choose algorithms, module boundaries, data structures, and internal APIs as long as the externally observable requirements are met.

## What counts as finished

A phase is complete when (a) its acceptance tests pass locally, (b) its demonstrable capability works with the specified command or equivalent documented API, and (c) the committed evidence includes a run command and its observed results. Record incomplete checks and known limitations in `progress.md`; exploration can continue while earlier checks are being resolved, but completion records should reflect their dependencies.

**Milestone 1 — Verified single-request engine:** Complete Phases 0–6 on the same Apple Silicon machine, then record the focused benchmark described after Phase 6. This is a completed personal-project milestone and a useful stopping point.

**Milestone 2 — Batched serving engine:** Complete Phases 7–9 on that machine. This remains the full intended roadmap. An unexpected slowdown in either milestone's benchmark does **not** constitute failure if the experiment is sound. Optional enhancements are separate from both milestones.

## Learning outcomes

- Explain the transformations and shapes of a Qwen2 decoder block, including GQA, RoPE, RMSNorm, SwiGLU, and residual paths.
- Reproduce intermediate tensor outputs and final pretrained-model logits against an independent reference.
- Distinguish uncached prefill, cached autoregressive decode, static batching, and continuous batching.
- Define and enforce tensor ownership, per-request KV-cache ownership, resource budgets, and cancellation semantics.
- Design experiments for throughput, time-to-first-token (TTFT), token latency, and memory without confusing MLX's lazy evaluation with completed GPU work.

# 1. Fixed project contract

## 1.1 Model and supported limits

Use **Qwen/Qwen2.5-0.5B-Instruct**, the original BF16 Safetensors checkpoint. Record the *exact* Hugging Face revision and file hashes in `reference/manifest.json`; do not use an unpinned `main` checkout for golden fixtures. Download the weights rather than committing them. The model configuration is authoritative for constants and optional behavior.

| Parameter | Expected value |
|---|---:|
| Decoder layers | 24 |
| Hidden / MLP dimensions | 896 / 4864 |
| Query / KV heads | 14 / 2 |
| Attention head dimension | 64 |
| Vocabulary size | 151936 |
| RoPE base (`rope_theta`) | 1000000 |
| Normalization epsilon | 1e-6 |
| Word embeddings | Tied to output projection |
| Original checkpoint precision | BF16 |

The single-request engine supports **1–128 prompt token IDs** and **1–64 requested output tokens**. Phase 7 adds batches up to 4; Phase 8 adds **4 active decode requests** and **16 waiting requests**. The supported total context is 192 tokens for this project, not the full context window of the checkpoint. Apply the 128-token limit to incoming prompts, including any chat-template tokens. Internal model forward calls support prefixes of up to 192 tokens so that uncached generation can process growing prefixes beyond 128. The standalone `logits` command retains the 128-token prompt limit. Explicitly reject requests exceeding prompt, output, or total-context budgets. An 8 GB MacBook Air is a *design constraint*, not a promise of performance; measure peak usage and keep alternate apps closed for memory-heavy tests.

**Fixture policy:** Golden comparisons use explicit token IDs, not tokenized natural language, until Phase 5. Tokenization may be implemented using an existing small library or adapter; you are **not** required to write a tokenizer. All tests after Phase 0 must run offline against already downloaded model assets and committed fixtures.

## 1.2 Responsibilities and permitted dependencies

| Write yourself in Zig | May be delegated to established libraries |
|---|---|
| Model structure, Qwen2 layer wiring, tensor shapes, weights-to-layer mapping | MLX-C: tensor objects, arithmetic, Metal execution, synchronization |
| Attention/GQA, RoPE application, RMSNorm, residual computation, SwiGLU composition | Individual MLX ops; their implementations need not be rewritten |
| Forward pass, generation loop, greedy decoding and stopping behavior | Safetensors reading via MLX-C `mlx_load_safetensors` |
| Per-request KV-cache representation, lifecycle, position tracking | Tokenization and chat-template application via a pinned third-party implementation |
| Batched model invocation, scheduler, queues, request state, cancellation | Zig standard library or small HTTP/event-loop dependency |
| Verification scripts, benchmarks, reproducibility records | Python and MLX LM as **offline reference/testing tools only** |

**Disallowed:** calling `mlx_lm.generate`, llama.cpp, or an existing inference server in the production runtime; embedding an existing Qwen model implementation in place of your own; silently falling back to per-request execution while claiming a batched GPU step. MLX's optimized primitive operators (e.g. attention/normalization) may be used **after** the corresponding reference-composed primitive passes its fixtures; note such substitutions in the documentation.

**Resource/interop note:** MLX-C object lifetimes, error callbacks, device selection, and lazy execution must be mediated by a small Zig FFI layer. Do not assume that changing MLX's error handler makes all failures recoverable inside the current process. Test fatal/error paths in a child process if necessary.

## 1.3 Required commands, phased in gradually

These are **capability contracts**, not restrictions on exact argument parser or file layout. Document any alternate spelling in the README.

```sh
zig build
zig build test
zinfer probe                                  # Phase 0
zinfer inspect --model models/qwen           # Phase 1
zinfer block-test --fixture fixtures/block   # Phase 3 (test-only CLI permitted)
zinfer logits --model models/qwen --tokens fixtures/prompt_ids.json --out results/logits.json  # Phase 4
zinfer generate --model models/qwen --tokens fixtures/prompt_ids.json --max-tokens 16           # Phase 5
zinfer generate --model models/qwen --prompt "Hello" --max-tokens 16                           # Phase 5
zinfer serve --model models/qwen --port 8080 --max-active 4                                     # Phase 8
zinfer bench --workload fixtures/workloads/single.jsonl --output results/single.json           # Milestone 1
zinfer bench --workload fixtures/workloads/mixed.jsonl --output results/bench.json             # Phase 9
```

`logits` accepts token IDs and emits a final-position vector of 151,936 numbers, or a binary array plus explicit dtype/shape metadata. `generate` provides ordered token IDs, decoded text, finish reason (`eos` or `length`), and generated-token count in machine-readable form; `temperature=0` is the only required sampling policy. An empty prompt, invalid token ID, negative token limit, or an excessive budget fails with a useful diagnostic.

# 2. The ten phase gates

**Reading the checks:** All acceptance tests listed within a phase are required to mark that phase complete. Error tolerances are measured on the arrays specified, *not* on a single aggregate score, unless explicitly indicated. Golden fixtures include their reference versions and are checked into `fixtures/`; the test code that generated them goes into `reference/`.

## Phase 0 — Establish the Zig ↔ MLX-C boundary

**Objective:** Prove you can link the current pinned MLX-C version, execute a tensor computation on the Apple GPU, synchronize it, and manage resources from Zig.

**Build requirements:** Initialize the Zig project, implement array-handle ownership wrappers, device selection, explicit evaluation/synchronization, and a test-only error probe. Document exact Zig version, MLX-C version, macOS version, and build instructions.

**Acceptance tests:**

- **P0.1 Build from clean checkout.** `zig build` and `zig build test` work after the documented prerequisite installation.
- **P0.2 Correct matrix multiplication.** Given `A=[[1,2],[3,4]]` and `B=[[5,6],[7,8]]` (float32), GPU evaluation of `A×B` yields `[[19,22],[43,50]]` with maximum absolute error `≤1e-5`.
- **P0.3 GPU proof.** The run explicitly selects the MLX GPU device and logs that selection; force evaluation to complete before reading the result. CPU-only execution is not accepted.
- **P0.4 Resource and error discipline.** Construct/destroy 1,000 temporary handles without leaked *logical* handles; log the counter reaching zero. Install a custom MLX-C error handler in a test process; a separate subprocess triggers a deliberately invalid operation and verifies that the handler is invoked and a nonzero termination plus diagnostic is produced (the subprocess need not remain usable afterward).

**Demonstration:** `zinfer probe` prints device details and the verified matrix output.  
**Evidence:** `docs/environment.md`, `tests/ffi/`, probe output.  
**Exit criterion:** P0.1–P0.4 PASS.

## Phase 1 — Inspect and load the checkpoint

**Objective:** Load the pretrained model's named parameters correctly without implementing an unrelated serialization project.

**Build requirements:** Read `config.json`; load the original BF16 Safetensors through MLX-C or another mature reader; validate required parameter names, element types, and shapes; define the model's static configuration representation. You may choose whether tensor values stay lazy or materialize eagerly.

**Acceptance tests:**

- **P1.1 Inspection.** `zinfer inspect` reports the nine table values from §1.1 and the number of required named tensors; reject incompatible `model_type` or unsupported architectures.
- **P1.2 Critical weights.** Verify embedding matrix shape `[151936,896]`, layer-0 query weight `[896,896]`, layer-0 key/value weights `[128,896]`, and one MLP gate weight `[4864,896]`; perform generalized shape checks for all 24 layers.
- **P1.3 Precision and identity.** Check that all expected checkpoint tensors are BF16 and record checkpoint revision, file size, and SHA-256 hash. Tied output projection must reuse embeddings, not require a duplicate unlisted weight.
- **P1.4 Negative inputs.** Missing config, missing/renamed required parameter, wrong shape, and damaged/truncated tensor file all fail with nonzero exit status and a diagnostic. A third-party loader may detect corruption; Zig must propagate the failure cleanly.

**Demonstration:** An inspection summary of a real downloaded checkpoint.  
**Evidence:** `tests/checkpoint/`, `reference/manifest.json`, inspection output.  
**Exit criterion:** P1.1–P1.4 PASS.

## Phase 2 — Implement transformer primitives

**Objective:** Build and numerically validate the components of a Qwen2 transformer *independently of real model weights*.

**Build requirements:** Implement linear projection (with and without bias), RMSNorm, rotary positional embedding, causal masking, grouped-query attention (14 query heads and 2 KV heads), SiLU/SwiGLU, and the residual additions. Use MLX operations but own the composition and shape logic in Zig. A straightforward, unfused implementation is preferred for the first pass.

**Acceptance tests:**

- **P2.1 Numerical fixtures.** With fixed float32 synthetic inputs, compare each primitive elementwise with an independently generated Python/MLX reference using `atol=1e-3, rtol=1e-3` (both apply: `abs(actual−ref) ≤ atol + rtol·abs(ref)` for every element).
- **P2.2 Position dependence.** RoPE tests cover positions 0 and 37; identical vectors at these offsets produce the expected distinct rotated outputs.
- **P2.3 Causality.** For a three-token sequence, changing token 3's input cannot affect the attention outputs for tokens 1–2 (maximum absolute difference `≤1e-4`, float32).
- **P2.4 GQA mapping.** A fixture with 14 distinct query heads and 2 KV heads proves 7-to-1 mapping per KV head; changing KV head 0 only affects query heads 0–6, and similarly for head 1.
- **P2.5 Failure behavior.** Reject mismatched projection dimensions and inconsistent attention masks instead of broadcasting them silently.

**Demonstration:** A primitive test report listing shape, dtype, maximum absolute error, and pass/fail for each operation.  
**Evidence:** `tests/primitives/`, fixture generator, results report.  
**Exit criterion:** P2.1–P2.5 PASS.

## Phase 3 — Reproduce one pretrained transformer block

**Objective:** Run *one real Qwen2 decoder block* with checkpoint weights and establish layer-by-layer parity before multiplying complexity by 24.

**Build requirements:** Wire the complete first block: input RMSNorm → Q/K/V → RoPE → causal GQA → output projection → residual → post-attention RMSNorm → SwiGLU MLP → residual. Expose test-only intermediate tensors: Q, K, V, attention output, post-attention residual, and final hidden state.

**Acceptance tests:**

- **P3.1 Shape contract.** Input `[1,8,896]` produces output `[1,8,896]`; Q shape `[1,14,8,64]` and K/V `[1,2,8,64]` before the attention operation.
- **P3.2 Numerical parity.** With identical first-block weights and a fixed float32 hidden-state fixture, each named intermediate tensor matches the reference with `atol=3e-3, rtol=3e-3` after both implementations explicitly cast the block inputs/weights to float32 for this test.
- **P3.3 BF16 smoke test.** A separate BF16 weight/input run is finite and has output cosine similarity `≥0.999` against the matching reference; report maximum absolute and normalized error as diagnostics, not just PASS.
- **P3.4 Causality and determinism.** The first position's block output is unchanged within the float32 fixture tolerance when later tokens are replaced; repeated identical evaluation gives the same resulting token-facing float32 tensor within `1e-5` on the same environment.

**Demonstration:** `zinfer block-test` prints measured differences for each intermediate.  
**Evidence:** `fixtures/block/`, block parity results.  
**Exit criterion:** P3.1–P3.4 PASS.

## Phase 4 — Full uncached pretrained-model forward pass

**Objective:** Compute the next-token logits from real input token IDs using your own 24-layer implementation.

**Build requirements:** Add token embeddings, 24 decoder blocks, final RMSNorm, and tied output projection. Implement a test-only path that can expose hidden states from selected layers and logits from all sequence positions. No token-generation loop is required yet.

**Acceptance tests:**

- **P4.1 Shape/finite outputs.** Token sequences of lengths `1,2,8,32,64,128` each return finite final-position logits with shape `[151936]`. The internal test-only forward path also accepts lengths 191 and 192 with the same shape/finite checks.
- **P4.2 Independent logits oracle.** For five committed token-ID prompts, the final logit vector has cosine similarity `≥0.999` and median absolute difference `≤0.15` against reference BF16 execution; top-1 IDs agree on at least five prompts selected with reference top-1/top-2 margin `≥1.0`. Save the margin and actual discrepancies. If a difference is systematic, investigate it rather than relaxing criteria silently.
- **P4.3 Intermediate hidden states.** For two prompts, the final hidden state at layers 0, 11, and 23 has cosine similarity `≥0.999` with reference. Log maximum absolute errors for debugging.
- **P4.4 Causality.** On the test-only all-position path, appending one token leaves logits of the existing positions unchanged within `atol=0.15, rtol=0.03` for the same BF16 execution mode.
- **P4.5 Invalid inputs.** Reject an out-of-range token ID, zero-length token list, and >128-token prompt supplied to `logits`. Separately reject internal forward prefixes longer than 192; do not apply the public prompt limit to internal growing prefixes.

**Demonstration:** `zinfer logits` writes a correctly shaped output which can be checked against reference.  
**Evidence:** `fixtures/logits/`, reference generator, numeric report.  
**Exit criterion:** P4.1–P4.5 PASS.

## Phase 5 — Basic autoregressive generation (intentionally uncached)

**Objective:** Generate real text before tackling cache correctness; this is a **slow but trusted baseline**.

**Build requirements:** Implement greedy next-token selection (`argmax` with deterministic tie rule), append the chosen token to the prefix, rerun the uncached full forward pass, and stop on EOS or output-token budget. Integrate a pinned tokenizer and an optional chat template. Output token IDs as well as decoded text. The running engine must not call Python to compute model outputs.

**Acceptance tests:**

- **P5.1 Generation contract.** Use a committed explicit-ID prompt with reference winning-token margin `≥1.0` at each generated step. With `max_tokens=8`, the engine returns an ordered sequence of 1–8 generated token IDs, finish reason, and count; tokens equal the reference greedy sequence. Record each reference margin.
- **P5.2 Stop conditions.** EOS stops early and reports `eos`; a fixture that does not emit EOS stops exactly at the budget and reports `length`. Use a synthetic logit source/test seam for deterministic edge cases.
- **P5.3 Text path.** `--prompt "Hello"` applies the documented tokenizer/template, produces UTF-8-decoded text, and logs the exact input token IDs. A no-template flag or mode must exist for raw prompt comparisons.
- **P5.4 Input validation.** Empty prompt, nonexistent tokenizer data, zero/negative output limit, and a request exceeding the supported context are rejected explicitly.
- **P5.5 Context boundary.** A 128-token prompt with a 64-token output budget is accepted. Use a test-only fixed non-EOS continuation to exercise all 64 uncached generation iterations with real model forward calls; the final call processes a 191-token prefix and the resulting sequence has 192 tokens. This test isolates context accounting from the model's choice to emit EOS. Reject a 129-token prompt and a 65-token output budget.

**Demonstration:** One actual short model response with its underlying IDs and stop reason.  
**Evidence:** `tests/generation/`, one command transcript, reference outputs.  
**Exit criterion:** P5.1–P5.5 PASS.

## Phase 6 — KV-cached autoregressive generation

**Objective:** Reuse attention keys/values without recomputing the entire prefix while preserving generation semantics.

**Build requirements:** Implement per-layer, per-request K/V storage, correct absolute RoPE positions, cache append, read, reclamation, and reset. Prefill a prompt once and decode one token at a time afterward. Keep the uncached mode available as a correctness oracle. Choose a simple dense cache representation; paged attention is not required.

**Acceptance tests:**

- **P6.1 Greedy parity.** For prompt lengths `1,8,32,64,128`, use committed fixtures with reference top-1/top-2 margin `≥1.0` at every generated step. Cached and uncached greedy generation emit identical token IDs for 16 steps, unless EOS occurs first; terminate both modes identically. Record each reference margin. Near-tie fixtures are evaluated separately under P6.5 and do not establish exact sequence parity.
- **P6.2 KV shape and lengths.** After a prompt of P tokens and G generated steps, each layer's cache has exactly P plus the number of generated *tokens already forwarded through that layer*; shape is documented using 2 KV heads × head width 64. Instrument and assert cache length at every decode boundary (avoid a one-token off-by-one).
- **P6.3 Reuse proof.** After prefill, each decode step forwards **one** new input position rather than the entire prompt. Record per-step sequence lengths from the model execution path.
- **P6.4 Lifecycle.** Two independent sequential requests, an early EOS, and a cancelled simulation all release their request-owned cache handles. New requests cannot observe cached state from prior ones.
- **P6.5 Numerical drift.** Feed cached and uncached execution the same committed 16-token continuation, rather than feeding each mode its own sampled outputs, so every logit comparison uses an identical prefix. Require cosine similarity `≥0.999` at every compared step, including near-tie cases. Report maximum absolute error, top-1 agreement, and reference winning-token margins. Near ties may explain top-1 disagreement here, but do not exempt logits from numerical checks.
- **P6.6 Context boundary.** Repeat P5.5 with caching, forwarding the same fixed continuation and asserting cache lengths at every step. After selecting output token 64, the cache contains 191 forwarded tokens; the selected final token is not forwarded merely to fill the cache.

**Demonstration:** Generate the same response in cached and uncached modes and show the per-step forward lengths.  
**Evidence:** `tests/cache/`, cache traces, parity report.  
**Exit criterion:** P6.1–P6.6 PASS.

### Milestone 1 — Focused benchmark and completion record

After Phases 0–6, compare uncached versus cached generation for a fixed 32-token prompt and 16 outputs. Use a committed non-EOS workload with stable winning-token margins and identical weights/dtype in both modes. Record at least three repetitions per condition, raw values and medians for TTFT, total latency, generated tokens/s, and per-step decode latency. Synchronize relevant MLX work before stopping timers, separate cold start, and state whether loading and tokenization are included. Record peak observed process memory, hardware/toolchain/model revisions, and thermal/power caveats; interleave conditions where practical.

Document build, inspect, logits, generate, test, and the focused `bench` command in the README. Add a short note explaining the model and KV ownership, and a concise `docs/performance.md` with the hypothesis, measured outcome, and limitations. Record Milestone 1 completion in `progress.md`. No server, load generator, or batching is required for this milestone; Phase 9 extends these same benchmark records. A measured slowdown is an acceptable result.

## Phase 7 — Static batched inference

**Objective:** Prove the model can process more than one independent sequence in **the same MLX batched computation** before introducing network concurrency.

**Build requirements:** Add a batch dimension across model execution and KV state. Initially accept equal-length prompts and equal cache lengths; separate input sequences must never attend to or read each other's KV states. Execute batched prefill and one batched decode step for B=2 and B=4; no live scheduler is needed yet.

**Acceptance tests:**

- **P7.1 Real batching.** Instrument the core model call to confirm tensor batch dimension `B=2` and `B=4` at **every** transformer block for at least one forward pass; exactly one per-layer logical batched forward path must process the B sequences, rather than B per-request model invocations.
- **P7.2 Independent parity.** Given 2 and 4 equal-length distinct prompts, batched next-token top-1 IDs equal separate cached runs for each request, with reference top-1 margin `≥1.0`; each resulting final logit vector has cosine similarity `≥0.999` to separate execution.
- **P7.3 No information leakage.** Modify only batch member 1's prompt while holding member 0 constant; compare member 0's entire final logit vector elementwise against its baseline and report maximum absolute difference. Establish tight `atol`/`rtol` from repeated identical runs on the pinned environment, record them in the fixture metadata before running perturbation tests, and require every element to satisfy them. Cosine similarity alone does not establish isolation. Swap batch order, restore the original request order for comparison, and apply the same elementwise checks.
- **P7.4 Cache isolation.** Cache lengths and K/V values are tracked independently by batch member. Snapshot the unaffected member's cache before modifying, removing, or clearing another member; afterward its length and stored K/V values must be exactly unchanged. Dense arrays and repacking are permitted.
- **P7.5 Boundary.** Accept B=1, 2, 4; reject B=0 and B>4 with documented errors. In this phase only, unequal input lengths may be rejected as unsupported.

**Demonstration:** A trace with one `[B,...]` GPU execution and per-member outputs, plus a comparison against B independent runs.  
**Evidence:** `tests/batch/`, execution trace, parity report.  
**Exit criterion:** P7.1–P7.5 PASS.

## Phase 8 — Continuous batching, request lifecycle, and minimal server

**Objective:** Admit, stream, finish, and cancel requests while the decode engine is running. Different arrival times must work, and requests that can share a step must actually batch together.

**Build requirements:** Implement a FIFO waiting queue (up to 16), up to 4 active decode requests, prefill/decode scheduling, IDs, cancellation, EOS and length termination, and an HTTP server with SSE streaming. You may serialize prefill on the single GPU. For mixed context lengths you may use **equal-cache-length buckets** and run separate batched groups rather than implement padded heterogeneous attention; document when this costs throughput. Joining/leaving must work at step boundaries without leaking state.

**Bucket scheduling:** Equal prompt lengths do not imply equal cache lengths after staggered arrivals. Document how newly admitted requests can join an existing decode group. A permitted starting policy is to pause the leading group while a newly admitted request takes a bounded number of single-token catch-up steps, then merge at equal cache lengths. Set and record a finite catch-up bound; when the gap exceeds it or a request finishes first, resume fair progress across separate buckets. Round-robin bucket scheduling is sufficient outside catch-up. Alternative policies are allowed if they demonstrate P8.1 and fair progress without requiring heterogeneous attention.

**Required HTTP subset:** `GET /health`; `POST /v1/completions` with `{"prompt": string, "max_tokens": integer, "stream": boolean, "temperature": 0}`; and a documented `DELETE /requests/{id}` cancellation endpoint (or an equivalent explicit cancellation operation). Streaming uses ordered SSE token events with request ID and token ID, followed by exactly one terminal event (`eos`, `length`, `cancelled`, or `error`). `stream=false` returns the complete ordered token-ID sequence and terminal reason. Do not claim full OpenAI API compatibility.

**Acceptance tests:**

- **P8.1 Overlap.** Start A, then admit B after A has completed two single-token decode steps. Use same-length prompt fixtures that remain active long enough for the documented policy to merge them (for catch-up, use a bound of at least two steps). Both terminate correctly; no tokens from one are attributed to the other. Confirm at least one shared decode step with tensor batch dimension `B≥2` in the actual execution trace, and log cache lengths plus any catch-up pauses.
- **P8.2 Arrival/departure.** While two requests are decoding, introduce a third, let the first finish, and start a fourth from the waiting queue. Each output must equal its isolated cached greedy baseline (fixtures use non-tie prompts).
- **P8.3 Cancellation.** Cancel an active request after ≥2 streamed tokens; no further tokens for it are emitted after the cancellation acknowledgement, exactly one terminal cancellation event appears, its cache is released, and unaffected requests complete correctly.
- **P8.4 Different prompt lengths.** Two prompts of lengths 8 and 32 both complete correctly when submitted concurrently. The engine may run them in separate equal-cache-length groups, but cannot reject them merely because their positions differ.
- **P8.5 Backpressure/capacity.** Never exceed 4 active or 16 waiting requests. A 21st simultaneous pending request is rejected with a machine-readable overload status rather than silently dropped. (Use a test barrier so the first 20 are still pending.)
- **P8.6 Transport correctness.** Streaming token events are ordered and terminated exactly once; nonstreaming output matches them for the same fixed prompt. Invalid JSON, unsupported temperature, oversized request, and unknown cancellation ID return documented 4xx responses; server remains healthy afterward.
- **P8.7 Fair progress.** Under a finite workload of ≤20 valid requests and no permanent client disconnections, every admitted request eventually runs to completion or explicit cancellation. Record FIFO queue admission and start order in an integration trace.

**Demonstration:** Launch server; stream two overlapping requests, cancel one, and show B≥2 in the execution trace.  
**Evidence:** `tests/serving/`, scripted curl or client harness, request-state trace.  
**Exit criterion:** P8.1–P8.7 PASS.

## Phase 9 — Benchmark, characterize, and finish

**Objective:** Extend the Milestone 1 measurements to batching and serving, making a defensible performance claim *or reporting a measured limitation* using reproducible experiments. This completes Milestone 2.

**Build requirements:** Add a load generator, JSON/CSV raw result export, and a compact `docs/performance.md` (or equivalent). Compare **uncached vs cached single-request decode**, and **B=1 versus B=2 and B=4** with fixed prompt/output lengths. Include short-prompt and long-prompt workloads and a mixed-arrival workload. Measure CPU/GPU memory where possible, and explain the limitations of your instrumentation.

**Acceptance tests:**

- **P9.1 Reproducibility.** A clean checkout can build, download the pinned model via documented instructions, run all P0–P8 tests, and reproduce the benchmark invocation using only documented local dependencies.
- **P9.2 Timing correctness.** Before stopping any GPU inference measurement, synchronize/evaluate the relevant MLX work. Report cold-start separately; state clearly whether tokenization, queue delay, and model load are included in each metric.
- **P9.3 Controlled comparison.** Run at least 3 repetitions per condition, record per-run raw values and medians for TTFT, total request latency, generated tokens/s, and per-step decode latency. Use identical weights/dtype and deterministic generation; interleave condition order where practical to reduce thermal drift.
- **P9.4 At least four comparisons.** Compare: (a) uncached versus cached for 32-token input + 16 outputs; (b) B=1 versus B=2 for 32+16; (c) B=1 versus B=4 for 32+16; and (d) 8-token versus 128-token prefills. For each, state the hypothesis, measured outcome, and plausible explanation.
- **P9.5 Resource reporting.** Record chip model, unified memory, macOS, Zig/MLX-C versions, model hash, workload, peak observed process memory, and thermal/power caveats. If an experiment OOMs, report it as a result and describe the smallest passing configuration; this does not fail the gate if mandatory B≤4 functional tests still pass.
- **P9.6 Final demo and documentation.** README explains how to build, inspect, generate, serve, test, and benchmark. A concise architecture note explains layers, KV ownership, batch assembly, and scheduler. An end-to-end script generates text, streams two overlapping requests, and prints the benchmark summary.

**Demonstration:** One reproducible command suite plus a brief written conclusion identifying the dominant bottleneck or uncertainty.  
**Evidence:** `results/raw/`, `docs/performance.md`, `docs/architecture.md`, final README.  
**Exit criterion:** P9.1–P9.6 PASS. **A smaller or negative speedup is allowed and should be reported.**

# 3. Acceptance-test conventions

**Precision comparisons.** Where the text says `atol` and `rtol`, every element must satisfy `|actual−reference| ≤ atol + rtol·|reference|`. When the text requests cosine similarity, compare flattened float32 representations of the named tensors, after synchronizing MLX. Run all reference fixtures on the same model revision and explicitly record whether the oracle used float32 or BF16. Never hide NaN/Inf behind aggregate similarities. **Threshold exceptions are not automatically permitted**; if legitimate numerical differences make a test impossible on a pinned environment, document the discrepancy and revise the specification explicitly before declaring PASS.

**Greedy parity fixtures.** Exact token-sequence comparisons use committed prompts with recorded reference winning-token margins at every compared step. When testing numerical drift, feed both implementations the same fixed continuation so that a near-tie argmax cannot cause unrelated later prefixes. Report near-tie disagreements explicitly; they do not waive numerical tolerances or replace stable-margin generation tests.

**Reference independence.** Production code does not call MLX LM or Python. The oracle is a separately pinned reference environment and/or independent mathematical fixtures; commit generation scripts, input IDs, fixture SHA-256 checksums, and a `requirements` lock or resolved package list. Do not generate expected output by invoking the exact Zig function being tested.

**Concurrency semantics.** A token is counted as *generated* when selected by the sampler; it is stored into KV only on a subsequent forward invocation. That distinction matters for the P6 cache-length assertions and for EOS. Once a cancellation is acknowledged, subsequent output for the cancelled request is forbidden. A single compute worker is permissible and recommended for a first implementation.

**Bounded memory.** Enforce prompt, output, context, batch, and queue limits. Before admitting a request, use a documented conservative memory estimate for weights, live caches, expected temporary allocations, and headroom; reject requests that exceed the configured engine budget with a useful diagnostic (a documented machine-readable error in serving mode). Test rejection with an artificially small budget rather than inducing system-wide memory pressure. This estimate is admission control, not a guarantee that all backend allocations will succeed. Handle recoverable allocation errors by failing the affected request and reclaiming its state. Document unrecoverable MLX/Metal failures and possible process termination; per-request survival of fatal backend errors and worker-process isolation are outside the core scope. Test fatal paths in subprocesses. MLX may retain allocations in a caching allocator; do not equate process RSS with live logical tensor ownership.

**Completion evidence format.** Add a `progress.md` with one record per phase and milestone: date, status, command, model/toolchain revisions, test output path, any deliberate limitations. The checklist below tracks demonstrated capabilities.

# 4. Project structure (suggested, not prescriptive)

```text
zinfer/
├── build.zig
├── src/
│   ├── main.zig                 # inspect / logits / generate / serve / bench
│   ├── mlx/                     # FFI wrappers and ownership
│   ├── model/                   # config, weights, primitives, Qwen2 blocks
│   ├── inference/               # prefill, decode, KV cache, sampling
│   └── serving/                 # queue, requests, scheduler, HTTP/SSE
├── reference/                   # pinned Python or MLX-LM oracle, manifest
├── fixtures/                    # small, committed deterministic fixtures
├── tests/                       # phase-specific correctness/integration tests
├── bench/                       # workload generation and result processing
├── results/                     # numeric reports, traces, raw benchmarks
├── docs/                        # environment, architecture, performance
├── progress.md                  # gate status and reproducer commands
└── README.md
```

## Milestone checklist

Milestone 1:

- [ ] P0: Zig/MLX-C GPU probe and ownership/error probe PASS.
- [ ] P1: Real checkpoint inspection, required weight validation, and negative cases PASS.
- [ ] P2: Transformer primitives PASS elementwise reference fixtures.
- [ ] P3: First pretrained transformer block PASS intermediate-tensor parity.
- [ ] P4: Full 24-layer model PASS reference-logit and intermediate-state comparisons.
- [ ] P5: Uncached greedy generation produces a real model response and correct stop reasons.
- [ ] P6: Cached generation matches the uncached baseline and reuses previous K/V.
- [ ] Focused benchmark: Reproducible single-request comparison, usage documentation, architecture note, and Milestone 1 completion record are committed.

Milestone 2:

- [ ] P7: Static B=2 and B=4 execution uses real batched model operations.
- [ ] P8: Dynamic arrivals/departures, cancellation, queue limits, and HTTP/SSE tests PASS.
- [ ] P9: Reproducible benchmark results and concise documentation are committed.

# 5. Explicitly excluded from the core

Not required: training or fine-tuning; multiple models; quantization; custom Metal kernels; speculative decoding; paged attention; tensor or pipeline parallelism; multi-machine serving; prefix sharing; speculative scheduling; authentication; persistent storage; full OpenAI API compatibility; browser UI; Docker/Kubernetes deployment; production uptime; telemetry stacks; formal academic report; video walkthrough; GitHub release tagging.

**Possible extensions after Milestone 2:** graph compilation versus eager MLX; memory-optimized KV layout; padded/masked heterogeneous-length decode in one batch; quantized weights; chunked prefill; multiple scheduling policies; optional direct Metal kernel; comparative benchmarks with mlx-lm. Each extension includes its own correctness and performance regression tests, but does not retroactively change either milestone's completion criteria.

# 6. Engineering risks and decision guidance

| Risk | Symptom | Containment |
|---|---|---|
| Zig and MLX-C version mismatch | Link failures, different API signatures | Pin versions; isolate foreign calls in Phase 0 |
| MLX-C error handling exits process | Test stops before useful diagnostic | Test unsafe paths in subprocess; install handler deliberately |
| BF16 arithmetic differences | Correct-looking text but imperfect oracle parity | Compare float32 primitives, block intermediates, then BF16 logits |
| Cache position off-by-one | Repeated/missing generated tokens | Record forwarded-token count versus selected-token count |
| Batch incorrectly implemented as a loop | Apparent concurrency without GPU batching | Assert B dimension at each layer and count logical model invocations |
| Mixed-length decode complexity | Shape mismatch or cross-request leaks | Begin with equal-length static batches; use bounded catch-up and fair scheduling for dynamic buckets |
| 8 GB memory or thermal pressure | OOM, inconsistent throughput | Keep 0.5B BF16 model, cap concurrency, report real usage |
| Overengineering interfaces | More server code than inference learning | No full API compatibility; one minimal server endpoint |

# 7. Review record

## Changes from v1 to v2

**Feasibility review.** Split weight loading, primitives, one block, and full model into separate gates. Safetensors parser implementation is optional because MLX-C exposes a reader. Keep the execution backend on Metal and scope limited to one small checkpoint.

**Correctness review.** Require intermediate tensor comparisons, fixed oracle revisions, explicit stop reasons, a distinct uncached baseline, and a real batched execution trace. Avoid treating plausible text as numerical proof.

**Systems review.** Separate static batching from request scheduling. Introduce queue bounds, lifecycle invariants, cancellation semantics, and mixed-length requests as explicit tests. Permit cache-length bucketing to avoid turning padded attention into a prerequisite.

**Self-directed learning review.** Remove weeks, points, formal reports, mandatory screencasts, and release administration. Preserve rigorous gates and an independently demonstrable artifact at each phase.

## Changes from v2 to v2.1

Define a completed single-request milestone after Phase 6 and a focused benchmark, while retaining Phases 7–9 as the full batching/serving roadmap. Clarify public prompt limits versus internal growing prefixes and test the full 128+64 boundary. Add an explicit policy for batching staggered arrivals, stable-margin sequence fixtures and fixed-continuation logit comparisons, elementwise/cache-state isolation checks, and bounded-memory admission control with documented fatal-backend behavior. Keep measurable acceptance checks while replacing assessment-style completion language.

These are internal review passes, **not** independent subagent audits. External review of Zig/MLX-C ABI compatibility, BF16 parity thresholds, and batching strategy remains welcome. A phase-0 prototype is the definitive feasibility test.

# 8. Primary reference sources

1. [MLX-C documentation](https://ml-explore.github.io/mlx-c/) — API, synchronization, GPU ops, I/O, and error management.
2. [MLX-C error management](https://ml-explore.github.io/mlx-c/build/html/error.html) — handler registration and error behavior.
3. [MLX-LM Qwen2 reference implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py) — numerical oracle, not runtime dependency.
4. [Qwen2.5-0.5B-Instruct model and config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — original checkpoint and model constants.
5. [MLX-C operations](https://ml-explore.github.io/mlx-c/build/html/ops.html) — array operations and mathematical kernels.
