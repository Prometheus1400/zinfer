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
zig build             # build and install into zig-out/
zig build run -- inspect --model qwen
zig build native      # build all native dependencies
zig build tokenizers  # build only the tokenizer C library
zig build minijinja   # build only the MiniJinja C library
zig build bindings    # translate only the C headers
zig build test-chat-template # test the wrapper and local Qwen chat template
```

Options: `-Dmetal=false`, `-Dnative-jobs=2` (default: 4), and
`-Doptimize=ReleaseFast`. Native libraries always build in Release mode.

## Layout

- `build.zig`: native build orchestration, C translation, linking, and installation.
- `cmake/build-mlx.cmake`: serializes native configuration and compilation so
  editor background builds and terminal builds can share the cache safely.
- `cmake/build-tokenizers.cmake`: builds the Rust tokenizer C library under a
  separate lock. Generated `@import("tokenizers_c")` bindings and static linking
  are available inside the runtime module.
- `cmake/tokenizers.h`: the translation unit including `tokenizers_c.h`.
- `cmake/build-minijinja.cmake`: builds the static MiniJinja C library under a
  separate lock using upstream's Cargo lockfile.
- `src/zinfer/root.zig`: the public runtime API, imported with `@import("zinfer")`.
- `src/zinfer/models/`: shared model loading and the Qwen implementation.
- `src/zinfer/chat_template.zig`: the owning `ChatTemplate` wrapper and its tests.
- `src/zinfer/mlx/headers.h`: the single translation unit including `mlx/c/mlx.h`.
- `src/cli/`: command-line argument parsing and the executable entry point.
- `vendor/mlx-c`: unmodified upstream submodule.
- `vendor/tokenizers-cpp`: unmodified upstream tokenizer submodule.
- `vendor/minijinja`: unmodified upstream template engine submodule.
- `.zig-cache/native-mlx/{cpu,metal}`: separate incremental native build trees.
- `.zig-cache/native-tokenizers`: copied crate sources and Cargo's incremental
  native build artifacts.
- `.zig-cache/native-minijinja`: Cargo's incremental MiniJinja build artifacts.
- `zig-out/bin` and `zig-out/lib`: executable, shared libraries, and GPU
  `mlx.metallib`. Keep these folders together when moving the installed output.

Generated bindings and compiled dependencies are ignored by Git.

## Chat templates

Applications can access chat templates through the runtime module:

```zig
const ChatTemplate = @import("zinfer").ChatTemplate;

var template = try ChatTemplate.init(gpa, config.chat_template);
defer template.deinit();

const prompt = try template.render(gpa, &.{
    .{ .role = "user", .content = "hi" },
}, .{});
defer gpa.free(prompt);
```

Pass the `chat_template` string parsed from `tokenizer_config.json` to `init`.
`render` accepts text messages and adds the assistant prefix by default; use
`.{ .add_generation_prompt = false }` to omit it. The returned prompt owns its
bytes and can be passed to the tokenizer's `encode` method. Tool schemas and
multimodal content are not exposed by this wrapper. The generation pipeline
does not apply the template automatically yet.
