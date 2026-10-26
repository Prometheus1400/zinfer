# zinfer

Zig project scaffold with MLX-C built, linked, and available as generated C
bindings. The executable currently prints `Hello, world!`.

## Prerequisites

- Apple Silicon macOS (native builds only).
- Zig **0.17.0**.
- CMake **3.25+** and Ninja (`brew install cmake ninja`).
- Xcode with the Metal compiler available through `xcrun --find metal` for
  the default GPU build. Use `-Dmetal=false` for a CPU-only native build.

MLX-C is pinned as a Git submodule at
`57731b00e8bed710e0750e7356da9f18f8eb3e20` (0.7.0), selecting MLX **v0.32.4**
through upstream CMake. The first native build downloads and compiles dependencies;
subsequent builds reuse CMake/Ninja's cached objects.

## Commands

```sh
git submodule update --init --recursive
zig build             # build and install into zig-out/
zig build run         # print Hello, world!
zig build native      # build only MLX-C and MLX
zig build bindings    # translate only the C headers
```

Options: `-Dmetal=false`, `-Dnative-jobs=2` (default: 4), and
`-Doptimize=ReleaseFast`. Native libraries always build in Release mode.

## Layout

- `build.zig`: native build orchestration, C translation, linking, and installation.
- `cmake/build-mlx.cmake`: serializes native configuration and compilation so
  editor background builds and terminal builds can share the cache safely.
- `src/main.zig`: hello world. Generated bindings are available through
  `@import("mlx_c")` when needed.
- `src/mlx/headers.h`: the single translation unit including `mlx/c/mlx.h`.
- `vendor/mlx-c`: unmodified upstream submodule.
- `.zig-cache/native-mlx/{cpu,metal}`: separate incremental native build trees.
- `zig-out/bin` and `zig-out/lib`: executable, shared libraries, and GPU
  `mlx.metallib`. Keep these folders together when moving the installed output.

Generated bindings and compiled dependencies are ignored by Git.
