//! Build zinfer on Apple Silicon macOS with Zig 0.17.
//!
//! Common commands:
//!   zig build                    Build and install the CLI and static library.
//!   zig build cli                Install the CLI as zig-out/bin/zinfer.
//!   zig build zinfer             Install the static library as zig-out/lib/libzinfer.a.
//!   zig build check              Check the CLI and all library tests without running them.
//!   zig build run-cli -- chat --model qwen
//!   zig build run-cli -- serve   Start the HTTP server at http://127.0.0.1:8080.
//!
//! Build options (can be combined with any step):
//!   -Doptimize=fast              Optimize Zig code for speed (default: debug).
//!   -Dmetal=false                Disable the Metal backend (enabled by default).
//!   -Dlog-level=info             Set the application log level: err, warn, info, debug.
//!   -Dnative-jobs=8              Limit parallel native compilation jobs (default: 4).
//!
//! Other steps:
//!   zig build native             Build all native dependencies.
//!   zig build tokenizers         Build only the tokenizer C library.
//!   zig build minijinja          Build only the MiniJinja C library.
//!   zig build bindings           Translate the C headers into Zig modules.
//!   zig build test-qwen          Compare batch and individual output; needs models/qwen.
//!   zig build test-engine --test-timeout 60s
//!                               Check engine streaming end to end; needs models/qwen.
//!
//! Initialize vendor submodules before the first build:
//!   git submodule update --init --recursive
//! Native dependencies are cached in .zig-cache; installed libraries go in zig-out/lib.

const std = @import("std");

pub fn build(b: *std.Build) void {
    // Build configuration. The CMake scripts compile for the host, so the Zig
    // target must match the host as well.
    const target = b.resolveTargetQuery(.{});
    if (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)
        @panic("zinfer currently supports native Apple Silicon macOS builds");
    const optimize = b.standardOptimizeOption(.{});
    const log_level = b.option(std.log.Level, "log-level", "Log level: err, warn, info, or debug (default: debug in Debug builds, info in release builds)") orelse
        if (optimize == .debug) std.log.Level.debug else std.log.Level.info;
    const metal = b.option(bool, "metal", "Enable the Metal GPU backend (requires Xcode's Metal compiler)") orelse true;
    const jobs = b.option(u32, "native-jobs", "Maximum parallel native compilation jobs") orelse 4;

    // Share one generated options module across the runtime and CLI.
    const build_options = b.addOptions();
    build_options.addOption(bool, "metal", metal);
    build_options.addOption(std.log.Level, "log_level", log_level);
    const options_module = build_options.createModule();
    const mlx_build_dir = std.Build.LazyPath.cache_root.path(b, if (metal) "native-mlx/metal" else "native-mlx/cpu");
    const mlx_library_dir = mlx_build_dir.path(b, "lib");

    // Native dependencies: MLX-C/MLX, Tokenizers, and MiniJinja.
    // Always run their build scripts so CMake/Cargo can check for changed inputs.
    // The scripts reuse cached outputs; the MLX script also locks its build tree
    // to keep terminal builds and editor background checks from racing.
    const mlx_native = b.addSystemCommand(&.{"cmake"});
    mlx_native.addDirectoryArg2(b.path("vendor/mlx-c"), .{
        .prefix = "-DMLX_C_SOURCE=",
        .make_absolute = true,
    });
    mlx_native.addDirectoryArg2(mlx_build_dir, .{
        .prefix = "-DNATIVE_BUILD=",
        .make_absolute = true,
    });
    mlx_native.addArgs(&.{
        if (metal) "-DMETAL=ON" else "-DMETAL=OFF",
        b.fmt("-DJOBS={d}", .{jobs}),
        "-P",
    });
    mlx_native.addFileArg(b.path("cmake/build-mlx.cmake"));
    mlx_native.has_side_effects = true;
    if (metal) {
        const metal_compiler = b.addSystemCommand(&.{ "xcrun", "--find", "metal" });
        metal_compiler.has_side_effects = true;
        mlx_native.step.dependOn(&metal_compiler.step);
    }
    const native_step = b.step("native", "Build the pinned MLX, tokenizer, and MiniJinja libraries");
    native_step.dependOn(&mlx_native.step);

    // Tokenizers and MiniJinja produce static libraries. Their generated paths
    // also tell Zig to finish these builds before linking an application.
    const tokenizers_native = b.addSystemCommand(&.{"cmake"});
    tokenizers_native.addDirectoryArg2(b.path("vendor/tokenizers-cpp"), .{
        .prefix = "-DTOKENIZERS_SOURCE=",
        .make_absolute = true,
    });
    tokenizers_native.addFileArg2(b.path("cmake/tokenizers.Cargo.lock"), .{
        .prefix = "-DTOKENIZERS_LOCK=",
        .make_absolute = true,
    });
    tokenizers_native.addDirectoryArg2(std.Build.LazyPath.cache_root.path(b, "native-tokenizers"), .{
        .prefix = "-DNATIVE_BUILD=",
        .make_absolute = true,
    });
    const tokenizer_library = tokenizers_native.addOutputFileArg2("libtokenizers_c.a", .{
        .prefix = "-DOUTPUT_LIBRARY=",
        .make_absolute = true,
    });
    tokenizers_native.addArgs(&.{ b.fmt("-DJOBS={d}", .{jobs}), "-P" });
    tokenizers_native.addFileArg(b.path("cmake/build-tokenizers.cmake"));
    tokenizers_native.has_side_effects = true;
    native_step.dependOn(&tokenizers_native.step);
    b.step("tokenizers", "Build the pinned tokenizer C library").dependOn(&tokenizers_native.step);

    const minijinja_native = b.addSystemCommand(&.{"cmake"});
    minijinja_native.addDirectoryArg2(b.path("vendor/minijinja"), .{
        .prefix = "-DMINIJINJA_SOURCE=",
        .make_absolute = true,
    });
    minijinja_native.addDirectoryArg2(std.Build.LazyPath.cache_root.path(b, "native-minijinja"), .{
        .prefix = "-DNATIVE_BUILD=",
        .make_absolute = true,
    });
    const minijinja_library = minijinja_native.addOutputFileArg2("libminijinja_cabi.a", .{
        .prefix = "-DOUTPUT_LIBRARY=",
        .make_absolute = true,
    });
    minijinja_native.addArgs(&.{ b.fmt("-DJOBS={d}", .{jobs}), "-P" });
    minijinja_native.addFileArg(b.path("cmake/build-minijinja.cmake"));
    minijinja_native.has_side_effects = true;
    native_step.dependOn(&minijinja_native.step);
    b.step("minijinja", "Build the pinned MiniJinja C library").dependOn(&minijinja_native.step);

    // C bindings. Each native library gets its own translated Zig module.
    // The bindings step translates headers without building an executable.
    const bindings_step = b.step("bindings", "Translate the native C headers");
    const translate_mlx = b.addTranslateC(.{
        .root_source_file = b.path("src/zinfer/mlx/headers.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_mlx.addIncludePath(b.path("vendor/mlx-c"));
    const mlx_bindings = translate_mlx.createModule();
    bindings_step.dependOn(&translate_mlx.step);

    const translate_tokenizers = b.addTranslateC(.{
        .root_source_file = b.path("cmake/tokenizers.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_tokenizers.addIncludePath(b.path("vendor/tokenizers-cpp/include"));
    const tokenizer_bindings = translate_tokenizers.createModule();
    tokenizer_bindings.addObjectFile(tokenizer_library);
    bindings_step.dependOn(&translate_tokenizers.step);

    const translate_minijinja = b.addTranslateC(.{
        .root_source_file = b.path("cmake/minijinja.h"),
        .target = target,
        .optimize = optimize,
    });
    translate_minijinja.addIncludePath(b.path("vendor/minijinja/minijinja-cabi/include"));
    const minijinja_bindings = translate_minijinja.createModule();
    minijinja_bindings.addObjectFile(minijinja_library);
    bindings_step.dependOn(&translate_minijinja.step);

    // Zig modules. The public zinfer module combines the MLX wrapper, model
    // code, tokenizer, and chat templates for use by the CLI and library.
    const metrics_dep = b.dependency("metrics", .{
        .target = target,
        .optimize = optimize,
    });
    const metrics_module = metrics_dep.module("metrics");
    const mlx = b.createModule(.{
        .root_source_file = b.path("src/zinfer/mlx/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mlx.addImport("mlx_c", mlx_bindings);
    mlx.addImport("build_options", options_module);
    mlx.addLibraryPath(mlx_library_dir);
    mlx.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no, .needed = true });
    // Cached executables can load MLX directly from its native build directory.
    mlx.addRPath(mlx_library_dir);

    const zinfer = b.addModule("zinfer", .{
        .root_source_file = b.path("src/zinfer/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    zinfer.addImport("mlx", mlx);
    zinfer.addImport("tokenizers_c", tokenizer_bindings);
    zinfer.addImport("minijinja_c", minijinja_bindings);
    zinfer.addImport("build_options", options_module);
    zinfer.addImport("metrics", metrics_module);

    // Each target installs its own artifact and the shared runtime assets.
    // Applications import the Zig module directly; they do not need this archive.
    const zinfer_step = b.step("zinfer", "Build and install the zinfer static library");
    const zinfer_lib = b.addLibrary(.{
        .name = "zinfer",
        .root_module = zinfer,
        .linkage = .static,
    });
    zinfer_lib.step.dependOn(&mlx_native.step);
    const install_zinfer = b.addInstallArtifact(zinfer_lib, .{});
    zinfer_step.dependOn(&install_zinfer.step);

    // The CLI includes the HTTP server through its serve subcommand.
    const cli_step = b.step("cli", "Build and install the zinfer CLI");
    const cli_exe = b.addExecutable(.{
        .name = "zinfer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    cli_exe.root_module.addImport("metrics", metrics_module);
    cli_exe.root_module.addImport("zinfer", zinfer);
    cli_exe.root_module.addImport("build_options", options_module);
    // Installed executables find libraries beside bin/, without DYLD overrides.
    cli_exe.root_module.addRPathSpecial("@executable_path/../lib");
    cli_exe.step.dependOn(&mlx_native.step);
    const install_cli = b.addInstallArtifact(cli_exe, .{});
    cli_step.dependOn(&install_cli.step);

    // Runtime assets. MLX's shared libraries and precompiled Metal kernels must
    // accompany the installed executables; the Rust libraries are linked statically.
    const install_libraries = b.addInstallDirectory(.{
        .source_dir = mlx_library_dir,
        .install_dir = .lib,
        .install_subdir = "",
        .include_extensions = &.{".dylib"},
    });
    install_libraries.step.dependOn(&mlx_native.step);
    for ([_]*std.Build.Step{ cli_step, zinfer_step }) |step| {
        step.dependOn(&install_libraries.step);
        b.getInstallStep().dependOn(step);
    }
    if (metal) {
        const metallib = b.addInstallFileWithDir(
            mlx_build_dir.path(b, "_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib"),
            .lib,
            "mlx.metallib",
        );
        metallib.step.dependOn(&mlx_native.step);
        for ([_]*std.Build.Step{ cli_step, zinfer_step }) |step| {
            step.dependOn(&metallib.step);
        }
    }

    // Run steps forward arguments after `--` and install runtime assets first.
    const run_cli = b.addRunArtifact(cli_exe);
    run_cli.addPassthruArgs();
    run_cli.step.dependOn(cli_step);
    b.step("run-cli", "Run the zinfer CLI").dependOn(&run_cli.step);
    b.step("run", "Alias for run-cli").dependOn(&run_cli.step);

    // Opt-in model integration check. Ordinary builds do not load a checkpoint.
    const qwen_tests = b.addTest(.{
        .root_module = zinfer,
        .filters = &.{"Qwen batch matches individual greedy responses"},
    });
    qwen_tests.step.dependOn(&mlx_native.step);
    const run_qwen_tests = b.addRunArtifact(qwen_tests);
    run_qwen_tests.stdio = .inherit;
    b.step("test-qwen", "Compare batched and individual greedy Qwen responses (requires models/qwen)").dependOn(&run_qwen_tests.step);

    const engine_tests = b.addTest(.{
        .name = "engine-test",
        .root_module = zinfer,
        .filters = &.{"Engine streams requests end to end"},
    });
    engine_tests.step.dependOn(&mlx_native.step);
    const run_engine_tests = b.addRunArtifact(engine_tests);
    b.step("test-engine", "Test engine request submission, streaming, and shutdown (requires models/qwen)").dependOn(&run_engine_tests.step);

    // ZLS detects this step for build-on-save diagnostics. Separate compile
    // artifacts avoid emitting application/test binaries or running model tests.
    const check = b.step("check", "Check the CLI and all library tests without running them");
    const check_cli = b.addExecutable(.{
        .name = cli_exe.name,
        .root_module = cli_exe.root_module,
    });
    const check_tests = b.addTest(.{
        .name = "zinfer-check",
        .root_module = zinfer,
    });
    for ([_]*std.Build.Step.Compile{ check_cli, check_tests }) |artifact| {
        check.dependOn(&artifact.step);
    }
}
