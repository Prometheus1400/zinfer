const std = @import("std");

pub fn build(b: *std.Build) void {
    // CMake builds for the host. Cross compilation needs a matching native
    // toolchain and is deliberately outside this initial integration.
    const target = b.resolveTargetQuery(.{});
    if (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)
        @panic("zinfer currently supports native Apple Silicon macOS builds");
    const optimize = b.standardOptimizeOption(.{});
    const metal = b.option(bool, "metal", "Enable the Metal GPU backend (requires Xcode's Metal compiler)") orelse true;
    const jobs = b.option(u32, "native-jobs", "Maximum parallel C++ compilation jobs") orelse 4;
    const native_dir = std.Build.LazyPath.cache_root.path(b, if (metal) "native-mlx/metal" else "native-mlx/cpu");
    const library_dir = native_dir.path(b, "lib");

    // One lock covers both configure and build, including calls from ZLS.
    const native = b.addSystemCommand(&.{"cmake"});
    native.addDirectoryArg2(b.path("vendor/mlx-c"), .{
        .prefix = "-DMLX_C_SOURCE=",
        .make_absolute = true,
    });
    native.addDirectoryArg2(native_dir, .{
        .prefix = "-DNATIVE_BUILD=",
        .make_absolute = true,
    });
    native.addArgs(&.{
        if (metal) "-DMETAL=ON" else "-DMETAL=OFF",
        b.fmt("-DJOBS={d}", .{jobs}),
        "-P",
    });
    native.addFileArg(b.path("cmake/build-mlx.cmake"));
    // Always let CMake/Ninja check their inputs, including submodule updates.
    native.has_side_effects = true;
    if (metal) {
        const metal_compiler = b.addSystemCommand(&.{ "xcrun", "--find", "metal" });
        metal_compiler.has_side_effects = true;
        native.step.dependOn(&metal_compiler.step);
    }
    b.step("native", "Build the pinned MLX-C and MLX libraries").dependOn(&native.step);

    const translate = b.addTranslateC(.{
        .root_source_file = b.path("src/mlx/headers.h"),
        .target = target,
        .optimize = optimize,
    });
    translate.addIncludePath(b.path("vendor/mlx-c"));
    const bindings = translate.createModule();
    b.step("bindings", "Translate upstream MLX-C headers into a Zig module").dependOn(&translate.step);

    const app = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    app.addImport("mlx_c", bindings);
    app.addLibraryPath(library_dir);
    app.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no, .needed = true });
    // Installed binaries use sibling libraries; cached binaries use the
    // native build directory. Both paths are embedded, without DYLD overrides.
    app.addRPathSpecial("@executable_path/../lib");
    app.addRPath(library_dir);

    const exe = b.addExecutable(.{
        .name = "zinfer",
        .root_module = app,
    });
    exe.step.dependOn(&native.step);
    b.installArtifact(exe);

    const install_libraries = b.addInstallDirectory(.{
        .source_dir = library_dir,
        .install_dir = .lib,
        .install_subdir = "",
        .include_extensions = &.{".dylib"},
    });
    install_libraries.step.dependOn(&native.step);
    b.getInstallStep().dependOn(&install_libraries.step);
    if (metal) {
        const metallib = b.addInstallFileWithDir(
            native_dir.path(b, "_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib"),
            .lib,
            "mlx.metallib",
        );
        metallib.step.dependOn(&native.step);
        b.getInstallStep().dependOn(&metallib.step);
    }

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    b.step("run", "Run zinfer").dependOn(&run.step);
}
