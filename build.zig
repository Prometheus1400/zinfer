const std = @import("std");

pub fn build(b: *std.Build) void {
    // CMake builds for the host. Cross compilation needs a matching native
    // toolchain and is deliberately outside this initial integration.
    const target = b.resolveTargetQuery(.{});
    if (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)
        @panic("zinfer currently supports native Apple Silicon macOS builds");
    const optimize = b.standardOptimizeOption(.{});
    const metal = b.option(bool, "metal", "Enable the Metal GPU backend (requires Xcode's Metal compiler)") orelse true;
    const build_options = b.addOptions();
    build_options.addOption(bool, "metal", metal);
    const options_module = build_options.createModule();
    const jobs = b.option(u32, "native-jobs", "Maximum parallel native compilation jobs") orelse 4;
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
    const native_step = b.step("native", "Build the pinned MLX, tokenizer, and MiniJinja libraries");
    native_step.dependOn(&native.step);

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

    const translate = b.addTranslateC(.{
        .root_source_file = b.path("src/mlx/headers.h"),
        .target = target,
        .optimize = optimize,
    });
    translate.addIncludePath(b.path("vendor/mlx-c"));
    const bindings = translate.createModule();
    const bindings_step = b.step("bindings", "Translate the native C headers");
    bindings_step.dependOn(&translate.step);

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

    const chat_template = b.createModule(.{
        .root_source_file = b.path("src/chat_template.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    chat_template.addImport("minijinja_c", minijinja_bindings);
    const chat_template_tests = b.addTest(.{ .root_module = chat_template });
    const run_chat_template_tests = b.addRunArtifact(chat_template_tests);
    run_chat_template_tests.stdio = .inherit;
    run_chat_template_tests.setCwd(b.path("."));
    b.step("test-chat-template", "Run MiniJinja wrapper and Qwen template tests").dependOn(&run_chat_template_tests.step);

    const mlx = b.createModule(.{
        .root_source_file = b.path("src/mlx/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mlx.addImport("mlx_c", bindings);
    mlx.addImport("build_options", options_module);
    mlx.addLibraryPath(library_dir);
    mlx.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no, .needed = true });
    mlx.addRPath(library_dir);

    const mlx_tests = b.addTest(.{ .root_module = mlx });
    mlx_tests.step.dependOn(&native.step);
    const run_tests = b.addRunArtifact(mlx_tests);
    run_tests.stdio = .inherit;
    const test_step = b.step("test", "Run MLX, model, and chat template tests");
    test_step.dependOn(&run_tests.step);
    test_step.dependOn(&run_chat_template_tests.step);

    const context = b.createModule(.{
        .root_source_file = b.path("src/context.zig"),
        .target = target,
        .optimize = optimize,
    });
    context.addImport("mlx", mlx);

    const models_test_module = b.createModule(.{
        .root_source_file = b.path("src/models.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    models_test_module.addImport("mlx", mlx);
    models_test_module.addImport("context", context);
    models_test_module.addImport("mlx_c", bindings);
    models_test_module.addImport("tokenizers_c", tokenizer_bindings);
    models_test_module.addImport("chat_template", chat_template);
    models_test_module.addImport("build_options", options_module);
    const models_tests = b.addTest(.{ .root_module = models_test_module });
    models_tests.step.dependOn(&native.step);
    const run_models_tests = b.addRunArtifact(models_tests);
    run_models_tests.stdio = .inherit;
    run_models_tests.setCwd(b.path("."));
    run_models_tests.step.dependOn(&run_tests.step);
    test_step.dependOn(&run_models_tests.step);

    const app = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    app.addImport("mlx", mlx);
    app.addImport("context", context);
    app.addImport("tokenizers_c", tokenizer_bindings);
    app.addImport("chat_template", chat_template);
    // Installed binaries use sibling libraries; cached binaries use the
    // native build directory. Both paths are embedded, without DYLD overrides.
    app.addRPathSpecial("@executable_path/../lib");

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
    run.addPassthruArgs();
    run.step.dependOn(b.getInstallStep());
    b.step("run", "Run zinfer").dependOn(&run.step);
}
