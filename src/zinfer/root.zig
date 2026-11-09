pub const mlx = @import("mlx");
pub const models = @import("models/root.zig");
pub const engine = @import("engine.zig");

pub const Context = @import("context.zig").Context;

comptime {
    if (@import("builtin").is_test) {
        _ = models.qwen;
        _ = engine;
    }
}
