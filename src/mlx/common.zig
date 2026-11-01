const std = @import("std");

pub fn check(errno: c_int) !void {
    if (errno != 0) {
        return error.MlxGenericError;
    }
}

pub fn ensure(items: anytype) !void {
    const T = @TypeOf(items);
    const info = @typeInfo(T);

    if (info != .@"struct") {
        @compileError("Expected a tuple or struct, got " ++ @typeName(T));
    }

    inline for (info.@"struct".field_types) |ItemType| {
        if (comptime !std.meta.hasMethod(ItemType, "ensure")) {
            @compileError("Type '" ++ @typeName(ItemType) ++ "' is missing required method 'ensure'");
        }
    }

    inline for (info.@"struct".field_names) |name| {
        try @field(items, name).ensure();
    }
}
