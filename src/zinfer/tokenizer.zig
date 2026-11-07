const mlx = @import("mlx");
const std = @import("std");
const tok = @import("tokenizers_c");

const Context = @import("context.zig").Context;

pub const Tokenizer = struct {
    const Self = @This();

    p: tok.TokenizerHandle,

    pub fn fromFile(ctx: Context, file: []const u8) !Self {
        const json_content = try std.Io.Dir.cwd().readFileAlloc(ctx.io, file, ctx.gpa, .unlimited);
        defer ctx.gpa.free(json_content);

        const p = tok.tokenizers_new_from_str(json_content.ptr, json_content.len);
        return .{ .p = p };
    }

    pub fn encode(self: Self, gpa: std.mem.Allocator, input: []const u8) ![]c_int {
        var res: tok.TokenizerEncodeResult = undefined;
        tok.tokenizers_encode(self.p, input.ptr, input.len, 0, &res);
        defer tok.tokenizers_free_encode_results(&res, 1);
        return try gpa.dupe(c_int, res.token_ids[0..res.len]);
    }

    /// Returns text borrowed until the next decode call or tokenizer deinit.
    pub fn decode(self: Self, gpa: std.mem.Allocator, ids: []const u32) ![]const u8 {
        tok.tokenizers_decode(self.p, ids.ptr, ids.len, 1);

        var ptr: [*c]const u8 = undefined;
        var len: usize = undefined;
        tok.tokenizers_get_decode_str(self.p, &ptr, &len);

        return gpa.dupe(u8, ptr[0..len]);
    }

    pub fn deinit(self: *Self) void {
        tok.tokenizers_free(self.p);
    }
};
