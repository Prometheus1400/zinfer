const std = @import("std");

const log = std.log.scoped(.args);

pub const InspectArgs = struct {
    model: []const u8 = "",
};

pub const PromptArgs = struct {
    model: []const u8 = "",
    prompt: []const u8 = "",
    stream: bool = false,
};

pub const Command = union(enum) {
    inspect: InspectArgs,
    prompt: PromptArgs,
};

pub const Args = struct {
    command: ?Command = null,
    help: bool = false,
};

pub fn parseArgs(args: std.process.Args) !Args {
    var logging_iter = args.iterate();
    log.debug("Args:", .{});
    while (logging_iter.next()) |arg| {
        log.debug("> {s}", .{arg});
    }

    var ret: Args = .{};
    var parsing_iter = args.iterate();
    // skip first arg
    _ = parsing_iter.skip();

    var i: u8 = 0;
    while (parsing_iter.next()) |arg| {
        if (std.mem.eql(u8, "--help", arg)) {
            ret.help = true;
        }

        if (std.mem.eql(u8, "inspect", arg) and i == 0) {
            ret.command = .{ .inspect = InspectArgs{} };
        } else if (std.mem.eql(u8, "prompt", arg) and i == 0) {
            ret.command = .{ .prompt = PromptArgs{} };
        }

        if (ret.command) |*c| {
            switch (c.*) {
                .inspect => |*x| {
                    if (std.mem.eql(u8, "--model", arg)) {
                        const model_str = parsing_iter.next() orelse return error.NoModelProvided;
                        x.model = model_str;
                    }
                },
                .prompt => |*x| {
                    if (std.mem.eql(u8, "--model", arg) or std.mem.eql(u8, "-m", arg)) {
                        const model_str = parsing_iter.next() orelse return error.NoModelProvided;
                        x.model = model_str;
                    }
                    if (std.mem.eql(u8, "--prompt", arg) or std.mem.eql(u8, "-p", arg)) {
                        const prompt_str = parsing_iter.next() orelse return error.NoPromptProvided;
                        x.prompt = prompt_str;
                    }
                    if (std.mem.eql(u8, "--stream", arg) or std.mem.eql(u8, "-s", arg)) {
                        x.stream = true;
                    }
                },
            }
        }
        i += 1;
    }

    return ret;
}
