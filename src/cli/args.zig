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

pub const ChatArgs = struct {
    model: []const u8 = "",
};

pub const ConnectArgs = struct {
    model: []const u8 = "",
    port: u16 = 0,
};

pub const ServeArgs = struct {
    model: []const u8 = "qwen",
    port: u16 = 8080,
    prefill_max_batch_size: usize = 8,
    decode_max_batch_size: usize = 32,
};

pub const Command = union(enum) {
    inspect: InspectArgs,
    prompt: PromptArgs,
    chat: ChatArgs,
    connect: ConnectArgs,
    serve: ServeArgs,
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
        } else if (std.mem.eql(u8, "chat", arg) and i == 0) {
            ret.command = .{ .chat = ChatArgs{} };
        } else if (std.mem.eql(u8, "connect", arg) and i == 0) {
            ret.command = .{ .connect = ConnectArgs{} };
        } else if (std.mem.eql(u8, "serve", arg) and i == 0) {
            ret.command = .{ .serve = ServeArgs{} };
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
                .chat => |*x| {
                    if (std.mem.eql(u8, "--model", arg) or std.mem.eql(u8, "-m", arg)) {
                        const model_str = parsing_iter.next() orelse return error.NoModelProvided;
                        x.model = model_str;
                    }
                },
                .connect => |*x| {
                    if (std.mem.eql(u8, "--model", arg) or std.mem.eql(u8, "-m", arg)) {
                        const model_str = parsing_iter.next() orelse return error.NoModelProvided;
                        x.model = model_str;
                    }
                    if (std.mem.eql(u8, "--port", arg) or std.mem.eql(u8, "-p", arg)) {
                        const port = try std.fmt.parseInt(
                            u16,
                            parsing_iter.next() orelse return error.NoModelProvided,
                            10,
                        );
                        x.port = port;
                    }
                },
                .serve => |*x| {
                    if (i != 0) {
                        if (std.mem.eql(u8, "--model", arg) or std.mem.eql(u8, "-m", arg)) {
                            x.model = parsing_iter.next() orelse return error.NoModelProvided;
                        } else if (std.mem.eql(u8, "--port", arg)) {
                            x.port = try std.fmt.parseInt(u16, parsing_iter.next() orelse return error.MissingPort, 10);
                        } else if (std.mem.eql(u8, "--prefill-max-batch-size", arg)) {
                            x.prefill_max_batch_size = try std.fmt.parseInt(usize, parsing_iter.next() orelse return error.MissingPrefillMaxBatchSize, 10);
                            if (x.prefill_max_batch_size == 0) return error.InvalidBatchSize;
                        } else if (std.mem.eql(u8, "--decode-max-batch-size", arg)) {
                            x.decode_max_batch_size = try std.fmt.parseInt(usize, parsing_iter.next() orelse return error.MissingDecodeMaxBatchSize, 10);
                            if (x.decode_max_batch_size == 0) return error.InvalidBatchSize;
                        } else if (!std.mem.eql(u8, "--help", arg)) {
                            return error.UnknownArgument;
                        }
                    }
                },
            }
        }
        i += 1;
    }

    return ret;
}
