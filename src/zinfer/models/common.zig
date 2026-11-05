const template = @import("chat_template");

pub const Conversation = struct {
    id: []const u8,
    messages: []const template.Message,
};
