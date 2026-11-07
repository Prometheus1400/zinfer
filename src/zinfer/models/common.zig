const std = @import("std");
const template = @import("chat_template.zig");

const Tensor = @import("mlx").Tensor;
const Stream = @import("mlx").Stream;
const Queue = std.Io.Queue;
const ChatTemplate = template.ChatTemplate;
const Tokenizer = @import("../tokenizer.zig").Tokenizer;
const ConversationState = @import("conversation_state.zig").ConversationState;

/// A conversation is a slice of messages and an ID to be able to look up the conversation
/// state, which consists of generated token ids, kv cache, etc.
///
/// 'id' and 'messages' are views (not owned)
pub const Conversation = struct {
    id: []const u8,
    messages: []const template.Message,
};

/// Pad a batch so it's rectangular. Each input gets left-padded to reach the size of the
/// longest input in the batch.
///
/// Caller owns the returned 2d array slice.
pub fn padBatch(gpa: std.mem.Allocator, batch: []const []const c_int) ![][]const c_int {
    var max_seq_length: usize = 0;
    for (batch) |row| {
        max_seq_length = @max(max_seq_length, row.len);
    }

    var padded_batch = try std.ArrayList([]const c_int).initCapacity(gpa, batch.len);
    errdefer {
        for (padded_batch.items) |row| {
            gpa.free(row);
        }
        padded_batch.deinit(gpa);
    }

    for (batch) |row| {
        const offset = max_seq_length - row.len;
        const new_row_buf = try gpa.alloc(c_int, max_seq_length);
        @memset(new_row_buf[0..offset], 0); // TODO: need pad token here?
        @memcpy(new_row_buf[offset..], row);
        padded_batch.appendAssumeCapacity(new_row_buf);
    }

    return padded_batch.toOwnedSlice(gpa);
}

/// Copies a rectangular batch of token IDs into a tensor.
///
/// Caller owns the returned Tensor
pub fn batchToTensor(gpa: std.mem.Allocator, batch: []const []const c_int) !Tensor {
    if (batch.len == 0) return error.EmptyBatch;
    const max_seq_len = batch[0].len;
    const flattened = try gpa.alloc(c_int, batch.len * max_seq_len);
    defer gpa.free(flattened);
    for (batch, 0..) |row, i| {
        if (row.len != max_seq_len) return error.InconsistentMaxSeqLen;
        @memcpy(flattened[i * max_seq_len .. i * max_seq_len + max_seq_len], row);
    }
    return try Tensor.fromSlice(c_int, flattened, &[_]c_int{ @intCast(batch.len), @intCast(max_seq_len) });
}

/// Renders each item in the batch into the chat template and tokenizes.
///
/// The caller owns the returned 2d slice of token ids.
pub fn applyChatTemplateAndTokenize(gpa: std.mem.Allocator, batch: []const Conversation, chat_template: ChatTemplate, tokenizer: Tokenizer) ![][]c_int {
    var encodings: [][]c_int = try gpa.alloc([]c_int, batch.len);
    errdefer {
        for (encodings) |encoding| {
            gpa.free(encoding);
        }
        gpa.free(encodings);
    }
    for (batch, 0..) |conversation, i| {
        const prompt = try chat_template.render(gpa, conversation.messages, .{});
        errdefer gpa.free(prompt);
        encodings[i] = try tokenizer.encode(gpa, prompt);
    }
    return encodings;
}

/// Removes the prefix that has already been cached and returns the uncached tokens.
///
/// Returns a view (not owned).
pub fn uncached_tokens(
    token_ids: []const c_int,
    conversation_id: []const u8,
    conversation_states: *std.StringHashMap(ConversationState),
) ![]const c_int {
    const conversation = conversation_states.getPtr(conversation_id) orelse return error.NoConversationState;

    // If a prefix matches then we can remove the cached tokens from the slice
    if (std.mem.cutPrefix(c_int, token_ids, conversation.cached_token_ids.items)) |new_tokens| {
        return new_tokens;
    }

    // No match, clear any conversation state
    conversation.reset();
    return token_ids;
}

/// Removes the prefix's for a batch that have already been cached and returns the uncached tokens.
///
/// Returns a slice (owned) where each element is a view into the batch's row (not owned).
pub fn uncached_tokens_batch(
    gpa: std.mem.Allocator,
    token_ids_batch: []const []const c_int,
    conversation_ids: []const []const u8,
    conversation_states: *std.StringHashMap(ConversationState),
) ![]const []const c_int {
    const uncached_tokens_batch_list: [][]const c_int = try gpa.alloc([]const c_int, token_ids_batch.len);
    errdefer gpa.free(uncached_tokens_batch_list);

    for (token_ids_batch, conversation_ids, 0..) |token_ids, conversation_id, i| {
        uncached_tokens_batch_list[i] = try uncached_tokens(token_ids, conversation_id, conversation_states);
    }

    return uncached_tokens_batch_list;
}

/// Decodes the token into a string and writes it into the queue.
///
/// The queue reader/recevier owns the string and is responsible for freeing it.
pub fn decodeAndEmitToken(gpa:std.mem.Allocator, io: std.Io, id: c_int, tokenizer: Tokenizer, output: *Queue([]const u8)) !void {
    const token = try tokenizer.decode(gpa, &[1]u32{@intCast(id)});
    errdefer gpa.free(token);
    _ = try output.put(io, &[1][]const u8{token}, 1);
}
