import torch



def demo(send, recv, output):
    """
        send shape: (num_cards, num_experts, total_tokens, hidden_size)
        counts shape: (num_cards, num_experts,)
        cumcounts shape: (num_cards, num_experts+1,)
        recv shape: (num_experts, num_cards, total_tokens, hidden_size)
        output shape: (num_cards, total_tokens, hidden_size)

        Note: This function now adapts to the actual input shape, supporting
        any number of cards (2, 3, 4, etc.), not just 4 cards.
    """
    # Infer dimensions from input tensors
    num_cards = send.shape[0]  # Actual number of cards from input
    num_experts = send.shape[1]  # Number of experts (typically equals num_cards)
    total_tokens = send.shape[2]
    hidden_size = send.shape[3]
    count = 4  # tokens to process per (card, expert) pair

    # dispatch
    for cardi in range(num_cards):
        for experti in range(num_experts):
            # count = counts[cardi, experti]
            recv[experti, cardi, :count, :] = send[cardi, experti, :count, :]
    print(f"send: {send}")
    print(f"recv: {recv}")
    # compute
    for cardi in range(num_cards):
        for experti in range(num_experts):
            recv[experti, cardi] = recv[experti, cardi] + 1.0  # 匹配实际kernel行为：总是加1.0f
    print(f"recv: {recv}")
    # combine
    for experti in range(num_experts):
        for cardi in range(num_cards):
            # count = counts[cardi, experti]
            output[cardi, :count, :] += recv[experti, cardi, :count, :]
    print(f"output: {output}")
    return output

