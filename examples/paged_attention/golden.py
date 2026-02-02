"""
Paged Attention Golden Implementation

Implements the online softmax algorithm for paged attention computation.
Used as reference for validation of the simulation results.
"""

import numpy as np


def paged_attention_golden(
    query: np.ndarray,      # (batch, num_heads, head_dim)
    key_cache: np.ndarray,  # (total_blocks, block_size, kv_head_num, head_dim)
    value_cache: np.ndarray,  # (total_blocks, block_size, kv_head_num, head_dim)
    block_table: np.ndarray,  # (batch, block_num)
    context_lens: np.ndarray,  # (batch,)
    scale_value: float = 1.0,
) -> np.ndarray:
    """
    Paged attention with online softmax.
    
    Returns:
        out: (batch, num_heads, head_dim)
    """
    batch, num_heads, head_dim = query.shape
    _, block_size, kv_head_num, _ = key_cache.shape
    block_num = block_table.shape[1]
    
    # Number of query heads per KV head (for GQA)
    heads_per_kv = num_heads // kv_head_num
    
    out = np.zeros((batch, num_heads, head_dim), dtype=np.float32)
    
    for b_idx in range(batch):
        cur_seq = context_lens[b_idx]
        bn_this_batch = (cur_seq + block_size - 1) // block_size
        
        for h_idx in range(num_heads):
            # Get the corresponding KV head
            kv_h_idx = h_idx // heads_per_kv
            
            # Query for this batch and head
            qi = query[b_idx, h_idx, :]  # (head_dim,)
            
            # Initialize online softmax accumulators
            mi = -np.inf  # max
            li = 0.0      # sum
            oi = np.zeros(head_dim, dtype=np.float32)  # output
            
            for bn in range(bn_this_batch):
                cur_block_idx = block_table[b_idx, bn]
                
                # Get K and V for this block
                # key_cache: (total_blocks, block_size, kv_head_num, head_dim)
                kj = key_cache[cur_block_idx, :, kv_h_idx, :]  # (block_size, head_dim)
                vj = value_cache[cur_block_idx, :, kv_h_idx, :]  # (block_size, head_dim)
                
                # Handle last block with partial tokens
                if bn == bn_this_batch - 1:
                    valid_tokens = cur_seq - bn * block_size
                else:
                    valid_tokens = block_size
                
                # QK matmul: qi @ kj.T -> (block_size,)
                sij = qi @ kj.T  # (block_size,)
                
                # Apply scale
                sij_scale = sij * scale_value
                
                # Mask invalid positions
                if valid_tokens < block_size:
                    sij_scale[valid_tokens:] = -np.inf
                
                # Row max
                mij = np.max(sij_scale)
                
                # Exp and row sum
                pij = np.exp(sij_scale - mij)
                lij = np.sum(pij)
                
                # PV matmul: pij @ vj -> (head_dim,)
                oi_new = pij @ vj  # (head_dim,)
                
                # Online softmax update
                if bn == 0:
                    mi = mij
                    li = lij
                    oi = oi_new
                else:
                    mi_new = max(mi, mij)
                    alpha = np.exp(mi - mi_new)
                    beta = np.exp(mij - mi_new)
                    li = alpha * li + beta * lij
                    oi = alpha * oi + beta * oi_new
                    mi = mi_new
            
            # Final normalize
            oi = oi / li
            out[b_idx, h_idx, :] = oi
    
    return out


def generate_test_data(
    batch: int = 2,
    num_heads: int = 4,
    kv_head_num: int = 1,
    head_dim: int = 128,
    block_size: int = 64,
    block_num: int = 4,
    context_len: int = 256,
    seed: int = 42,
):
    """Generate test data for paged attention."""
    np.random.seed(seed)
    
    total_blocks = batch * block_num
    
    # Generate random tensors
    query = np.random.randn(batch, num_heads, head_dim).astype(np.float32) * 0.1
    key_cache = np.random.randn(total_blocks, block_size, kv_head_num, head_dim).astype(np.float32) * 0.1
    value_cache = np.random.randn(total_blocks, block_size, kv_head_num, head_dim).astype(np.float32) * 0.1
    
    # Block table: simple sequential mapping
    block_table = np.zeros((batch, block_num), dtype=np.int32)
    for b in range(batch):
        for bn in range(block_num):
            block_table[b, bn] = b * block_num + bn
    
    # Context lengths
    context_lens = np.full(batch, context_len, dtype=np.int32)
    
    scale_value = 1.0 / np.sqrt(head_dim)
    
    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "context_lens": context_lens,
        "scale_value": scale_value,
        "batch": batch,
        "num_heads": num_heads,
        "kv_head_num": kv_head_num,
        "head_dim": head_dim,
        "block_size": block_size,
        "block_num": block_num,
    }


if __name__ == "__main__":
    # Test the golden implementation
    data = generate_test_data()
    
    print("=== Paged Attention Golden Test ===")
    print(f"batch={data['batch']}, num_heads={data['num_heads']}, head_dim={data['head_dim']}")
    print(f"block_size={data['block_size']}, block_num={data['block_num']}")
    
    out = paged_attention_golden(
        data["query"],
        data["key_cache"],
        data["value_cache"],
        data["block_table"],
        data["context_lens"],
        data["scale_value"],
    )
    
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"Output mean: {out.mean():.4f}")
    print("Golden test passed!")
