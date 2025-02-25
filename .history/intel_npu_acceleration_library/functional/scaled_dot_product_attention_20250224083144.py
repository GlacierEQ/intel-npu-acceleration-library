#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from intel_npu_acceleration_library.backend import run_factory, SDPA, SimpleSDPA
from typing import Optional
from functools import partial
import torch
import logging

logger = logging.getLogger(__name__)

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Execute Scaled Dot Product Attention (SDPA) kernel on NPU.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        attn_mask (Optional[torch.Tensor]): Attention mask tensor of shape (batch_size, num_heads, seq_len, seq_len).
            Defaults to None.
        dropout_p (float): Dropout probability. Currently must be 0.0 as dropout is not supported yet.
            Defaults to 0.0.
        is_causal (bool): If True, applies a causal mask. Defaults to False.
        scale (Optional[float]): Custom scaling factor. Currently must be None as custom scaling is not supported yet.
            Defaults to None.

    Raises:
        ValueError: If input tensors have invalid shapes or types
        NotImplementedError: If dropout_p != 0 or scale is not None, as these features are not yet implemented

    Returns:
        torch.Tensor: The attention output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    # Input validation
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        logger.error("Invalid tensor dimensions: query=%s, key=%s, value=%s", query.shape, key.shape, value.shape)
        raise ValueError("query, key, and value must be 4D tensors")
    if attn_mask is not None and attn_mask.ndim != 4:
        logger.error("Invalid attn_mask dimension: %s", attn_mask.shape)
        raise ValueError("attn_mask must be a 4D tensor or None")
    
    if dropout_p != 0:
        raise NotImplementedError("Dropout is not yet supported. Please use dropout_p=0.0")
    if scale is not None:
        raise NotImplementedError("Custom scaling is not yet supported. Please use scale=None")

    if attn_mask is None:
        backend_cls = partial(SimpleSDPA, is_causal=is_causal)  # type: ignore
        return run_factory([query, key, value], [], backend_cls)
    else:
        backend_cls = partial(SDPA, is_causal=is_causal)  # type: ignore
        return run_factory([query, key, value, attn_mask], [], backend_cls)
