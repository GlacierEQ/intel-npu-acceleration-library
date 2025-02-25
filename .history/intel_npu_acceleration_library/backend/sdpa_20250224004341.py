# DEPRECATED: This historical backup is no longer maintained. Use /intel_npu_acceleration_library/backend/sdpa.py instead.
#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from intel_npu_acceleration_library.backend.factory import NNFactory
from typing import Tuple
import numpy as np


class SDPA(NNFactory):
    """Implementation of a ScaledDotProductAttention NPU operation."""

    def __init__(
        self,
        query_shapes: Tuple[int, int],
        key_shapes: Tuple[int, int],
        value_shapes: Tuple[int, int],
        mask_shapes: Tuple[int, int],
        is_causal: bool = False,
        profile: bool = False,
        device: str = "NPU",
    ):
        """Initialize the SDPA.

        Args:
            query_shapes (Tuple[int, int]): shape of the query tensor
            key_shapes (Tuple[int, int]): shape of the key tensor
            value_shapes (Tuple[int, int]): shape of the value tensor
            mask_shapes (Tuple[int, int]): shape of the mask tensor
            is_causal (bool, optional): If the SDPA mask is is_causal or not. Defaults to False.
            profile (bool, optional): Enable/Disable profiling. Defaults to False.
            device (str, optional): Target device, default to "NPU".
        """
        super().__init__(profile, device)

        self.query = self.parameter(query_shapes)
        self.key = self.parameter(key_shapes)
        self.value = self.parameter(value_shapes)
        self.mask = self.parameter(mask_shapes)

        _ = self.scaled_dot_product_attention(  # type: ignore[attr-defined]
            self.query, self.key, self.value, self.mask, is_causal
        )
        self.compile()

    def run(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Run the scaled dot product attention kernel.

        Args:
            query (np.ndarray): Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            key (np.ndarray): Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            value (np.ndarray): Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            mask (np.ndarray): Attention mask tensor of shape (batch_size, num_heads, seq_len, seq_len)

        Returns:
            np.ndarray: The attention output tensor of shape (batch_size, num_heads, seq_len, head_dim)

        Raises:
            ValueError: If input tensors have invalid shapes or types
        """
        # Input validation
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or mask.ndim != 4:
            raise ValueError("All input tensors must be 4-dimensional")
            
        if query.shape != self.query.shape:
            raise ValueError(f"Query tensor shape {query.shape} doesn't match expected shape {self.query.shape}")
        if key.shape != self.key.shape:
            raise ValueError(f"Key tensor shape {key.shape} doesn't match expected shape {self.key.shape}")
        if value.shape != self.value.shape:
            raise ValueError(f"Value tensor shape {value.shape} doesn't match expected shape {self.value.shape}")
        if mask.shape != self.mask.shape:
            raise ValueError(f"Mask tensor shape {mask.shape} doesn't match expected shape {self.mask.shape}")

        return super().run(query, key, value, mask)


class SimpleSDPA(NNFactory):
    """Implementation of a ScaledDotProductAttention NPU operation."""

    def __init__(
        self,
        query_shapes: Tuple[int, int],
        key_shapes: Tuple[int, int],
        value_shapes: Tuple[int, int],
        is_causal: bool = False,
        profile: bool = False,
        device: str = "NPU",
    ):
        """Initialize the SDPA.

        Args:
            query_shapes (Tuple[int, int]): shape of the query tensor
            key_shapes (Tuple[int, int]): shape of the key tensor
            value_shapes (Tuple[int, int]): shape of the value tensor
            is_causal (bool, optional): If the SDPA mask is is_causal or not. Defaults to False.
            profile (bool, optional): Enable/Disable profiling. Defaults to False.
            device (str, optional): Target device, default to "NPU".
        """
        super().__init__(profile, device)

        self.query = self.parameter(query_shapes)
        self.key = self.parameter(key_shapes)
        self.value = self.parameter(value_shapes)

        _ = self.scaled_dot_product_attention_simple(  # type: ignore[attr-defined]
            self.query, self.key, self.value, is_causal
        )
        self.compile()

    def run(self, query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """Run the scaled dot product attention kernel without mask.

        Args:
            query (np.ndarray): Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            key (np.ndarray): Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            value (np.ndarray): Value tensor of shape (batch_size, num_heads, seq_len, head_dim)

        Returns:
            np.ndarray: The attention output tensor of shape (batch_size, num_heads, seq_len, head_dim)

        Raises:
            ValueError: If input tensors have invalid shapes or types
        """
        # Input validation
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("All input tensors must be 4-dimensional")
            
        if query.shape != self.query.shape:
            raise ValueError(f"Query tensor shape {query.shape} doesn't match expected shape {self.query.shape}")
        if key.shape != self.key.shape:
            raise ValueError(f"Key tensor shape {key.shape} doesn't match expected shape {self.key.shape}")
        if value.shape != self.value.shape:
            raise ValueError(f"Value tensor shape {value.shape} doesn't match expected shape {self.value.shape}")

        return super().run(query, key, value)
