# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stand-ins for the vLLM / vLLM-Omni internals the vendored spatial-shard decode imports."""

import logging
from typing import Callable

import torch
from torch import nn

DecodedChunkConsumer = Callable[[torch.Tensor], None]


def init_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _persistent_input_buffer(module: nn.Module, shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    """One zero-initialised input buffer per conv, reallocated only when the shape, dtype or device changes
    (verbatim from vllm_omni/diffusion/distributed/autoencoders/wan_decoder_utils.py)."""
    memory_format = getattr(module, "input_memory_format", torch.contiguous_format)
    buf = module._input_buf
    if (
        buf is None
        or buf.shape != shape
        or buf.dtype != reference.dtype
        or buf.device != reference.device
        or not buf.is_contiguous(memory_format=memory_format)
    ):
        buf = torch.empty(shape, dtype=reference.dtype, device=reference.device, memory_format=memory_format).zero_()
        module._input_buf = buf
    return buf
