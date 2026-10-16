# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stand-ins for the two vLLM modules the vendored fast path imports."""

import logging

import triton
import triton.language as tl
from triton.language.extra import libdevice as tldevice

HAS_TRITON = True


def init_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["HAS_TRITON", "init_logger", "tl", "tldevice", "triton"]
