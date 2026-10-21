# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatial sharding of WinoVae: one rank per slice of the frame along the height or the width.

Every buffer a spatial 3x3 convolution reads carries one halo line on each side of the split
axis (:class:`FrameGeometry`): the norm kernels write their slabs and frame-buffer rows into
that layout, the Kernel Factory convolutions return their products in it, and only the border
lines travel between neighbouring ranks. Edge ranks keep zeros in their outer halo (the global
zero padding of the convolution).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .jit_norm import FrameGeometry


def _align_up(n: int, align: int) -> int:
    return -(-n // align) * align


class Shard:
    """This rank's place in the spatial split and the exchanges it needs.

    ``axis``: "h" (ranks own consecutive row bands) or "w" (column bands). ``pg`` is the
    process group of the VAE ranks (the attention gather); ``adj_groups[i]`` is the group of
    ranks ``i`` and ``i + 1``, through which neighbours trade border lines with batched
    point-to-point sends (measured 10% faster end to end than two-rank all-gathers: a 720p
    decode exchanges ~2900 lines).
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        axis: str,
        pg: dist.ProcessGroup,
        adj_groups: list[dist.ProcessGroup | None],
    ) -> None:
        if axis not in ("h", "w"):
            raise ValueError("axis must be 'h' or 'w'")
        if not 0 <= rank < world_size:
            raise ValueError("rank out of range")
        if len(adj_groups) != world_size - 1:
            raise ValueError("one adjacent group per neighbouring pair of ranks")
        self.rank = rank
        self.world_size = world_size
        self.axis = axis
        self.pg = pg
        self.left_group = adj_groups[rank - 1] if rank > 0 else None
        self.right_group = adj_groups[rank] if rank + 1 < world_size else None
        if (rank > 0 and self.left_group is None) or (
            rank + 1 < world_size and self.right_group is None
        ):
            raise ValueError("missing adjacent process group")
        # Global rank ids of the neighbours (point-to-point peers are global ranks).
        self.left = dist.get_global_rank(self.left_group, 0) if rank > 0 else None
        self.right = dist.get_global_rank(self.right_group, 1) if rank + 1 < world_size else None
        # Communication buffers live as long as the shard: a temporary handed to NCCL and
        # dropped right after is a race inside a captured CUDA graph (the allocator may hand
        # its block to a later allocation with no graph edge to the pending transfer).
        self._buffers: dict[tuple, torch.Tensor] = {}

    @property
    def first(self) -> bool:
        return self.rank == 0

    @property
    def last(self) -> bool:
        return self.rank == self.world_size - 1

    # ---- geometry -----------------------------------------------------------------------

    def geometry(self, height: int, width: int, align: int = 1) -> FrameGeometry:
        """The physical frame for a ``height x width`` slice: one halo line on each side of
        the split axis, the physical extent rounded up to ``align`` lines (kernel tiles)."""
        if self.axis == "h":
            return FrameGeometry(height, width, _align_up(height + 2, align), width, 1, 0)
        return FrameGeometry(height, width, height, _align_up(width + 2, align), 0, 1)

    def conv_padding(self, padding: tuple[int, ...]) -> tuple[int, ...]:
        """``padding`` with the split axis set to zero (the halo supplies that context);
        the last two entries are (height, width)."""
        pad = list(padding)
        pad[-2 if self.axis == "h" else -1] = 0
        return tuple(pad)

    # ---- the exchanges ------------------------------------------------------------------

    def _line(self, phys: torch.Tensor, index: int) -> torch.Tensor:
        """Line ``index`` along the split axis of ``phys [..., rows, cols, C]``."""
        return phys[..., index, :, :] if self.axis == "h" else phys[..., :, index, :]

    def _buffer(self, role: str, like: torch.Tensor) -> torch.Tensor:
        """A persistent contiguous buffer of ``like``'s shape and dtype for ``role``."""
        key = (role, tuple(like.shape), like.dtype, like.device)
        buf = self._buffers.get(key)
        if buf is None:
            buf = torch.empty(like.shape, dtype=like.dtype, device=like.device)
            self._buffers[key] = buf
        return buf

    def _staged(self, role: str, line: torch.Tensor | None) -> torch.Tensor | None:
        """``line`` copied into the persistent send buffer of ``role``."""
        if line is None:
            return None
        buf = self._buffer(role, line)
        buf.copy_(line)
        return buf

    def _swap(
        self, send_left: torch.Tensor | None, send_right: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Send ``send_left`` to the left neighbour and ``send_right`` to the right one;
        returns what they sent back (same shapes), None where there is no neighbour. One
        batched send/receive for both neighbours on the world group (a batch must use a
        single group; the peers are addressed by global rank)."""
        ops = []
        recv_left = recv_right = None
        if self.left is not None:
            assert send_left is not None
            recv_left = self._buffer("recv_left", send_left)
            ops.append(dist.P2POp(dist.isend, send_left, self.left, group=self.pg))
            ops.append(dist.P2POp(dist.irecv, recv_left, self.left, group=self.pg))
        if self.right is not None:
            assert send_right is not None
            recv_right = self._buffer("recv_right", send_right)
            ops.append(dist.P2POp(dist.isend, send_right, self.right, group=self.pg))
            ops.append(dist.P2POp(dist.irecv, recv_right, self.right, group=self.pg))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        return recv_left, recv_right

    def exchange(self, phys: torch.Tensor, geo: FrameGeometry) -> None:
        """Fill the halo lines of ``phys [..., rows, cols, C]`` (interior at ``geo``) with the
        neighbours' border lines. Edge ranks keep their outer halo as it is (zeros)."""
        lo = geo.row0 if self.axis == "h" else geo.col0
        n = geo.height if self.axis == "h" else geo.width
        send_left = self._staged(
            "send_left", self._line(phys, lo) if self.left is not None else None
        )
        send_right = self._staged(
            "send_right", self._line(phys, lo + n - 1) if self.right is not None else None
        )
        recv_left, recv_right = self._swap(send_left, send_right)
        if recv_left is not None:
            self._line(phys, lo - 1).copy_(recv_left)
        if recv_right is not None:
            self._line(phys, lo + n).copy_(recv_right)

    def zero_outside(self, phys: torch.Tensor, geo: FrameGeometry) -> None:
        """Zero every line outside the logical extent (the outer halo of the edge ranks and
        the alignment lines)."""
        if self.axis == "h":
            phys[..., : geo.row0, :, :].zero_()
            phys[..., geo.row0 + geo.height :, :, :].zero_()
        else:
            phys[..., :, : geo.col0, :].zero_()
            phys[..., :, geo.col0 + geo.width :, :].zero_()

    def pad_frames(self, x: torch.Tensor) -> torch.Tensor:
        """A logical channels-last ``[N, C, (T,) H, W]`` tensor with one exchanged halo line
        on each side of the split axis (a copy; the direct convolution paths)."""
        spatial = list(x.shape[-2:])
        spatial[0 if self.axis == "h" else 1] += 2
        shape = (*x.shape[:-2], *spatial)
        mf = torch.channels_last_3d if x.dim() == 5 else torch.channels_last
        out = torch.empty(shape, dtype=x.dtype, device=x.device, memory_format=mf).zero_()
        h, w = x.shape[-2:]
        if self.axis == "h":
            out[..., 1 : h + 1, :].copy_(x)
        else:
            out[..., :, 1 : w + 1].copy_(x)
        geo = self.geometry(h, w)
        phys = out.permute(0, 2, 3, 4, 1) if x.dim() == 5 else out.permute(0, 2, 3, 1)
        self.exchange(phys, geo)
        return out

    def pad_for_downsample(self, x: torch.Tensor) -> torch.Tensor:
        """The downsampler's ``ZeroPad2d((0, 1, 0, 1))`` under sharding: ``x [N, C, H, W]``
        channels-last -> ``[N, C, H+1, W+1]`` where the appended line on the split axis is the
        right neighbour's first line (zeros on the last rank)."""
        out = F.pad(x, (0, 1, 0, 1)).contiguous(memory_format=torch.channels_last)
        self.fill_pad_line(out)
        return out

    def fill_pad_line(self, padded: torch.Tensor) -> None:
        """``padded [N, C, H+1, W+1]`` channels-last with zero pad lines: overwrite the pad
        line on the split axis with the right neighbour's first line (last rank: kept zero)."""
        phys = padded.permute(0, 2, 3, 1)
        rows, cols = phys.shape[1], phys.shape[2]
        n = (rows if self.axis == "h" else cols) - 1
        first = self._line(phys, 0)
        # The right neighbour needs nothing from us; it sends its first line.
        recv_left, recv_right = self._swap(
            self._staged("send_left", first if self.left is not None else None),
            self._staged("send_right", first if self.right is not None else None),
        )
        del recv_left
        if recv_right is not None:
            self._line(phys, n).copy_(recv_right)

    def gather_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """All ranks' ``rows [N, C]`` concatenated (rank order; the attention keys and values
        do not depend on the order)."""
        staged = self._staged("gather_in", rows)
        out = self._buffer("gather_out", rows.new_empty((self.world_size, *rows.shape)))
        dist.all_gather(list(out.unbind(0)), staged, group=self.pg)
        return out.view(-1, rows.shape[-1])
