"""Causal layouts for differentiable shared-prefix recommendation training.

A query is stored as ``prefix, candidate_0, candidate_1, ...``. Candidate
tokens attend to their own causal branch and the shared prefix, never another
candidate. Positions restart for each query and each candidate starts at the
prefix length. The layout is usable during training: the prefix stays in the
same autograd graph as all of its candidates.

Dense masks are a small-input reference only. ``block_attention_mask`` builds
FlexAttention block indices directly from intervals; it deliberately does not
call ``create_block_mask``, which can allocate a token-by-token dense mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable, Iterable

import torch


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class QueryLayout:
    prefix_length: int
    branch_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        _positive_int(self.prefix_length, "prefix_length")
        lengths = tuple(self.branch_lengths)
        for index, length in enumerate(lengths):
            _positive_int(length, f"branch_lengths[{index}]")
        object.__setattr__(self, "branch_lengths", lengths)


@dataclass(frozen=True)
class PackedTreeLayout:
    """Half-open token ranges and masks for a padding-free packed batch."""

    queries: tuple[QueryLayout, ...]
    prefix_ranges: tuple[tuple[int, int], ...]
    branch_ranges: tuple[tuple[tuple[int, int], ...], ...]
    score_indices: tuple[tuple[int, ...], ...]
    position_ids: torch.Tensor
    token_query_ids: torch.Tensor
    token_branch_ids: torch.Tensor

    @classmethod
    def from_queries(cls, queries: Iterable[QueryLayout]) -> PackedTreeLayout:
        queries = tuple(queries)
        if not queries:
            raise ValueError("at least one query is required")
        prefixes, branches, scores = [], [], []
        positions, query_ids, branch_ids = [], [], []
        cursor = 0
        for query_index, query in enumerate(queries):
            if not isinstance(query, QueryLayout):
                raise TypeError("queries must contain QueryLayout instances")
            prefixes.append((cursor, cursor + query.prefix_length))
            positions.extend(range(query.prefix_length))
            query_ids.extend([query_index] * query.prefix_length)
            branch_ids.extend([-1] * query.prefix_length)
            cursor += query.prefix_length
            query_branches, query_scores = [], []
            for branch_index, length in enumerate(query.branch_lengths):
                query_branches.append((cursor, cursor + length))
                query_scores.append(cursor + length - 1)
                positions.extend(range(query.prefix_length, query.prefix_length + length))
                query_ids.extend([query_index] * length)
                branch_ids.extend([branch_index] * length)
                cursor += length
            branches.append(tuple(query_branches))
            scores.append(tuple(query_scores))
        return cls(
            queries, tuple(prefixes), tuple(branches), tuple(scores),
            torch.tensor([positions], dtype=torch.long),
            torch.tensor(query_ids, dtype=torch.long),
            torch.tensor(branch_ids, dtype=torch.long),
        )

    @property
    def token_count(self) -> int:
        return self.token_query_ids.numel()

    @property
    def query_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (prefix[0], branches[-1][1] if branches else prefix[1])
            for prefix, branches in zip(self.prefix_ranges, self.branch_ranges)
        )

    def causal_allowed(self, query_index: int, key_index: int) -> bool:
        """Inspect the exact token rule, without allocating a dense mask."""
        if not (0 <= query_index < self.token_count and 0 <= key_index < self.token_count):
            return False
        same_query = self.token_query_ids[query_index] == self.token_query_ids[key_index]
        same_branch = self.token_branch_ids[query_index] == self.token_branch_ids[key_index]
        prefix_key = self.token_branch_ids[key_index] == -1
        return bool(same_query & (prefix_key | same_branch) & (key_index <= query_index))

    def attention_mask_mod(self, device: torch.device | str = "cpu") -> Callable:
        """FlexAttention's vectorizable token rule, including padded tile bounds."""
        query_ids = self.token_query_ids.to(device)
        branch_ids = self.token_branch_ids.to(device)
        length = self.token_count

        def mask_mod(batch, head, q_idx, kv_idx):
            valid = (q_idx >= 0) & (kv_idx >= 0) & (q_idx < length) & (kv_idx < length)
            # FlexAttention may visit the tail of a partial block. Clamp before
            # indexing; validity still masks all out-of-range tokens.
            q = q_idx.clamp(0, length - 1)
            k = kv_idx.clamp(0, length - 1)
            return (
                valid
                & (query_ids[q] == query_ids[k])
                & ((branch_ids[k] == -1) | (branch_ids[q] == branch_ids[k]))
                & (kv_idx <= q_idx)
            )

        return mask_mod

    def dense_attention_mask(
        self, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Additive [1,1,L,L] reference mask, never an automatic fallback."""
        if not dtype.is_floating_point:
            raise ValueError("attention mask dtype must be floating point")
        indexes = torch.arange(self.token_count, device=device)
        allowed = self.attention_mask_mod(device)(None, None, indexes[:, None], indexes[None, :])
        return torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill(
            ~allowed, torch.finfo(dtype).min,
        )[None, None]

    def _sparse_block_rows(self, block_size: int) -> tuple[tuple[int, ...], ...]:
        """Enumerate intersecting blocks from prefix/branch intervals.

        Entries are conservatively partial blocks; ``mask_mod`` applies exact
        causality inside them. Memory is block-scale, never a dense L x L mask.
        """
        block_size = _positive_int(block_size, "block_size")
        count = (self.token_count + block_size - 1) // block_size
        rows = [set() for _ in range(count)]
        for prefix, branches in zip(self.prefix_ranges, self.branch_ranges):
            for segment, visible_prefix in [(prefix, None)] + [(branch, prefix) for branch in branches]:
                start, end = segment
                for q_block in range(start // block_size, (end - 1) // block_size + 1):
                    q_end = min(end, (q_block + 1) * block_size)
                    # Same-segment keys that can precede at least one query
                    # token, plus the query's entire shared prefix.
                    intervals = [(start, q_end)]
                    if visible_prefix is not None:
                        intervals.append(visible_prefix)
                    for k_start, k_end in intervals:
                        rows[q_block].update(range(k_start // block_size, (k_end - 1) // block_size + 1))
        return tuple(tuple(sorted(row)) for row in rows)

    def block_attention_mask(
        self, device: torch.device | str = "cpu", block_size: int = 128,
    ):
        """Build a sparse FlexAttention BlockMask directly from interval geometry.

        PyTorch without FlexAttention raises an explicit error. This API never
        substitutes quadratic dense attention for a production packed batch.
        The backend owns its block-scale reverse-index bookkeeping for backward.
        """
        try:
            from torch.nn.attention.flex_attention import BlockMask
        except ImportError as error:
            raise RuntimeError("packed training requires PyTorch FlexAttention") from error
        rows = self._sparse_block_rows(block_size)
        count = len(rows)
        # 2.6 infers sequence length from the index-table width. Later versions
        # accept exact seq_lengths, but a square block table is portable to both.
        indices = torch.zeros((1, 1, count, count), dtype=torch.int32)
        counts = torch.tensor([[[len(row) for row in rows]]], dtype=torch.int32)
        for index, row in enumerate(rows):
            indices[0, 0, index, :len(row)] = torch.tensor(row, dtype=torch.int32)
        kwargs = {
            "kv_num_blocks": counts.to(device),
            "kv_indices": indices.to(device),
            "BLOCK_SIZE": block_size,
            "mask_mod": self.attention_mask_mod(device),
        }
        if "seq_lengths" in inspect.signature(BlockMask.from_kv_blocks).parameters:
            kwargs["seq_lengths"] = (self.token_count, self.token_count)
        return BlockMask.from_kv_blocks(**kwargs)


@dataclass(frozen=True)
class PackedSegmentLayout:
    """Independent causal A inputs, concatenated without padding tokens."""

    tree: PackedTreeLayout

    @property
    def ranges(self) -> tuple[tuple[int, int], ...]:
        return self.tree.prefix_ranges

    @property
    def offsets(self) -> tuple[int, ...]:
        return tuple(start for start, _ in self.ranges) + (self.token_count,)

    @property
    def token_count(self) -> int:
        return self.tree.token_count

    @property
    def position_ids(self) -> torch.Tensor:
        return self.tree.position_ids

    @property
    def token_segment_ids(self) -> torch.Tensor:
        return self.tree.token_query_ids

    def causal_allowed(self, query_index: int, key_index: int) -> bool:
        return self.tree.causal_allowed(query_index, key_index)

    def attention_mask_mod(self, device: torch.device | str = "cpu") -> Callable:
        return self.tree.attention_mask_mod(device)

    def dense_attention_mask(self, dtype=torch.float32, device="cpu") -> torch.Tensor:
        return self.tree.dense_attention_mask(dtype, device)

    def block_attention_mask(self, device="cpu", block_size=128):
        return self.tree.block_attention_mask(device, block_size)


def packed_segments(lengths: Iterable[int]) -> PackedSegmentLayout:
    """Create reset RoPE positions and isolated causal attention for A texts."""
    return PackedSegmentLayout(PackedTreeLayout.from_queries(QueryLayout(length, ()) for length in lengths))
