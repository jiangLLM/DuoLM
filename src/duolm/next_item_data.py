"""Chronological recommendation queries, distinct from the rating protocol.

Metadata never edits a user's timeline. Test identifiers may be selected, but
this module deliberately has no API that loads or prepares test labels.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "books-next-item-v1"
SPLIT_NAMESPACE = "amz-split-20260927|"
CAP_NAMESPACE = "amz-cap-20260927|Books|"
SELECTOR_NAMESPACE = "books-next-item-dev-selector-v1|"
DEFAULT_CAPS = {"train": 50_000, "dev": 5_000, "test": 10_000}


def _identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty, unpadded string")
    return value


def _integer(value: object, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def user_split(user_id: str) -> str:
    """Preserve the original Books user boundary exactly."""
    _identity(user_id, "user_id")
    bucket = int(_digest(SPLIT_NAMESPACE + user_id), 16) % 10_000
    return "train" if bucket < 8_000 else "dev" if bucket < 9_000 else "test"


def _train_or_dev(split: str) -> None:
    if split not in ("train", "dev"):
        raise ValueError("only train/dev labels are allowed; test is sealed")


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    item_id: str
    rating: int
    timestamp: int

    def __post_init__(self) -> None:
        _identity(self.item_id, "history item_id")
        if _integer(self.rating, "history rating") > 5:
            raise ValueError("history rating must be 1 through 5")
        _integer(self.timestamp, "history timestamp")


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    item_id: str
    rating: int | None
    timestamp: int

    def __post_init__(self) -> None:
        _identity(self.item_id, "event item_id")
        _integer(self.timestamp, "event timestamp")
        if self.rating is not None and _integer(self.rating, "event rating") > 5:
            raise ValueError("event rating must be 1 through 5 or unknown")


@dataclass(frozen=True, slots=True)
class RecommendationContext:
    query_id: str
    user_id: str
    split: str
    history: tuple[HistoryEntry, ...]
    target_item_id: str
    target_timestamp: int
    seen_item_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecommendationQuery:
    query_id: str
    user_id: str
    split: str
    history: tuple[HistoryEntry, ...]
    candidate_ids: tuple[str, ...]
    positive_index: int | None
    target_item_id: str
    target_timestamp: int
    seen_item_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identity(self.query_id, "query_id")
        _identity(self.user_id, "user_id")
        _train_or_dev(self.split)
        if user_split(self.user_id) != self.split:
            raise ValueError("user does not belong to the declared original hash split")
        _identity(self.target_item_id, "target_item_id")
        _integer(self.target_timestamp, "target_timestamp")
        if not isinstance(self.history, tuple) or not 1 <= len(self.history) <= 10:
            raise ValueError("history must be a tuple with 1 through 10 entries")
        if not all(isinstance(entry, HistoryEntry) for entry in self.history):
            raise ValueError("history entries must be HistoryEntry instances")
        times = [entry.timestamp for entry in self.history]
        if times != sorted(set(times)) or times[-1] >= self.target_timestamp:
            raise ValueError("history timestamps must be strictly increasing before target")
        if not isinstance(self.seen_item_ids, tuple):
            raise ValueError("seen_item_ids must be a tuple of all past item identities")
        for item in self.seen_item_ids:
            _identity(item, "seen item_id")
        if len(set(self.seen_item_ids)) != len(self.seen_item_ids):
            raise ValueError("seen_item_ids must be unique")
        seen = set(self.seen_item_ids)
        if not {entry.item_id for entry in self.history} <= seen:
            raise ValueError("seen_item_ids must include every visible history item")
        if self.target_item_id in seen:
            raise ValueError("target must be a new item, absent from all past events")
        if not isinstance(self.candidate_ids, tuple) or len(self.candidate_ids) < 2:
            raise ValueError("a complete candidate group needs at least two items")
        for item in self.candidate_ids:
            _identity(item, "candidate item_id")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate identities must be unique")
        if seen.intersection(self.candidate_ids):
            raise ValueError("candidates must exclude all past items, not just visible history")
        if self.positive_index is None:
            if self.split == "train":
                raise ValueError("training groups require a positive_index")
            if self.target_item_id in self.candidate_ids:
                raise ValueError("a retrieval miss cannot contain the target")
        else:
            index = _integer(self.positive_index, "positive_index", 0)
            if index >= len(self.candidate_ids) or self.candidate_ids[index] != self.target_item_id:
                raise ValueError("positive_index must identify the target item")

    @property
    def retrieval_hit(self) -> bool:
        return self.positive_index is not None

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, **asdict(self),
                "history": [asdict(entry) for entry in self.history],
                "candidate_ids": list(self.candidate_ids), "seen_item_ids": list(self.seen_item_ids)}

    @classmethod
    def from_dict(cls, row: Mapping) -> RecommendationQuery:
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported recommendation query schema")
        required = set(cls.__dataclass_fields__) | {"schema_version"}
        if set(row) != required:
            raise ValueError(f"query fields must exactly match {sorted(required)}")
        _train_or_dev(row["split"])
        if user_split(row["user_id"]) != row["split"]:
            raise ValueError("user does not belong to the declared original hash split")
        if not isinstance(row["history"], list):
            raise ValueError("JSON history must be a list")
        for field in ("candidate_ids", "seen_item_ids"):
            if not isinstance(row[field], list):
                raise ValueError(f"JSON {field} must be a list")
        try:
            history = tuple(HistoryEntry(**entry) for entry in row["history"])
        except TypeError as error:
            raise ValueError("invalid history fields") from error
        return cls(**{key: value for key, value in row.items()
                     if key not in {"schema_version", "history", "candidate_ids", "seen_item_ids"}},
                   history=history, candidate_ids=tuple(row["candidate_ids"]),
                   seen_item_ids=tuple(row["seen_item_ids"]))


@dataclass(frozen=True, slots=True)
class Cohorts:
    train: tuple[str, ...]
    dev: tuple[str, ...]
    test: tuple[str, ...]
    checkpoint_selector: tuple[str, ...]


def select_cohorts(user_ids: Iterable[str], *, caps: Mapping[str, int] | None = None,
                   selector_count: int = 2_000) -> Cohorts:
    """Select IDs only; the test IDs do not authorize reading their records.

    `user_ids` must be the externally audited eligible identity pool. Insufficient
    availability is an error, never silently repaired by borrowing another split.
    """
    caps = dict(DEFAULT_CAPS if caps is None else caps)
    if set(caps) != {"train", "dev", "test"}:
        raise ValueError("caps must specify train, dev and test")
    for split, count in caps.items():
        _integer(count, f"{split} cap", 0)
    _integer(selector_count, "selector_count", 0)
    if selector_count > caps["dev"]:
        raise ValueError("checkpoint selector cannot exceed the dev cohort")
    pools = {split: [] for split in caps}
    for user in set(user_ids):
        pools[user_split(user)].append(user)
    selected = {}
    for split, pool in pools.items():
        pool.sort(key=lambda user: (_digest(CAP_NAMESPACE + split + "|" + user), user))
        if len(pool) < caps[split]:
            raise ValueError(f"{split} has {len(pool)} eligible identities, needs {caps[split]}")
        selected[split] = tuple(pool[:caps[split]])
    selector = tuple(sorted(selected["dev"],
                            key=lambda user: (_digest(SELECTOR_NAMESPACE + user), user))[:selector_count])
    return Cohorts(**selected, checkpoint_selector=selector)


def load_timeline_csv(path: str | Path, user_ids: Iterable[str], *, split: str
                      ) -> dict[str, tuple[TimelineEvent, ...]]:
    """Parse labels only after matching an explicitly authorized train/dev ID.

    Unknown ratings remain in the timeline. Missing identity or time prevents
    lawful adjacency and therefore fails the selected cohort preparation.
    """
    _train_or_dev(split)
    allowed = set(user_ids)
    if any(user_split(user) != split for user in allowed):
        raise ValueError("requested identities cross the original split boundary")
    rows: dict[str, list[TimelineEvent]] = {user: [] for user in sorted(allowed)}
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"user_id", "parent_asin", "rating", "timestamp"} <= set(reader.fieldnames or ()):
            raise ValueError("ratings CSV requires user_id,parent_asin,rating,timestamp")
        for line, row in enumerate(reader, 2):
            user = row["user_id"].strip()
            if user not in allowed:
                continue  # No rating/time conversion for sealed or unselected users.
            item = _identity(row["parent_asin"].strip(), "CSV parent_asin")
            try:
                timestamp = _integer(int(row["timestamp"]), "CSV timestamp")
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid selected timestamp at CSV line {line}") from error
            try:
                number = Decimal(row["rating"])
                rating = int(number) if number.is_finite() and number in (1, 2, 3, 4, 5) else None
            except (InvalidOperation, ValueError, TypeError):
                rating = None
            rows[user].append(TimelineEvent(item, rating, timestamp))
    return {user: tuple(events) for user, events in rows.items()}


def chronological_contexts(user_id: str, events: Sequence[TimelineEvent], *, split: str,
                            known_item_ids: Iterable[str], history_limit: int = 10
                            ) -> tuple[tuple[RecommendationContext, ...], dict[str, int]]:
    """Build immediate transitions before checking metadata; dev uses last only.

    Exact duplicate rows collapse. Conflicting or unknown ratings remain in
    chronology and invalidate a query when they occur in its visible history.
    Tied history timestamps are rejected rather than given arbitrary order.
    """
    _train_or_dev(split)
    if user_split(user_id) != split:
        raise ValueError("user does not belong to the requested split")
    if not 1 <= _integer(history_limit, "history_limit") <= 10:
        raise ValueError("history_limit must be 1 through 10")
    # A full Books catalog is shared by tens of thousands of users. Prepared
    # set/frozenset indexes must be reused rather than copied per user.
    known = known_item_ids if isinstance(known_item_ids, (set, frozenset)) else frozenset(known_item_ids)
    groups: dict[int, dict[str, set[int | None]]] = defaultdict(lambda: defaultdict(set))
    audit = Counter()
    for event in events:
        if not isinstance(event, TimelineEvent):
            raise ValueError("events must be TimelineEvent instances")
        ratings = groups[event.timestamp][event.item_id]
        if event.rating in ratings:
            audit["exact_duplicate_rows_collapsed"] += 1
        ratings.add(event.rating)
    ordered = [(time, sorted(items.items())) for time, items in sorted(groups.items())]
    past: list[tuple[int, str, set[int | None]]] = []
    seen: set[str] = set()
    result = []
    for position, (time, group) in enumerate(ordered):
        if position and (split == "train" or position == len(ordered) - 1):
            audit["positions_considered"] += 1
            recent = past[-history_limit:]
            reason = None
            if len(group) != 1:
                reason = "multiple_immediate_next_items"
            elif group[0][0] in seen:
                reason = "revisited_target"
            elif len(past) > history_limit and past[-history_limit - 1][0] == recent[0][0]:
                reason = "history_cut_through_time_group"
            elif len({row[0] for row in recent}) != len(recent):
                reason = "tied_history_timestamps"
            elif any(len(row[2]) != 1 or None in row[2] for row in recent):
                reason = "unknown_or_conflicting_history_rating"
            elif group[0][0] not in known:
                reason = "missing_target_metadata"
            elif any(row[1] not in known for row in recent):
                reason = "missing_history_metadata"
            if reason:
                audit[reason] += 1
            else:
                target = group[0][0]
                history = tuple(HistoryEntry(item, next(iter(ratings)), timestamp)
                                for timestamp, item, ratings in recent)
                query_id = _digest(f"{SCHEMA_VERSION}|{user_id}|{time}|{target}")
                result.append(RecommendationContext(query_id, user_id, split, history,
                                                    target, time, tuple(sorted(seen))))
                audit["legal_queries"] += 1
        past.extend((time, item, ratings) for item, ratings in group)
        seen.update(item for item, _ in group)
    audit["time_groups"] = len(ordered)
    return tuple(result), dict(audit)


def select_temporal_queries(contexts: Sequence[RecommendationContext], *, limit: int = 5
                            ) -> tuple[RecommendationContext, ...]:
    """Uniform temporal-quantile midpoints, without replacement or duplication."""
    _integer(limit, "query limit")
    ordered = tuple(sorted(contexts, key=lambda context: (context.target_timestamp, context.query_id)))
    if len({context.user_id for context in ordered}) > 1:
        raise ValueError("temporal selection must operate on one user at a time")
    if len({context.query_id for context in ordered}) != len(ordered):
        raise ValueError("context query IDs must be unique")
    if len(ordered) <= limit:
        return ordered
    indices = [((2 * index + 1) * len(ordered)) // (2 * limit) for index in range(limit)]
    return tuple(ordered[index] for index in indices)


class TrainingCandidateSampler:
    """Reusable frozen catalog index for positive + four random + three hard.

    Construct once per prepared catalog, then call ``sample`` for every query.
    Construction costs O(C log C), with C catalog items. Each query visits only
    its past items and retrieval input; it never scans or sorts the C-item
    catalog. Uniform random ranks are mapped past the excluded indices using
    binary search, so even an almost fully excluded catalog needs no retries.
    Future user events are unavailable to the sampler. Unobserved items are
    contrasts, not negative star-rating labels. Three hard contrasts are drawn
    uniformly without replacement from the legal, deduplicated Top-200 results.
    """

    __slots__ = ("_catalog_ids", "_item_index", "seed")

    def __init__(self, catalog_ids: Iterable[str], *, seed: int = 17) -> None:
        self.seed = _integer(seed, "candidate seed", 0)
        catalog = {_identity(item, "catalog item_id") for item in catalog_ids}
        self._catalog_ids = tuple(sorted(catalog))
        self._item_index = {item: index for index, item in enumerate(self._catalog_ids)}

    def sample(self, context: RecommendationContext, *,
               retrieved_ids: Sequence[str]) -> RecommendationQuery:
        if context.split != "train":
            raise ValueError("training_candidates requires a train context")
        if context.target_item_id not in self._item_index:
            raise ValueError("target is absent from the frozen candidate catalog")
        excluded = set(context.seen_item_ids) | {context.target_item_id}
        hard_pool, hard_seen = [], set()
        for item in retrieved_ids[:200]:
            _identity(item, "retrieved item_id")
            if item in self._item_index and item not in excluded and item not in hard_seen:
                hard_pool.append(item)
                hard_seen.add(item)
        if len(hard_pool) < 3:
            raise ValueError("retrieval Top-200 must supply three distinct legal hard negatives")
        rng = random.Random(int(_digest(
            f"{SCHEMA_VERSION}|candidates|{self.seed}|{context.query_id}"), 16))
        hard = rng.sample(hard_pool, 3)
        excluded.update(hard)
        indices = sorted(self._item_index[item] for item in excluded if item in self._item_index)
        available = len(self._catalog_ids) - len(indices)
        if available < 4:
            raise ValueError("catalog must supply four additional distinct random negatives")
        # If excluded indices are e_j, e_j-j is the first remaining-item rank
        # shifted by that exclusion. Consecutive exclusions have identical
        # ranks, so bisect_right includes the complete block at once.
        blocked_ranks = tuple(index - position for position, index in enumerate(indices))
        random_ids = [self._catalog_ids[rank + bisect_right(blocked_ranks, rank)]
                      for rank in rng.sample(range(available), 4)]
        candidates = [context.target_item_id, *random_ids, *hard]
        rng.shuffle(candidates)
        return _query(context, tuple(candidates))


def training_candidates(context: RecommendationContext, *, catalog_ids: Iterable[str],
                         retrieved_ids: Sequence[str], seed: int = 17) -> RecommendationQuery:
    """Compatibility helper; reuse TrainingCandidateSampler for bulk preparation."""
    return TrainingCandidateSampler(catalog_ids, seed=seed).sample(context, retrieved_ids=retrieved_ids)


def evaluation_candidates(context: RecommendationContext, retrieved_ids: Sequence[str], *,
                           count: int = 100) -> RecommendationQuery:
    """Keep retrieval misses; never insert the target into a Top-N evaluation."""
    if context.split != "dev":
        raise ValueError("evaluation_candidates allows dev only; test is sealed")
    _integer(count, "retrieval count", 2)
    seen = set(context.seen_item_ids)
    candidates = []
    for item in retrieved_ids:
        _identity(item, "retrieved item_id")
        if item not in seen and item not in candidates:
            candidates.append(item)
        if len(candidates) == count:
            break
    if len(candidates) != count:
        raise ValueError("retriever must supply the requested distinct unseen candidates")
    return _query(context, tuple(candidates))


def _query(context: RecommendationContext, candidates: tuple[str, ...]) -> RecommendationQuery:
    positive = candidates.index(context.target_item_id) if context.target_item_id in candidates else None
    fields = asdict(context)
    fields["history"] = context.history
    return RecommendationQuery(**fields, candidate_ids=candidates, positive_index=positive)


def write_queries(path: str | Path, queries: Iterable[RecommendationQuery]) -> None:
    """Write one complete candidate group per JSONL row, never individual pairs."""
    path = Path(path)
    seen = set()
    split = None
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            for query in queries:
                if not isinstance(query, RecommendationQuery):
                    raise ValueError("rows must be RecommendationQuery instances")
                if query.query_id in seen:
                    raise ValueError("duplicate query_id")
                if split is not None and query.split != split:
                    raise ValueError("one dataset cannot mix train and dev queries")
                split = query.split
                seen.add(query.query_id)
                stream.write(json.dumps(query.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            if not seen:
                raise ValueError("query dataset is empty")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_queries(path: str | Path, *, split: str, expected_sha256=None) -> tuple[RecommendationQuery, ...]:
    _train_or_dev(split)  # Reject test before opening any file.
    result = []
    seen = set()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for line, text in enumerate(stream, 1):
            digest.update(text)
            if not text.strip():
                raise ValueError(f"empty recommendation row at line {line}")
            query = RecommendationQuery.from_dict(json.loads(text))
            if query.split != split:
                raise ValueError(f"declared split differs from requested split at line {line}")
            if query.query_id in seen:
                raise ValueError(f"duplicate query_id at line {line}")
            seen.add(query.query_id)
            result.append(query)
    if not result:
        raise ValueError("query dataset is empty")
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise ValueError("parsed query bytes differ from frozen input hash")
    return tuple(result)
