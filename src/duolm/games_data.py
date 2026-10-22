"""Versioned Games chronology and time-legal fixed-candidate data.

The Books split, cap ordering, visible-history and temporal-quantile rules are
preserved. Games alone supports sealed test labels; the normal loader enumerates
only the five development files and the shared first-observation sidecar.
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
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

from .next_item_data import (
    CAP_NAMESPACE, DEFAULT_CAPS, SELECTOR_NAMESPACE, SPLIT_NAMESPACE,
    HistoryEntry, RecommendationContext, TimelineEvent, _identity, _integer,
    select_temporal_queries, user_split,
)

PROTOCOL = "games-candidate-ranking-v1"
PREPARATION_SCHEMA = "games-candidate-ranking-preparation-v1"
FIRST_SEEN_SOURCE = "amazon-reviews-2023-video-games-raw-0core"
TIME_FILTER = "first-observed<=cutoff"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _split(split):
    if split not in ("train", "dev", "test"):
        raise ValueError("Games split must be train/dev/test")


@dataclass(frozen=True, slots=True)
class GamesQuery:
    """Duck-typed RecommendationQuery with an independent test-aware schema."""
    query_id: str
    user_id: str
    split: str
    history: tuple[HistoryEntry, ...]
    candidate_ids: tuple[str, ...]
    positive_index: int
    target_item_id: str
    target_timestamp: int
    seen_item_ids: tuple[str, ...]

    def __post_init__(self):
        for name in ("query_id", "user_id", "target_item_id"):
            _identity(getattr(self, name), name)
        _split(self.split)
        if user_split(self.user_id) != self.split:
            raise ValueError("user does not belong to the original hash split")
        _integer(self.target_timestamp, "target timestamp")
        if not isinstance(self.history, tuple) or not 1 <= len(self.history) <= 10:
            raise ValueError("history must contain 1 through 10 entries")
        if not all(isinstance(entry, HistoryEntry) for entry in self.history):
            raise ValueError("history entries must be HistoryEntry instances")
        times = [entry.timestamp for entry in self.history]
        if times != sorted(set(times)) or times[-1] >= self.target_timestamp:
            raise ValueError("history timestamps must strictly increase before target")
        if not isinstance(self.seen_item_ids, tuple):
            raise ValueError("seen_item_ids must be a tuple")
        for item in self.seen_item_ids:
            _identity(item, "seen item")
        seen = set(self.seen_item_ids)
        if len(seen) != len(self.seen_item_ids) or not {h.item_id for h in self.history} <= seen:
            raise ValueError("seen identities must be unique and include all visible history")
        if self.target_item_id in seen:
            raise ValueError("target must be absent from all past events")
        if not isinstance(self.candidate_ids, tuple) or len(self.candidate_ids) < 2:
            raise ValueError("candidate group must contain at least two items")
        for item in self.candidate_ids:
            _identity(item, "candidate item")
        if len(set(self.candidate_ids)) != len(self.candidate_ids) or seen.intersection(self.candidate_ids):
            raise ValueError("candidates must be unique and exclude all past items")
        index = _integer(self.positive_index, "positive_index", 0)
        if index >= len(self.candidate_ids) or self.candidate_ids[index] != self.target_item_id:
            raise ValueError("positive_index must identify the included target")

    @property
    def retrieval_hit(self):
        return True

    def to_dict(self):
        return {"schema_version": PROTOCOL, **asdict(self),
                "history": [asdict(h) for h in self.history],
                "candidate_ids": list(self.candidate_ids), "seen_item_ids": list(self.seen_item_ids)}

    @classmethod
    def from_dict(cls, row):
        if row.get("schema_version") != PROTOCOL or set(row) != set(cls.__dataclass_fields__) | {"schema_version"}:
            raise ValueError("unsupported Games query schema/fields")
        if not all(isinstance(row[name], list) for name in ("history", "candidate_ids", "seen_item_ids")):
            raise ValueError("JSON history/candidate/seen fields must be lists")
        try:
            return cls(**{key: value for key, value in row.items()
                          if key not in {"schema_version", "history", "candidate_ids", "seen_item_ids"}},
                       history=tuple(HistoryEntry(**entry) for entry in row["history"]),
                       candidate_ids=tuple(row["candidate_ids"]), seen_item_ids=tuple(row["seen_item_ids"]))
        except TypeError as error:
            raise ValueError("invalid Games query fields") from error


def chronological_contexts(user_id, events, *, split, known_item_ids, history_limit=10):
    """Books chronology verbatim, with Games identities and a real test split.

    Metadata checks occur after taking recent original events. Evaluation only
    considers the final original time group; an invalid target never falls back.
    """
    _split(split)
    if user_split(user_id) != split:
        raise ValueError("user does not belong to the requested original split")
    if not 1 <= _integer(history_limit, "history_limit") <= 10:
        raise ValueError("history_limit must be 1 through 10")
    known = known_item_ids if isinstance(known_item_ids, (set, frozenset)) else frozenset(known_item_ids)
    groups = defaultdict(lambda: defaultdict(set))
    audit = Counter()
    for event in events:
        if not isinstance(event, TimelineEvent):
            raise ValueError("events must be TimelineEvent instances")
        ratings = groups[event.timestamp][event.item_id]
        if event.rating in ratings:
            audit["exact_duplicate_rows_collapsed"] += 1
        ratings.add(event.rating)
    ordered = [(time, sorted(items.items())) for time, items in sorted(groups.items())]
    past, seen, result = [], set(), []
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
                query_id = hashlib.sha256(f"{PROTOCOL}|{user_id}|{time}|{target}".encode()).hexdigest()
                result.append(RecommendationContext(query_id, user_id, split, history,
                                                    target, time, tuple(sorted(seen))))
                audit["legal_queries"] += 1
        past.extend((time, item, ratings) for item, ratings in group)
        seen.update(item for item, _ in group)
    audit["time_groups"] = len(ordered)
    return tuple(result), dict(audit)


def select_cohorts(user_ids, *, caps=None, selector_count=2000):
    """Original hash ordering with upper bounds; never borrow another split."""
    caps = dict(DEFAULT_CAPS if caps is None else caps)
    if set(caps) != {"train", "dev", "test"}:
        raise ValueError("caps must specify all three splits")
    _integer(selector_count, "selector_count", 0)
    pools = {split: [] for split in caps}
    for user in set(user_ids):
        pools[user_split(user)].append(user)
    result = {}
    for split, cap in caps.items():
        _integer(cap, "cohort cap", 0)
        result[split] = sorted(pools[split], key=lambda user: (
            hashlib.sha256((CAP_NAMESPACE + split + "|" + user).encode()).hexdigest(), user))[:cap]
    result["checkpoint_selector"] = sorted(result["dev"], key=lambda user: (
        hashlib.sha256((SELECTOR_NAMESPACE + user).encode()).hexdigest(), user))[:selector_count]
    return result


def load_timelines(stream, user_ids=None):
    """Read original CSV events; bad identity/time invalidates a whole user."""
    allowed = set(user_ids) if user_ids is not None else None
    rows, rejected, counts = defaultdict(list), {}, Counter()
    reader = csv.DictReader(stream)
    if not {"user_id", "parent_asin", "rating", "timestamp"} <= set(reader.fieldnames or ()):
        raise ValueError("ratings CSV requires user_id,parent_asin,rating,timestamp")
    for row in reader:
        counts["csv_rows"] += 1
        user = (row.get("user_id") or "").strip()
        if not user:
            raise ValueError("CSV contains an unknown user identity")
        if allowed is not None and user not in allowed:
            continue
        counts["selected_rows"] += 1
        item = (row.get("parent_asin") or "").strip()
        try:
            _identity(item, "parent_asin")
            timestamp = _integer(int(row["timestamp"]), "timestamp")
        except (ValueError, TypeError):
            rejected[user] = "unknown_identity_or_timestamp"
            continue
        try:
            number = Decimal(row["rating"])
            rating = int(number) if number.is_finite() and number in (1, 2, 3, 4, 5) else None
        except (ValueError, TypeError, InvalidOperation):
            rating = None
        rows[user].append(TimelineEvent(item, rating, timestamp))
    return dict(rows), rejected, dict(counts)


def history_hash(context, items):
    """Content identity excludes the target and includes its observation cutoff."""
    cutoff = context.target_timestamp if hasattr(context, "target_timestamp") else context["cutoff_timestamp"]
    history = context.history if hasattr(context, "history") else tuple(HistoryEntry(**h) for h in context["history"])
    seen = context.seen_item_ids if hasattr(context, "seen_item_ids") else context["seen_item_ids"]
    return digest({"protocol": PROTOCOL, "cutoff_timestamp": cutoff,
                   "history": [(h.item_id, h.rating, h.timestamp, items[h.item_id]) for h in history],
                   "seen_item_ids": list(seen)})


def first_seen_times(records):
    if not isinstance(records, dict) or not records:
        raise ValueError("first-observation sidecar must be a nonempty mapping")
    result = {}
    for item, row in records.items():
        _identity(item, "first-observed item")
        if not isinstance(row, dict) or set(row) != {"timestamp", "source"} or row["source"] != FIRST_SEEN_SOURCE:
            raise ValueError("first observations must identify raw 0-core provenance")
        result[item] = _integer(row["timestamp"], "first-observed timestamp")
    return result


class CandidateUnavailable(ValueError):
    """The fixed original target cannot form the declared candidate group."""


class TimeAwareSampler:
    """One (first observation, ID) index; uniform allowed ranks per cutoff.

    Build O(C log C) once. Per query O(P log P + K log P), with P all-past
    identities and K sampled negatives. No per-query catalog scans or retries.
    """
    def __init__(self, first_seen, *, seed=17):
        self.seed = _integer(seed, "candidate seed", 0)
        times = first_seen_times(first_seen)
        ordered = sorted(times, key=lambda item: (times[item], item))
        self.catalog = tuple(ordered)
        self.timestamps = tuple(times[item] for item in ordered)
        self.indices = {item: index for index, item in enumerate(ordered)}

    def _rng(self, context, purpose):
        return random.Random(int(digest([PROTOCOL, self.seed, context.query_id, purpose]), 16))

    def _allowed(self, context, extra=()):
        cutoff = bisect_right(self.timestamps, context.target_timestamp)
        target = self.indices.get(context.target_item_id)
        if target is None or target >= cutoff:
            raise CandidateUnavailable("target_not_first_observed_by_cutoff")
        excluded = set(context.seen_item_ids) | {context.target_item_id} | set(extra)
        blocked = sorted(self.indices[item] for item in excluded
                         if item in self.indices and self.indices[item] < cutoff)
        return cutoff, blocked

    def available_count(self, context):
        cutoff, blocked = self._allowed(context)
        return cutoff - len(blocked)

    def _uniform(self, context, count, *, extra=(), purpose="random"):
        cutoff, blocked = self._allowed(context, extra)
        available = cutoff - len(blocked)
        if available < count:
            raise CandidateUnavailable(f"insufficient_time_legal_pool_need_{count}_have_{available}")
        blocked_ranks = tuple(index - position for position, index in enumerate(blocked))
        ranks = self._rng(context, purpose).sample(range(available), count)
        return [self.catalog[rank + bisect_right(blocked_ranks, rank)] for rank in ranks]

    def _query(self, context, negatives):
        candidates = sorted([context.target_item_id, *negatives])
        self._rng(context, "order").shuffle(candidates)
        return GamesQuery(**{key: value for key, value in asdict(context).items() if key != "history"},
                          history=context.history,
                          candidate_ids=tuple(candidates), positive_index=candidates.index(context.target_item_id))

    def random_candidates(self, context, *, count=100):
        _integer(count, "candidate count", 2)
        return self._query(context, self._uniform(context, count - 1, purpose="random-evaluation"))

    def mixed_candidates(self, context, *, retrieved_ids):
        if context.split != "train":
            raise ValueError("mixed8 requires an original train context")
        cutoff, _ = self._allowed(context)
        excluded, pool, seen = set(context.seen_item_ids) | {context.target_item_id}, [], set()
        for item in retrieved_ids[:200]:
            _identity(item, "retrieved item")
            index = self.indices.get(item)
            if index is not None and index < cutoff and item not in excluded and item not in seen:
                pool.append(item)
                seen.add(item)
        if len(pool) < 3:
            raise CandidateUnavailable("insufficient_legal_history_top200_hard_pool")
        hard = self._rng(context, "hard").sample(pool, 3)
        random_items = self._uniform(context, 4, extra=hard, purpose="random-training")
        return self._query(context, [*random_items, *hard])


def write_queries(path, queries):
    with Path(path).open("w", encoding="utf-8") as stream:
        seen = set()
        for query in queries:
            if not isinstance(query, GamesQuery) or query.query_id in seen:
                raise ValueError("Games queries must be valid and have unique IDs")
            seen.add(query.query_id)
            stream.write(json.dumps(query.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _read_json(path, expected=None):
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None and sha != expected:
        raise ValueError(f"frozen file checksum mismatch: {Path(path).name}")
    return json.loads(raw), sha


def read_queries(path, *, split, expected_sha256=None):
    _split(split)
    result, seen, hasher = [], set(), hashlib.sha256()
    with Path(path).open("rb") as stream:
        for line in stream:
            hasher.update(line)
            query = GamesQuery.from_dict(json.loads(line))
            if query.split != split or query.query_id in seen:
                raise ValueError("Games query split/unique identity mismatch")
            seen.add(query.query_id)
            result.append(query)
    if expected_sha256 is not None and hasher.hexdigest() != expected_sha256:
        raise ValueError("frozen Games query checksum mismatch")
    return tuple(result)


def manifest_identity(manifest):
    return digest({key: value for key, value in manifest.items() if key != "identity_sha256"})


def _validate_manifest(manifest):
    if (manifest.get("schema") != PREPARATION_SCHEMA or manifest.get("protocol") != PROTOCOL
            or manifest.get("identity_sha256") != manifest_identity(manifest)
            or manifest.get("test_labels_sealed") is not True
            or manifest.get("time_filter") != TIME_FILTER
            or manifest.get("train_policy") != "mixed-history"
            or manifest.get("seed") != 17 or manifest.get("query_cap") != 5
            or (manifest.get("train_candidates"), manifest.get("dev_candidates"), manifest.get("test_candidates")) != (8, 100, 100)):
        raise ValueError("invalid frozen Games preparation/protocol identity")
    files = manifest.get("files")
    expected_files = {"train.jsonl", "dev.jsonl", "sealed-test.jsonl", "catalog.json", "first-seen.json", "cohorts.json"}
    valid_sha = lambda value: isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    if not isinstance(files, dict) or set(files) != expected_files or not all(valid_sha(sha) for sha in files.values()):
        raise ValueError("Games preparation must bind exactly its six frozen data files")
    sources = manifest.get("sources")
    if (not isinstance(sources, dict) or not {"raw0core_reviews_sha256", "ratings_sha256", "metadata_sha256", "catalog_sha256"} <= sources.keys()
            or not all(valid_sha(sha) for sha in sources.values())):
        raise ValueError("Games preparation requires raw0core/5core/metadata/catalog source hashes")
    if not all(valid_sha(manifest.get(key)) for key in ("context_manifest_sha256", "retrieval_manifest_sha256")):
        raise ValueError("Games preparation requires frozen context/retrieval identities")
    if manifest.get("timestamp_audit", {}).get("unit") != "unix-milliseconds":
        raise ValueError("Games requires audited Unix millisecond sources with no implicit conversion")
    if manifest.get("selector_namespace") != SELECTOR_NAMESPACE:
        raise ValueError("Games selector namespace differs from Books")
    truncation = manifest.get("catalog_truncation", {})
    from .prepare_next_item import TOKENIZER_REPO, TOKENIZER_REVISION
    from .build_next_item_assets import ENCODER_REPO, ENCODER_REVISION
    for name, repo, revision in (("b_tokenizer", TOKENIZER_REPO, TOKENIZER_REVISION),
                                 ("a_tokenizer", ENCODER_REPO, ENCODER_REVISION)):
        identity = truncation.get(name, {})
        if identity.get("repo") != repo or identity.get("revision") != revision:
            raise ValueError("Games catalog requires the frozen Books tokenizer revisions")
    if truncation.get("max_item_tokens") != 512:
        raise ValueError("Games catalog requires the Books 512-token budget")
    retrieval = manifest.get("retrieval", {})
    files = manifest.get("files", {})
    if (retrieval.get("schema") != "games-next-item-retrieval-v1"
            or retrieval.get("history_only") is not True or retrieval.get("time_filter") != TIME_FILTER
            or retrieval.get("catalog_sha256") != files.get("catalog.json")
            or retrieval.get("first_seen_sha256") != files.get("first-seen.json")
            or not all(valid_sha(retrieval.get(key)) for key in ("inputs_sha256", "results_sha256"))
            or not isinstance(retrieval.get("retriever"), dict) or not retrieval["retriever"]):
        raise ValueError("Games retrieval must bind history-only time-filtered catalog/sidecar")


def _validate_queries(queries, items, first_seen, *, count):
    for query in queries:
        if len(query.candidate_ids) != count:
            raise ValueError("Games candidate count differs from the frozen protocol")
        needed = set(query.candidate_ids) | {h.item_id for h in query.history}
        if not needed <= items.keys():
            raise ValueError("Games query references missing catalog content")
        if any(first_seen[item] > query.target_timestamp for item in query.candidate_ids):
            raise ValueError("Games candidate is first observed after the query cutoff")


def _validate_cohorts(cohorts, manifest, config):
    caps = manifest.get("caps")
    if caps != DEFAULT_CAPS:
        raise ValueError("Games cohort upper bounds differ from Books")
    for split in ("train", "dev", "test"):
        users = cohorts.get(split)
        if (not isinstance(users, list) or len(users) != len(set(users))
                or len(users) > caps[split] or any(user_split(user) != split for user in users)):
            raise ValueError("Games cohort split/identity/upper bound mismatch")
        if config is not None and len(users) > config["data"][split + "_users"]:
            raise ValueError("Games actual cohort exceeds configured upper bound")
    selector_count = min(2000, len(cohorts["dev"]))
    expected = sorted(cohorts["dev"], key=lambda user: (
        hashlib.sha256((SELECTOR_NAMESPACE + user).encode()).hexdigest(), user))[:selector_count]
    if config is not None and len(expected) > config["data"]["selection_users"]:
        raise ValueError("Games selector exceeds configured upper bound")
    if cohorts.get("checkpoint_selector") != expected:
        raise ValueError("Games selector differs from the fixed Books hash ordering")
    actual = {key: len(cohorts[key]) for key in (*DEFAULT_CAPS, "checkpoint_selector")}
    if manifest.get("actual_cohorts") != actual:
        raise ValueError("Games actual cohort declaration mismatch")


def load_dataset(data_dir, config=None, *, formal=True):
    """Return (train, dev, items, cohorts, file_hashes); never open sealed test.

    Each critical file is hashed once while parsing this first load/restore.
    Query tuples and canonical catalog are then reused by training/evaluation.
    """
    root = Path(data_dir)
    manifest, manifest_sha = _read_json(root / "preparation-manifest.json")
    _validate_manifest(manifest)
    if config is not None:
        if config.get("schema_version") != PROTOCOL:
            raise ValueError("Games dataset requires a Games configuration")
        if (config.get("candidate_seed", config.get("seed")) != 17 or config["data"].get("query_cap") != 5
                or config["data"].get("history_max") != 10 or config["data"].get("train_candidates") != 8
                or config["data"].get("eval_candidates") != 100
                or config["data"].get("train_policy", "mixed-history") != "mixed-history"):
            raise ValueError("Games configured sampling/history/candidate contract differs from preparation")
    expected = manifest["files"]
    items, catalog_sha = _read_json(root / "catalog.json", expected["catalog.json"])
    cohorts, cohorts_sha = _read_json(root / "cohorts.json", expected["cohorts.json"])
    records, first_seen_sha = _read_json(root / "first-seen.json", expected["first-seen.json"])
    first_seen = first_seen_times(records)
    if not isinstance(items, dict) or not items or any(
            not isinstance(item, str) or not isinstance(text, str) or not text.strip() for item, text in items.items()):
        raise ValueError("Games catalog must map identities to canonical nonempty content")
    if set(items) != set(first_seen):
        raise ValueError("Games first-observation sidecar must cover exactly the catalog")
    _validate_cohorts(cohorts, manifest, config)
    train = read_queries(root / "train.jsonl", split="train", expected_sha256=expected["train.jsonl"])
    dev = read_queries(root / "dev.jsonl", split="dev", expected_sha256=expected["dev.jsonl"])
    _validate_queries(train, items, first_seen, count=8)
    _validate_queries(dev, items, first_seen, count=100)
    train_counts = Counter(q.user_id for q in train)
    unavailable = manifest.get("unavailable_dev_users")
    reasons = manifest.get("unavailable_dev_reasons")
    if (not isinstance(unavailable, list) or len(set(unavailable)) != len(unavailable)
            or not set(unavailable) <= set(cohorts["dev"]) or not isinstance(reasons, dict)
            or set(reasons) != set(unavailable)):
        raise ValueError("unavailable dev users require complete declared reasons")
    dev_users = {q.user_id for q in dev}
    if (not train or set(train_counts) != set(cohorts["train"]) or max(train_counts.values()) > 5
            or dev_users.intersection(unavailable) or dev_users | set(unavailable) != set(cohorts["dev"])
            or len(dev) != len(dev_users) or len(train) != manifest.get("train_queries")
            or len(dev) != manifest.get("dev_queries")):
        raise ValueError("Games query coverage/count/cap mismatch")
    if len({q.query_id for q in (*train, *dev)}) != len(train) + len(dev):
        raise ValueError("Games query IDs must be unique across train/dev")
    cohorts["unavailable_dev_users"] = unavailable
    files = {"train": expected["train.jsonl"], "dev": expected["dev.jsonl"],
             "catalog": catalog_sha, "cohorts": cohorts_sha, "first_seen": first_seen_sha,
             "preparation_manifest": manifest_sha}
    return train, dev, items, cohorts, files


def load_sealed_test(data_dir, *, expected_identity_sha256):
    """Explicit final evaluation only; require the caller's frozen data identity."""
    root = Path(data_dir)
    manifest, _ = _read_json(root / "preparation-manifest.json")
    _validate_manifest(manifest)
    if expected_identity_sha256 != manifest["identity_sha256"]:
        raise ValueError("sealed test requires the expected frozen preparation identity")
    items, _ = _read_json(root / "catalog.json", manifest["files"]["catalog.json"])
    cohorts, _ = _read_json(root / "cohorts.json", manifest["files"]["cohorts.json"])
    records, _ = _read_json(root / "first-seen.json", manifest["files"]["first-seen.json"])
    _validate_cohorts(cohorts, manifest, None)
    test = read_queries(root / "sealed-test.jsonl", split="test",
                        expected_sha256=manifest["files"]["sealed-test.jsonl"])
    _validate_queries(test, items, first_seen_times(records), count=100)
    unavailable = manifest.get("unavailable_test_users")
    reasons = manifest.get("unavailable_test_reasons")
    if (not isinstance(unavailable, list) or len(set(unavailable)) != len(unavailable)
            or not set(unavailable) <= set(cohorts["test"]) or not isinstance(reasons, dict)
            or set(reasons) != set(unavailable)):
        raise ValueError("unavailable test users require complete declared reasons")
    test_users = {q.user_id for q in test}
    if (test_users.intersection(unavailable) or test_users | set(unavailable) != set(cohorts["test"])
            or len(test) != len(test_users) or len(test) != manifest.get("test_queries")):
        raise ValueError("sealed test coverage/count mismatch")
    return test, items, cohorts, manifest
