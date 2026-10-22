"""Private BCG admission, options, and serving-only replay counters."""

import hashlib
import json
import os

ADMISSION_ENVIRONMENT = "MIXLM_NEXTBOOK_BCG_ADMISSION"
ADMISSION_PROTOCOL = "mixlm-nextbook-bcg-admission-v1"


def _admission_payload(buckets):
    value = {"protocol": ADMISSION_PROTOCOL, "mode": "breakable", "token_buckets": list(buckets)}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def configure_graph_admission(mode, buckets):
    # Called by the private server after configure_runtime and before Engine.
    # The mark is bound to the same verified scalar runtime and inherited key.
    os.environ.pop(ADMISSION_ENVIRONMENT, None)
    if mode == "disabled":
        return
    from .nextbook_score_head import runtime_config, _new_mark
    config = runtime_config()
    if (mode != "breakable" or config is None or config.get("arithmetic_mode") != "sglang"
            or config.get("variant") != "V3" or config.get("m") != 1):
        raise ValueError("Private BCG admission requires the verified sglang scalar runtime")
    value = {"protocol": ADMISSION_PROTOCOL, "mode": mode, "token_buckets": list(buckets),
             "mark": _new_mark(config, _admission_payload(buckets))}
    os.environ[ADMISSION_ENVIRONMENT] = json.dumps(value, sort_keys=True, separators=(",", ":"))


def allow_scalar_mis_bcg(server_args, cfg):
    # Do not relax ordinary MIS, implicit graph defaults, Full, or TC paths.
    if (cfg.cuda_graph_config.prefill.backend != "breakable"
            or ("prefill", "backend") not in getattr(server_args, "_cuda_graph_config_locked", ())):
        return False
    raw = os.environ.get(ADMISSION_ENVIRONMENT)
    if not raw:
        return False
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return False
    buckets = cfg.cuda_graph_config.prefill.bs
    if (type(value) is not dict or set(value) != {"protocol", "mode", "token_buckets", "mark"}
            or value["protocol"] != ADMISSION_PROTOCOL or value["mode"] != "breakable"
            or not buckets or value["token_buckets"] != list(buckets)):
        return False
    from .nextbook_score_head import runtime_config, _valid_mark
    config = runtime_config()
    return (config is not None and config.get("arithmetic_mode") == "sglang"
            and config.get("variant") == "V3" and config.get("m") == 1
            and _valid_mark(value["mark"], config, _admission_payload(buckets)))


PROTOCOL = "mixlm-nextbook-bcg-replay-v1"
_COUNTERS = {"graph_batches": 0, "graph_requests": 0, "raw_tokens": 0,
             "padded_tokens": 0, "padding_tokens": 0, "debug_eager_replays": 0}
_TOKEN_BUCKETS = {}
_RUNNER_INFO = {"backend": None, "captured_token_buckets": [], "debug_eager": None}


def record_capture(backend, token_buckets):
    # Startup metadata only; this does not count dummy/warmup forwards as hits.
    _RUNNER_INFO.update(backend=type(backend).__name__, captured_token_buckets=list(token_buckets),
                        debug_eager=bool(getattr(backend, "_debug_eager", False)))


def authenticated_bcg_batch(batch):
    marks = getattr(batch, "mixlm_nextbook_scores", None)
    if not marks or not any(mark is not None for mark in marks):
        return False
    # Same signed admission and all-scalar invariant as the unchanged eager head.
    from .nextbook_score_head import _marked, _mark_mode
    return bool(_marked(batch)) and all(_mark_mode(mark) == "mis" for mark in marks)


def record_replay(batch, raw_tokens, padded_tokens, *, debug_eager=False):
    # Called after backend.replay only, never from capture or warmup.
    if debug_eager:
        _COUNTERS["debug_eager_replays"] += 1
        return
    _COUNTERS["graph_batches"] += 1
    _COUNTERS["graph_requests"] += batch.batch_size
    _COUNTERS["raw_tokens"] += raw_tokens
    _COUNTERS["padded_tokens"] += padded_tokens
    _COUNTERS["padding_tokens"] += padded_tokens - raw_tokens
    key = str(padded_tokens)
    _TOKEN_BUCKETS[key] = _TOKEN_BUCKETS.get(key, 0) + 1


def graph_report():
    return {"protocol": PROTOCOL, **_COUNTERS, **_RUNNER_INFO,
            "captured_token_buckets": list(_RUNNER_INFO["captured_token_buckets"]),
            "token_buckets": dict(_TOKEN_BUCKETS)}


def parse_token_buckets(value):
    try:
        buckets = [int(part) for part in value.split(",")]
    except (ValueError, AttributeError) as exc:
        raise ValueError("Prefill graph token buckets must be comma-separated integers") from exc
    if not buckets or buckets != sorted(set(buckets)) or buckets[0] <= 0 or buckets[-1] > 32768:
        raise ValueError("Prefill graph token buckets must be unique, increasing, and within 1..32768")
    return buckets


def prefill_graph_options(mode, buckets, arithmetic_mode="sglang"):
    if mode == "disabled":
        return {}
    if mode != "breakable" or arithmetic_mode != "sglang":
        raise ValueError("Private prefill graph trial requires breakable with sglang arithmetic")
    return {"disable_cuda_graph": False,
            "cuda_graph_config": {"decode": {"backend": "disabled"},
                                  "prefill": {"backend": "breakable", "bs": list(buckets)}}}
