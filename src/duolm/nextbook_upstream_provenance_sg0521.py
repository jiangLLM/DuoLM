"""One-time ready-state provenance; never called by the scoring forward path."""
from functools import lru_cache
import hashlib
import importlib.metadata
from pathlib import Path
import sys

MODULES = {
    "sglang.srt.managers.scheduler": "srt/managers/scheduler.py",
    "sglang.srt.models.qwen3": "srt/models/qwen3.py",
    "sglang.srt.layers.attention.flashinfer_backend": "srt/layers/attention/flashinfer_backend.py",
    "sglang.srt.layers.layernorm": "srt/layers/layernorm.py",
}

@lru_cache(maxsize=1)
def worker_provenance():
    from .nextbook_sglang_overlay import verify_overlay
    version = importlib.metadata.version("sglang")
    package = sys.modules.get("sglang")
    if version != "0.5.21" or package is None or not getattr(package, "__file__", None):
        raise ValueError("Actual SGLang 0.5.21 worker package required")
    root = Path(package.__file__).resolve().parent
    manifest = verify_overlay(root.parent)
    if manifest["supported_sglang_version"] != version:
        raise ValueError("Worker package version differs from overlay")
    hashes, paths = {}, {}
    for name, relative in MODULES.items():
        module = sys.modules.get(name)
        if module is None or not getattr(module, "__file__", None):
            raise ValueError("Expected SGLang worker module was not already loaded: " + name)
        path = Path(module.__file__).resolve()
        if path != root / relative or not path.is_file():
            raise ValueError("Worker imported a source outside its fresh overlay: " + name)
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        if value != manifest["overlay_files_sha256"].get(relative):
            raise ValueError("Loaded worker source differs from overlay: " + name)
        hashes[relative], paths[relative] = value, str(path)
    return {"version": version, "package_path": str(root),
            "loaded_source_files_sha256": hashes, "loaded_source_paths": paths}
