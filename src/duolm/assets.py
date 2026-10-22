"""Download pinned public models and official Amazon Reviews 2023 sources.

Run explicitly with --workspace and optionally --models-only/--data-only.
Network libraries for model downloads are imported only when requested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.request import urlopen


MODEL_ASSETS = (
    {"name": "encoder", "repo": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
     "files": {
         "config.json": "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
         "model.safetensors": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
         "tokenizer.json": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
         "tokenizer_config.json": "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
         "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
         "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
         "generation_config.json": "28396d421a2108acce96383f6a7de78008f7f1b17f807958f3c14c51dbfb65fb",
         "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
         "config_sentence_transformers.json": "10667c72ddb772627bf1780cb7f86af8e2ae0032b8c243c731172064105c6961",
     }},
    {"name": "reader", "repo": "Qwen/Qwen3-0.6B-Base", "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
     "files": {
         "config.json": "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
         "model.safetensors": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
         "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
         "tokenizer_config.json": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
         "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
         "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
         "generation_config.json": "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
     }},
)
DATA_ASSETS = (
    {"name": "Books.csv.gz",
     "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Books.csv.gz",
     "bytes": 144479502, "sha256": "28c5104d0c2f7a7e842da3431606e21a245d8fb317e3434cd20b5c5a7b521615"},
    {"name": "meta_Books.jsonl.gz",
     "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz",
     "bytes": 4942125770, "sha256": "d41f2348d3e0f3099d2a7ee45b304970282199f92a7ccdfe8ba4b1f753009a10"},
)

GAMES_DATA_ASSETS = (
    {"name": "Video_Games.csv.gz",
     "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz",
     "bytes": 12801967, "sha256": "a2bde5f3b945960d161538c200dd87845e6ee471b46da96410dde61613c6901c"},
    {"name": "Video_Games.jsonl.gz",
     "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz",
     "bytes": 814206092, "sha256": "52d8762c94a5b99d3bccdb31c15cc6c99e05c89f9042cd3753b3b623a44bc193"},
    {"name": "meta_Video_Games.jsonl.gz",
     "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz",
     "bytes": 103096082, "sha256": "ced9696b0874121327a438bfe9d3d01aba45da158e58f94f4920352d75fc1fc9"},
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(workspace, relative):
    workspace = Path(workspace).expanduser().resolve()
    path = workspace / relative
    if not path.resolve().is_relative_to(workspace):
        raise ValueError(f"asset path escapes workspace: {path}")
    current = path
    while current != workspace:
        if current.is_symlink():
            raise ValueError(f"asset paths must not be symlinks: {current}")
        current = current.parent
    return path


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("refusing an asset metadata symlink")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _check_file(path, expected_sha256=None, expected_bytes=None):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"asset is missing or is a symlink: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"asset byte count mismatch: {path}")
    actual = file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"asset SHA256 mismatch: {path}; preserve and inspect the file before retrying")
    return actual


def fetch_data(workspace, asset, *, opener=None):
    """Stream atomically, verify any pinned checksum, and record downloaded bytes."""
    path = _safe_path(workspace, "raw/" + asset["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "data_start", "name": asset["name"], "url": asset["url"]}), flush=True)
    receipt = _safe_path(workspace, "raw/" + asset["name"] + ".download.json")
    expected_sha256, expected_bytes = asset.get("sha256"), asset.get("bytes")
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if prior.get("url") != asset["url"]:
            raise ValueError("download receipt identifies a different source")
        if (not isinstance(prior.get("sha256"), str) or len(prior["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in prior["sha256"])
                or isinstance(prior.get("bytes"), bool) or not isinstance(prior.get("bytes"), int)
                or prior["bytes"] < 1):
            raise ValueError("download receipt is missing its byte count or SHA256")
        expected_sha256 = expected_sha256 or prior.get("sha256")
        expected_bytes = expected_bytes or prior.get("bytes")
    elif path.exists() and expected_sha256 is None:
        raise ValueError("untracked download has no reference hash; supply it directly to preparation or use a fresh workspace")
    if not path.exists():
        part = _safe_path(workspace, "raw/" + asset["name"] + ".part")
        open_url = urlopen if opener is None else opener
        with open_url(asset["url"], timeout=60) as response, part.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        _check_file(part, expected_sha256, expected_bytes)
        if part.stat().st_size == 0:
            raise ValueError("empty public download")
        os.replace(part, path)
    digest = _check_file(path, expected_sha256, expected_bytes)
    record = {**asset, "path": str(path), "sha256": digest, "bytes": path.stat().st_size,
              "checksum_origin": "pinned-reference" if asset.get("sha256") else "download-observation"}
    _atomic_json(_safe_path(workspace, "raw/" + asset["name"] + ".download.json"), record)
    print(json.dumps({"event": "data_verified", **record}), flush=True)
    return record


def fetch_model(workspace, asset, *, snapshot_downloader=None):
    path = _safe_path(workspace, "models/" + asset["name"])
    path.mkdir(parents=True, exist_ok=True)
    source_path = _safe_path(workspace, f"models/{asset['name']}/source_revision.json")
    if source_path.exists():
        previous = json.loads(source_path.read_text())
        if previous.get("repo") != asset["repo"] or previous.get("revision") != asset["revision"]:
            raise ValueError(f"existing model provenance conflicts with pinned model: {source_path}")
    for filename in asset["files"]:
        _safe_path(workspace, f"models/{asset['name']}/{filename}")
    if not all((path / filename).is_file() for filename in asset["files"]):
        if snapshot_downloader is None:
            from huggingface_hub import snapshot_download
            snapshot_downloader = snapshot_download
        print(json.dumps({"event": "model_start", "repo": asset["repo"], "revision": asset["revision"]}), flush=True)
        snapshot_downloader(repo_id=asset["repo"], revision=asset["revision"], local_dir=path,
                            allow_patterns=list(asset["files"]), max_workers=4)
    hashes = {filename: _check_file(path / filename, digest) for filename, digest in asset["files"].items()}
    source = {"repo": asset["repo"], "revision": asset["revision"], "path": str(path)}
    _atomic_json(source_path, source)
    snapshot_files = {name: {"bytes": (path / name).stat().st_size, "sha256": digest}
                      for name, digest in hashes.items()}
    snapshot_files["source_revision.json"] = {
        "bytes": source_path.stat().st_size, "sha256": file_sha256(source_path)}
    _atomic_json(path / "snapshot-manifest.json", {
        "repo_id": asset["repo"], "revision": asset["revision"],
        "files": snapshot_files})
    record = {"name": asset["name"], **source, "files_sha256": hashes}
    print(json.dumps({"event": "model_verified", "repo": asset["repo"], "revision": asset["revision"]}), flush=True)
    return record


def download_assets(workspace, mode="all", *, domain="books"):
    if mode not in {"all", "models", "data"}:
        raise ValueError("asset mode must be all, models, or data")
    if domain not in {"books", "games"}:
        raise ValueError("asset domain must be books or games")
    data_assets = DATA_ASSETS if domain == "books" else GAMES_DATA_ASSETS
    workspace = Path(workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    manifest_path = _safe_path(workspace, "assets_manifest.json")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("schema_version") != 1:
            raise ValueError("existing assets manifest has an unsupported schema")
    else:
        previous = {"schema_version": 1, "models": [], "data": []}
    manifest = {"schema_version": 1, "workspace": str(workspace),
                "models": previous.get("models", []), "data": previous.get("data", []),
                "verified_this_run": []}
    if mode in {"all", "models"}:
        manifest["models"] = [fetch_model(workspace, asset) for asset in MODEL_ASSETS]
        manifest["verified_this_run"].extend(asset["name"] for asset in MODEL_ASSETS)
    if mode in {"all", "data"}:
        manifest["data"] = [fetch_data(workspace, asset) for asset in data_assets]
        manifest["verified_this_run"].extend(asset["name"] for asset in data_assets)
    _atomic_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--domain", choices=("books", "games"), default="books")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--models-only", action="store_const", dest="mode", const="models")
    modes.add_argument("--data-only", action="store_const", dest="mode", const="data")
    modes.add_argument("--download-all", action="store_const", dest="mode", const="all")
    parser.set_defaults(mode="all")
    args = parser.parse_args(argv)
    manifest = download_assets(args.workspace, args.mode, domain=args.domain)
    print(json.dumps({"event": "assets_ready", "workspace": manifest["workspace"],
                      "verified_this_run": manifest["verified_this_run"]}), flush=True)


if __name__ == "__main__":
    main()
