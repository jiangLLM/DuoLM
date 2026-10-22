"""All downloads are small in-memory fixtures; never contact a remote service."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from duolm import assets


class AssetsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "workspace"

    def tearDown(self):
        self.temporary.cleanup()

    def test_public_model_revisions_and_official_raw_sha_are_pinned(self):
        self.assertEqual([(item["repo"], item["revision"]) for item in assets.MODEL_ASSETS], [
            ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
            ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")])
        self.assertEqual([(item["name"], item["sha256"]) for item in assets.DATA_ASSETS], [
            ("Books.csv.gz", "28c5104d0c2f7a7e842da3431606e21a245d8fb317e3434cd20b5c5a7b521615"),
            ("meta_Books.jsonl.gz", "d41f2348d3e0f3099d2a7ee45b304970282199f92a7ccdfe8ba4b1f753009a10")])
        self.assertTrue(all(item["url"].startswith("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/")
                            for item in (*assets.DATA_ASSETS, *assets.GAMES_DATA_ASSETS)))
        self.assertEqual([(item["name"], item["sha256"]) for item in assets.GAMES_DATA_ASSETS], [
            ("Video_Games.csv.gz", "a2bde5f3b945960d161538c200dd87845e6ee471b46da96410dde61613c6901c"),
            ("Video_Games.jsonl.gz", "52d8762c94a5b99d3bccdb31c15cc6c99e05c89f9042cd3753b3b623a44bc193"),
            ("meta_Video_Games.jsonl.gz", "ced9696b0874121327a438bfe9d3d01aba45da158e58f94f4920352d75fc1fc9")])
        for item in assets.MODEL_ASSETS:
            self.assertIn("model.safetensors", item["files"])
            self.assertIn("tokenizer.json", item["files"])
            self.assertTrue(all(len(digest) == 64 for digest in item["files"].values()))

    def data_fixture(self, content=b"offline source bytes"):
        return {"name": "tiny.csv.gz", "url": "https://example.invalid/fixture", "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest()}

    def test_data_published_only_after_checksum_then_reused_without_network(self):
        content = b"offline source bytes"
        specification = self.data_fixture(content)
        calls = []

        def opener(url, timeout):
            calls.append((url, timeout))
            return io.BytesIO(content)

        with contextlib.redirect_stdout(io.StringIO()):
            record = assets.fetch_data(self.workspace, specification, opener=opener)
            again = assets.fetch_data(self.workspace, specification,
                                      opener=lambda *args, **kwargs: self.fail("verified existing bytes must not redownload"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(record, again)
        path = self.workspace / "raw/tiny.csv.gz"
        self.assertEqual(path.read_bytes(), content)
        self.assertEqual(json.loads(path.with_name(path.name + ".download.json").read_text()), record)
        self.assertFalse(path.with_name(path.name + ".part").exists())

    def test_corrupt_new_or_existing_data_is_never_accepted(self):
        specification = self.data_fixture(b"valid")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            assets.fetch_data(self.workspace, specification, opener=lambda *args, **kwargs: io.BytesIO(b"wrong"))
        self.assertFalse((self.workspace / "raw/tiny.csv.gz").exists())
        self.assertFalse((self.workspace / "raw/tiny.csv.gz.download.json").exists())
        (self.workspace / "raw/tiny.csv.gz").write_bytes(b"wrong")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            assets.fetch_data(self.workspace, specification,
                              opener=lambda *args, **kwargs: self.fail("must not overwrite existing corruption"))
        self.assertEqual((self.workspace / "raw/tiny.csv.gz").read_bytes(), b"wrong")

    def test_model_download_receives_exact_revision_and_hashes_before_provenance(self):
        content = {"config.json": b'{"hidden_size":1024}', "model.safetensors": b"fake test weights"}
        specification = {**assets.MODEL_ASSETS[0], "files": {name: hashlib.sha256(value).hexdigest() for name, value in content.items()}}
        calls = []

        def download(**kwargs):
            calls.append(kwargs)
            for filename, value in content.items():
                (kwargs["local_dir"] / filename).write_bytes(value)

        with contextlib.redirect_stdout(io.StringIO()):
            record = assets.fetch_model(self.workspace, specification, snapshot_downloader=download)
            assets.fetch_model(self.workspace, specification, snapshot_downloader=lambda **kwargs: self.fail("no latest/network lookup"))
        self.assertEqual(calls[0]["repo_id"], "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(calls[0]["revision"], "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
        self.assertEqual(set(calls[0]["allow_patterns"]), set(content))
        source = json.loads((self.workspace / "models/encoder/source_revision.json").read_text())
        self.assertEqual(source["revision"], record["revision"])
        (self.workspace / "models/encoder/model.safetensors").write_bytes(b"corruption")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            assets.fetch_model(self.workspace, specification, snapshot_downloader=lambda **kwargs: self.fail("don't replace corrupt bytes"))

    def test_model_revision_conflict_is_rejected_before_download(self):
        path = self.workspace / "models/reader"
        path.mkdir(parents=True)
        (path / "source_revision.json").write_text(json.dumps({"repo": assets.MODEL_ASSETS[1]["repo"], "revision": "bad-revision"}))
        with self.assertRaisesRegex(ValueError, "provenance conflicts"):
            assets.fetch_model(self.workspace, assets.MODEL_ASSETS[1],
                               snapshot_downloader=lambda **kwargs: self.fail("must reject before download"))

    def test_unpinned_games_source_receipt_detects_later_corruption(self):
        specification = {"name": "tiny-games.gz", "url": "https://example.invalid/games"}
        with contextlib.redirect_stdout(io.StringIO()):
            report = assets.fetch_data(self.workspace, specification,
                                       opener=lambda *args, **kwargs: io.BytesIO(b"original"))
            self.assertEqual(report["checksum_origin"], "download-observation")
            self.assertEqual(report["sha256"], hashlib.sha256(b"original").hexdigest())
            assets.fetch_data(self.workspace, specification,
                              opener=lambda *args, **kwargs: self.fail("no second download"))
            (self.workspace / "raw/tiny-games.gz").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                assets.fetch_data(self.workspace, specification,
                                  opener=lambda *args, **kwargs: self.fail("no overwrite"))

    def test_downloaded_model_emits_preparation_and_training_compatible_snapshot(self):
        from duolm.build_next_item_assets import verify_public_snapshot
        from duolm.train_next_item import validate_local_snapshot
        content = {"config.json": b"{}", "tokenizer.json": b"fixture",
                   "model.safetensors": b"synthetic weights for inventory validation"}
        specification = {**assets.MODEL_ASSETS[0],
                         "files": {name: hashlib.sha256(value).hexdigest() for name, value in content.items()}}

        def download(**kwargs):
            for filename, value in content.items():
                (kwargs["local_dir"] / filename).write_bytes(value)

        with contextlib.redirect_stdout(io.StringIO()):
            assets.fetch_model(self.workspace, specification, snapshot_downloader=download)
        manifest = self.workspace / "models/encoder/snapshot-manifest.json"
        digest = verify_public_snapshot(manifest.parent, repo=specification["repo"], revision=specification["revision"])
        self.assertEqual(digest, assets.file_sha256(manifest))
        validate_local_snapshot({"repo": specification["repo"], "revision": specification["revision"],
                                 "local_path": str(manifest.parent), "snapshot_manifest_sha256": digest})

    def test_workspace_symlinks_cannot_redirect_asset_writes(self):
        self.workspace.mkdir()
        outside = self.workspace.parent / "outside"
        outside.mkdir()
        (self.workspace / "raw").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes workspace"):
            assets.fetch_data(self.workspace, self.data_fixture(), opener=lambda *args, **kwargs: self.fail("unsafe path"))

    def test_cli_modes_and_incremental_manifest_only_fetch_requested_assets(self):
        with patch.object(assets, "fetch_model", side_effect=lambda workspace, spec: {"name": spec["name"]}) as model_fetch, \
             patch.object(assets, "fetch_data", side_effect=lambda workspace, spec: {"name": spec["name"]}) as data_fetch, \
             contextlib.redirect_stdout(io.StringIO()):
            assets.main(["--workspace", str(self.workspace), "--models-only"])
            self.assertEqual(model_fetch.call_count, 2)
            self.assertEqual(data_fetch.call_count, 0)
            assets.main(["--workspace", str(self.workspace), "--data-only"])
            self.assertEqual(model_fetch.call_count, 2)
            self.assertEqual(data_fetch.call_count, 2)
            manifest = json.loads((self.workspace / "assets_manifest.json").read_text())
            self.assertEqual(len(manifest["models"]), 2)
            self.assertEqual(len(manifest["data"]), 2)
            self.assertEqual(manifest["verified_this_run"], [item["name"] for item in assets.DATA_ASSETS])
            assets.main(["--workspace", str(self.workspace), "--download-all"])
            self.assertEqual((model_fetch.call_count, data_fetch.call_count), (4, 4))


if __name__ == "__main__":
    unittest.main()
