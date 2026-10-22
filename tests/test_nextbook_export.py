"""Serving assets against tiny real Qwen3 backbones and frozen identities."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from duolm import nextbook_export as exporter
from duolm.next_item_model import GroupRanker
from duolm import train_next_item as trainer


REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def tokenizer():
    words = ["[PAD]", "[EOS]", "[UNK]", "Item", ":", "a", "different", "book", "short", "longer", "history"]
    backend = Tokenizer(models.WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]")


def fixture(root, *, tied=True, domain="book"):
    source = root / "source"
    package = source / "src/duolm"
    package.mkdir(parents=True)
    config = json.loads((REPO / f"configs/{'games' if domain == 'game' else 'books'}/student.json").read_text())
    names = exporter.source_files(config)
    for name in names:
        shutil.copyfile(REPO / "src/duolm" / name, package / name)
    spec = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      head_dim=8, max_position_embeddings=512, attention_dropout=0,
                      tie_word_embeddings=tied, use_cache=False)
    spec._attn_implementation = "sdpa"
    torch.manual_seed(19)
    a, b = Qwen3Model(copy.deepcopy(spec)), Qwen3Model(copy.deepcopy(spec))
    ta, tb = tokenizer(), tokenizer()
    paths, public_specs = {}, {}
    for side, model, tokens in (("a", a, ta), ("b", b, tb)):
        path = root / (side + "-model")
        model.save_pretrained(path)
        tokens.save_pretrained(path)
        report = {"repo_id": "fixture-" + side, "revision": side * 40,
                  "files": {p.name: exporter._record(p) for p in path.iterdir() if p.is_file()}}
        write_json(path / "snapshot-manifest.json", report)
        paths[side] = path
        public_specs[side] = {"repo": report["repo_id"], "revision": report["revision"],
                              "local_path": str(path), "snapshot_manifest_sha256": exporter._hash(path / "snapshot-manifest.json")}
    model = GroupRanker(a, b, ta, tb, variant="V3", m=1, domain=domain, a_readout="last_token",
                        shared_candidate_header=True, score_readout="candidate", attention_backend="sdpa",
                        max_encoder_packed_tokens=512, max_reader_packed_tokens=512, max_logical_tokens=512)
    model.bridge.set_reference(.37, {"source": "training_query_content", "items": 3, "content_hash": "0" * 64})
    # Prove the export uses trained checkpoint values, not public initialization.
    with torch.no_grad():
        model.a.layers[0].mlp.down_proj.weight.add_(.025)
        model.b.embed_tokens.weight.mul_(1.7)
        model.b.layers[1].self_attn.o_proj.weight.add_(.01)
        model.bridge.log_gain.fill_(.12)
        model.head.weight.copy_(torch.linspace(-.4, .3, 16)[None])
    model.eval()
    config["model"].update(a_model=public_specs["a"], b_model=public_specs["b"])
    config["training"].update(max_encoder_packed_tokens=512, max_reader_packed_tokens=512, max_logical_tokens=512)
    config["loss"]["recipe"] = "ce"
    config["training"]["teacher_cache_on_device"] = False
    if domain == "game":
        from tests.test_games_assets import build_fixture as prepare_games_fixture
        games_root = root / "games-data"
        games_root.mkdir()
        data = prepare_games_fixture(games_root)["prepared"]
        _, _, items, _, games_hashes = trainer.load_dataset(data, config, formal=True)
    else:
        data = root / "data"
        data.mkdir()
        items = {"item-b": "a different longer book", "item-a": "a short book", "item-c": "history book"}
        write_json(data / "catalog.json", items)
        for name in ("train.jsonl", "dev.jsonl", "cohorts.json", "preparation-manifest.json"):
            (data / name).write_text("{}\n")
        (data / "test.jsonl").write_text("sealed test data must never be opened")
    template = model.metadata()["student_template"]
    template["flex_attention_kernel"] = "forced_prefill"
    template["sha256"] = exporter._digest({key: value for key, value in template.items() if key != "sha256"})
    identity = {"config_sha256": exporter._digest(config),
                "source_sha256": {name: exporter._hash(package / name) for name in names},
                "data_sha256": {key: exporter._hash(data / name) for key, name in
                                  {"catalog": "catalog.json", "train": "train.jsonl", "dev": "dev.jsonl",
                                   "cohorts": "cohorts.json", "preparation_manifest": "preparation-manifest.json"}.items()},
                "student_template": template}
    if domain == "game":
        identity["data_sha256"] = games_hashes
        prompt = trainer.domain_template(domain)
        identity.update(domain=domain, schema_version=config["schema_version"], prompt_template={**prompt, "sha256": exporter._digest(prompt)})
    run = root / "run"
    run.mkdir()
    write_json(run / "run_manifest.json", {"protocol": trainer.training_protocol(identity), "identity": identity, "config": config})
    state = {"step": 7, "cursor": 3, "best_step": 7, "best_ndcg": .4, "best_checkpoint": "checkpoints/step-00000007"}
    checkpoint = trainer.save_checkpoint(run, model, torch.optim.AdamW(model.parameters()), state, identity)
    args = SimpleNamespace(source=source, checkpoint=checkpoint, data=data, a_model=paths["a"], b_model=paths["b"],
                           output=root / "export", vector_device="cpu", vector_dtype="float32", batch_size=2)
    return args, model, items


def resign(root, mutate):
    manifest = json.loads((root / "manifest.json").read_text())
    mutate(manifest)
    manifest["manifest_signature"] = exporter._digest({key: value for key, value in manifest.items() if key != "manifest_signature"})
    write_json(root / "manifest.json", manifest)


class NextbookExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_games_export_and_compiler_preserve_domain_prompt(self):
        from duolm.nextbook_serving_inputs import NextBookCompiler
        with tempfile.TemporaryDirectory() as temporary:
            args, model, _ = fixture(Path(temporary), domain="game")
            manifest = exporter.export_nextbook(args)
            self.assertEqual(manifest["model_metadata"]["domain"], "game")
            compiler = NextBookCompiler(args.output)
            expected = tuple(model.tokenizer_b.encode(trainer.domain_template("game")["instruction"], add_special_tokens=False))
            self.assertEqual(compiler._instruction, expected)
            row = json.loads((args.data / "dev.jsonl").read_text().splitlines()[0])
            query = {"query_id": row["query_id"], "history": [{"item_id": h["item_id"], "rating": h["rating"]} for h in row["history"]], "candidate_ids": row["candidate_ids"]}
            prepared = compiler.compile(query)
            self.assertEqual(prepared.candidate_ids, tuple(row["candidate_ids"]))
            self.assertIn("first_seen", manifest["data_sha256"])
            self.assertNotIn("first-seen.json", manifest["files"])
            self.assertTrue(prepared.query_ids[:len(expected)] == list(expected))

    def test_roundtrip_reader_and_exact_fp32_scalar_with_native_hf_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, model, _ = fixture(Path(temporary))
            manifest = exporter.export_nextbook(args)
            self.assertEqual(manifest["checkpoint_step"], 7)
            self.assertEqual(manifest["vector_count"], 3)
            self.assertEqual(manifest["model_metadata"]["student_template"], manifest["checkpoint_identity"]["student_template"])
            self.assertEqual(manifest["encoding"]["compute_dtype"], "float32")
            self.assertIn("pending", manifest["encoding"]["numerical_acceptance"])
            actual = AutoModelForCausalLM.from_pretrained(args.output / "reader", local_files_only=True, torch_dtype=torch.float32)
            self.assertEqual(actual.config.architectures, ["Qwen3ForCausalLM"])
            for key, value in model.b.state_dict().items():
                self.assertTrue(torch.equal(value, actual.model.state_dict()[key]), key)
            self.assertTrue(torch.equal(actual.lm_head.weight, model.b.embed_tokens.weight))
            head = load_file(str(args.output / "scalar_head.safetensors"))["head.weight"]
            self.assertEqual(head.dtype, torch.float32)
            self.assertTrue(torch.equal(head, model.head.weight))
            self.assertFalse(manifest["reader"]["vocabulary_head"]["trained_score_head"])
            self.assertEqual(exporter.verify_nextbook_export(args.output, verify_vectors=False), manifest)

    def test_catalog_vectors_match_independent_eos_endpoint_and_checkpoint_rms(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, model, items = fixture(Path(temporary), tied=False)
            exporter.export_nextbook(args)
            vectors = np.load(args.output / "vectors.npy", mmap_mode="r", allow_pickle=False)
            self.assertIsInstance(vectors, np.memmap)
            self.assertEqual(vectors.dtype, np.float16)
            with torch.inference_mode():
                for row, item in enumerate(sorted(items)):
                    ids = model.tokenizer_a.encode("Item: " + items[item], add_special_tokens=False) + [model.tokenizer_a.eos_token_id]
                    hidden = model.a(input_ids=torch.tensor(ids)[None], position_ids=torch.arange(len(ids))[None],
                                     attention_mask=torch.ones((1, len(ids)), dtype=torch.long), use_cache=False).last_hidden_state[0, -1:]
                    expected = model.bridge(model.projection(hidden)).half().numpy()[0]
                    np.testing.assert_array_equal(vectors[row], expected)
            rows = [json.loads(line) for line in (args.output / "index.jsonl").read_text().splitlines()]
            self.assertEqual(rows, [{"item_id": item, "row": row} for row, item in enumerate(sorted(items))])
            reader = load_file(str(args.output / "reader/model.safetensors"))
            self.assertFalse(bool(reader["lm_head.weight"].any()))

    def test_encoder_packs_respect_frozen_dense_reference_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, model, items = fixture(Path(temporary))
            model.dense_reference_limit = 8
            args.batch_size = 3
            args.output.mkdir()
            with torch.inference_mode():
                encoding = exporter._write_vectors(args.output, model, items, args)
            self.assertEqual(encoding["encoder_pack_tokens"], 8)
            self.assertEqual(np.load(args.output / "vectors.npy").shape, (3, 16))

    def test_rejects_checkpoint_bytes_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _ = fixture(Path(temporary))
            with (args.checkpoint / "training.pt").open("ab") as stream:
                stream.write(b"tampered")
            with patch.object(exporter.torch, "load") as loader, self.assertRaisesRegex(ValueError, "checksum/size"):
                exporter.export_nextbook(args)
            loader.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_rejects_source_config_and_data_identity_changes(self):
        for field in ("source", "config", "data"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                args, _, _ = fixture(Path(temporary))
                if field == "source":
                    path = args.source / "src/duolm/next_item_model.py"
                    path.write_text(path.read_text() + "\n# changed source\n")
                elif field == "config":
                    path = args.checkpoint.parent.parent / "run_manifest.json"
                    value = json.loads(path.read_text())
                    value["config"]["seed"] += 1
                    write_json(path, value)
                else:
                    (args.data / "train.jsonl").write_text("altered training file")
                with self.assertRaisesRegex(ValueError, "hash differs|checksum differs"):
                    exporter.export_nextbook(args)
                self.assertFalse(args.output.exists())

    def test_output_no_clobber_and_file_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _ = fixture(Path(temporary))
            exporter.export_nextbook(args)
            before = exporter._hash(args.output / "manifest.json")
            with self.assertRaisesRegex(ValueError, "output exists"):
                exporter.export_nextbook(args)
            self.assertEqual(exporter._hash(args.output / "manifest.json"), before)
            with (args.output / "index.jsonl").open("a") as stream:
                stream.write('{}\n')
            with self.assertRaisesRegex(ValueError, "checksum/size"):
                exporter.verify_nextbook_export(args.output)

    def test_resigned_export_rejects_extra_training_files_and_tokenizer_disguise(self):
        for filename, disguise in (("train.jsonl", False), ("reader/test.jsonl", False),
                                   ("reader/train.jsonl", True)):
            with self.subTest(filename=filename, disguise=disguise), tempfile.TemporaryDirectory() as temporary:
                args, _, _ = fixture(Path(temporary))
                exporter.export_nextbook(args)
                extra = args.output / filename
                extra.write_text('{"private_training_row": true}\n')
                def update(manifest):
                    record = exporter._record(extra)
                    manifest["files"][filename] = record
                    if disguise:
                        name = extra.name
                        manifest["native_tokenizer_sha256"][name] = record["sha256"]
                        manifest["public_models"]["b"]["files"][name] = record
                resign(args.output, update)
                with self.assertRaisesRegex(ValueError, "allowed serving files|tokenizer allowlist"):
                    exporter.verify_nextbook_export(args.output)
                # Disabling array-content validation cannot bypass the same
                # public serving inventory boundary used before publication.
                with self.assertRaisesRegex(ValueError, "allowed serving files|tokenizer allowlist"):
                    exporter.verify_nextbook_export(args.output, verify_vectors=False)

    def test_vectors_nan_and_index_alias_are_rejected_after_checksum_resigning(self):
        for field in ("vectors", "index"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                args, _, _ = fixture(Path(temporary))
                exporter.export_nextbook(args)
                filename = "vectors.npy" if field == "vectors" else "index.jsonl"
                if field == "vectors":
                    vectors = np.load(args.output / filename, mmap_mode="r+", allow_pickle=False)
                    vectors[0, 0] = np.nan
                    vectors.flush()
                    del vectors
                else:
                    rows = [json.loads(line) for line in (args.output / filename).read_text().splitlines()]
                    rows[1]["item_id"] = rows[0]["item_id"]
                    (args.output / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
                def update(manifest):
                    record = exporter._record(args.output / filename)
                    manifest["files"][filename] = record
                    manifest[field].update(record)
                resign(args.output, update)
                with self.assertRaisesRegex(ValueError, "nonfinite|sorted unique"):
                    exporter.verify_nextbook_export(args.output)

    def test_model_scalar_shape_rejected_even_with_rebound_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _ = fixture(Path(temporary))
            path = args.checkpoint / "training.pt"
            payload = torch.load(path, weights_only=True)
            payload["model"]["head.weight"] = torch.zeros((5, 16))
            torch.save(payload, path)
            receipt = json.loads((args.checkpoint / "manifest.json").read_text())
            receipt["files"]["training.pt"] = exporter._record(path)
            write_json(args.checkpoint / "manifest.json", receipt)
            with self.assertRaises(RuntimeError):
                exporter.export_nextbook(args)
            self.assertFalse(args.output.exists())

    def test_changed_template_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _ = fixture(Path(temporary))
            exporter.export_nextbook(args)
            resign(args.output, lambda manifest: manifest["model_metadata"]["student_template"].update(a_eos=9))
            with self.assertRaisesRegex(ValueError, "template"):
                exporter.verify_nextbook_export(args.output)


if __name__ == "__main__":
    unittest.main()
