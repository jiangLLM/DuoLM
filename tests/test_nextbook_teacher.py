"""Tiny real Qwen3 tests for the separate full-text next-book teacher path."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from duolm import nextbook_teacher_export as exporter
from duolm.nextbook_teacher_inputs import NextBookTeacherCompiler
from duolm.next_item_model import GroupRanker
from duolm.ranking_packing import PackedTreeLayout, QueryLayout
from duolm import train_next_item as trainer

REPO = Path(__file__).resolve().parents[1]
ref_spec = importlib.util.spec_from_file_location("teacher_reference", REPO / "scripts/build_nextbook_teacher_reference.py")
reference = importlib.util.module_from_spec(ref_spec)
ref_spec.loader.exec_module(reference)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def fixture(root, *, tied=True, public_dtype=None, domain="book", step=4000):
    source = root / "source"
    package = source / "src/duolm"
    package.mkdir(parents=True)
    config_path = REPO / f"configs/{'games' if domain == 'game' else 'books'}/teacher.json"
    source_names = exporter.source_files(json.loads(config_path.read_text()))
    for name in source_names:
        shutil.copyfile(REPO / "src/duolm" / name, package / name)
    words = ["[PAD]", "[EOS]", "[UNK]", "a", "book", "different", "longer", "history", "short",
             "Predict", "the", "user", "s", "next", "interaction", "from", "past", ".", "'", ":",
             "History", "Item", "User", "rating", "1", "2", "3", "4", "5", "Candidate", "Next", "-", "score"]
    tokenizer = Tokenizer(models.WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokens = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]")
    native = Qwen3Config(vocab_size=40, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                        max_position_embeddings=512, attention_dropout=0, tie_word_embeddings=tied, use_cache=False)
    native._attn_implementation = "sdpa"
    torch.manual_seed(71)
    model = GroupRanker(None, Qwen3Model(native), None, tokens, variant="T", attention_backend="sdpa", domain=domain,
                        max_reader_packed_tokens=512, max_encoder_packed_tokens=512, max_logical_tokens=512)
    public = root / "public"
    model.b.save_pretrained(public)
    tokens.save_pretrained(public)
    if public_dtype is not None:
        config = json.loads((public / "config.json").read_text())
        config.pop("torch_dtype", None)
        config["dtype"] = public_dtype
        config["transformers_version"] = "4.51.0"
        write_json(public / "config.json", config)
    report = {"repo_id": "fixture-b", "revision": "b" * 40,
              "files": {path.name: exporter._record(path) for path in public.iterdir() if path.is_file()}}
    write_json(public / "snapshot-manifest.json", report)
    # Deliberately remove unused public weights. Instantiation may never read them.
    (public / "model.safetensors").unlink()
    with torch.no_grad():
        model.b.embed_tokens.weight.mul_(1.7)
        model.b.layers[0].self_attn.o_proj.weight.add_(.02)
        model.head.weight.copy_(torch.linspace(-.4, .3, native.hidden_size)[None])
    model.eval()
    config = json.loads(config_path.read_text())
    config["model"]["b_model"] = {"repo": report["repo_id"], "revision": report["revision"],
                                  "snapshot_manifest_sha256": exporter._hash(public / "snapshot-manifest.json")}
    config["training"].update(max_reader_packed_tokens=512, max_encoder_packed_tokens=512, max_logical_tokens=512)
    if domain == "game":
        from tests.test_games_assets import build_fixture as prepare_games_fixture
        games_root = root / "games-data"
        games_root.mkdir()
        data = prepare_games_fixture(games_root)["prepared"]
        _, _, items, _, games_hashes = trainer.load_dataset(data, config, formal=True)
    else:
        data = root / "data"
        data.mkdir()
        items = {"a": "a short book", "b": "a different longer book book", "h": "history book"}
        write_json(data / "catalog.json", items)
        for name in ("dev.jsonl", "cohorts.json", "preparation-manifest.json"):
            (data / name).write_text("{}\n")
    identity = {"config_sha256": exporter._digest(config),
                "source_sha256": {name: exporter._hash(package / name) for name in source_names},
                "data_sha256": {"train": "a" * 64, **{key: exporter._hash(data / name) for key, name in
                    {"catalog": "catalog.json", "dev": "dev.jsonl", "cohorts": "cohorts.json",
                     "preparation_manifest": "preparation-manifest.json"}.items()}}}
    if domain == "game":
        identity["data_sha256"] = games_hashes
        prompt = trainer.domain_template(domain)
        identity.update(domain=domain, schema_version=config["schema_version"], prompt_template={**prompt, "sha256": exporter._digest(prompt)})
    run = root / "run"
    run.mkdir()
    write_json(run / "run_manifest.json", {"protocol": trainer.training_protocol(identity), "identity": identity, "config": config})
    state = {"step": step, "cursor": 32 * step, "best_step": step, "best_ndcg": .4,
             "best_checkpoint": f"checkpoints/step-{step:08d}"}
    checkpoint = trainer.save_checkpoint(run, model, torch.optim.AdamW(model.parameters()), state, identity)
    proof = root / "restore-proof.json"
    write_json(proof, {"repo_id": "fixture-archive", "revision": "a" * 40, "prefix": "teacher4000",
                       "archive": {"bytes": 10, "sha256": "b" * 64}, "all_members_verified": True,
                       "restored_files": {str(checkpoint / name): exporter._record(checkpoint / name)
                                          for name in ("training.pt", "manifest.json")}})
    args = SimpleNamespace(source=source, checkpoint=checkpoint, data=data, b_model=public, output=root / "export",
                           provenance=proof, expected_checkpoint_sha256=exporter._hash(checkpoint / "training.pt"))
    return args, model, items


def query(candidates=None):
    return {"query_id": "query-one", "history": [{"item_id": "h", "rating": 5}],
            "candidate_ids": candidates or ["b", "a"]}


class NextbookTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_public_games_checkpoint_exports_without_archive_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, model, _ = fixture(Path(temporary), domain="game", step=7)
            args.provenance = None
            manifest = exporter.export_nextbook_teacher(args)
            self.assertEqual(manifest["checkpoint_step"], 7)
            self.assertEqual(manifest["archived_provenance"]["evidence"]["kind"], "local_checkpoint_manifest")
            self.assertEqual(manifest["model_metadata"]["teacher_template"]["instruction"], trainer.domain_template("game")["instruction"])
            compiler = NextBookTeacherCompiler(args.output)
            row = json.loads((args.data / "dev.jsonl").read_text().splitlines()[0])
            request = {"query_id": row["query_id"], "history": [{"item_id": h["item_id"], "rating": h["rating"]} for h in row["history"]], "candidate_ids": row["candidate_ids"]}
            prepared = compiler.compile(request)
            actual = reference.independent_scores(model.b, model.head.weight, prepared, token_budget=512)
            self.assertEqual(len(actual), 100)
            self.assertEqual(manifest["data_verification"]["first_seen"]["status"], "verified_local_bytes")
            self.assertEqual(compiler._instruction, tuple(model.tokenizer_b.encode(trainer.domain_template("game")["instruction"], add_special_tokens=False)))

    def test_export_roundtrip_exact_trained_weights_no_public_weights_no_train(self):
        for tied in (True, False):
            with self.subTest(tied=tied), tempfile.TemporaryDirectory() as temporary:
                args, model, _ = fixture(Path(temporary), tied=tied)
                manifest = exporter.export_nextbook_teacher(args)
                self.assertEqual(manifest["kind"], exporter.KIND)
                self.assertEqual(manifest["checkpoint_step"], 4000)
                self.assertFalse(manifest["public_models"]["b"]["public_weights_loaded"])
                self.assertIn("model.safetensors", manifest["public_models"]["b"]["unused_unverified_files"])
                self.assertEqual(manifest["data_verification"]["train"]["status"], "archived_identity_only_bytes_absent")
                self.assertEqual(manifest["data_verification"]["dev"]["status"], "verified_local_bytes")
                loaded = AutoModelForCausalLM.from_pretrained(args.output / "reader", local_files_only=True,
                                                            torch_dtype=torch.float32, attn_implementation="sdpa")
                for name, expected in model.b.state_dict().items():
                    self.assertTrue(torch.equal(expected, loaded.model.state_dict()[name]), name)
                head = load_file(str(args.output / "scalar_head.safetensors"))["head.weight"]
                self.assertEqual(head.dtype, torch.float32)
                self.assertTrue(torch.equal(head, model.head.weight))
                prepared = NextBookTeacherCompiler(args.output).compile(query())
                expected = reference.independent_scores(model.b, model.head.weight, prepared, token_budget=512)
                actual = reference.independent_scores(loaded.model, head, prepared, token_budget=512)
                torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=0, atol=0)
                if tied:
                    self.assertTrue(torch.equal(loaded.lm_head.weight, model.b.embed_tokens.weight))
                else:
                    self.assertFalse(bool(loaded.lm_head.weight.any()))

    def test_exact_separate_tokenization_suffix_endpoints_and_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, model, items = fixture(Path(temporary))
            exporter.export_nextbook_teacher(args)
            compiler = NextBookTeacherCompiler(args.output)
            prepared = compiler.compile(query())
            q = SimpleNamespace(query_id="query-one", history=(SimpleNamespace(item_id="h", rating=5),), candidate_ids=("b", "a"))
            prefix, branches = model._query_embeddings(q, items)
            embed = model.b.get_input_embeddings()
            torch.testing.assert_close(embed(torch.tensor(prepared.query_ids)), prefix, atol=0, rtol=0)
            for ids, expected in zip(prepared.item_ids, branches):
                torch.testing.assert_close(embed(torch.tensor(ids)), expected, atol=0, rtol=0)
                suffix = tuple(model.tokenizer_b.encode("\nNext-book score:", add_special_tokens=False))
                self.assertEqual(ids[-len(suffix):], suffix)
            self.assertNotEqual(len(prepared.item_ids[0]), len(prepared.item_ids[1]))
            trained = model([q], items)["scores"][0]
            independently = reference.independent_scores(model.b, model.head.weight, prepared, token_budget=512)
            torch.testing.assert_close(torch.tensor(independently), trained, atol=2e-6, rtol=2e-6)
            self.assertEqual(prepared.candidate_ids, ("b", "a"))

    def test_public_bf16_config_does_not_round_trained_fp32_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, model, _ = fixture(Path(temporary), public_dtype="bfloat16")
            self.assertFalse(torch.equal(model.b.embed_tokens.weight, model.b.embed_tokens.weight.bfloat16().float()))
            manifest = exporter.export_nextbook_teacher(args)
            self.assertEqual(manifest["native_reader_config"]["dtype"], "bfloat16")
            self.assertEqual(manifest["native_reader_config"]["transformers_version"], "4.51.0")
            saved = load_file(str(args.output / "reader/model.safetensors"))
            for name, expected in model.b.state_dict().items():
                self.assertEqual(saved["model." + name].dtype, torch.float32)
                self.assertTrue(torch.equal(saved["model." + name], expected), name)
            loaded = AutoModelForCausalLM.from_pretrained(args.output / "reader", local_files_only=True,
                                                        torch_dtype=torch.float32, attn_implementation="sdpa")
            prepared = NextBookTeacherCompiler(args.output).compile(query())
            head = load_file(str(args.output / "scalar_head.safetensors"))["head.weight"]
            actual = reference.independent_scores(loaded.model, head, prepared, token_budget=512)
            expected = reference.independent_scores(model.b, model.head.weight, prepared, token_budget=512)
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), atol=0, rtol=0)

    def test_split_shared_logical_positions_isolation_and_candidate_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, model, _ = fixture(Path(temporary))
            exporter.export_nextbook_teacher(args)
            prepared = NextBookTeacherCompiler(args.output).compile(query(["b", "a", "h"]))
            budget = len(prepared.query_ids) + 1 + max(map(len, prepared.item_ids)) + 1
            batches = prepared.shared_batches(budget)
            self.assertEqual(len(batches), 3)
            self.assertEqual([i for batch in batches for i in batch.candidate_indices], [0, 1, 2])
            self.assertEqual([item for batch in batches for item in batch.candidate_ids], ["b", "a", "h"])
            self.assertTrue(all(batch.packed_length <= budget and batch.full_input_hash == prepared.input_hash for batch in batches))
            singles = prepared.independent_batches()
            self.assertEqual(len(singles), 3)
            self.assertEqual(len({batch.input_hash for batch in singles}), 3)
            self.assertEqual([batch.candidate_indices for batch in singles], [(0,), (1,), (2,)])
            self.assertEqual([batch.score_kwargs()["items"] for batch in singles], [[list(row)] for row in prepared.item_ids])
            independently = reference.independent_scores(model.b, model.head.weight, prepared, token_budget=512)
            layout = PackedTreeLayout.from_queries([QueryLayout(len(prepared.query_ids), tuple(map(len, prepared.item_ids)))])
            ids = list(prepared.query_ids) + [token for row in prepared.item_ids for token in row]
            with torch.no_grad():
                shared = model.b(input_ids=torch.tensor([ids]), position_ids=layout.position_ids,
                                 attention_mask=layout.dense_attention_mask(), use_cache=False).last_hidden_state[0]
                values = torch.nn.functional.linear(shared[list(layout.score_indices[0])].float(), model.head.weight).flatten()
            torch.testing.assert_close(values, torch.tensor(independently), atol=2e-6, rtol=2e-6)
            for (start, end), length in zip(layout.branch_ranges[0], prepared.logical_lengths):
                self.assertEqual(layout.position_ids[0, start:end].tolist(), list(range(len(prepared.query_ids), length)))
            a, b = layout.branch_ranges[0][:2]
            self.assertFalse(layout.causal_allowed(b[1] - 1, a[1] - 1))
            # Mutating another candidate cannot influence the first score.
            changed = NextBookTeacherCompiler(args.output).compile(query(["b", "h"]))
            other = reference.independent_scores(model.b, model.head.weight, changed, token_budget=512)
            self.assertAlmostEqual(independently[0], other[0], places=6)
            with self.assertRaisesRegex(ValueError, "physical token budget"):
                prepared.shared_batches(budget - 1)
            with self.assertRaisesRegex(ValueError, "independent token budget"):
                prepared.independent_causal_batches(min(prepared.logical_lengths) - 1)

    def test_fp16_reference_keeps_fp32_head_and_rotary_and_padding_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, model, _ = fixture(Path(temporary))
            exporter.export_nextbook_teacher(args)
            prepared = NextBookTeacherCompiler(args.output).compile(query())
            expected = reference.independent_scores(model.b, model.head.weight, prepared, token_budget=512)
            original = model.b.rotary_emb.inv_freq.clone()
            reference.cast_half_preserve_rotary(model.b)
            self.assertEqual(model.b.rotary_emb.inv_freq.dtype, torch.float32)
            self.assertTrue(torch.equal(original, model.b.rotary_emb.inv_freq))
            self.assertEqual(model.head.weight.dtype, torch.float32)
            actual = reference.independent_scores(model.b, model.head.weight, prepared, token_budget=512)
            separately = reference.independent_scores(model.b, model.head.weight, prepared, token_budget=512, max_batch_size=1)
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(separately), atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), atol=.002, rtol=.002)

    def test_wrong_checkpoint_and_provenance_rejected_before_deserialization(self):
        for mutation in ("hash", "checkpoint", "proof", "source", "dev"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                args, _, _ = fixture(Path(temporary))
                if mutation == "hash":
                    args.expected_checkpoint_sha256 = "0" * 64
                elif mutation == "checkpoint":
                    with (args.checkpoint / "training.pt").open("ab") as stream:
                        stream.write(b"changed")
                elif mutation == "proof":
                    proof = json.loads(args.provenance.read_text())
                    proof["all_members_verified"] = False
                    write_json(args.provenance, proof)
                elif mutation == "source":
                    path = args.source / "src/duolm/next_item_model.py"
                    path.write_text(path.read_text() + "\n# changed\n")
                else:
                    (args.data / "dev.jsonl").write_text("changed")
                with patch.object(exporter.torch, "load") as loader, self.assertRaises(ValueError):
                    exporter.export_nextbook_teacher(args)
                loader.assert_not_called()
                self.assertFalse(args.output.exists())

    def test_five_class_or_student_state_rejected_even_with_rebound_receipts(self):
        for mutation in ("head", "student"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                args, _, _ = fixture(Path(temporary))
                path = args.checkpoint / "training.pt"
                payload = torch.load(path, weights_only=True)
                if mutation == "head":
                    payload["model"]["head.weight"] = torch.ones(5, 16)
                else:
                    payload["model"]["a.embed_tokens.weight"] = torch.zeros(40, 16)
                torch.save(payload, path)
                manifest_path = args.checkpoint / "manifest.json"
                receipt = json.loads(manifest_path.read_text())
                receipt["files"]["training.pt"] = exporter._record(path)
                write_json(manifest_path, receipt)
                proof = json.loads(args.provenance.read_text())
                proof["restored_files"] = {str(args.checkpoint / name): exporter._record(args.checkpoint / name)
                                           for name in ("training.pt", "manifest.json")}
                write_json(args.provenance, proof)
                args.expected_checkpoint_sha256 = exporter._hash(path)
                with self.assertRaisesRegex(ValueError, "scalar head|exact B"):
                    exporter.export_nextbook_teacher(args)
                self.assertFalse(args.output.exists())

    def test_no_clobber_inventory_tamper_and_v3_rejection(self):
        from duolm.nextbook_export import verify_nextbook_export
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _ = fixture(Path(temporary))
            exporter.export_nextbook_teacher(args)
            with self.assertRaisesRegex(ValueError, "output exists"):
                exporter.export_nextbook_teacher(args)
            with self.assertRaisesRegex(ValueError, "unsupported next-book export"):
                verify_nextbook_export(args.output)
            (args.output / "train.jsonl").write_text("private row")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                exporter.verify_nextbook_teacher_export(args.output)

    def test_compiler_rejects_invalid_queries_and_explicit_logical_overflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _ = fixture(Path(temporary))
            exporter.export_nextbook_teacher(args)
            compiler = NextBookTeacherCompiler(args.output)
            for value in ({**query(), "extra": True}, query(["missing"]), query(["a", "a"]),
                          {**query(), "history": [{"item_id": "h", "rating": True}]},
                          {**query(), "history": [{"item_id": "h", "rating": 5}] * 11}):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    compiler.compile(value)
            compiler.logical_limit = 1
            with self.assertRaisesRegex(ValueError, "logical position limit"):
                compiler.compile(query())

    def test_bounded_text_token_cache_warm_requests_do_not_retokenize(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, model, _ = fixture(Path(temporary))
            exporter.export_nextbook_teacher(args)
            compiler = NextBookTeacherCompiler(args.output, tokenizer=model.tokenizer_b, token_cache_capacity=3)
            cold = compiler.compile(query())
            before = compiler.token_cache_report()
            with patch.object(compiler._tokenizer, "encode", side_effect=AssertionError("warm item was retokenized")):
                warm = compiler.compile(query())
            self.assertEqual(warm, cold)
            after = compiler.token_cache_report()
            self.assertEqual(after["misses"], before["misses"])
            self.assertEqual(after["hits"] - before["hits"], 3)
            bounded = NextBookTeacherCompiler(args.output, token_cache_capacity=2)
            bounded.compile(query())
            self.assertEqual(bounded.token_cache_report()["entries"], 2)

    def test_pretokenized_catalog_matches_lazy_inputs_with_no_eviction(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, model, items = fixture(Path(temporary))
            exporter.export_nextbook_teacher(args)
            lazy = NextBookTeacherCompiler(args.output, tokenizer=model.tokenizer_b, token_cache_capacity=1)
            queries = [query(["b", "a", "h"]), query(["h", "b"]), query(["a", "h", "b"])]
            expected = [lazy.compile(value) for value in queries]
            self.assertEqual(lazy.token_cache_report()["policy"], "bounded_lru")
            for batched in (True, False):
                tokenizer = model.tokenizer_b if batched else SimpleNamespace(encode=model.tokenizer_b.encode)
                compiler = NextBookTeacherCompiler(args.output, tokenizer=tokenizer, token_cache_capacity=1,
                                                   pretokenize_catalog=True)
                report = compiler.token_cache_report()
                self.assertEqual(report["policy"], "pretokenized_catalog")
                self.assertEqual(report["startup_tokenization"], "batch_encode_plus" if batched else "encode")
                self.assertEqual(report["capacity"], len(items))
                self.assertEqual(report["entries"], len(items))
                self.assertEqual(report["eviction"], "disabled")
                with self.assertRaises(TypeError):
                    compiler._token_cache["new"] = (1,)
                with patch.object(tokenizer, "encode", side_effect=AssertionError("catalog item retokenized")):
                    actual = [compiler.compile(value) for value in queries]
                    self.assertEqual(actual, expected)
                    self.assertEqual([row.input_hash for row in actual], [row.input_hash for row in expected])
                after = compiler.token_cache_report()
                self.assertEqual(after["entries"], len(items))
                self.assertEqual(after["misses"], 0)
                self.assertEqual(after["hits"], sum(len(value["history"]) + len(value["candidate_ids"]) for value in queries))

    def test_pretokenized_batch_validates_vocabulary_and_disables_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, model, items = fixture(Path(temporary))
            exporter.export_nextbook_teacher(args)
            with patch.object(model.tokenizer_b, "batch_encode_plus", wraps=model.tokenizer_b.batch_encode_plus) as encode:
                NextBookTeacherCompiler(args.output, tokenizer=model.tokenizer_b, pretokenize_catalog=True)
                encode.assert_called_once_with(list(items.values()), add_special_tokens=False, padding=False,
                                               truncation=False, return_attention_mask=False, return_token_type_ids=False)
            for invalid in ([], [True], [-1], [model.b.config.vocab_size]):
                with self.subTest(invalid=invalid), patch.object(model.tokenizer_b, "batch_encode_plus",
                        return_value={"input_ids": [invalid] * len(items)}), self.assertRaisesRegex(ValueError, "native vocabulary"):
                    NextBookTeacherCompiler(args.output, tokenizer=model.tokenizer_b, pretokenize_catalog=True)


def reference_workload(args, groups=3):
    requests = []
    for index in range(groups):
        value = {**query(), "query_id": "reference-" + str(index)}
        requests.append({"request_id": reference.digest(value), "query": value, "episodes": value["candidate_ids"]})
    workload = {"protocol": "mixlm-nextbook-capacity-v1", "dev_sha256": exporter._hash(args.data / "dev.jsonl"),
                "users": groups, "history": 1, "candidate_counts": [2], "seed": 17,
                "label_dependent_selection": False, "requests": requests}
    workload["workload_sha256"] = reference.digest(workload)
    path = args.output.parent / "workload.json"
    write_json(path, workload)
    return workload, SimpleNamespace(export=args.output, workload=path, output=args.output.parent / "reference.json",
                                     device="cpu", token_budget=512, batch_size=2, fp32_groups=1)


class TeacherReferenceCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_candidate_count_and_fp32_selection_preserve_parent_order(self):
        rows = [{"request_id": str(index), "query": {"candidate_ids": list(range(count))}}
                for index, count in enumerate((10, 50, 100, 10, 50, 100))]
        workload = {"requests": rows}
        selected, diagnostic = reference.select_requests(workload, 100, 1)
        self.assertEqual([row["request_id"] for row in selected], ["2", "5"])
        self.assertEqual([row["request_id"] for row in diagnostic], ["2"])
        self.assertEqual(reference.select_requests(workload), (rows, rows))
        self.assertEqual(reference.select_requests(workload, 100, 0), (selected, []))
        for count, fp32 in ((True, 1), (2, None), (100, -1), (100, 3), (100, True)):
            with self.subTest(count=count, fp32=fp32), self.assertRaises(ValueError):
                reference.select_requests(workload, count, fp32)

    def test_fp16_first_partial_fp32_coverage_reloads_original_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, trained, _ = fixture(Path(temporary))
            exporter.export_nextbook_teacher(args)
            workload, build_args = reference_workload(args)
            phases = []
            original_score = reference.independent_scores
            def score(model, *positional, **keywords):
                phases.append(next(model.parameters()).dtype)
                return original_score(model, *positional, **keywords)
            with patch.object(reference, "independent_scores", side_effect=score), \
                    patch.object(reference, "_load_reader", wraps=reference._load_reader) as load:
                result = reference.build(build_args)
            self.assertEqual(load.call_count, 2)
            self.assertEqual(phases, [torch.float16] * 3 + [torch.float32])
            ids = [row["request_id"] for row in workload["requests"]]
            self.assertEqual(result["workload_sha256"], workload["workload_sha256"])
            self.assertEqual(result["reference_request_ids"], ids)
            self.assertEqual(result["fp32_reference_request_ids"], ids[:1])
            self.assertEqual(result["fp32_coverage"], {"requested_groups": 1, "selected_groups": 3,
                                                      "diagnostic_groups": 1, "selection": "fixed_first_n"})
            self.assertTrue(all("fp16" in result["references"][item] for item in ids))
            self.assertTrue(all("fp32" not in result["references"][item] and "fp16_vs_fp32" not in result["references"][item]
                                for item in ids[1:]))
            prepared = NextBookTeacherCompiler(args.output).compile(workload["requests"][0]["query"])
            expected = original_score(trained.b, trained.head.weight, prepared, token_budget=512)
            self.assertEqual(result["references"][ids[0]]["fp32"], expected)
            progress = json.loads(build_args.output.with_name("reference.progress.json").read_text())
            self.assertTrue(progress["complete"])
            self.assertEqual(progress["completed_request_ids"], {"fp16": ids, "fp32": ids[:1]})
            self.assertEqual(progress["progress_sha256"], reference.digest({key: value for key, value in progress.items()
                                                                          if key != "progress_sha256"}))
            build_args.output.unlink()
            with patch.object(reference, "_load_reader") as load, self.assertRaisesRegex(ValueError, "not resumed"):
                reference.build(build_args)
            load.assert_not_called()

    def test_interruption_preserves_atomic_ten_group_fp16_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _ = fixture(Path(temporary))
            exporter.export_nextbook_teacher(args)
            workload, build_args = reference_workload(args, groups=11)
            build_args.fp32_groups = 0
            original_score, calls = reference.independent_scores, []
            def interrupt(model, *positional, **keywords):
                calls.append(next(model.parameters()).dtype)
                if len(calls) == 11:
                    raise RuntimeError("simulated interruption")
                return original_score(model, *positional, **keywords)
            with patch.object(reference, "independent_scores", side_effect=interrupt), \
                    self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                reference.build(build_args)
            self.assertFalse(build_args.output.exists())
            progress_path = build_args.output.with_name("reference.progress.json")
            progress = json.loads(progress_path.read_text())
            ids = [row["request_id"] for row in workload["requests"][:10]]
            self.assertFalse(progress["complete"])
            self.assertEqual(progress["completed_request_ids"], {"fp16": ids, "fp32": []})
            self.assertEqual(set(progress["references"]), set(ids))
            self.assertEqual(progress["fp32_reference_request_ids"], [])
            self.assertEqual(progress["script_sha256"], reference.digest(Path(reference.__file__).read_text()))
            original = progress_path.read_bytes()
            with patch.object(reference.os, "replace", side_effect=OSError("publish failed")), self.assertRaises(OSError):
                reference._write_progress(progress_path, {"changed": True})
            self.assertEqual(progress_path.read_bytes(), original)
            self.assertEqual(list(progress_path.parent.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
