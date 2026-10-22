"""Offline real-Qwen proofs for separate, portable weights-only artifacts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from duolm import inference_checkpoint as portable, train_next_item as trainer
from duolm.next_item_model import GroupRanker, domain_template


class InferenceCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.public = {}
        self.metadata = {}
        definition = Qwen3Config(vocab_size=64, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, max_position_embeddings=2048, attention_dropout=0, use_cache=False)
        for name in ('encoder', 'reader'):
            folder = self.root / name
            folder.mkdir()
            definition.save_pretrained(folder)
            vocab = {word: i for i, word in enumerate(['[UNK]', '[EOS]', 'a', 'book', 'game',
                'history', 'first', 'second', 'third', 'interesting', 'puzzle', 'novel'])}
            raw = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
            raw.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', eos_token='[EOS]')
            tokenizer.save_pretrained(folder)
            self.public[name] = {'repo': 'fixtures/' + name, 'revision': 'a' * 40,
                'files': {path.name: portable._hash(path) for path in folder.iterdir()}}
            self.metadata[name] = folder
        self.asset_patch = patch.object(portable, '_ASSETS', self.public)
        self.asset_patch.start()
        self.addCleanup(self.asset_patch.stop)
        self.query = SimpleNamespace(history=(SimpleNamespace(item_id='h', rating=5),),
                                     candidate_ids=('c1', 'c2'), positive_index=0)
        self.items = {'h': 'interesting book', 'c1': 'first novel', 'c2': 'second puzzle'}

    def fixture(self, domain='book', variant='V3'):
        torch.manual_seed(233)
        def backbone(side):
            definition = Qwen3Config.from_pretrained(self.metadata[side], local_files_only=True)
            definition._attn_implementation = 'sdpa'
            return Qwen3Model(definition)
        def tokenizer(side):
            return PreTrainedTokenizerFast.from_pretrained(self.metadata[side], local_files_only=True)
        options = {'variant': variant, 'm': 1, 'a_readout': 'last_token' if variant == 'V3' else 'perceiver',
            'shared_candidate_header': variant == 'V3', 'score_readout': 'candidate' if variant == 'V3' else 'legacy',
            'resampler_layers': 2, 'resampler_heads': 2, 'input_assembly': 'legacy'}
        limits = dict(max_reader_packed_tokens=1024, max_encoder_packed_tokens=1024, max_logical_tokens=1024)
        model = GroupRanker(backbone('encoder') if variant == 'V3' else None, backbone('reader'),
            tokenizer('encoder') if variant == 'V3' else None, tokenizer('reader'), domain=domain, **options, **limits)
        if variant == 'V3':
            model.calibrate_bridge([self.query], self.items)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        result = model.loss([self.query], self.items)
        result['loss'].backward()
        optimizer.step()
        model.eval()
        cfg = {'schema_version': 'games-candidate-ranking-v1' if domain == 'game' else 'books-candidate-ranking-v1',
            'model': {**options, 'b_model': {**{k: self.public['reader'][k] for k in ('repo', 'revision')},
                'local_path': '/private/must-never-export/model'}},
            'training': limits, 'loss': {'teacher_cache': '/private/must-never-export/cache'},
            'evaluation': {'compute_dtype': 'bfloat16' if domain == 'game' else 'float32'}}
        if variant == 'V3':
            cfg['model']['a_model'] = {k: self.public['encoder'][k] for k in ('repo', 'revision')}
        if domain == 'game':
            cfg['model']['domain'] = domain
        identity = {'config_sha256': portable._digest(cfg), 'data_sha256': {'train': 'b' * 64},
                    'source_sha256': {'next_item_model.py': 'c' * 64},
                    'teacher': {'private-cache-receipt': '/private/must-never-export/cache'}}
        if variant == 'V3':
            identity['student_template'] = model.metadata()['student_template']
        if domain == 'game':
            prompt = domain_template('game')
            identity.update(domain=domain, prompt_template={**prompt, 'sha256': portable._digest(prompt)})
        run = self.root / (domain + variant)
        run.mkdir()
        cfg_path = run / 'config.json'
        portable._write_json(cfg_path, cfg)
        checkpoint = trainer.save_checkpoint(run, model, optimizer, {'step': 1, 'cursor': 1}, identity)
        return model, checkpoint, cfg_path

    def export(self, checkpoint, cfg, output='artifact', **kwargs):
        config = portable._json(cfg)
        options = dict(b_model=self.metadata['reader'])
        if config['model']['variant'] == 'V3':
            options['a_model'] = self.metadata['encoder']
        options.update(kwargs)
        path = self.root / output
        portable.export_checkpoint(checkpoint, cfg, path, **options)
        return path

    def resign(self, path, *changed):
        manifest = portable._json(path / 'manifest.json')
        for name in changed:
            manifest['files'][name] = portable._record(path / name)
        manifest.pop('manifest_sha256')
        manifest['manifest_sha256'] = portable._digest(manifest)
        portable._write_json(path / 'manifest.json', manifest)

    def test_books_and_games_teacher_student_exact_state_scores_loss_and_offline_load(self):
        for domain in ('book', 'game'):
            for variant in ('T', 'V3'):
                with self.subTest(domain=domain, variant=variant):
                    model, checkpoint, cfg = self.fixture(domain, variant)
                    path = self.export(checkpoint, cfg, domain + variant + '-artifact')
                    with patch('transformers.AutoModel.from_pretrained', side_effect=AssertionError('no base weight downloads')):
                        loaded = portable.load_ranker(path)
                    self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
                    self.assertFalse(any(t.is_meta for t in loaded.buffers()))
                    with torch.no_grad():
                        expected = model.loss([self.query], self.items)
                        actual = loaded.loss([self.query], self.items)
                    for left, right in zip(expected['scores'], actual['scores']):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                    torch.testing.assert_close(expected['loss'], actual['loss'], rtol=0, atol=0)
                    for key, value in model.state_dict().items():
                        if isinstance(value, torch.Tensor):
                            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
                        else:
                            self.assertEqual(value, loaded.state_dict()[key])
                    text = ''.join(p.read_text() for p in path.rglob('*.json'))
                    self.assertNotIn('must-never-export', text)
                    self.assertNotIn('rng_python', text)
                    self.assertNotIn('private-cache-receipt', text)
                    self.assertFalse((path / 'training.pt').exists())
                    self.assertEqual(portable.verify_checkpoint(path)['architecture']['domain'], domain)
                    fresh_items = {**self.items, 'new': 'third game novel'}
                    fresh_query = SimpleNamespace(history=self.query.history, candidate_ids=('new', 'c1'))
                    with torch.no_grad():
                        torch.testing.assert_close(model([fresh_query], fresh_items)['scores'][0],
                            loaded([fresh_query], fresh_items)['scores'][0], rtol=0, atol=0)

    def test_corrupt_checksum_unexpected_file_symlink_and_transport_metadata(self):
        _, checkpoint, cfg = self.fixture()
        path = self.export(checkpoint, cfg)
        (path / '.gitattributes').write_text('*.safetensors filter=lfs diff=lfs merge=lfs -text\n')
        (path / '.cache' / 'huggingface').mkdir(parents=True)
        (path / '.cache' / 'huggingface' / 'receipt').write_text('download metadata')
        portable.verify_checkpoint(path)
        (path / 'optimizer.pt').write_text('private')
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            portable.verify_checkpoint(path)
        (path / 'optimizer.pt').unlink()
        target = path / 'a' / 'config.json'
        original = target.read_bytes()
        target.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            portable.verify_checkpoint(path)
        target.unlink()
        target.symlink_to(self.metadata['encoder'] / 'config.json')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            portable.verify_checkpoint(path)

    def test_shape_dtype_missing_nonfinite_and_uncalibrated_fail_after_rehash(self):
        _, checkpoint, cfg = self.fixture()
        path = self.export(checkpoint, cfg)
        original = load_file(str(path / 'model.safetensors'))
        for problem in ('shape', 'dtype', 'missing', 'nonfinite', 'uncalibrated'):
            with self.subTest(problem=problem):
                state = deepcopy(original)
                if problem == 'shape':
                    state['head.weight'] = state['head.weight'][:, :2]
                elif problem == 'dtype':
                    state['head.weight'] = state['head.weight'].bfloat16()
                elif problem == 'missing':
                    del state['head.weight']
                elif problem == 'nonfinite':
                    state['head.weight'][0, 0] = float('nan')
                else:
                    state['bridge.calibrated'].fill_(False)
                save_file(state, str(path / 'model.safetensors'))
                self.resign(path, 'model.safetensors')
                with self.assertRaises(ValueError):
                    portable.load_ranker(path)

    def test_metadata_template_and_provenance_allowlists(self):
        _, checkpoint, cfg = self.fixture()
        path = self.export(checkpoint, cfg)
        extra = portable._json(path / 'extra-state.json')
        extra['bridge._extra_state']['calibration']['raw_ids'] = ['private-user']
        portable._write_json(path / 'extra-state.json', extra)
        self.resign(path, 'extra-state.json')
        with self.assertRaisesRegex(ValueError, 'calibration'):
            portable.load_ranker(path)
        extra['bridge._extra_state']['calibration'].pop('raw_ids')
        portable._write_json(path / 'extra-state.json', extra)
        self.resign(path, 'extra-state.json')
        manifest = portable._json(path / 'manifest.json')
        manifest['template']['instruction'] = 'wrong prompt'
        portable._write_json(path / 'manifest.json', manifest)
        self.resign(path)
        with self.assertRaisesRegex(ValueError, 'template'):
            portable.load_ranker(path)
        provenance = portable._json(path / 'provenance.json')
        provenance['full_resume_supported'] = True
        portable._write_json(path / 'provenance.json', provenance)
        self.resign(path, 'provenance.json')
        with self.assertRaisesRegex(ValueError, 'full resume'):
            portable.load_ranker(path)

    def test_export_rejects_config_checkpoint_pin_and_metadata_changes_and_no_clobber(self):
        _, checkpoint, cfg = self.fixture()
        with self.assertRaisesRegex(ValueError, 'authorized checkpoint'):
            self.export(checkpoint, cfg, expected_checkpoint_sha256='0' * 64)
        original = cfg.read_bytes()
        config = portable._json(cfg)
        config['model']['m'] = 2
        portable._write_json(cfg, config)
        with self.assertRaisesRegex(ValueError, 'configuration hash'):
            self.export(checkpoint, cfg)
        cfg.write_bytes(original)
        path = self.export(checkpoint, cfg)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.export(checkpoint, cfg)
        metadata = self.metadata['reader'] / 'config.json'
        metadata.write_text(metadata.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'metadata checksum'):
            self.export(checkpoint, cfg, 'invalid')
        self.assertFalse((self.root / 'invalid').exists())
        self.assertFalse(list(self.root.glob('.duolm-export-*')))
        self.assertTrue(path.exists())

    def test_rehashed_tokenizer_configuration_must_still_match_public_pin(self):
        _, checkpoint, cfg = self.fixture()
        path = self.export(checkpoint, cfg)
        metadata = portable._json(path / 'a' / 'tokenizer_config.json')
        metadata['eos_token'] = 'wrong'
        portable._write_json(path / 'a' / 'tokenizer_config.json', metadata)
        self.resign(path, 'a/tokenizer_config.json')
        with self.assertRaisesRegex(ValueError, 'pinned public metadata'):
            portable.load_ranker(path)

    def test_original_template_mismatch_and_nonallowlisted_extra_state_rejected(self):
        _, checkpoint, cfg = self.fixture()
        payload_path = checkpoint / 'training.pt'
        original = torch.load(payload_path, weights_only=True)
        original_receipt = portable._json(checkpoint / 'manifest.json')
        for problem in ('template', 'extra_state'):
            with self.subTest(problem=problem):
                payload, receipt = deepcopy(original), deepcopy(original_receipt)
                if problem == 'template':
                    template = payload['identity']['student_template']
                    template['instruction'] = 'different semantics'
                    template.pop('sha256')
                    template['sha256'] = portable._digest(template)
                    receipt['identity'] = payload['identity']
                else:
                    payload['model']['bridge._extra_state']['calibration']['private'] = 'raw data'
                torch.save(payload, payload_path)
                receipt['files']['training.pt'] = portable._record(payload_path)
                portable._write_json(checkpoint / 'manifest.json', receipt)
                with self.assertRaisesRegex(ValueError, 'template|calibration'):
                    self.export(checkpoint, cfg, problem)
                self.assertFalse((self.root / problem).exists())

    def test_export_cli_includes_card_license_notice_and_verify(self):
        _, checkpoint, cfg = self.fixture(variant='T')
        docs = {}
        for name in ('README.md', 'LICENSE', 'NOTICE'):
            docs[name] = self.root / name
            docs[name].write_text('Public documentation ' + name)
        output = self.root / 'cli'
        with patch('sys.stdout'):
            portable.main(['export', '--checkpoint', str(checkpoint), '--config', str(cfg),
                '--output', str(output), '--b-model', str(self.metadata['reader']),
                '--model-card', str(docs['README.md']), '--license-file', str(docs['LICENSE']),
                '--notice-file', str(docs['NOTICE'])])
            portable.main(['verify', '--artifact', str(output)])
        self.assertTrue({'README.md', 'LICENSE', 'NOTICE'} <= portable._inventory(output))


if __name__ == '__main__':
    unittest.main()
