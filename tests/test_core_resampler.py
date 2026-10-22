"""Tiny offline tests for masked Perceiver readout and calibrated RMS scaling."""

import copy
import unittest

import torch

from duolm.resampler import CalibratedRMSBridge, PerceiverResampler


class ResamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_readout(self):
        torch.manual_seed(7)
        return PerceiverResampler(16, max_latents=20, layers=2, heads=8)

    def test_shapes_padding_and_dynamic_count_match_separate_calls(self):
        readout = self.make_readout()
        hidden = torch.randn(2, 11, 16)
        mask = torch.tensor([[1] * 5 + [0] * 6, [1] * 11], dtype=torch.bool)
        batched = readout(hidden, mask, [3, 8])
        self.assertEqual(tuple(batched.shape), (2, 8, 16))
        self.assertTrue(torch.equal(batched[0, 3:], torch.zeros_like(batched[0, 3:])))
        for row, (length, count) in enumerate(((5, 3), (11, 8))):
            single = readout(hidden[row:row+1, :length], mask[row:row+1, :length], [count])
            torch.testing.assert_close(batched[row, :count], single[0], atol=1e-6, rtol=1e-5)
        perturbed = hidden.clone()
        perturbed[0, 5:] = torch.randn_like(perturbed[0, 5:]) * 100000
        torch.testing.assert_close(readout(perturbed, mask, [3, 8]), batched)

    def test_every_valid_encoder_token_and_query_gets_gradient_no_padding_gradient(self):
        readout = self.make_readout()
        hidden = torch.randn(2, 11, 16, requires_grad=True)
        mask = torch.tensor([[1] * 5 + [0] * 6, [1] * 11], dtype=torch.bool)
        output = readout(hidden, mask, [3, 8])
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue((hidden.grad.abs().sum(-1)[mask] > 0).all())
        self.assertEqual(hidden.grad[~mask].abs().sum().item(), 0)
        self.assertTrue((readout.queries.grad[:8].abs().sum(-1) > 0).all())
        self.assertEqual(readout.queries.grad[8:].abs().sum().item(), 0)
        for layer in readout.layers:
            self.assertEqual(layer.attention.dropout, 0.0)
            self.assertTrue(layer.attention.batch_first)
            self.assertGreater(layer.attention.in_proj_weight.grad.abs().sum().item(), 0)

    def test_counts_change_latent_communication_but_never_expose_unused_queries(self):
        readout = self.make_readout()
        hidden = torch.randn(1, 7, 16)
        mask = torch.ones(1, 7, dtype=torch.bool)
        expected = readout(hidden, mask, [3]).detach()
        with torch.no_grad():
            readout.queries[3:].fill_(1000)
        torch.testing.assert_close(readout(hidden, mask, [3]), expected)

    def test_bf16_autocast_retains_fp32_master_gradients(self):
        readout = self.make_readout()
        hidden = torch.randn(2, 7, 16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = readout(hidden, torch.ones(2, 7, dtype=torch.bool), [8, 4])
        result.float().square().mean().backward()
        self.assertTrue(all(p.dtype == torch.float32 for p in readout.parameters()))
        self.assertTrue(all(p.grad is not None and p.grad.dtype == torch.float32 for p in readout.parameters()))
        self.assertGreater(hidden.grad.abs().sum().item(), 0)

    def test_invalid_counts_masks_and_heads_are_rejected(self):
        with self.assertRaises(ValueError):
            PerceiverResampler(16, 8, heads=3)
        readout = self.make_readout()
        hidden = torch.randn(2, 5, 16)
        mask = torch.ones(2, 5, dtype=torch.bool)
        for counts in ([0, 1], [1, 21], [1], [True, 1], [1.0, 2]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                readout(hidden, mask, counts)
        mask[1] = False
        with self.assertRaisesRegex(ValueError, "valid token"):
            readout(hidden, mask, [2, 2])


class BridgeTests(unittest.TestCase):
    def test_explicit_reference_gain_and_gradient(self):
        bridge = CalibratedRMSBridge()
        states = torch.randn(3, 8, 16, requires_grad=True) * 3
        states.retain_grad()
        with self.assertRaisesRegex(RuntimeError, "calibrated"):
            bridge(states)
        bridge.set_reference(0.025, {"source": "test_train_only"})
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = bridge(states)
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output.square().mean(-1).sqrt(), torch.full((3, 8), 0.025), atol=1e-7, rtol=1e-5)
        before = output.detach()
        with torch.no_grad():
            bridge.log_gain.fill_(torch.log(torch.tensor(2.0)))
        torch.testing.assert_close(bridge(states), before * 2)
        bridge(states).sum().backward()
        self.assertGreater(states.grad.abs().sum().item(), 0)
        self.assertGreater(bridge.log_gain.grad.abs().sum().item(), 0)
        self.assertFalse(bridge.rms_reference.requires_grad)
        self.assertIsNone(bridge.rms_reference.grad)

    def test_calibration_state_roundtrip_is_exact_and_metadata_copied(self):
        bridge = CalibratedRMSBridge()
        metadata = {"items": ["one", "two"]}
        bridge.set_reference(0.03, metadata)
        metadata["items"].clear()
        report = bridge.metadata()
        self.assertEqual(report["calibration"]["items"], ["one", "two"])
        report["calibration"]["items"].clear()
        restored = CalibratedRMSBridge()
        restored.load_state_dict(copy.deepcopy(bridge.state_dict()))
        self.assertEqual(restored.metadata(), bridge.metadata())
        states = torch.randn(8, 16)
        torch.testing.assert_close(restored(states), bridge(states), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "fixed"):
            restored.set_reference(0.04, {})
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                CalibratedRMSBridge().set_reference(invalid, {})


if __name__ == "__main__":
    unittest.main()
