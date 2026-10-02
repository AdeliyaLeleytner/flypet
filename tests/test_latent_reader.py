import unittest
import numpy as np

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Optional PyTorch dependencies are not installed")
from flypet.latent_bridge import NeuralReader
from flypet.latent_reader_data import read_phase, parse_reply, split_for


class LatentReaderTests(unittest.TestCase):
    def test_phase_cut_and_neuron_mask(self):
        arrays = {
            "phase_end_tick": np.array([10, 20]),
            "spike_tick": np.array([2, 3, 12, 13, 14]),
            "spike_neuron_index": np.array([0, 1, 0, 1, 2]),
            "voltage_mv": np.full((2, 3), -52, dtype=np.float32),
            "synaptic_mv": np.zeros((2, 3), dtype=np.float32),
            "refractory_remaining_ms": np.zeros((2, 3), dtype=np.float32),
        }
        a = read_phase(arrays, 0, np.array([1, 2]))
        self.assertEqual(a.shape, (2, 9))
        self.assertGreater(a[0, 5], 0)
        self.assertEqual(a[1, 5], 0)
        arrays["spike_tick"][2:] = 19
        arrays["voltage_mv"][1] = 100
        np.testing.assert_array_equal(a, read_phase(arrays, 0, np.array([1, 2])))

    def test_features_and_identity_reach_prefix_gradients(self):
        torch.manual_seed(4)
        model = NeuralReader(
            5,
            9,
            [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0]],
            64,
            0.02,
            tokens=4,
            width=16,
        )
        x = torch.randn(2, 5, 9, requires_grad=True)
        prefix, aux = model(x)
        self.assertEqual(prefix.shape, (2, 4, 64))
        self.assertEqual(aux.shape, (2, 3))
        self.assertFalse(torch.equal(prefix[0], prefix[1]))
        (prefix[:, :, :8].sum() + aux.square().sum()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_strict_numeric_schema(self):
        self.assertEqual(parse_reply('{"approach_hz":2,"avoid_hz":1,"valence":0.25}'), [2, 1, 0.25])
        self.assertIsNone(parse_reply('{"approach_hz":-1,"avoid_hz":1,"valence":0.25}'))
        self.assertIsNone(parse_reply('{"approach_hz":2,"avoid_hz":1,"valence":"0.25"}'))
        self.assertIsNone(parse_reply('{"approach_hz":2,"avoid_hz":1,"valence":2}'))

    def test_family_split_is_fixed(self):
        self.assertEqual(split_for("pair00-opposed"), "development_test")
        self.assertEqual(split_for("pair01-reward_a"), "validation")
        self.assertEqual(split_for("pair01-naive"), "train")

    def test_frozen_qwen_prefix_gradient_and_generation_api(self):
        from transformers import Qwen3Config, AutoModelForCausalLM

        torch.manual_seed(41)
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            eos_token_id=2,
            pad_token_id=0,
            bos_token_id=1,
        )
        llm = AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
        reader = NeuralReader(5, 9, [[0, 0, 0]] * 5, 64, 0.02, tokens=4, width=16)
        soft, _ = reader(torch.randn(2, 5, 9))
        ids = torch.tensor([[1, 4, 5, 6, 2], [1, 7, 8, 9, 2]])
        embeddings = torch.cat((soft, llm.get_input_embeddings()(ids)), dim=1)
        labels = torch.cat((torch.full((2, 4), -100), ids), dim=1)
        loss = llm(
            inputs_embeds=embeddings,
            attention_mask=torch.ones(labels.shape),
            labels=labels,
            use_cache=False,
        ).loss
        loss.backward()
        self.assertGreater(reader.output[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in llm.parameters()))
        with torch.no_grad():
            reply = llm.generate(
                inputs_embeds=embeddings[:, :6].detach(),
                attention_mask=torch.ones((2, 6)),
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=2,
            )
        self.assertEqual(reply.shape[0], 2)
        self.assertLessEqual(reply.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
