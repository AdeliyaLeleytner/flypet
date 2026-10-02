import threading, unittest
import numpy as np

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Optional PyTorch dependencies are not installed")
from transformers import Qwen3Config, AutoModelForCausalLM
from flypet.latent_inference import LatentModels
from flypet.latent_dialogue import DialogueReader, CodecDialogueReader


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages) + "\nAssistant:"

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        out = {"input_ids": [ord(c) % 128 for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids, **kwargs):
        return "test response"


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 64)

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, inputs_embeds, **kwargs):
        self.recorded = inputs_embeds.detach().clone()
        return torch.tensor([[1]])


class FeatureTests(unittest.TestCase):
    def test_current_reading_does_not_receive_reference_or_difference(self):
        torch.manual_seed(23)
        rng = np.random.default_rng(23)
        model = LatentModels.__new__(LatentModels)
        model.device = torch.device("cpu")
        model.lock = threading.RLock()
        model.reader_config = {
            "kind": "paired_dialogue",
            "architecture": "codec",
            "separate_context": True,
        }
        model.tokenizer = CharacterTokenizer()
        model.llm = RecordingModel()
        model.reader = CodecDialogueReader(
            5,
            9,
            [[0, 0, 0]] * 5,
            3,
            64,
            0.02,
            tokens=4,
            width=16,
            auxiliary_dim=9,
            mbon_positions=[1, 3],
            codec_tokens=2,
            codec_width=16,
        ).eval()
        current = {
            "fine": rng.normal(size=(5, 9)).astype(np.float32),
            "population": rng.normal(size=(3, 9)).astype(np.float32),
        }
        reference = {k: v.copy() for k, v in current.items()}
        model.read(current, reference, "Current response?", mode="current")
        before = model.llm.recorded
        reference["fine"] *= 7
        model.read(current, reference, "Current response?", mode="current")
        torch.testing.assert_close(before, model.llm.recorded, rtol=0, atol=0)
        model.read(current, reference, "What changed?", mode="comparison")
        comparison = model.llm.recorded
        reference["fine"] *= 3
        model.read(current, reference, "What changed?", mode="comparison")
        self.assertGreater(float((comparison - model.llm.recorded).abs().sum()), 0)

    def test_codec_context_uses_hidden_states_and_difference_without_readout_values(self):
        torch.manual_seed(21)
        rng = np.random.default_rng(21)
        model = LatentModels.__new__(LatentModels)
        model.device = torch.device("cpu")
        model.lock = threading.RLock()
        model.reader_config = {"kind": "paired_dialogue", "architecture": "codec"}
        model.tokenizer = CharacterTokenizer()
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
        model.llm = AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
        model.reader = CodecDialogueReader(
            5,
            9,
            [[0, 0, 0]] * 5,
            3,
            64,
            0.02,
            tokens=4,
            width=16,
            auxiliary_dim=9,
            mbon_positions=[1, 3],
            codec_tokens=2,
            codec_width=16,
        ).eval()
        current = {
            "fine": rng.normal(size=(5, 9)).astype(np.float32),
            "population": rng.normal(size=(3, 9)).astype(np.float32),
        }
        reference = {k: v.copy() for k, v in current.items()}
        goal = "Bring the probe response close to +0.45."
        first = model.agent_features(current, reference, goal)
        with torch.no_grad():
            model.reader.codec.readout[-1].weight.add_(1000)
        np.testing.assert_array_equal(first, model.agent_features(current, reference, goal))
        reference["fine"] = rng.normal(size=(5, 9)).astype(np.float32) * 4
        self.assertGreater(
            float(np.linalg.norm(first - model.agent_features(current, reference, goal))), 1e-5
        )

    def test_neural_state_reaches_frozen_llm_policy_features(self):
        torch.manual_seed(18)
        rng = np.random.default_rng(18)
        model = LatentModels.__new__(LatentModels)
        model.device = torch.device("cpu")
        model.lock = threading.RLock()
        model.reader_config = {"kind": "paired_dialogue"}
        model.tokenizer = CharacterTokenizer()
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
        model.llm = AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
        model.reader = DialogueReader(
            5, 9, [[0, 0, 0]] * 5, 3, 64, 0.02, tokens=4, width=16, auxiliary_dim=9
        ).eval()
        reference = {
            "fine": rng.normal(size=(5, 9)).astype(np.float32),
            "population": rng.normal(size=(3, 9)).astype(np.float32),
        }
        current = {k: v.copy() for k, v in reference.items()}
        goal = "Bring the probe response close to +0.45."
        first = model.agent_features(current, reference, goal)
        current["measured_valence"] = -999
        np.testing.assert_array_equal(first, model.agent_features(current, reference, goal))
        current["fine"] = rng.normal(size=(5, 9)).astype(np.float32) * 3
        second = model.agent_features(current, reference, goal)
        self.assertEqual(first.shape, (128,))
        self.assertGreater(float(np.linalg.norm(first - second)), 1e-5)


if __name__ == "__main__":
    unittest.main()
