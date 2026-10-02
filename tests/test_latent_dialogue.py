import unittest
import tempfile
from pathlib import Path
import numpy as np

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Optional PyTorch dependencies are not installed")
from flypet.latent_dialogue import (
    DialogueReader,
    FocusedDialogueReader,
    CodecDialogueReader,
    prompt_parts,
)


class DialogueTests(unittest.TestCase):
    def test_codec_parameters_stay_frozen_and_readout_head_is_not_an_input(self):
        torch.manual_seed(19)
        model = CodecDialogueReader(
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
        )
        fine = torch.randn(2, 5, 9)
        pop = torch.randn(2, 3, 9)
        original = {k: v.clone() for k, v in model.codec.state_dict().items()}
        before = model(fine, pop)[0].detach()
        with torch.no_grad():
            model.codec.readout[-1].weight.add_(100)
        torch.testing.assert_close(model(fine, pop)[0].detach(), before)
        model.codec.load_state_dict(original)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        soft, _ = model(fine, pop)
        soft[:, -2:, :8].square().sum().backward()
        optimizer.step()
        for key, value in model.codec.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        self.assertGreater(float(model.codec_projection[0].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.codec.parameters()))

    def test_mbon_tokens_preserve_warm_start_and_use_only_declared_neurons(self):
        torch.manual_seed(14)
        base = DialogueReader(
            5, 9, [[0, 0, 0]] * 5, 3, 64, 0.02, tokens=4, width=16, auxiliary_dim=9
        ).eval()
        focused = FocusedDialogueReader(
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
            extra_mbon_tokens=4,
        ).eval()
        result = focused.load_state_dict(base.state_dict(), strict=False)
        self.assertFalse(result.unexpected_keys)
        fine = torch.randn(4, 5, 9)
        pop = torch.randn(4, 3, 9)
        before, _ = base(fine, pop)
        soft, aux, readout, mbon, latent = focused.forward_extended(fine, pop)
        torch.testing.assert_close(soft[:, :4], before)
        other = fine.clone()
        other[:, [0, 2, 4]] += 20
        changed, *_ = focused.forward_extended(other, pop)
        torch.testing.assert_close(changed[:, 4:], soft[:, 4:])
        self.assertGreater(float((changed[:, :4] - soft[:, :4]).detach().abs().sum()), 0)
        delta = focused.change_head(torch.cat((latent[:2], latent[2:]), 1))
        (readout.square().mean() + mbon.square().mean() + delta.square().mean()).backward()
        self.assertGreater(float(focused.mbon_queries.grad.abs().sum()), 0)
        self.assertGreater(float(focused.input[0].weight.grad.abs().sum()), 0)

    def test_population_and_fine_features_both_reach_latent(self):
        torch.manual_seed(2)
        model = DialogueReader(
            5, 9, [[0, 0, 0]] * 5, 3, 64, 0.02, tokens=4, width=16, auxiliary_dim=9
        )
        fine = torch.randn(2, 5, 9, requires_grad=True)
        pop = torch.randn(2, 3, 9, requires_grad=True)
        prefix, aux = model(fine, pop)
        self.assertEqual(prefix.shape, (2, 4, 64))
        self.assertEqual(aux.shape, (2, 9))
        (prefix[:, :, :8].sum() + aux.square().sum()).backward()
        self.assertGreater(fine.grad.abs().sum().item(), 0)
        self.assertGreater(pop.grad.abs().sum().item(), 0)

    def test_lora_and_neural_encoder_receive_answer_gradients_and_reload(self):
        from transformers import Qwen3Config, AutoModelForCausalLM
        from peft import (
            LoraConfig,
            get_peft_model,
            set_peft_model_state_dict,
            get_peft_model_state_dict,
        )

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
        base = AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
        model = get_peft_model(
            base,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        reader = DialogueReader(
            5, 9, [[0, 0, 0]] * 5, 3, 64, 0.02, tokens=4, width=16, auxiliary_dim=9
        )
        soft, _ = reader(torch.randn(4, 5, 9), torch.randn(4, 3, 9))
        embeddings = model.get_input_embeddings()
        vectors = []
        labels = []
        for j in range(2):
            tail = [7, 8, 9, 2] if j else [5, 6, 2]
            v = torch.cat(
                (
                    embeddings(torch.tensor([1, 3])),
                    soft[j],
                    embeddings(torch.tensor([4])),
                    soft[j + 2],
                    embeddings(torch.tensor(tail)),
                )
            )
            vectors.append(v)
            labels.append(torch.tensor([-100] * (len(v) - len(tail)) + tail))
        width = max(map(len, vectors))
        batch = embeddings(torch.zeros((2, width), dtype=torch.long))
        mask = torch.zeros((2, width), dtype=torch.long)
        target = torch.full((2, width), -100, dtype=torch.long)
        for j, (v, l) in enumerate(zip(vectors, labels)):
            batch[j, : len(v)] = v
            mask[j, : len(v)] = 1
            target[j, : len(v)] = l
        loss = model(inputs_embeds=batch, attention_mask=mask, labels=target, use_cache=False).loss
        loss.backward()
        self.assertGreater(reader.output[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for n, p in model.named_parameters()
                if "lora_" in n
            )
        )
        state = get_peft_model_state_dict(model)
        result = set_peft_model_state_dict(model, state)
        self.assertFalse(result.unexpected_keys)
        with torch.no_grad():
            generated = model.generate(
                inputs_embeds=batch[:, :8].detach(),
                attention_mask=torch.ones((2, 8), dtype=torch.long),
                max_new_tokens=2,
                do_sample=False,
                eos_token_id=2,
                pad_token_id=0,
            )
        self.assertEqual(generated.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
