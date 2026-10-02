"""Inference-only latent language/brain bridges; no simulator or text planner."""

from pathlib import Path
import json
import threading
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from .latent_bridge import NeuralReader
from .latent_dialogue import (
    DialogueReader,
    FocusedDialogueReader,
    CodecDialogueReader,
    SYSTEM,
    prompt_parts,
    prompt_text_parts,
    comparison_prompt_parts,
    comparison_prompt_text_parts,
    current_prompt_parts,
)
from .latent_writer import hidden_features
from .neural_records import verify_bundle


class LatentModels:
    def __init__(self, directory, device=None):
        self.directory = Path(directory)
        self.manifest = verify_bundle(self.directory)
        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        dtype = torch.bfloat16 if self.device.type != "cpu" else torch.float32
        if self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            dtype = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.manifest["model"], revision=self.manifest["revision"]
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.llm = (
            AutoModelForCausalLM.from_pretrained(
                self.manifest["model"],
                revision=self.manifest["revision"],
                dtype=dtype,
                attn_implementation="sdpa",
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self.lock = threading.RLock()
        self.writer_config = (
            json.loads((self.directory / "writer_config.json").read_text())
            if (self.directory / "writer_config.json").exists()
            else None
        )
        if self.writer_config:
            with np.load(self.directory / "writer_preprocessing.npz", allow_pickle=False) as z:
                self.writer_preprocessing = {k: z[k] for k in z.files}
            state = load_file(str(self.directory / "writer.safetensors"), device="cpu")
            if self.writer_config["kind"] == "ridge":
                self.writer_weight = state["weight"].numpy()
                self.writer_bias = state["bias"].numpy()
            else:
                from torch import nn

                w = self.writer_config
                self.writer = nn.Sequential(
                    nn.Linear(w["dim"], w["hidden"]),
                    nn.SiLU(),
                    nn.Linear(w["hidden"], len(w["channels"])),
                )
                self.writer.load_state_dict({k.removeprefix("net."): v for k, v in state.items()})
                self.writer.to(self.device).eval()
            if self.writer_config.get("dopamine_head"):
                from torch import nn

                h = self.writer_config["dopamine_head"]
                self.dopamine_head = nn.Sequential(
                    nn.Linear(h["dim"], h["hidden"]),
                    nn.SiLU(),
                    nn.Linear(h["hidden"], h["outputs"]),
                )
                self.dopamine_head.load_state_dict(
                    load_file(str(self.directory / "dopamine_head.safetensors"))
                )
                self.dopamine_head.to(self.device).eval()
        self.reader_config = (
            json.loads((self.directory / "reader_config.json").read_text())
            if (self.directory / "reader_config.json").exists()
            else None
        )
        if self.reader_config:
            r = self.reader_config
            kwargs = {
                k: r[k]
                for k in [
                    "n_neurons",
                    "channels",
                    "metadata",
                    "embedding_dim",
                    "embedding_std",
                    "tokens",
                    "width",
                ]
            }
            if r["kind"] == "paired_dialogue":
                kwargs.update(n_populations=r["n_populations"], auxiliary_dim=r["auxiliary_dim"])
                if r.get("architecture") in ("focused", "codec"):
                    kwargs.update(
                        mbon_positions=r["mbon_positions"], extra_mbon_tokens=r["extra_mbon_tokens"]
                    )
                    if r["architecture"] == "codec":
                        kwargs.update(codec_tokens=r["codec_tokens"], codec_width=r["codec_width"])
                        self.reader = CodecDialogueReader(**kwargs)
                    else:
                        self.reader = FocusedDialogueReader(**kwargs)
                else:
                    self.reader = DialogueReader(**kwargs)
            else:
                self.reader = NeuralReader(**kwargs)
            self.reader.load_state_dict(
                load_file(str(self.directory / "reader.safetensors"), device="cpu")
            )
            self.reader.to(self.device).eval()
            with np.load(self.directory / "reader_preprocessing.npz", allow_pickle=False) as z:
                self.reader_preprocessing = {k: z[k] for k in z.files}
            if r.get("lora_directory"):
                from peft import PeftModel

                self.llm = PeftModel.from_pretrained(
                    self.llm, self.directory / r["lora_directory"]
                ).eval()

    @torch.inference_mode()
    def write(self, text):
        if not self.writer_config:
            raise RuntimeError("Writer not loaded")
        with self.lock:
            x = (
                hidden_features(self.llm, self.tokenizer, [text], self.writer_config["pooling"])
                .cpu()
                .numpy()
            )
            p = self.writer_preprocessing
            x = np.clip((x - p["mean"]) / p["std"], -10, 10).astype(np.float32)
            if self.writer_config["kind"] == "ridge":
                channels = x @ self.writer_weight + self.writer_bias
            else:
                channels = self.writer(torch.tensor(x, device=self.device)).cpu().numpy()
            channels = np.clip(channels[0], 0, 1) * self.writer_config["max_rate_hz"]
            if self.writer_config.get("dopamine_head"):
                channels[-2:] = (
                    self.dopamine_head(torch.tensor(x, device=self.device))
                    .sigmoid()[0]
                    .cpu()
                    .numpy()
                    * self.writer_config["dopamine_head"]["max_rate_hz"]
                )
            drive = p["basis"] @ channels
            return {
                "drive_hz": drive.astype(np.float64),
                "channel_rates_hz": channels,
                "input_root_ids": p["input_root_ids"].copy(),
            }

    @torch.inference_mode()
    def agent_features(self, current, reference, goal):
        """Contextual task-token states after attention to both neural prefixes.

        No decoded reply, numeric readout, stimulus recipe or memory weights are
        provided to the action policy. The text contains the agent's goal only.
        """
        if self.reader_config["kind"] != "paired_dialogue":
            raise ValueError("Agent requires the paired-state reader")
        with self.lock:
            emb = self.llm.get_input_embeddings()
            fine = torch.tensor(np.stack([reference["fine"], current["fine"]]), device=self.device)
            pop = torch.tensor(
                np.stack([reference["population"], current["population"]]), device=self.device
            )
            soft, _ = self.reader(fine, pop)
            codec = self.reader_config.get("architecture") == "codec"
            formatter = comparison_prompt_text_parts if codec else prompt_text_parts
            texts = formatter(self.tokenizer, goal, self.reader_config.get("system", SYSTEM))
            right = texts[-1]
            ids = [self.tokenizer(s, add_special_tokens=False)["input_ids"] for s in texts]
            pieces = [
                emb(torch.tensor(ids[0], device=self.device)),
                soft[0].to(emb.weight.dtype),
                emb(torch.tensor(ids[1], device=self.device)),
                soft[1].to(emb.weight.dtype),
            ]
            if codec:
                pieces.extend(
                    [
                        emb(torch.tensor(ids[2], device=self.device)),
                        self.reader.pair_difference(fine)[0].to(emb.weight.dtype),
                    ]
                )
            prefix = torch.cat(pieces)
            vector = torch.cat((prefix, emb(torch.tensor(ids[-1], device=self.device))))[None]
            start = right.index(goal)
            end = start + len(goal)
            offsets = self.tokenizer(right, add_special_tokens=False, return_offsets_mapping=True)[
                "offset_mapping"
            ]
            selected = torch.tensor(
                [len(prefix) + i for i, (a, b) in enumerate(offsets) if b > start and a < end],
                device=self.device,
            )
            output = self.llm(
                inputs_embeds=vector,
                attention_mask=torch.ones(vector.shape[:2], device=self.device, dtype=torch.long),
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = output.hidden_states
            return (
                torch.cat(
                    [
                        hidden[layer][0, selected].float().mean(0)
                        for layer in (len(hidden) // 2, len(hidden) - 1)
                    ]
                )
                .cpu()
                .numpy()
            )

    @torch.inference_mode()
    def read(
        self,
        current,
        reference=None,
        question="Describe your current neural response in a short sentence.",
        max_new_tokens=100,
        mode="current",
    ):
        if not self.reader_config:
            raise RuntimeError("Reader not loaded")
        if mode not in ("current", "comparison"):
            raise ValueError("Unknown reader context mode")
        reference = reference or current
        with self.lock:
            r = self.reader_config
            emb = self.llm.get_input_embeddings()
            if r.get("separate_context") and mode == "current":
                fine = torch.tensor(current["fine"][None], device=self.device)
                pop = torch.tensor(current["population"][None], device=self.device)
                soft, _ = self.reader(fine, pop)
                left, right = current_prompt_parts(self.tokenizer, question)
                vector = torch.cat(
                    (
                        emb(torch.tensor(left, device=self.device)),
                        soft[0].to(emb.weight.dtype),
                        emb(torch.tensor(right, device=self.device)),
                    )
                )[None]
            elif r["kind"] == "paired_dialogue":
                fine = torch.tensor(
                    np.stack([reference["fine"], current["fine"]]), device=self.device
                )
                pop = torch.tensor(
                    np.stack([reference["population"], current["population"]]), device=self.device
                )
                soft, _ = self.reader(fine, pop)
                codec = r.get("architecture") == "codec"
                formatter = comparison_prompt_parts if codec else prompt_parts
                parts = formatter(self.tokenizer, question, r.get("system", SYSTEM))
                pieces = [
                    emb(torch.tensor(parts[0], device=self.device)),
                    soft[0].to(emb.weight.dtype),
                    emb(torch.tensor(parts[1], device=self.device)),
                    soft[1].to(emb.weight.dtype),
                ]
                if codec:
                    pieces.extend(
                        [
                            emb(torch.tensor(parts[2], device=self.device)),
                            self.reader.pair_difference(fine)[0].to(emb.weight.dtype),
                        ]
                    )
                pieces.append(emb(torch.tensor(parts[-1], device=self.device)))
                vector = torch.cat(pieces)[None]
            else:
                soft, _ = self.reader(torch.tensor(current["fine"][None], device=self.device))
                rendered = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": r["question"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                ids = torch.tensor(
                    self.tokenizer(rendered, add_special_tokens=False)["input_ids"],
                    device=self.device,
                )[None]
                vector = torch.cat((soft.to(emb.weight.dtype), emb(ids)), dim=1)
            mask = torch.ones(vector.shape[:2], device=self.device, dtype=torch.long)
            output = self.llm.generate(
                inputs_embeds=vector,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
