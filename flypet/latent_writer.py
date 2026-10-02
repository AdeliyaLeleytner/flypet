"""Contextual language representations for the learned neural actuator."""

from contextlib import nullcontext
import torch


@torch.inference_mode()
def hidden_features(model, tokenizer, texts, pooling="content_mean"):
    sentinel = "__FLYPET_USER_CONTENT_71a3__"
    skeleton = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix, suffix = skeleton.split(sentinel, 1)
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text in texts
    ]
    batch = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = batch.pop("offset_mapping")
    masks = []
    for text, prompt, mapping in zip(texts, rendered, offsets):
        start = len(prefix)
        end = start + len(text)
        if not text.strip() or prompt != prefix + text + suffix:
            raise ValueError("Unexpected chat-template content placement")
        mask = [int(int(b) > start and int(a) < end and int(b) > int(a)) for a, b in mapping]
        if not any(mask):
            raise ValueError("No user-content tokens")
        masks.append(mask)
    device = model.get_input_embeddings().weight.device
    batch = batch.to(device)
    batch["position_ids"] = (batch["attention_mask"].cumsum(-1) - 1).masked_fill(
        batch["attention_mask"] == 0, 0
    )
    # A conversational reader LoRA must not move the writer's fixed input space.
    context = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    with context:
        result = model(**batch, output_hidden_states=True, use_cache=False)
    middle = model.config.num_hidden_layers // 2
    if pooling == "last":
        values = [result.hidden_states[middle][:, -1], result.hidden_states[-1][:, -1]]
    elif pooling == "content_mean":
        mask = torch.tensor(masks, device=device, dtype=torch.float32)
        values = [
            (result.hidden_states[layer].float() * mask[:, :, None]).sum(dim=1)
            / mask.sum(dim=1, keepdim=True)
            for layer in (middle, -1)
        ]
    else:
        raise ValueError("Unknown writer pooling")
    return torch.cat(values, dim=-1).float()
