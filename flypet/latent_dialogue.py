"""Rich neural token encoder and paired-observation prompt construction."""

import torch
from torch import nn
from .latent_bridge import NeuralReader


class DialogueReader(NeuralReader):
    def __init__(
        self,
        n_neurons,
        channels,
        metadata,
        n_populations,
        embedding_dim,
        embedding_std,
        tokens=16,
        width=128,
        auxiliary_dim=150,
    ):
        super().__init__(n_neurons, channels, metadata, embedding_dim, embedding_std, tokens, width)
        self.population_input = nn.Sequential(
            nn.Linear(channels, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.population_identity = nn.Embedding(n_populations, width)
        self.auxiliary = nn.Linear(tokens * width, auxiliary_dim)

    def forward(self, features, populations):
        h = self.input(features) + self.identity.weight[None]
        for i, layer in enumerate(self.categories):
            h = h + layer(self.metadata[:, i])[None]
        p = self.population_input(populations) + self.population_identity.weight[None]
        h = self.node_norm(torch.cat((h, p), dim=1))
        queries = self.queries[None].expand(len(features), -1, -1)
        pooled, _ = self.attention(queries, h, h, need_weights=False)
        latent = self.latent_norm(pooled + queries)
        return self.output(latent) * self.embedding_std, self.auxiliary(latent.flatten(1))


SYSTEM = (
    "You voice a simulated fruit fly through supplied neural observations. "
    "The reference and current neural states are the evidence. Describe neural responses, not subjective consciousness. "
    "Do not infer an unobserved training history. Answer the user briefly and concretely."
)

CORE_SYSTEM = SYSTEM + (
    " This interface supports current approach/avoidance responses and changes between observations. "
    "Do not claim reliable exact odor identification or precise temporal-bin reconstruction."
)
CURRENT_SYSTEM = (
    "You voice a simulated fruit fly through its supplied CURRENT neural observation. "
    "Describe the current approach/avoidance response, not a change over time or subjective consciousness. "
    "No reference observation or exposure history is supplied. Do not invent either. "
    "Answer briefly. Do not claim reliable exact odor identification or precise temporal-bin reconstruction."
)


def prompt_text_parts(tokenizer, question, system=SYSTEM):
    content = (
        "Reference neural observation:\n<<REFERENCE>>\nCurrent neural observation:\n<<CURRENT>>\n"
        + question
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    left, rest = rendered.split("<<REFERENCE>>", 1)
    middle, right = rest.split("<<CURRENT>>", 1)
    return left, middle, right


def prompt_parts(tokenizer, question, system=SYSTEM):
    return [
        list(tokenizer(s, add_special_tokens=False)["input_ids"])
        for s in prompt_text_parts(tokenizer, question, system)
    ]


def comparison_prompt_text_parts(tokenizer, question, system=CORE_SYSTEM):
    left, middle, right = prompt_text_parts(tokenizer, question, system)
    return left, middle, "\nNeural representation change:\n", right


def comparison_prompt_parts(tokenizer, question, system=CORE_SYSTEM):
    return [
        list(tokenizer(s, add_special_tokens=False)["input_ids"])
        for s in comparison_prompt_text_parts(tokenizer, question, system)
    ]


def current_prompt_parts(tokenizer, question):
    content = "Current neural observation:\n<<CURRENT>>\n" + question
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": CURRENT_SYSTEM}, {"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return [
        list(tokenizer(part, add_special_tokens=False)["input_ids"])
        for part in rendered.split("<<CURRENT>>", 1)
    ]


class FocusedDialogueReader(DialogueReader):
    """Warm-startable global tokens with optional extra MBON tokens.

    Readout/change/reconstruction heads supervise training only. Inference still
    passes continuous neural tokens, never their scalar predictions as text.
    """

    def __init__(self, *args, mbon_positions, extra_mbon_tokens=0, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.queries.shape[1]
        total = len(self.queries) + extra_mbon_tokens
        self.register_buffer("mbon_positions", torch.tensor(mbon_positions, dtype=torch.long))
        self.extra_mbon_tokens = extra_mbon_tokens
        self.mbon_queries = (
            nn.Parameter(torch.randn(extra_mbon_tokens, width) * 0.02)
            if extra_mbon_tokens
            else None
        )
        self.readout_head = nn.Linear(total * width, 3)
        self.change_head = nn.Sequential(
            nn.Linear(2 * total * width, width), nn.SiLU(), nn.Linear(width, 1)
        )
        self.mbon_head = (
            nn.Linear(extra_mbon_tokens * width, len(mbon_positions)) if extra_mbon_tokens else None
        )

    def forward_extended(self, features, populations):
        h = self.input(features) + self.identity.weight[None]
        for i, layer in enumerate(self.categories):
            h = h + layer(self.metadata[:, i])[None]
        p = self.population_input(populations) + self.population_identity.weight[None]
        h = self.node_norm(torch.cat((h, p), dim=1))
        q = self.queries[None].expand(len(features), -1, -1)
        pooled, _ = self.attention(q, h, h, need_weights=False)
        base = self.latent_norm(pooled + q)
        aux = self.auxiliary(base.flatten(1))
        mbon = None
        if self.extra_mbon_tokens:
            mq = self.mbon_queries[None].expand(len(features), -1, -1)
            mh = h[:, self.mbon_positions]
            mp, _ = self.attention(mq, mh, mh, need_weights=False)
            ml = self.latent_norm(mp + mq)
            mbon = self.mbon_head(ml.flatten(1))
            latent = torch.cat((base, ml), dim=1)
        else:
            latent = base
        flat = latent.flatten(1)
        return self.output(latent) * self.embedding_std, aux, self.readout_head(flat), mbon, flat

    def forward(self, features, populations):
        soft, aux, _, _, _ = self.forward_extended(features, populations)
        return soft, aux


class CodecDialogueReader(FocusedDialogueReader):
    """A frozen rich neural bottleneck with a trainable language projection."""

    def __init__(self, *args, codec_tokens=4, codec_width=128, extra_mbon_tokens=0, **kwargs):
        if extra_mbon_tokens:
            raise ValueError("Codec and attention MBON branches are separate variants")
        super().__init__(*args, extra_mbon_tokens=0, **kwargs)
        from .neural_codec import NeuralCodec

        self.codec = NeuralCodec(
            len(self.mbon_positions), self.input[0].in_features, codec_tokens, codec_width
        )
        self.codec.requires_grad_(False)
        dim = self.output[0].out_features
        self.codec_projection = nn.Sequential(nn.Linear(codec_width, dim), nn.LayerNorm(dim))
        hidden = self.codec.readout[0].out_features
        self.semantic_projection = nn.Sequential(nn.Linear(hidden, dim), nn.LayerNorm(dim))
        self.delta_projection = nn.Sequential(nn.Linear(hidden, dim), nn.LayerNorm(dim))

    def neural_hidden(self, features):
        detailed = self.codec.encode(features[:, self.mbon_positions])
        # The penultimate hidden representation, not the three readout values.
        hidden = self.codec.readout[1](self.codec.readout[0](detailed.flatten(1)))
        return detailed, hidden

    def pair_difference(self, joined_features):
        if len(joined_features) % 2:
            raise ValueError("Expected reference/current pairs")
        _, hidden = self.neural_hidden(joined_features)
        n = len(hidden) // 2
        return self.delta_projection(hidden[n:] - hidden[:n])[:, None] * self.embedding_std

    def forward_extended(self, features, populations):
        soft, aux, readout, mbon, latent = super().forward_extended(features, populations)
        detailed, hidden = self.neural_hidden(features)
        projected = self.codec_projection(detailed) * self.embedding_std
        semantic = self.semantic_projection(hidden)[:, None] * self.embedding_std
        return torch.cat((soft, projected, semantic), dim=1), aux, readout, mbon, latent
