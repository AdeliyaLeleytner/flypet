"""RSA, версия 2. Исправлены две вещи из первой попытки:

1. Центрирование. Раньше состояния и слова центрировались общим средним. Они лежат в разных
   областях пространства (16 мягких токенов против 1–3 обычных), общее среднее садится между
   двумя облаками, и после вычитания всё становится антипараллельным — косинус −1 у всех пар.
   Теперь каждое множество центрируется своим средним: сравниваем не абсолютное положение,
   а отклонение от среднего состояния против отклонения от среднего слова.

2. Способ чтения. Добавлен вариант с общей несущей фразой: и мозг, и слово ставятся перед
   одним и тем же хвостом «Это ощущается как», и берётся скрытое состояние на последней позиции.
   Так обе стороны читаются в одинаковом синтаксическом месте.
"""

import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from flypet.projector import (
    BrainReader,
    Projector,
    load_reader_checkpoint,
    load_feature_cache,
    save_feature_cache,
)

ap = argparse.ArgumentParser()
ap.add_argument("--per", type=int, default=6)
ap.add_argument("--feats", default="data/rsa_feats.npz")
ap.add_argument("--out", default="data/rsa_v2_clean_v1.json")
a = ap.parse_args()
if Path(a.out).exists():
    ap.error(f"Output exists: {a.out}; choose a fresh --out path to preserve historical results")
if a.per < 1:
    ap.error("--per must be positive")

WORDS = {
    "seen": [
        "сахар на хоботке",
        "горечь на хоботке",
        "надвигающийся объект",
        "касание антенн",
        "хоботок сильно",
        "побег сильно",
        "чистка антенн",
        "жар",
    ],
    "unseen": [
        "сладкое",
        "еда",
        "вкусно",
        "голод",
        "нектар",
        "опасность",
        "страх",
        "бежать",
        "хищник",
        "удар",
        "щекотно",
        "прикосновение",
        "чесаться",
        "жарко",
        "обжечься",
        "горячо",
        "вонь",
        "гниль",
        "тухлое",
        "фрукт",
        "спелый",
        "цветок",
    ],
    "unrelated": [
        "вторник",
        "алгебра",
        "кирпич",
        "налоговая декларация",
        "парламент",
        "бухгалтерия",
        "квадратный корень",
        "ипотека",
    ],
}
EXPECT = {
    "сахар": ["сладкое", "еда", "вкусно", "голод", "нектар"],
    "горечь": ["вонь", "гниль", "тухлое"],
    "побег": ["опасность", "страх", "бежать", "хищник", "удар"],
    "касание": ["щекотно", "прикосновение", "чесаться"],
    "жар": ["жарко", "обжечься", "горячо"],
    "фруктовый": ["фрукт", "спелый", "цветок"],
    "гнилой": ["вонь", "гниль", "тухлое"],
    "тишина": [],
}
CARRIER = " Это ощущается как"
LAYERS = [4, 8, 14, 20, 27]

checkpoint_path, _, reader_metadata = load_reader_checkpoint()
protocol = {"name": "rsa_v2_probes_v1", "per": a.per, "duration_ms": 250, "seed_offset": 100}
cached = (
    load_feature_cache(a.feats, reader_metadata, protocol=protocol)
    if Path(a.feats).exists()
    else None
)
rd = BrainReader(checkpoint_path)
if rd.checkpoint_sha256 != reader_metadata["checkpoint_sha256"]:
    raise RuntimeError("Reader checkpoint changed after feature-cache validation")
print(f"читатель: {rd.model_id}, {len(rd.feats)} признаков", flush=True)

# --- признаки проб (с кэшем)
if cached is not None:
    F, cats, feature_cache_metadata = cached
    print(f"взяты из кэша: {len(cats)} проб", flush=True)
else:
    from flypet.engine import Brain, StimInput
    from flypet import corrections
    from flypet.catalog import STIMULI
    from flypet.odor import DoorOdor
    from flypet.senses import CHANNELS

    b = corrections.apply(Brain())
    door = DoorOdor(rate_max=150.0)
    F, cats = [], []
    for i in range(a.per):
        s = 100 + i
        for cat, st in [
            ("сахар", [StimInput(STIMULI["sugar"].resolve(), 120 + 10 * i, key="s")]),
            ("горечь", [StimInput(STIMULI["bitter"].resolve(), 120 + 10 * i, key="b")]),
            ("побег", [StimInput(STIMULI["looming"].resolve(), 160 + 15 * i, key="l")]),
            ("касание", CHANNELS["touch"].stim("head", 0.6 + 0.07 * i, seed=i)),
            ("жар", CHANNELS["temperature"].stim(32 + i)),
            ("фруктовый", door.stim("ethyl acetate", scale=0.7 + 0.08 * i)),
            ("гнилой", door.stim("putrescine", scale=0.7 + 0.08 * i)),
            ("тишина", []),
        ]:
            F.append(rd.features(b.run(st, duration_ms=250, seed=s)))
            cats.append(cat)
        print(f"  повтор {i + 1}/{a.per}", flush=True)
    F = np.stack(F)
    feature_cache_metadata = save_feature_cache(a.feats, F, cats, rd.checkpoint_metadata, protocol)
print(f"проб: {len(cats)}", flush=True)

carrier_ids = rd.tok(CARRIER, return_tensors="pt", add_special_tokens=False)["input_ids"].to(rd.dev)


@torch.no_grad()
def hidden(prefix_emb, mode):
    """mode='pool' — среднее по позициям префикса; mode='carrier' — последняя позиция после несущей фразы"""
    if mode == "carrier":
        emb = torch.cat([prefix_emb, rd.E(carrier_ids)], dim=1)
        o = rd.llm(inputs_embeds=emb, output_hidden_states=True)
        return {L: o.hidden_states[L][0, -1].float().cpu().numpy() for L in LAYERS}
    o = rd.llm(inputs_embeds=prefix_emb, output_hidden_states=True)
    return {L: o.hidden_states[L][0].mean(0).float().cpu().numpy() for L in LAYERS}


word_list = [(g, w) for g, ws in WORDS.items() for w in ws]
W_h = {
    m: {
        w: hidden(
            rd.E(rd.tok(w, return_tensors="pt", add_special_tokens=False)["input_ids"].to(rd.dev)),
            m,
        )
        for _, w in word_list
    }
    for m in ("pool", "carrier")
}

torch.manual_seed(0)
rand_proj = Projector(len(rd.feats), rd.tokens, rd.E.weight.shape[1]).to(rd.dev).eval()


@torch.no_grad()
def states(proj, mode):
    out = []
    for f in F:
        soft = proj(torch.as_tensor(f[None], device=rd.dev), rd.emb_std)
        out.append(hidden(soft, mode))
    return out


def analyse(proj, mode):
    S_h = states(proj, mode)
    res = {}
    for L in LAYERS:
        S = np.stack([h[L] for h in S_h])
        Wm = np.stack([W_h[mode][w][L] for _, w in word_list])
        Sc = S - S.mean(0)
        Wc = Wm - Wm.mean(0)  # каждое множество своим средним
        Sc /= np.linalg.norm(Sc, axis=1, keepdims=True) + 1e-9
        Wc /= np.linalg.norm(Wc, axis=1, keepdims=True) + 1e-9
        sim = Sc @ Wc.T
        groups = np.array([g for g, _ in word_list])
        rows = {}
        for cat in sorted(set(cats)):
            m = np.array([c == cat for c in cats])
            v = sim[m].mean(0)
            exp = np.array([w in EXPECT[cat] for _, w in word_list])
            order = np.argsort(-v)
            ranks = {word_list[i][1]: int(r) for r, i in enumerate(order)}
            rows[cat] = {
                "expected": float(v[exp].mean()) if exp.any() else None,
                "unseen_other": float(v[(groups == "unseen") & ~exp].mean()),
                "unrelated": float(v[groups == "unrelated"].mean()),
                "seen": float(v[groups == "seen"].mean()),
                "rank_expected": float(
                    np.mean([ranks[w] for _, w in word_list if w in EXPECT[cat]])
                )
                if exp.any()
                else None,
                "rank_unrelated": float(
                    np.mean([ranks[w] for g, w in word_list if g == "unrelated"])
                ),
                "top5": [word_list[i][1] for i in order[:5]],
            }
        res[L] = rows
    return res


out = {}
for mode in ("pool", "carrier"):
    out[mode] = {"trained": analyse(rd.proj, mode), "random": analyse(rand_proj, mode)}
    print(f"\n########## способ чтения: {mode}", flush=True)
    for tag in ("trained", "random"):
        print(f"\n=== {'обученный' if tag == 'trained' else 'случайный'} проектор")
        for L in LAYERS:
            rows = out[mode][tag][L]
            gaps = [
                r["expected"] - r["unrelated"] for r in rows.values() if r["expected"] is not None
            ]
            rk = [
                r["rank_unrelated"] - r["rank_expected"]
                for r in rows.values()
                if r["rank_expected"] is not None
            ]
            print(
                f"  слой {L:2d}: разрыв ожидаемые−несвязанные {np.mean(gaps):+.3f} | ожидаемые выше несвязанных на {np.mean(rk):+.1f} мест из {len(word_list)}"
            )
    L = 20
    print(f"\n  детали, слой {L}, обученный:")
    for cat, r in out[mode]["trained"][L].items():
        if r["expected"] is None:
            continue
        print(
            f"   {cat:11s} ожид {r['expected']:+.3f} прочие {r['unseen_other']:+.3f} несвяз {r['unrelated']:+.3f} | топ: {', '.join(r['top5'][:4])}"
        )
with open(a.out, "x") as output:
    json.dump(
        {
            "cats": cats,
            "layers": LAYERS,
            "result": out,
            "words": WORDS,
            "expect": EXPECT,
            "reader": rd.checkpoint_metadata,
            "feature_cache": feature_cache_metadata,
        },
        output,
        ensure_ascii=False,
        indent=1,
        default=float,
    )
print("\nсохранено в", a.out)
