"""Легло ли состояние мухи на то, что языковая модель уже знает о мире?

Берём скрытые состояния замороженной Qwen3, когда она читает мягкие токены мозга,
и когда она читает обычные слова. Сравниваем.

Главная ловушка: проектор обучался выдавать подписи с определёнными словами, поэтому
близость к ним тривиальна. Поэтому слова разделены на три группы:
  seen      — встречались в обучающих подписях (положительный контроль, должны быть близко)
  unseen    — по смыслу связаны, но в подписях НИКОГДА не встречались (настоящая проверка)
  unrelated — не связаны ни с чем (отрицательный контроль, должны быть далеко)
Плюс базовая линия: необученный проектор со случайными весами.
"""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, json, argparse

sys.path.insert(0, _ROOT + "")
import numpy as np, torch
from flypet.engine import Brain, StimInput
from flypet import corrections
from flypet.catalog import STIMULI
from flypet.projector import BrainReader, Projector
from flypet.odor import DoorOdor
from flypet.senses import CHANNELS

ap = argparse.ArgumentParser()
ap.add_argument("--per", type=int, default=5, help="проб на категорию")
ap.add_argument("--out", default="data/rsa_result.json")
a = ap.parse_args()

# ---------------------------------------------------------------- слова
WORDS = {
    # встречались в обучающих подписях
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
    # по смыслу связаны, но в подписях не было ни разу
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
    # не связаны ни с чем
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
# какие unseen-слова «правильные» для какой категории проб
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

# ---------------------------------------------------------------- пробы
print("строю мозг…", flush=True)
b = corrections.apply(Brain())
door = DoorOdor(rate_max=150.0)


def probes():
    out = []
    for i in range(a.per):
        s = 100 + i
        out.append(("сахар", [StimInput(STIMULI["sugar"].resolve(), 120 + 10 * i, key="sugar")], s))
        out.append(
            ("горечь", [StimInput(STIMULI["bitter"].resolve(), 120 + 10 * i, key="bitter")], s)
        )
        out.append(
            ("побег", [StimInput(STIMULI["looming"].resolve(), 160 + 15 * i, key="looming")], s)
        )
        out.append(("касание", CHANNELS["touch"].stim("head", 0.6 + 0.1 * i, seed=i), s))
        out.append(("жар", CHANNELS["temperature"].stim(32 + i), s))
        out.append(("фруктовый", door.stim("ethyl acetate", scale=0.7 + 0.1 * i), s))
        out.append(("гнилой", door.stim("putrescine", scale=0.7 + 0.1 * i), s))
        out.append(("тишина", [], s))
    return out


rd = BrainReader()
print(
    f"читатель загружен: {rd.model_id} на {rd.dev}, {len(rd.feats)} признаков, {rd.tokens} токенов",
    flush=True,
)
LAYERS = [4, 8, 14, 20, 27]


@torch.no_grad()
def hidden_from_embeds(emb):
    """emb: (1, T, d) -> dict слой -> вектор (усреднение по позициям)"""
    o = rd.llm(inputs_embeds=emb, output_hidden_states=True)
    return {L: o.hidden_states[L][0].mean(0).float().cpu().numpy() for L in LAYERS}


@torch.no_grad()
def soft_tokens(feat_vec, proj):
    x = torch.as_tensor(feat_vec[None], device=rd.dev)
    return proj(x, rd.emb_std)


# состояния мозга
cats, feats_all = [], []
for cat, stims, seed in probes():
    r = b.run(stims, duration_ms=250, seed=seed)
    feats_all.append(rd.features(r))
    cats.append(cat)
    print(f"  проба {cat:10s} готова", flush=True)
feats_all = np.stack(feats_all)

# слова
word_list = [(g, w) for g, ws in WORDS.items() for w in ws]
word_h = {}
for g, w in word_list:
    ids = rd.tok(w, return_tensors="pt", add_special_tokens=False)["input_ids"].to(rd.dev)
    word_h[w] = hidden_from_embeds(rd.E(ids))

# случайный проектор — базовая линия
torch.manual_seed(0)
rand_proj = Projector(len(rd.feats), rd.tokens, rd.E.weight.shape[1]).to(rd.dev).eval()


def analyse(proj, tag):
    st_h = [hidden_from_embeds(soft_tokens(f, proj)) for f in feats_all]
    res = {}
    for L in LAYERS:
        S = np.stack([h[L] for h in st_h])
        Wm = np.stack([word_h[w][L] for _, w in word_list])
        allv = np.vstack([S, Wm])
        mu = allv.mean(0)
        Sc, Wc = S - mu, Wm - mu
        Sc /= np.linalg.norm(Sc, axis=1, keepdims=True) + 1e-9
        Wc /= np.linalg.norm(Wc, axis=1, keepdims=True) + 1e-9
        sim = Sc @ Wc.T
        groups = np.array([g for g, _ in word_list])
        rows = {}
        for cat in sorted(set(cats)):
            m = np.array([c == cat for c in cats])
            v = sim[m].mean(0)
            exp = np.array([w in EXPECT[cat] for _, w in word_list])
            rows[cat] = {
                "seen": float(v[groups == "seen"].mean()),
                "unseen_expected": float(v[exp].mean()) if exp.any() else None,
                "unseen_other": float(v[(groups == "unseen") & ~exp].mean()),
                "unrelated": float(v[groups == "unrelated"].mean()),
                "top3": [word_list[i][1] for i in np.argsort(-v)[:3]],
            }
        res[L] = rows
    return res


print("\nанализ обученного проектора…", flush=True)
trained = analyse(rd.proj, "trained")
print("анализ случайного проектора…", flush=True)
random_ = analyse(rand_proj, "random")


def report(res, tag):
    print(f"\n===== {tag}")
    for L in LAYERS:
        rows = res[L]
        print(f"\n-- слой {L}")
        print(
            f"{'категория':12s} {'ожидаемые':>10s} {'прочие':>8s} {'несвязанные':>12s} {'разрыв':>8s}  топ-3 слова"
        )
        for cat, r in rows.items():
            if r["unseen_expected"] is None:
                continue
            gap = r["unseen_expected"] - r["unrelated"]
            print(
                f"{cat:12s} {r['unseen_expected']:10.3f} {r['unseen_other']:8.3f} {r['unrelated']:12.3f} {gap:+8.3f}  {', '.join(r['top3'])}"
            )


report(trained, "ОБУЧЕННЫЙ проектор")
report(random_, "СЛУЧАЙНЫЙ проектор (базовая линия)")

# сводка: средний разрыв «ожидаемые минус несвязанные» по слоям
print("\n===== сводка: разрыв (ожидаемые − несвязанные), усреднённый по категориям")
print(f"{'слой':>6s} {'обученный':>12s} {'случайный':>12s}")
for L in LAYERS:
    g_t = np.mean(
        [
            r["unseen_expected"] - r["unrelated"]
            for r in trained[L].values()
            if r["unseen_expected"] is not None
        ]
    )
    g_r = np.mean(
        [
            r["unseen_expected"] - r["unrelated"]
            for r in random_[L].values()
            if r["unseen_expected"] is not None
        ]
    )
    print(f"{L:6d} {g_t:+12.3f} {g_r:+12.3f}")
json.dump(
    {
        "layers": LAYERS,
        "cats": cats,
        "trained": trained,
        "random": random_,
        "words": WORDS,
        "expect": EXPECT,
    },
    open(a.out, "w"),
    ensure_ascii=False,
    indent=1,
    default=float,
)
print("\nсохранено в", a.out)
