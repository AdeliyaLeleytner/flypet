"""Функциональная проверка вместо геометрической.

Вопрос тот же: легло ли состояние мухи на то, что языковая модель уже знает? Но проверяем не
близость векторов, а поведение модели. Подаём мягкие токены и задаём вопрос, которого в
обучающих подписях не было ни разу, и смотрим на вероятности ответов.

Контроли: перемешанные состояния (ответ не должен следовать за категорией) и случайный проектор.
"""

import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from flypet.projector import BrainReader, Projector, load_reader_checkpoint, load_feature_cache

ap = argparse.ArgumentParser()
ap.add_argument("--feats", default="data/rsa_feats.npz")
ap.add_argument("--out", default="data/rsa_functional_clean_v1.json")
a = ap.parse_args()
if Path(a.out).exists():
    ap.error(f"Output exists: {a.out}; choose a fresh --out path to preserve historical results")

QUESTIONS = [
    ("\nВопрос: это приятно или опасно?\nОтвет:", [" приятно", " опасно"], "приятно/опасно"),
    ("\nВопрос: муха хочет это съесть, да или нет?\nОтвет:", [" да", " нет"], "хочет съесть"),
    ("\nВопрос: надо убегать, да или нет?\nОтвет:", [" да", " нет"], "надо убегать"),
    ("\nВопрос: это еда или угроза?\nОтвет:", [" еда", " угроза"], "еда/угроза"),
]
EXPECT = {  # какой ответ ждём для категории: 0 — первый вариант, 1 — второй
    "сахар": {"приятно/опасно": 0, "хочет съесть": 0, "надо убегать": 1, "еда/угроза": 0},
    "фруктовый": {"приятно/опасно": 0, "хочет съесть": 0, "надо убегать": 1, "еда/угроза": 0},
    "горечь": {"приятно/опасно": 1, "хочет съесть": 1, "надо убегать": None, "еда/угроза": 1},
    "гнилой": {"приятно/опасно": 1, "хочет съесть": 1, "надо убегать": None, "еда/угроза": 1},
    "побег": {"приятно/опасно": 1, "хочет съесть": 1, "надо убегать": 0, "еда/угроза": 1},
    "жар": {"приятно/опасно": 1, "хочет съесть": 1, "надо убегать": 0, "еда/угроза": 1},
    "касание": {
        "приятно/опасно": None,
        "хочет съесть": 1,
        "надо убегать": None,
        "еда/угроза": None,
    },
    "тишина": {"приятно/опасно": None, "хочет съесть": 1, "надо убегать": 1, "еда/угроза": None},
}

checkpoint_path, _, reader_metadata = load_reader_checkpoint()
F, cats, feature_cache_metadata = load_feature_cache(a.feats, reader_metadata)
rd = BrainReader(checkpoint_path)
if rd.checkpoint_sha256 != reader_metadata["checkpoint_sha256"]:
    raise RuntimeError("Reader checkpoint changed after feature-cache validation")
print(f"проб: {len(cats)}, категорий: {len(set(cats))}", flush=True)
torch.manual_seed(0)
rand_proj = Projector(len(rd.feats), rd.tokens, rd.E.weight.shape[1]).to(rd.dev).eval()


@torch.no_grad()
def answer_probs(feat, proj, q_text, options):
    soft = proj(torch.as_tensor(feat[None], device=rd.dev), rd.emb_std)
    q_ids = rd.tok(q_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(rd.dev)
    emb = torch.cat([soft, rd.E(q_ids)], dim=1)
    logits = rd.llm(inputs_embeds=emb).logits[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    out = []
    for o in options:
        ids = rd.tok(o, add_special_tokens=False)["input_ids"]
        out.append(float(lp[ids[0]]))  # вероятность первого токена варианта
    return out


def run(proj, shuffle=False, tag=""):
    idx = np.random.default_rng(0).permutation(len(cats)) if shuffle else np.arange(len(cats))
    rows = {}
    for q_text, options, qname in QUESTIONS:
        per_cat = {}
        for i, c in zip(idx, cats):
            p = answer_probs(F[i], proj, q_text, options)
            per_cat.setdefault(c, []).append(p[0] - p[1])  # >0 значит первый вариант вероятнее
        rows[qname] = {c: float(np.mean(v)) for c, v in per_cat.items()}
    return rows


print("обученный проектор…", flush=True)
trained = run(rd.proj)
print("перемешанные состояния…", flush=True)
shuffled = run(rd.proj, shuffle=True)
print("случайный проектор…", flush=True)
random_ = run(rand_proj)


def report(rows, tag):
    print(f"\n===== {tag}: log P(первый вариант) − log P(второй), по категориям")
    qs = list(rows)
    print(f"{'категория':12s}" + "".join(f"{q:>18s}" for q in qs))
    for c in sorted(rows[qs[0]]):
        print(f"{c:12s}" + "".join(f"{rows[q][c]:+18.2f}" for q in qs))


for r, t in (
    (trained, "ОБУЧЕННЫЙ"),
    (shuffled, "ПЕРЕМЕШАННЫЕ состояния"),
    (random_, "СЛУЧАЙНЫЙ проектор"),
):
    report(r, t)


# сводка: насколько ответы следуют за ожиданием
def score(rows):
    hits, tot = 0, 0
    for q, per in rows.items():
        for c, v in per.items():
            e = EXPECT.get(c, {}).get(q)
            if e is None:
                continue
            tot += 1
            hits += int((v > 0) == (e == 0))
    return hits, tot


print("\n===== совпадение с ожидаемым ответом")
for r, t in ((trained, "обученный"), (shuffled, "перемешанные"), (random_, "случайный")):
    h, n = score(r)
    print(f"  {t:14s} {h}/{n} = {100 * h / max(1, n):.0f} %")
# разброс между категориями: если модель отвечает одинаково на всё, разброс около нуля
print("\n===== разброс ответов между категориями (стандартное отклонение)")
for r, t in ((trained, "обученный"), (shuffled, "перемешанные"), (random_, "случайный")):
    sds = [float(np.std(list(per.values()))) for per in r.values()]
    print(f"  {t:14s} {np.mean(sds):.3f}")
with open(a.out, "x") as output:
    json.dump(
        {
            "trained": trained,
            "shuffled": shuffled,
            "random": random_,
            "reader": rd.checkpoint_metadata,
            "feature_cache": feature_cache_metadata,
        },
        output,
        ensure_ascii=False,
        indent=1,
        default=float,
    )
print("\nсохранено в", a.out)
