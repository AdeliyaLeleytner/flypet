"""Перевести бытовые слова питомца в реальные вещества из DoOR (один раз, с кэшем)."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys

sys.path.insert(0, _ROOT + "")
from flypet.odor import DoorOdor, resolve_object
from flypet.llm import get_llm

WORDS = [
    "яблоко",
    "банан",
    "груша",
    "вишня",
    "уксус",
    "кофе",
    "сыр",
    "хлеб",
    "рыба",
    "трава",
    "дождь",
    "дым",
    "бензин",
    "мята",
    "мыло",
    "молоток",
    "гвоздь",
    "отвёртка",
    "пила",
    "книга",
]
o = DoorOdor()
llm = get_llm()
print(f"LLM: {llm.name}; словарь DoOR: {len(o.well_covered(25))} веществ\n")
for w in WORDS:
    r = resolve_object(w, llm, o)
    c = r["compound"]
    if c:
        gl = o.glomerular(c)
        top = sorted(gl.items(), key=lambda x: -x[1])[:3]
        print(
            f"{w:10s} -> {c:24s} гломерулы: {[(g, round(v, 2)) for g, v in top]}  ({r['why'][:44]})",
            flush=True,
        )
    else:
        print(f"{w:10s} -> (не подобрано) {r['why'][:60]}", flush=True)
