_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys

sys.path.insert(0, _ROOT + "")
from flypet.odor import DoorOdor, resolve_object
from flypet.llm import get_llm

o = DoorOdor()
llm = get_llm()
for w in ["молоток", "гвоздь", "отвёртка", "пила", "книга", "бензин", "рыба"]:
    r = resolve_object(w, llm, o)
    c = r["compound"]
    if c:
        gl = o.glomerular(c)
        top = sorted(gl.items(), key=lambda x: -x[1])[:3]
        print(
            f"{w:10s} -> {c:22s} {[(g, round(v, 2)) for g, v in top]}  {r['why'][:50]}", flush=True
        )
    else:
        print(f"{w:10s} -> не пахнет для мухи: {r['why'][:60]}", flush=True)
