"""Реальные запахи в модели: совпадает ли близость кодов клеток Кеньона с близостью по рецепторам?"""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys

sys.path.insert(0, _ROOT + "")
import numpy as np
from flypet.engine import Brain
from flypet import corrections, connectome as C
from flypet.odor import DoorOdor

o = DoorOdor(rate_max=150.0)
b = corrections.apply(Brain())
ann = C.annotations()
cc = ann.cell_class.astype(str)
ct = ann.cell_type.astype(str)
KC = np.array([b.flyid2i[int(x)] for x in ann.index[cc == "Kenyon_Cell"]])
MBON = np.array([b.flyid2i[int(x)] for x in ann.index[cc == "MBON"]])
PN = np.array([b.flyid2i[int(x)] for x in ann.index[cc == "ALPN"]])

ODORS = [
    "ethyl acetate",
    "pentyl acetate",
    "ethyl butyrate",
    "acetic acid",
    "propanoic acid",
    "benzaldehyde",
    "carbon dioxide",
    "geosmin",
    "1-hexanol",
    "2,3-butanedione",
]
codes, mb = {}, {}
print(f"{'вещество':22s} {'гломерул':>8s} {'PN':>5s} {'KC':>6s} {'KC %':>6s} {'MBON':>5s}")
for name in ODORS:
    st = o.stim(name)
    if not st:
        print(f"{name:22s}  нет данных")
        continue
    r = b.run(st, duration_ms=200, seed=5)
    v = r.rate_vector(b.n)
    codes[name] = set(np.flatnonzero(v[KC] > 0).tolist())
    mb[name] = v[MBON]
    print(
        f"{name:22s} {len(st):8d} {int((v[PN] > 0).sum()):5d} {len(codes[name]):6d} "
        f"{100 * len(codes[name]) / len(KC):5.1f}% {int((v[MBON] > 0).sum()):5d}",
        flush=True,
    )


def jac(a, c):
    return len(a & c) / max(1, len(a | c))


pairs = [
    ("ethyl acetate", "pentyl acetate"),
    ("ethyl acetate", "ethyl butyrate"),
    ("pentyl acetate", "ethyl butyrate"),
    ("acetic acid", "propanoic acid"),
    ("ethyl acetate", "acetic acid"),
    ("ethyl acetate", "carbon dioxide"),
    ("acetic acid", "carbon dioxide"),
    ("benzaldehyde", "1-hexanol"),
    ("ethyl acetate", "benzaldehyde"),
    ("geosmin", "ethyl acetate"),
]
print(f"\n{'пара':42s} {'похожесть по рецепторам':>24s} {'общий код KC':>14s}")
rows = []
for a_, b_ in pairs:
    if a_ not in codes or b_ not in codes:
        continue
    s = o.similarity(a_, b_)
    j = jac(codes[a_], codes[b_])
    rows.append((s, j))
    print(f"{a_ + ' / ' + b_:42s} {s:24.2f} {j:14.2f}")
if len(rows) > 3:
    s = np.array([x[0] for x in rows])
    j = np.array([x[1] for x in rows])
    m = ~np.isnan(s)
    print(
        f"\nкорреляция «похожи по рецепторам» ↔ «похожи по коду KC»: r = {np.corrcoef(s[m], j[m])[0, 1]:.2f} (n={m.sum()})"
    )
