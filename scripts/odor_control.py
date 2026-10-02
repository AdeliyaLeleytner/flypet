"""Контроль: следовал ли старый (случайная проекция) энкодер за химией? Должен не следовать."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys

sys.path.insert(0, _ROOT + "")
import numpy as np
from flypet.engine import Brain
from flypet import corrections, connectome as C
from flypet.odor import DoorOdor
from flypet.mb import OdorEncoder

door = DoorOdor(rate_max=150.0)
old = OdorEncoder(k_active=3, rate_hz=100.0, frac=1.0)
b = corrections.apply(Brain())
ann = C.annotations()
cc = ann.cell_class.astype(str)
KC = np.array([b.flyid2i[int(x)] for x in ann.index[cc == "Kenyon_Cell"]])

ODORS = [
    "ethyl acetate",
    "pentyl acetate",
    "ethyl butyrate",
    "acetic acid",
    "propanoic acid",
    "benzaldehyde",
    "carbon dioxide",
    "1-hexanol",
    "2,3-butanedione",
]


def kc_code(stims):
    r = b.run(stims, duration_ms=200, seed=5)
    v = r.rate_vector(b.n)
    return set(np.flatnonzero(v[KC] > 0).tolist())


code_new = {n: kc_code(door.stim(n)) for n in ODORS}
code_old = {n: kc_code([old.stim(n, seed=0)]) for n in ODORS}
jac = lambda a, c: len(a & c) / max(1, len(a | c))

rows = []
for i, a_ in enumerate(ODORS):
    for b_ in ODORS[i + 1 :]:
        s = door.similarity(a_, b_)
        if np.isnan(s):
            continue
        rows.append((s, jac(code_new[a_], code_new[b_]), jac(code_old[a_], code_old[b_])))
S = np.array([r[0] for r in rows])
JN = np.array([r[1] for r in rows])
JO = np.array([r[2] for r in rows])
print(f"пар веществ: {len(rows)}")
print(f"\nDoOR-энкодер (реальные рецепторы):")
print(f"  корреляция химия ↔ общий код KC: r = {np.corrcoef(S, JN)[0, 1]:+.2f}")
print(f"  средний общий код: {JN.mean():.2f} (разброс {JN.min():.2f}–{JN.max():.2f})")
print(f"\nстарый энкодер (случайная проекция эмбеддинга):")
print(f"  корреляция химия ↔ общий код KC: r = {np.corrcoef(S, JO)[0, 1]:+.2f}")
print(f"  средний общий код: {JO.mean():.2f} (разброс {JO.min():.2f}–{JO.max():.2f})")
