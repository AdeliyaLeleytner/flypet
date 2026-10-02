"""Анатомия грибовидного тела по нашим данным: сходимость, расходимость, отделы."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys

sys.path.insert(0, _ROOT + "")
import numpy as np, pandas as pd
from collections import defaultdict
from flypet import connectome as C

ann = C.annotations()
cc = ann.cell_class.astype(str)
ct = ann.cell_type.astype(str)
flyid2i, i2flyid = C.id_maps()
i_pre, i_post, w = C.connections()
idx = lambda mask: np.array([flyid2i[int(x)] for x in ann.index[mask]], dtype=np.int64)
KC, MBON, DAN = idx(cc == "Kenyon_Cell"), idx(cc == "MBON"), idx(cc == "DAN")
PN, APL = idx(cc == "ALPN"), idx(ct == "APL")
S = lambda a: set(a.tolist())
kc, mbon, dan, pn, apl = map(S, (KC, MBON, DAN, PN, APL))
print(
    f"клеток Кеньона {len(kc)}, MBON {len(mbon)}, дофаминовых {len(dan)}, проекционных {len(pn)}, APL {len(apl)}"
)


def edges(a, b):
    m = np.isin(i_pre, list(a)) & np.isin(i_post, list(b))
    return i_pre[m], i_post[m], np.abs(w[m])


# PN -> KC: сколько входов у одной клетки Кеньона
p, q, c = edges(pn, kc)
in_per_kc = pd.Series(q).value_counts()
print(
    f"\nPN→KC: связей {len(p)}, у одной KC входов от {in_per_kc.median():.0f} PN (медиана), "
    f"10–90 %: {in_per_kc.quantile(0.1):.0f}–{in_per_kc.quantile(0.9):.0f}"
)
print(
    f"   контактов на связь: медиана {np.median(c):.0f}, 90-й перцентиль {np.percentile(c, 90):.0f}"
)
out_per_pn = pd.Series(p).value_counts()
print(f"   один PN расходится на {out_per_pn.median():.0f} KC (медиана)")

# KC -> MBON
p2, q2, c2 = edges(kc, mbon)
print(f"\nKC→MBON: связей {len(p2)}, контактов всего {int(c2.sum())}")
print(f"   у одного MBON входов от {pd.Series(q2).value_counts().median():.0f} KC (медиана)")
print(
    f"   одна KC выходит на {pd.Series(p2).value_counts().median():.0f} MBON (медиана), "
    f"максимум {pd.Series(p2).value_counts().max()}"
)

# KC -> APL -> KC (обратное торможение)
p3, q3, c3 = edges(kc, apl)
p4, q4, c4 = edges(apl, kc)
print(
    f"\nобратное торможение: KC→APL {len(p3)} связей, APL→KC {len(p4)} связей "
    f"({100 * len(set(q4.tolist())) / len(kc):.0f} % всех KC получают от APL)"
)

# отделы: DAN -> MBON
p5, q5, c5 = edges(dan, mbon)
comp = defaultdict(lambda: defaultdict(float))
for a, b, x in zip(p5, q5, c5):
    comp[ct[i2flyid[int(b)]]][ct[i2flyid[int(a)]][:4]] += x
rows = []
for mb_t, d in sorted(comp.items()):
    tot = sum(d.values())
    pam = sum(v for k, v in d.items() if k.startswith("PAM"))
    ppl = tot - pam
    rows.append((mb_t, tot, pam / tot))
df = pd.DataFrame(rows, columns=["MBON", "контактов от DAN", "доля PAM"])
pure = ((df["доля PAM"] > 0.9) | (df["доля PAM"] < 0.1)).mean()
print(
    f"\nотделы: у {100 * pure:.0f} % типов MBON дофамин приходит почти целиком либо от PAM (награда), либо от PPL1 (наказание)"
)
print(f"   типов MBON с прямым входом от DAN: {len(df)}")
print(df.sort_values("контактов от DAN", ascending=False).head(8).to_string(index=False))

# сколько отделов задевает одна KC
kc2mb = defaultdict(set)
for a, b in zip(p2, q2):
    kc2mb[int(a)].add(ct[i2flyid[int(b)]])
n = pd.Series([len(v) for v in kc2mb.values()])
print(
    f"\nодна клетка Кеньона выходит на {n.median():.0f} разных типов MBON (медиана), до {n.max()}"
)
