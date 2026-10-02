"""Запахи на измеренных данных вместо случайной проекции.

Источник: DoOR 2.0 (Münch & Galizia, Sci Rep 6:21841, 2016) — сводная матрица откликов
обонятельных рецепторов дрозофилы на 691 пахучее вещество, собранная из литературы.
Рецептор -> гломерула берётся из той же базы, гломерула -> наши ORN_* по аннотациям FlyWire.

Отличие от OdorEncoder в flypet/mb.py: там слово шло через эмбеддинг языковой модели и
случайную проекцию, то есть соответствие было выдумано. Здесь "уксусная кислота" попадает
на те рецепторы, которые уксусная кислота активирует в реальном эксперименте.
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from . import connectome as C
from .engine import StimInput

ROOT = Path(__file__).resolve().parent.parent
DOOR = ROOT / "data" / "door"


@lru_cache(maxsize=1)
def _load():
    rm = pd.read_csv(DOOR / "door_response_matrix.csv", sep=";", index_col=0)
    od = pd.read_csv(DOOR / "odor.csv", sep=";", index_col=0)
    mp = pd.read_csv(DOOR / "door_mappings.csv", sep=";")
    # отклик сверх спонтанного
    sfr = rm.loc["SFR"]
    resp = (rm.subtract(sfr, axis=1)).clip(lower=0.0)
    resp = resp.drop(index=[i for i in ("SFR",) if i in resp.index])
    # рецептор -> гломерула
    mp = mp[mp.glomerulus.notna() & (mp.glomerulus != "?")]
    r2g = {}
    for col in ("receptor", "OSN", "code", "code.OSN", "sensillum"):
        if col in mp.columns:
            for k, v in zip(mp[col].astype(str), mp.glomerulus.astype(str)):
                if k and k not in ("?", "nan"):
                    r2g.setdefault(k, v)
    # наши гломерулы
    ann = C.annotations()
    ct = ann.cell_type.astype(str)
    orn = ann[(ann.cell_class == "olfactory") & (ct != "")]
    gl2ids = {
        t.replace("ORN_", ""): [int(x) for x in orn.index[orn.cell_type.astype(str) == t]]
        for t in sorted(orn.cell_type.astype(str).unique())
    }
    # имя вещества -> ключ строки
    name2key, key2name = {}, {}
    for key, nm in zip(od.InChIKey.astype(str), od.Name.astype(str)):
        if isinstance(nm, str) and nm not in ("nan", ""):
            name2key.setdefault(nm.strip().lower(), key)
            key2name.setdefault(key, nm.strip())
    return resp, r2g, gl2ids, name2key, key2name


class DoorOdor:
    """rate_max — частота ORN при максимальном измеренном отклике (1.0)."""

    def __init__(self, rate_max: float = 150.0, min_response: float = 0.05):
        self.resp, self.r2g, self.gl2ids, self.name2key, self.key2name = _load()
        self.rate_max, self.min_response = rate_max, min_response
        self.glomeruli = sorted(self.gl2ids)

    # ---------- поиск вещества
    def find(self, query: str, limit: int = 12) -> list[tuple[str, int]]:
        """Вещества, чьё имя содержит query; вместе с числом измеренных рецепторов."""
        q = query.strip().lower()
        out = []
        for nm, key in self.name2key.items():
            if q in nm and key in self.resp.index:
                out.append((self.key2name[key], int(self.resp.loc[key].notna().sum())))
        return sorted(out, key=lambda x: -x[1])[:limit]

    def _key(self, name: str) -> str:
        k = self.name2key.get(name.strip().lower())
        if k is None or k not in self.resp.index:
            raise KeyError(f"вещества «{name}» нет в DoOR; попробуйте find()")
        return k

    # ---------- отклик по гломерулам
    def glomerular(self, name: str) -> dict[str, float]:
        """Гломерула -> отклик 0..1, усреднённый по рецепторам этой гломерулы. Только наши гломерулы."""
        row = self.resp.loc[self._key(name)]
        acc: dict[str, list[float]] = {}
        for rec, val in row.items():
            if np.isnan(val):
                continue
            g = self.r2g.get(str(rec))
            if g in self.gl2ids:
                acc.setdefault(g, []).append(float(val))
        return {g: float(np.mean(v)) for g, v in sorted(acc.items())}

    def coverage(self, name: str) -> tuple[int, int]:
        """Сколько наших гломерул измерено для этого вещества, из скольких всего."""
        return len(self.glomerular(name)), len(self.glomeruli)

    # ---------- стимул для симуляции
    def stim(self, name: str, scale: float = 1.0) -> list[StimInput]:
        """Список StimInput: каждая гломерула получает свою частоту по измеренному отклику."""
        gl = self.glomerular(name)
        out = []
        for g, v in gl.items():
            if v < self.min_response:
                continue
            ids = self.gl2ids[g]
            if ids:
                out.append(StimInput(ids, self.rate_max * v * scale, key=f"{name}:{g}"))
        return out

    # ---------- сравнение веществ
    def vector(self, name: str) -> np.ndarray:
        gl = self.glomerular(name)
        return np.array([gl.get(g, np.nan) for g in self.glomeruli], dtype=float)

    def similarity(self, a: str, b: str) -> float:
        """Косинус по гломерулам, измеренным у обоих веществ."""
        va, vb = self.vector(a), self.vector(b)
        m = ~np.isnan(va) & ~np.isnan(vb)
        if m.sum() < 5 or va[m].sum() == 0 or vb[m].sum() == 0:
            return float("nan")
        return float(va[m] @ vb[m] / (np.linalg.norm(va[m]) * np.linalg.norm(vb[m])))

    # ---------- словарь для промпта
    def well_covered(self, min_receptors: int = 25) -> list[str]:
        """Вещества, у которых измерено достаточно рецепторов — из них выбирает планировщик."""
        n = self.resp.notna().sum(1)
        names = []
        for key, c in n.items():
            if c >= min_receptors and key in self.key2name:
                nm = self.key2name[key]
                if nm.lower() not in ("sfr", "oil", "water", "solvent"):
                    names.append((nm, int(c)))
        seen, out = set(), []
        for nm, c in sorted(names, key=lambda x: -x[1]):
            if nm.lower() not in seen:
                seen.add(nm.lower())
                out.append(nm)
        return out


# ------------------------------------------------------------------ предмет -> вещество
RESOLVE_CACHE = ROOT / "data" / "odor_resolved.json"

RESOLVE_SYSTEM = """Ты химик-одорант. Тебе дают бытовой предмет или запах по-русски и список веществ,
для которых измерены отклики обонятельных рецепторов дрозофилы. Выбери из списка одно вещество,
которое лучше всего представляет запах этого предмета для мухи.
Правила: выбирай только из списка, точным написанием. Думай о летучих веществах, которые
реально пахнут у этого предмета (яблоко — сложные эфиры, уксус — уксусная кислота,
гниющий фрукт — этанол и кислоты, трава — зелёные C6-альдегиды и спирты).
Если подходящего вещества нет в списке или соответствие ненадёжно, верни пустую строку
и объясни ограничение в why. Отсутствие соответствия в базе не означает, что предмет
не пахнет или что настоящая муха не может его обнаружить. Одно выбранное вещество —
приближение к запаху предмета, а не измеренный химический состав самого предмета."""

RESOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "compound": {
            "type": "string",
            "description": "имя вещества из списка, точным написанием, или пустая строка",
        },
        "why": {
            "type": "string",
            "description": "одна короткая фраза по-русски, почему именно оно",
        },
    },
    "required": ["compound", "why"],
    "additionalProperties": False,
}


def resolve_object(
    word: str,
    llm,
    door: DoorOdor | None = None,
    cache: dict | None = None,
    *,
    cache_path: str | Path | None = RESOLVE_CACHE,
) -> dict:
    """Resolve a proposed proxy chemical; missing coverage is not biological absence.

    Errors are retryable and never cached. ``cache_path=None`` is in-memory only.
    FlyChat passes its own state directory so experiments cannot share resolver state.
    """
    import json

    door = door or DoorOdor()
    key = word.strip().lower()
    path = Path(cache_path) if cache_path is not None else None
    if cache is None:
        try:
            cache = json.loads(path.read_text()) if path is not None and path.exists() else {}
            if not isinstance(cache, dict):
                raise ValueError("resolver cache must be an object")
        except Exception as e:
            return {"status": "error", "compound": "", "why": f"{type(e).__name__}: {e}"}
    if key in cache:
        cached = cache[key]
        # Old caches collapsed exceptions into empty compounds. Retry those entries.
        if cached.get("status") != "error" and not cached.get("why", "").startswith("ошибка:"):
            comp = cached.get("compound", "")
            if comp and comp.lower() in door.name2key:
                return {**cached, "status": "resolved"}
            if not comp:
                return {**cached, "status": "unmapped"}
    try:
        catalog = door.well_covered(25)
        prompt = (
            f"Предмет: {word}\n\nСписок веществ ({len(catalog)}):\n"
            + ", ".join(catalog)
            + "\n\nВерни JSON."
        )
        r = llm.complete(RESOLVE_SYSTEM, prompt, schema=RESOLVE_SCHEMA)
        comp = (r.get("compound") or "").strip()
        if comp:
            chosen = next((c for c in catalog if c.lower() == comp.lower()), None)
            if chosen is None:
                raise ValueError(f"resolver returned chemical outside its catalog: {comp}")
            comp = chosen
        out = {
            "status": "resolved" if comp else "unmapped",
            "compound": comp,
            "why": r.get("why", ""),
        }
    except Exception as e:
        return {"status": "error", "compound": "", "why": f"{type(e).__name__}: {e}"}
    cache[key] = out
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    return out
