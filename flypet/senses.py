"""Все сенсорные каналы мухи одним интерфейсом, с явным происхождением каждого числа.

Практика помечать провенанс каждого параметра взята у Fly.exe (github.com/Ibtisam-Mohammad/Fly.exe).
Уровни:
  measured    — число взято из опубликованных измерений на дрозофиле
  literature  — направление и порядок известны из литературы, точная кривая подогнана нами
  engineering — наша конструкция, в природе такого параметра нет

Каждый канал отдаёт список StimInput для flypet.engine.Brain.run().
"""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
from . import connectome as C
from .engine import StimInput


@dataclass
class Provenance:
    level: str
    note: str


@lru_cache(maxsize=1)
def _groups() -> dict[str, list[int]]:
    """Готовые группы сенсорных нейронов по аннотациям FlyWire."""
    ann = C.annotations()
    ct = ann.cell_type.astype(str)
    cc = ann.cell_class.astype(str)
    sub = ann.cell_sub_class.astype(str)
    side = ann.side.astype(str)
    ids = lambda m: [int(x) for x in ann.index[m]]
    g = {
        # --- орган Джонстона: слух и ветер
        "JO_A": ids(ct.str.startswith("JO-A")),
        "JO_B": ids(ct.str.startswith("JO-B")),
        "JO_C": ids(ct.str.startswith("JO-C")),
        "JO_D": ids(ct.str.startswith("JO-D")),
        "JO_E": ids(ct.str.startswith("JO-E")),
        "JO_F": ids(ct.str.startswith("JO-F")),
        # --- вкус по модальностям
        "taste_sugar": ids((cc == "gustatory") & (sub == "sugar/water") & ct.str.startswith("LB")),
        "taste_bitter": ids((cc == "gustatory") & (sub == "bitter")),
        "taste_salt": ids((cc == "gustatory") & (sub == "low-salt")),
        "taste_peg": ids((cc == "gustatory") & (sub == "taste peg")),
        "taste_tarsal": ids((cc == "gustatory") & ct.isin(["claw_tpGRN", "dorsal_tpGRN"])),
        "taste_pharynx": ids((cc == "gustatory") & ct.str.startswith("PhG")),
        # --- температура и влажность
        "temp_hot": ids(ct == "TRN_VP2"),
        "temp_cold": ids(ct.str.startswith("TRN_VP3")),
        "temp_humid": ids(ct == "TRN_VP1m"),
        "hygro_dry": ids(ct == "HRN_VP4"),
        "hygro_moist": ids(ct == "HRN_VP5"),
        "hygro_cool": ids(ct.isin(["HRN_VP1d", "HRN_VP1l"])),
        # --- касание
        "touch_eye": ids((cc == "mechanosensory") & (sub == "eye bristle")),
        "touch_head": ids((cc == "mechanosensory") & (sub == "head bristle")),
        "touch_groom": ids((cc == "mechanosensory") & (sub == "grooming")),
        "touch_taste": ids((cc == "mechanosensory") & (sub == "taste peg")),
        # --- зрение
        "photo_R16": ids(ct == "R1-6"),
        "photo_R7": ids(ct == "R7"),
        "photo_R8": ids(ct == "R8"),
        "ocelli": ids((cc == "visual") & (sub == "ocellar")),
    }
    # стороны для направленных стимулов
    for k in list(g):
        for s in ("left", "right"):
            g[f"{k}__{s}"] = [i for i in g[k] if side.get(i, "") == s]
    return g


def group(name: str, side: str | None = None) -> list[int]:
    g = _groups()
    key = f"{name}__{side}" if side in ("left", "right") else name
    return g.get(key, [])


def _log_gauss(f, centre, octaves):
    return float(np.exp(-0.5 * (np.log2(max(f, 1e-6) / centre) / octaves) ** 2))


# ----------------------------------------------------------------------- каналы
class Sound:
    """Звук: частота в Гц и громкость 0..1.

    Настройка подгрупп органа Джонстона: A предпочитает высокие частоты с пиком около 400 Гц
    и отвечает в диапазоне 100–1000 Гц, B — низкие, ниже 100 Гц (Kamikouchi et al. 2009;
    Matsuo et al. 2014). Форма кривой (логнормальная, ширина в октавах) — наша подгонка.
    """

    provenance = {
        "центры настройки": Provenance("measured", "JO-A ~400 Гц, JO-B ~50 Гц"),
        "форма кривой": Provenance(
            "literature", "логнормальная, ширина 1,2 и 1,5 октавы — наша подгонка"
        ),
        "громкость→частота разрядов": Provenance("engineering", "линейно до 200 Гц"),
    }
    RATE_MAX = 200.0

    def stim(
        self, frequency_hz: float, amplitude: float = 1.0, side: str | None = None
    ) -> list[StimInput]:
        a = float(np.clip(amplitude, 0, 1))
        w = {
            "JO_A": _log_gauss(frequency_hz, 400, 1.2),
            "JO_B": _log_gauss(frequency_hz, 50, 1.5),
            "JO_D": 0.4 * _log_gauss(frequency_hz, 150, 2.0),
            "JO_F": 0.3 * _log_gauss(frequency_hz, 200, 2.0),
        }
        out = []
        for k, v in w.items():
            ids = group(k, side)
            if ids and v * a > 0.02:
                out.append(
                    StimInput(ids, self.RATE_MAX * v * a, key=f"sound{frequency_hz:.0f}Гц:{k}")
                )
        return out


class Wind:
    """Ветер: направление в градусах (0 спереди, +90 справа) и сила 0..1.

    Подгруппы C и E органа Джонстона отвечают на статичное отклонение антенны — ветер и
    гравитацию (Kamikouchi et al. 2009; Yorozu et al. 2009). Зависимость силы от направления
    (косинус с перекосом на ближнюю антенну) — наша конструкция.
    """

    provenance = {
        "какие подгруппы": Provenance("measured", "JO-C и JO-E — статичное отклонение"),
        "направленность": Provenance(
            "engineering", "косинус с полом 0,35: сзади слабее, но не ноль"
        ),
    }
    RATE_MAX = 200.0

    def stim(self, direction_deg: float = 0.0, speed: float = 1.0) -> list[StimInput]:
        s = float(np.clip(speed, 0, 1))
        d = np.radians(direction_deg)
        lr = {"left": 0.5 * (1 - np.sin(d)), "right": 0.5 * (1 + np.sin(d))}
        front = 0.35 + 0.65 * (
            0.5 + 0.5 * np.cos(d)
        )  # сзади антенна тоже отклоняется, просто слабее
        out = []
        for k in ("JO_C", "JO_E", "JO_D"):
            for sd, bal in lr.items():
                ids = group(k, sd)
                r = self.RATE_MAX * s * front * bal * (1.0 if k != "JO_D" else 0.5)
                if ids and r > 4:
                    out.append(StimInput(ids, r, key=f"wind{direction_deg:.0f}°:{k}/{sd}"))
        return out


class Gravity(Wind):
    """Наклон тела: те же C и E, что и ветер — они не различают источник отклонения."""

    provenance = {
        "какие подгруппы": Provenance("measured", "JO-C и JO-E отвечают и на гравитацию"),
        "угол→сила": Provenance("engineering", "наша конструкция"),
    }


class Taste:
    """Вкус по модальностям: сладкое, горькое, соль, вода; место — хоботок, лапки, глотка.

    Функциональные подклассы вкусовых нейронов размечены в аннотациях FlyWire (Schlegel 2024).
    Зависимость частоты разрядов от концентрации — наша конструкция: у Weiss et al. 2011 есть
    матрица 16 горьких веществ × 31 волосок, но её стык с типами LB1–LB4 ещё не сделан.
    """

    provenance = {
        "кто за что отвечает": Provenance(
            "measured", "подклассы sugar/water, bitter, low-salt из аннотаций"
        ),
        "концентрация→разряды": Provenance("engineering", "насыщение, полумаксимум на 0,3"),
        "по веществам": Provenance("engineering", "не сделано: нужен стык Weiss 2011 с типами LB"),
    }
    RATE_MAX = 200.0
    SITES = {
        "labellum": {
            "sugar": "taste_sugar",
            "bitter": "taste_bitter",
            "salt": "taste_salt",
            "water": "taste_sugar",
        },
        "tarsi": {
            "sugar": "taste_tarsal",
            "bitter": "taste_tarsal",
            "salt": "taste_tarsal",
            "water": "taste_tarsal",
        },
        "pharynx": {
            "sugar": "taste_pharynx",
            "bitter": "taste_pharynx",
            "salt": "taste_pharynx",
            "water": "taste_pharynx",
        },
    }

    def stim(
        self,
        modality: str,
        concentration: float = 1.0,
        site: str = "labellum",
        side: str | None = None,
    ) -> list[StimInput]:
        c = float(np.clip(concentration, 0, 1))
        rate = self.RATE_MAX * c / (c + 0.3)
        name = self.SITES.get(site, self.SITES["labellum"]).get(modality)
        ids = group(name, side) if name else []
        return [StimInput(ids, rate, key=f"taste:{modality}@{site}")] if ids and rate > 4 else []


class Temperature:
    """Температура в °C. Нейтраль около 25 °C: выше растёт ответ TRN_VP2, ниже — TRN_VP3.

    Направление настройки измерено (Gallio et al. 2011; Frank et al. 2015), форма зависимости
    от градусов — наша конструкция.
    """

    provenance = {
        "кто на что": Provenance("measured", "VP2 нагрев, VP3 холод, VP1m влажность"),
        "°C→разряды": Provenance("engineering", "линейно от нейтрали 25 °C, насыщение на ±10 °C"),
    }
    RATE_MAX = 200.0
    NEUTRAL = 25.0

    def stim(self, celsius: float) -> list[StimInput]:
        d = (celsius - self.NEUTRAL) / 10.0
        out = []
        if d > 0.02:
            out.append(
                StimInput(
                    group("temp_hot"), self.RATE_MAX * min(1.0, d), key=f"temp{celsius:.0f}°:hot"
                )
            )
        elif d < -0.02:
            out.append(
                StimInput(
                    group("temp_cold"), self.RATE_MAX * min(1.0, -d), key=f"temp{celsius:.0f}°:cold"
                )
            )
        return [s for s in out if s.ids and s.rate_hz > 4]


class Humidity:
    """Влажность в процентах. Нейтраль 60 %: суше — HRN_VP4, влажнее — HRN_VP5."""

    provenance = {
        "кто на что": Provenance(
            "measured", "VP4 сухость, VP5 влага (Enjin et al. 2016; Knecht et al. 2016)"
        ),
        "%→разряды": Provenance("engineering", "линейно от нейтрали 60 %"),
    }
    RATE_MAX = 200.0
    NEUTRAL = 60.0

    def stim(self, percent: float) -> list[StimInput]:
        d = (percent - self.NEUTRAL) / 40.0
        out = []
        if d > 0.02:
            out.append(
                StimInput(
                    group("hygro_moist"),
                    self.RATE_MAX * min(1.0, d),
                    key=f"hum{percent:.0f}%:moist",
                )
            )
        elif d < -0.02:
            out.append(
                StimInput(
                    group("hygro_dry"), self.RATE_MAX * min(1.0, -d), key=f"hum{percent:.0f}%:dry"
                )
            )
        return [s for s in out if s.ids and s.rate_hz > 4]


class Touch:
    """Касание щетинок: место (глаз, голова, груминговые) и сила 0..1."""

    provenance = {
        "расположение щетинок": Provenance(
            "measured", "подклассы eye/head bristle, grooming из аннотаций"
        ),
        "сила→разряды": Provenance("engineering", "линейно до 200 Гц"),
        "сколько щетинок задето": Provenance("engineering", "доля популяции пропорциональна силе"),
    }
    RATE_MAX = 200.0
    SITES = {
        "eye": "touch_eye",
        "head": "touch_head",
        "groom": "touch_groom",
        "labellum": "touch_taste",
    }

    def stim(
        self, site: str = "head", intensity: float = 1.0, side: str | None = None, seed: int = 0
    ) -> list[StimInput]:
        i = float(np.clip(intensity, 0, 1))
        ids = group(self.SITES.get(site, "touch_head"), side)
        if not ids or i < 0.02:
            return []
        k = max(1, int(round(len(ids) * min(1.0, 0.2 + 0.8 * i))))
        rng = np.random.default_rng(seed)
        sel = sorted(rng.choice(ids, k, replace=False).tolist()) if k < len(ids) else ids
        return [StimInput(sel, self.RATE_MAX * i, key=f"touch:{site}")]


class Light:
    """Свет: яркость 0..1 на фоторецепторы. Работает плохо и это известно:
    оптическая доля в основном градуальная, а LIF заставляет её спайковать."""

    provenance = {
        "кто фоторецепторы": Provenance("measured", "R1-6, R7, R8 из аннотаций"),
        "яркость→разряды": Provenance("engineering", "линейно"),
        "ограничение": Provenance("engineering", "сигнал не проходит дальше ламины, см. docs"),
    }
    RATE_MAX = 100.0

    def stim(
        self, brightness: float = 1.0, uv: float = 0.0, subsample: int = 400, seed: int = 0
    ) -> list[StimInput]:
        rng = np.random.default_rng(seed)
        out = []
        for name, level in (("photo_R16", brightness), ("photo_R7", uv)):
            ids = group(name)
            if ids and level > 0.02:
                if len(ids) > subsample:
                    ids = sorted(rng.choice(ids, subsample, replace=False).tolist())
                out.append(
                    StimInput(ids, self.RATE_MAX * float(np.clip(level, 0, 1)), key=f"light:{name}")
                )
        return out


CHANNELS = {
    "sound": Sound(),
    "wind": Wind(),
    "gravity": Gravity(),
    "taste": Taste(),
    "temperature": Temperature(),
    "humidity": Humidity(),
    "touch": Touch(),
    "light": Light(),
}


def describe_channels() -> str:
    lines = []
    for name, ch in CHANNELS.items():
        lines.append(f"{name} ({ch.__class__.__name__}):")
        for k, p in ch.provenance.items():
            lines.append(f"   {k:28s} [{p.level}] {p.note}")
    return "\n".join(lines)
