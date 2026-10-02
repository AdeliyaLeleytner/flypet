"""Talk to the fly: text -> stimulus plan (LLM) -> whole-brain simulation -> readout -> fly's reply (LLM).

Usage:
    python -m flypet.chat                       # interactive
    python -m flypet.chat --once "сахар на хоботок"
    FLYPET_LLM=ollama python -m flypet.chat     # local Qwen via Ollama
"""

from __future__ import annotations
import argparse, json, math, os, re, sys, time
from datetime import datetime
from pathlib import Path
from .catalog import STIMULI
from .engine import Brain, StimInput
from . import analysis as A, corrections
from .llm import get_llm
from .mb import MushroomBody, OdorEncoder
from .odor import DoorOdor, resolve_object
from .senses import CHANNELS, describe_channels
from .vnc import Vnc

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.environ.get("FLYPET_STATE_DIR", ROOT / "data"))
SESSIONS = STATE_DIR / "sessions"
MEMORY_PATH = STATE_DIR / "memory.npz"

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "is_stimulus": {
            "type": "boolean",
            "description": "true if the message describes something happening to the fly that maps onto the catalog",
        },
        "interpretation": {
            "type": "string",
            "description": "one short Russian sentence: how the message was interpreted",
        },
        "duration_ms": {"type": "integer", "minimum": 100, "maximum": 1000},
        "stimuli": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "side": {"type": "string", "enum": ["left", "right", "both"]},
                    "rate_hz": {"type": "number", "minimum": 0, "maximum": 300},
                },
                "required": ["key", "side", "rate_hz"],
                "additionalProperties": False,
            },
        },
        "odor_text": {
            "type": "string",
            "description": "object the fly is given to smell, as a short noun phrase in Russian (e.g. 'яблоко'), or empty string",
        },
        "reinforcement": {
            "type": "string",
            "enum": ["none", "reward", "punish"],
            "description": "reward = food/sugar/praise given together with the smell (PAM dopamine); punish = shock, bitter, heat, shaking together with the smell (PPL1 dopamine)",
        },
        "channels": {
            "type": "array",
            "description": "физические каналы с параметрами; предпочитай их каталогу, когда в сообщении есть число или направление",
            "items": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "enum": [
                            "sound",
                            "wind",
                            "gravity",
                            "taste",
                            "temperature",
                            "humidity",
                            "touch",
                            "light",
                        ],
                    },
                    "params": {
                        "type": "object",
                        "description": "параметры канала, см. список ниже",
                        "additionalProperties": True,
                    },
                },
                "required": ["channel", "params"],
                "additionalProperties": False,
            },
        },
        "reply_if_no_stimulus": {
            "type": "string",
            "description": "if is_stimulus is false: what the fly should answer, in Russian, else empty",
        },
    },
    "required": [
        "is_stimulus",
        "interpretation",
        "duration_ms",
        "stimuli",
        "channels",
        "odor_text",
        "reinforcement",
        "reply_if_no_stimulus",
    ],
    "additionalProperties": False,
}


def catalog_text() -> str:
    lines = []
    for k, s in STIMULI.items():
        lines.append(
            f"- {k}: {s.label_ru}. {s.description}. default {s.default_rate_hz} Hz. expected: {s.expected}"
        )
    return "\n".join(lines)


PLANNER_SYSTEM = """Ты переводчик между человеком и компьютерной моделью мозга дрозофилы (leaky integrate-and-fire модель на полном коннектоме FlyWire, 138 639 нейронов, Shiu et al. 2024).
Человек описывает, что происходит с мухой. Твоя задача: выбрать, какие сенсорные нейроны из каталога активировать, с какой стороны и с какой частотой, и на сколько миллисекунд.
Правила:
- Используй только ключи из каталога. Один стимул может быть слабее или сильнее: частота 20–300 Гц (по умолчанию из каталога). Несколько стимулов можно сочетать.
- Сторона: left/right/both. Если не сказано, both.
- Длительность 200–500 мс обычно достаточно; 100 для короткого касания, до 1000 для долгого воздействия.
- Если сообщение не описывает воздействие на муху (вопрос, приветствие, разговор), поставь is_stimulus=false и предложи ответ в reply_if_no_stimulus, честно: муха-модель не чувствует ничего, пока на её сенсоры ничего не подано. Можно ссылаться на историю разговора.
- Если описанное воздействие не покрывается каталогом (например, боль в лапке, музыка Моцарта), подбери ближайшее и скажи об этом в interpretation.
- Запахи предметов: любой предмет, который подносят понюхать, идёт в odor_text (короткое существительное). Это отдельный канал: слово превращается в набор гломерул обонятельных нейронов.
- Подкрепление: еда, сахар, похвала вместе с запахом → reinforcement=reward (дофамин PAM, муха запоминает запах как хороший); удар током, горечь, жар, встряска вместе с запахом → punish (дофамин PPL1). Без запаха подкрепление не запоминается. "Запомни, что X вкусно" = odor_text X + reward.
Примеры (сообщение → JSON):
- "Капнула сахар на хоботок" → {"is_stimulus": true, "interpretation": "сахар на лабеллуме", "duration_ms": 300, "stimuli": [{"key": "sugar", "side": "both", "rate_hz": 150}], "reply_if_no_stimulus": ""}
- "На тебя летит мухобойка!" → {"is_stimulus": true, "interpretation": "надвигающийся объект, детекторы LC4/LPLC2", "duration_ms": 300, "stimuli": [{"key": "looming", "side": "both", "rate_hz": 200}], "reply_if_no_stimulus": ""}
- "Дую на левую антенну" → {"is_stimulus": true, "interpretation": "ветер на левую антенну, орган Джонстона", "duration_ms": 300, "stimuli": [{"key": "antenna_touch", "side": "left", "rate_hz": 150}], "reply_if_no_stimulus": ""}
- "Дай понюхать яблоко" → {"is_stimulus": true, "interpretation": "запах яблока", "duration_ms": 200, "stimuli": [], "odor_text": "яблоко", "reinforcement": "none", "reply_if_no_stimulus": ""}
- "Вот яблоко, и я даю тебе сахар" → {"is_stimulus": true, "interpretation": "запах яблока с наградой", "duration_ms": 200, "stimuli": [{"key": "sugar", "side": "both", "rate_hz": 150}], "odor_text": "яблоко", "reinforcement": "reward", "reply_if_no_stimulus": ""}
- "Понюхай молоток и получи разряд тока" → {"is_stimulus": true, "interpretation": "запах молотка с наказанием", "duration_ms": 200, "stimuli": [], "odor_text": "молоток", "reinforcement": "punish", "reply_if_no_stimulus": ""}
- "Как ты?" → {"is_stimulus": false, "interpretation": "вопрос, стимула нет", "duration_ms": 100, "stimuli": [], "odor_text": "", "reinforcement": "none", "reply_if_no_stimulus": "Сенсоры пусты, чувствовать нечего. Последнее, что было, — сахар на хоботке."}
Каждое новое сообщение интерпретируй заново, не повторяй прошлый стимул, если его не назвали.
Физические каналы — используй их, когда в сообщении есть частота, направление, градусы, концентрация или сила:
- sound: {"frequency_hz": 30…1000, "amplitude": 0…1, "side": "left"/"right"/null} — звук; 400 Гц ловят одни нейроны, 50 Гц другие
- wind: {"direction_deg": 0 спереди, 90 справа, -90 слева, 180 сзади, "speed": 0…1} — поток воздуха на антенны
- gravity: те же параметры — наклон тела
- taste: {"modality": "sugar"/"bitter"/"salt"/"water", "concentration": 0…1, "site": "labellum"/"tarsi"/"pharynx", "side": …}
- temperature: {"celsius": 10…40} — нейтраль 25
- humidity: {"percent": 0…100} — нейтраль 60
- touch: {"site": "head"/"eye"/"groom"/"labellum", "intensity": 0…1, "side": …}
- light: {"brightness": 0…1, "uv": 0…1} — предупреждение: зрение в этой модели почти не работает
Каталог стимулов (грубее, без параметров — бери, если числа не названы):
""" + catalog_text()

REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string", "description": "реплика мухи, 1-3 предложения по-русски"}
    },
    "required": ["reply"],
    "additionalProperties": False,
}

NARRATOR_SYSTEM = """Ты — голос компьютерной модели мозга дрозофилы (полный коннектом FlyWire, модель Shiu et al. 2024). Человек только что подал стимул, симуляция отработала, и у тебя есть показания: какие нейроны поведения (мотонейроны хоботка, нисходящие нейроны побега, ходьбы, поворота, груминга) активировались и с какой частотой.
Напиши ответ мухи от первого лица по-русски, 1–3 коротких предложения. Правила:
- Опирайся ТОЛЬКО на показания. Если поведенческий канал молчит, не придумывай реакцию. Активность выходных нейронов — сигнал, связанный с действием, а не наблюдение движения: физическое тело здесь не симулируется. Говори «активировался сигнал к прыжку», а не «я прыгнула».
- Можно упомянуть, что чувствуется стимул (это сенсорные нейроны, их подали). Эмоции — только как осторожная интерпретация в стиле мухи, без человеческих метафор про "радость" и "любовь".
- Не выдумывай числа. Можно назвать одно-два ключевых: частоту мотонейрона хоботка или гигантского волокна.
- Помни предыдущие реплики (история дана), чтобы ответ был связным: например, сравни с прошлым разом.
- Если есть строка «что вторая модель прочитала прямо из нейронов», это независимое описание той же пробы, сделанное по 16 мягким токенам из 11 тысяч нейронов; если оно расходится с показаниями, верь показаниям и можешь упомянуть расхождение одной фразой.
- Запах и память: если дан запах, есть строка «валентность»: наивная (какой была бы без памяти) и с памятью, и разница. Разница больше +0.1 — запах теперь тянет (приближение), меньше −0.1 — отталкивает. Это память грибовидного тела, изменённые синапсы KC→MBON, а не история разговора. Если разница мала, память об этом запахе нейтральна.
- Если есть строка «тело», это мотонейроны брюшной нервной цепочки второго коннектома (MaleCNS): они связаны с мышцами ног, крыльев, прыжка, шеи, брюшка. Это нейронный моторный выход, не измеренное сокращение или движение тела. Хоботок сюда не входит, он читается в мозге.
- Запах предмета сопоставляет с одним веществом языковой интерфейс: это приближение, а не измерение состава предмета. Статус unmapped означает отсутствие соответствия в данных, error — техническую ошибку. Ни один из них не доказывает отсутствие запаха или способности настоящей мухи его чувствовать.
- Никаких приветствий и объяснений про модель, только реплика мухи. Без эмодзи.
Верни JSON вида {"reply": "..."}."""


class FlyChat:
    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        log: bool = True,
        *,
        state_dir: str | Path | None = None,
        seed: int = 1000,
        enable_door: bool | None = None,
        enable_vnc: bool | None = None,
        enable_reader: bool | None = None,
        require_components: tuple[str, ...] = (),
        strict_door: bool = False,
        load_memory: bool = True,
        persist_memory: bool = True,
    ):
        self.state_dir = (
            Path(state_dir)
            if state_dir is not None
            else Path(os.environ.get("FLYPET_STATE_DIR", STATE_DIR))
        )
        self.memory_path = self.state_dir / "memory.npz"
        self.resolve_cache_path = self.state_dir / "odor_resolved.json"
        self.seed = int(seed)
        self._simulation_count = 0
        self.strict_door = strict_door
        self.persist_memory = persist_memory
        self.require_components = set(require_components)
        unknown = self.require_components - {"door", "vnc", "reader"}
        if unknown:
            raise ValueError(f"unknown required components: {sorted(unknown)}")
        self.component_status = {}
        self._llm = None
        self._llm_args = (backend, model)
        self.brain = corrections.apply(Brain())
        self.mb = MushroomBody(self.brain)
        self.enc = None  # legacy projection is lazy and never used by strict DoOR cases
        self.door = self._component("door", enable_door, lambda: DoorOdor(rate_max=150.0))
        if load_memory and self.memory_path.exists():
            try:
                self.mb.load(self.memory_path)
            except Exception as e:
                # A corrupt checkpoint must never silently become a naive animal.
                raise RuntimeError(f"could not load memory {self.memory_path}") from e
        self.history: list[dict] = []
        self.vnc = self._component("vnc", enable_vnc, Vnc)

        def make_reader():
            from .projector import BrainReader, DEFAULT_PT

            if not DEFAULT_PT.exists():
                raise FileNotFoundError(DEFAULT_PT)
            return BrainReader()

        self.reader = self._component("reader", enable_reader, make_reader)
        sessions = self.state_dir / "sessions"
        if log:
            sessions.mkdir(parents=True, exist_ok=True)
        self.log_path = sessions / f"{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl" if log else None

    def _component(self, name, enabled, factory):
        if enabled is None:
            enabled = os.environ.get(f"FLYPET_{name.upper()}", "1") != "0"
        if not enabled:
            self.component_status[name] = {"status": "disabled"}
            if name in self.require_components:
                raise RuntimeError(f"required component {name} is disabled")
            return None
        try:
            result = factory()
        except Exception as e:
            self.component_status[name] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            if name in self.require_components:
                raise RuntimeError(f"required component {name} is unavailable") from e
            print(f"[{name}] unavailable: {e}")
            return None
        self.component_status[name] = {"status": "ready"}
        return result

    def reset(self, *, memory: bool = False, history: bool = True, persist_memory: bool = False):
        """Explicit session reset; learned checkpoints are retained unless requested."""
        if persist_memory and not memory:
            raise ValueError("persist_memory reset requires memory=True")
        if memory:
            self.mb.reset()
            if persist_memory:
                self.mb.save(self.memory_path)
        if history:
            self.history.clear()
        self._simulation_count = 0

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm(*self._llm_args)
        return self._llm

    def _history_text(self, n: int = 6) -> str:
        if not self.history:
            return "(история пуста)"
        out = []
        for h in self.history[-n:]:
            out.append(
                f"Человек: {h['user']}\nСтимул: {h['plan'].get('interpretation', '')}\nПоказания: {h.get('behaviours_short', '')}\nМуха: {h['reply']}"
            )
        return "\n---\n".join(out)

    def plan(self, user: str) -> dict:
        prompt = f"История разговора:\n{self._history_text()}\n\nНовое сообщение человека: {user}\n\nВерни JSON по схеме."
        p = self.llm.complete(PLANNER_SYSTEM, prompt, schema=PLAN_SCHEMA)
        p["stimuli"] = [s for s in p.get("stimuli", []) if s.get("key") in STIMULI]
        p["odor_text"] = (p.get("odor_text") or "").strip()
        p["reinforcement"] = p.get("reinforcement") or "none"
        p["channels"] = [c for c in (p.get("channels") or []) if c.get("channel") in CHANNELS]
        if p["odor_text"] or p["reinforcement"] != "none" or p["channels"]:
            p["is_stimulus"] = True
        return self._dedupe(p)

    def odor_stimulus(
        self, word: str, *, compound: str | None = None, scale: float = 1.0, seed: int | None = None
    ):
        """Natural-language proxy or direct chemical -> measured receptor responses."""
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("odor_scale must be finite and nonnegative")
        info = {
            "source": "door",
            "status": "error",
            "compound": "",
            "why": "DoOR unavailable",
            "mapping": "direct" if compound is not None else "llm_proxy",
            "glomeruli": [],
            "n_glomeruli": 0,
        }
        if self.door is not None:
            try:
                r = (
                    {"compound": compound, "status": "resolved", "why": "explicit chemical input"}
                    if compound is not None
                    else resolve_object(
                        word, self.llm, self.door, cache_path=self.resolve_cache_path
                    )
                )
                info.update(r)
                comp = r.get("compound", "")
                if r["status"] == "unmapped":
                    return [], info
                if r["status"] == "error":
                    raise RuntimeError(r["why"])
                gl = self.door.glomerular(comp)
                st = self.door.stim(comp, scale=scale)
                info.update(
                    status="resolved" if gl else "unmapped",
                    scale=scale,
                    glomeruli=[(g, float(v)) for g, v in sorted(gl.items(), key=lambda x: -x[1])],
                    n_glomeruli=len(st),
                    n_measured_glomeruli=len(gl),
                )
                if not gl:
                    info["why"] = "chemical has no measured receptor mapping into this connectome"
                return st, info
            except Exception as e:
                info.update(status="error", why=f"{type(e).__name__}: {e}")
        if self.strict_door or compound is not None or "door" in self.require_components:
            raise RuntimeError(f"DoOR stimulus failed: {info['why']}")
        # Interactive compatibility only: explicitly report the synthetic fallback.
        if self.enc is None:
            self.enc = OdorEncoder(k_active=3, rate_hz=100.0, frac=1.0)
        st = self.enc.stim(word, seed=self.seed if seed is None else seed)
        st.rate_hz *= scale
        return [st], {
            **info,
            "source": "random",
            "fallback_reason": info["why"],
            "why": "synthetic projection; measured DoOR input unavailable",
            "glomeruli": [(g, None) for g in self.enc.glomerular_code(word)],
            "n_glomeruli": 3,
        }

    CHANNEL_COVERS = {
        "sound": {"sound"},
        "wind": {"antenna_touch"},
        "gravity": {"antenna_touch"},
        "taste": {"sugar", "bitter", "salt_low", "water", "sugar_tarsal", "pharyngeal"},
        "temperature": {"heat", "cold"},
        "humidity": {"dry_air", "moist_air"},
        "touch": {"head_touch", "eye_touch", "grooming_bristles"},
        "light": {"light", "uv_light"},
    }

    def _dedupe(self, p: dict) -> dict:
        """Если канал уже покрывает модальность, убрать грубый дубль из каталога."""
        covered = set()
        for c in p.get("channels", []):
            covered |= self.CHANNEL_COVERS.get(c.get("channel"), set())
        if covered:
            dropped = [s["key"] for s in p["stimuli"] if s["key"] in covered]
            p["stimuli"] = [s for s in p["stimuli"] if s["key"] not in covered]
            if dropped:
                p["deduped"] = dropped
        return p

    def associations(self) -> list[str]:
        return [e["note"] for e in self.mb.log if e.get("n_synapses_changed", 0) > 0]

    def simulate(self, p: dict, seed: int | None = None, *, read_brain: bool = True):
        if not read_brain and "reader" in self.require_components:
            raise ValueError("cannot skip a required brain reader")
        probe_seed = self.seed + self._simulation_count if seed is None else int(seed)
        inputs = []
        for s in p.get("stimuli", []):
            st = STIMULI[s["key"]]
            side = None if s.get("side", "both") == "both" else s["side"]
            ids = st.resolve(side=side)
            if not ids and side:  # Shiu's sets are one-sided; fall back to whatever exists
                ids = st.resolve()
            rate = s.get("rate_hz")
            rate = st.default_rate_hz if rate is None else float(rate)
            if not math.isfinite(rate) or rate < 0:
                raise ValueError("stimulus rate_hz must be finite and nonnegative")
            inputs.append(StimInput(ids, rate, key=f"{s['key']}/{s.get('side', 'both')}"))
        for ch in p.get("channels", []) or []:
            c = CHANNELS.get(ch.get("channel"))
            if c is None:
                continue
            inputs.extend(c.stim(**(ch.get("params") or {})))
        compound = p.get("odor_compound")
        odor = p.get("odor_text") or compound or ""
        reinf = p.get("reinforcement", "none")
        odor_stim, odor_info = (
            self.odor_stimulus(
                odor, compound=compound, scale=float(p.get("odor_scale", 1.0)), seed=probe_seed
            )
            if odor
            else (None, None)
        )
        has_odor_input = bool(odor_stim and any(st.ids and st.rate_hz > 0 for st in odor_stim))
        if odor_stim:
            inputs.extend(odor_stim)
        if has_odor_input and reinf == "reward":
            inputs.append(self.mb.reward())
        elif has_odor_input and reinf == "punish":
            inputs.append(self.mb.punishment())
        dur = int(p.get("duration_ms", 300))
        res = self.brain.run(inputs, duration_ms=dur, seed=probe_seed)
        self._simulation_count += 1
        summary = A.summarize(res)
        summary["simulation"] = {
            "seed": probe_seed,
            "duration_ms": dur,
            "reader_requested": read_brain,
            "inputs": [
                {"ids": [int(i) for i in st.ids], "rate_hz": float(st.rate_hz), "key": st.key}
                for st in inputs
            ],
            "components": {k: dict(v) for k, v in self.component_status.items()},
        }
        if self.vnc is not None:
            try:
                vr = self.vnc.run_from_brain(res, duration_ms=min(dur, 200), seed=probe_seed)
                summary["body"] = vr.brief()
            except Exception as e:
                if "vnc" in self.require_components:
                    raise RuntimeError("required VNC simulation failed") from e
                summary["body"] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        if read_brain and self.reader is not None:
            try:
                summary["brain_reading"] = self.reader.describe(res)
            except Exception as e:
                if "reader" in self.require_components:
                    raise RuntimeError("required brain reader failed") from e
                summary["brain_reading"] = f"(ошибка чтения: {e})"
        if odor:
            summary["odor_source"] = odor_info
            summary["odor_input_present"] = has_odor_input
        if has_odor_input:
            learned = self.mb.valence(res)
            # наивный близнец: ТЕ ЖЕ входы и ТО ЖЕ зерно, отличается только память —
            # иначе разница смешивает эффект памяти с пуассоновским шумом и с разным набором стимулов
            mem = self.mb.mem.copy()
            try:
                self.mb.mem[:] = 1.0
                self.mb._push()
                naive = self.mb.valence(self.brain.run(inputs, duration_ms=dur, seed=probe_seed))
            finally:
                self.mb.mem = mem
                self.mb._push()
            glom = [g for g, _ in odor_info.get("glomeruli", [])]
            summary["valence"] = {
                "odor": odor,
                "glomeruli": glom,
                "naive_score": round(naive.score, 2),
                "score": round(learned.score, 2),
                "delta": round(learned.score - naive.score, 2),
                "approach_hz": round(learned.approach_hz),
                "avoid_hz": round(learned.avoid_hz),
                "top_mbons": learned.top[:4],
            }
            if reinf in ("reward", "punish"):
                e = self.mb.learn(res, note=f"{odor}:{reinf}")
                if self.persist_memory:
                    self.mb.save(self.memory_path)
                summary["learning"] = {
                    "reinforcement": reinf,
                    "synapses_changed": e["n_synapses_changed"],
                    "memory_strength": round(self.mb.memory_strength(), 4),
                }
            summary["associations"] = self.associations()[-12:]
        return res, summary

    def narrate(self, user: str, p: dict, summary: dict) -> str:
        short = A.behaviours_text(summary)
        top = ", ".join(
            f"{r['cell_type']}({r['super_class']},{r['side'][:1]}) {r['rate_hz']}Hz"
            for r in summary["top_downstream"][:12]
        )
        prompt = (
            f"История:\n{self._history_text()}\n\nСообщение человека: {user}\n"
            f"Как интерпретирован стимул: {p.get('interpretation', '')}\n"
            f"Поданные стимулы: {json.dumps(summary['stimulated'], ensure_ascii=False)}; длительность {summary['duration_ms']} мс\n"
            f"Активных нейронов ниже по цепи: {summary['n_active_downstream']} (по классам {summary['active_by_super_class']})\n"
            f"Поведенческие каналы:\n{short}\n"
            f"Самые активные нейроны: {top}\n"
            + (
                f"Вещество, поданное через DoOR (соответствие предмету — приближение интерфейса): {summary['odor_source']['compound']} "
                f"(гломерулы {summary['odor_source']['glomeruli']}); {summary['odor_source']['why']}\n"
                if summary.get("odor_source", {}).get("compound")
                and summary["odor_source"].get("source") == "door"
                and summary["odor_source"].get("status") == "resolved"
                else ""
            )
            + (
                f"Статус запахового входа: {summary['odor_source'].get('status', 'unknown')}; "
                f"источник: {summary['odor_source'].get('source')}; {summary['odor_source'].get('why', '')}. "
                "Отсутствие соответствия или ошибка не доказывают отсутствие запаха; синтетическая проекция не является измеренным откликом.\n"
                if summary.get("odor_source")
                else ""
            )
            + (
                f"Запах '{summary['valence']['odor']}' (гломерулы {summary['valence']['glomeruli']}): валентность наивная {summary['valence']['naive_score']:+.2f}, с памятью {summary['valence']['score']:+.2f}, разница {summary['valence']['delta']:+.2f}; MBON приближения {summary['valence']['approach_hz']} Гц, избегания {summary['valence']['avoid_hz']} Гц\n"
                if "valence" in summary
                else ""
            )
            + (
                f"Обучение: {summary['learning']['reinforcement']}, изменено синапсов KC→MBON {summary['learning']['synapses_changed']}\n"
                if "learning" in summary
                else ""
            )
            + (
                f"Моторный нейронный выход (MaleCNS; движение тела не симулируется): {summary['body']['мышцы']}; "
                f"самые активные: {', '.join(summary['body']['мотонейроны'])}\n"
                if summary.get("body", {}).get("мотонейроны")
                else ""
            )
            + (
                f"Что вторая модель (проектор состояния → Qwen) прочитала прямо из нейронов, без показаний: «{summary['brain_reading']}»\n"
                if summary.get("brain_reading")
                else ""
            )
            + (
                f"Ассоциации в памяти (запах:подкрепление): {summary['associations']}\n"
                if summary.get("associations")
                else ""
            )
            + "\nОтвет мухи в JSON:"
        )
        out = self.llm.complete(NARRATOR_SYSTEM, prompt, schema=REPLY_SCHEMA)
        if isinstance(out, str):
            try:
                out = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
            except Exception:
                return out.strip()
        return str(out.get("reply", "") if isinstance(out, dict) else out).strip()

    def turn(self, user: str, verbose: bool = True) -> dict:
        t0 = time.time()
        p = self.plan(user)
        if verbose:
            chans = [(c["channel"], c["params"]) for c in p.get("channels", [])]
            print(
                f"  [план] {p.get('interpretation', '')} | каталог {[(s['key'], s['side'], s['rate_hz']) for s in p['stimuli']]} | каналы {chans} | {p.get('duration_ms')} мс",
                flush=True,
            )
        entry = {"user": user, "plan": p, "t": datetime.now().isoformat()}
        if p.get("is_stimulus") and (
            p["stimuli"] or p.get("odor_text") or p.get("odor_compound") or p.get("channels")
        ):
            res, summary = self.simulate(p)
            reply = self.narrate(user, p, summary)
            entry["summary"] = summary
            entry["behaviours_short"] = (
                "; ".join(
                    l.split(" — ")[0] + (" — " + l.split(" — ")[1][:60] if " — " in l else "")
                    for l in A.behaviours_text(summary).splitlines()
                    if "silent" not in l
                )
                or "все каналы молчат"
            )
            if verbose:
                print(
                    f"  [симуляция] {summary['duration_ms']} мс, {summary['wall_s']} с; активных нейронов {summary['n_active_downstream']}, спайков {summary['n_spikes_total']}"
                )
                print("  [показания]\n    " + A.behaviours_text(summary).replace("\n", "\n    "))
                if summary.get("body", {}).get("мышцы"):
                    print(f"  [тело, ВНЦ MaleCNS] {summary['body']['мышцы']}")
                    print("    мотонейроны: " + ", ".join(summary["body"]["мотонейроны"]))
        else:
            reply = p.get("reply_if_no_stimulus") or "…"
            entry["behaviours_short"] = "стимула не было"
        entry["reply"] = reply
        entry["wall_s"] = round(time.time() - t0, 1)
        self.history.append(entry)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, help="claude-cli | anthropic | ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--once", action="append", help="send this message and exit (repeatable)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    chat = FlyChat(a.backend, a.model)
    print(
        f"LLM: {chat.llm.name} ({chat.llm.model}); мозг: {chat.brain.n} нейронов. Пиши, что делаешь с мухой. Пустая строка — выход.",
        flush=True,
    )
    msgs = a.once or []
    if msgs:
        for m in msgs:
            print(f"\nТы: {m}")
            e = chat.turn(m, verbose=not a.quiet)
            print(f"Муха: {e['reply']}   [{e['wall_s']} с]", flush=True)
        return
    while True:
        try:
            m = input("\nТы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not m:
            break
        e = chat.turn(m, verbose=not a.quiet)
        print(f"Муха: {e['reply']}   [{e['wall_s']} с]", flush=True)


if __name__ == "__main__":
    main()
