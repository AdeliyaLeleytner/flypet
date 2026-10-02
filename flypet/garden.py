"""Validated garden inputs. Bounds are also served to the controls."""

import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .catalog import STIMULI

ODORS = {
    "apple": "hexyl acetate",
    "pear": "ethyl acetate",
    "plum": "ethanol",
    "melon": "methyl hexanoate",
    "mushroom": "geosmin",
    "grape": "ethyl acetate",
}
STIMULUS_KEYS = {
    "sugar": "sugar",
    "bitter": "bitter",
    "touch": "antenna_touch",
    "looming": "looming",
}
Side = Literal["both", "left", "right"]


def number(label, default, minimum, maximum, step=0.1):
    return {"label": label, "default": default, "min": minimum, "max": maximum, "step": step}


def choice(label, default, options):
    return {"label": label, "default": default, "options": options}


CHANNEL_FIELDS = {
    "sound": {
        "frequency_hz": number("Frequency, Hz", 400, 20, 1000, 10),
        "amplitude": number("Amplitude", 0.5, 0, 1),
    },
    "wind": {
        "direction_deg": number("Direction, °", 0, -180, 180, 15),
        "speed": number("Strength", 0.5, 0, 1),
    },
    "gravity": {
        "direction_deg": number("Tilt direction, °", 0, -180, 180, 15),
        "speed": number("Deflection strength", 0.5, 0, 1),
    },
    "taste": {
        "modality": choice("Taste", "sugar", ["sugar", "bitter", "salt", "water"]),
        "concentration": number("Concentration", 0.5, 0, 1),
        "site": choice("Site", "labellum", ["labellum", "tarsi", "pharynx"]),
    },
    "temperature": {"celsius": number("Temperature, °C", 25, 10, 40, 1)},
    "humidity": {"percent": number("Humidity, %", 60, 0, 100, 5)},
    "touch": {
        "site": choice("Site", "head", ["head", "eye", "groom", "labellum"]),
        "intensity": number("Touch intensity", 0.5, 0, 1),
    },
    "light": {"brightness": number("Brightness", 0.5, 0, 1), "uv": number("UV", 0, 0, 1)},
}
CHANNEL_LABELS = dict(
    zip(
        CHANNEL_FIELDS,
        ["Sound", "Wind", "Tilt", "Taste", "Temperature", "Humidity", "Touch", "Light"],
    )
)


def validate_channel(name, values):
    if name not in CHANNEL_FIELDS:
        raise ValueError("unknown physical channel")
    fields = CHANNEL_FIELDS[name]
    if set(values) - fields.keys():
        raise ValueError("unknown channel parameter")
    result = {}
    for key, spec in fields.items():
        value = values.get(key, spec["default"])
        if "options" in spec:
            if value not in spec["options"]:
                raise ValueError(f"invalid {key}")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not spec["min"] <= value <= spec["max"]
        ):
            raise ValueError(f"{key} is outside the supported range")
        result[key] = value
    return result


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class GardenStimulus(StrictModel):
    key: str
    rate_hz: float | None = Field(default=None, ge=0, le=300)

    @field_validator("key")
    @classmethod
    def known_key(cls, key):
        if key not in STIMULI:
            raise ValueError("unknown catalog stimulus")
        return key


class GardenReq(StrictModel):
    action: Literal[
        "smell", "sugar", "bitter", "touch", "looming", "baseline", "stimuli", "channel", "chat"
    ]
    target: Literal["apple", "pear", "plum", "melon", "mushroom", "grape"] | None = None
    compound: str | None = Field(default=None, min_length=1, max_length=128)
    odor_text: str | None = Field(default=None, min_length=1, max_length=120)
    reinforcement: Literal["none", "reward", "punish"] = "none"
    odor_scale: float = Field(default=1, ge=0, le=1)
    stimuli: list[GardenStimulus] = Field(default_factory=list, max_length=4)
    channel: str | None = None
    params: dict = Field(default_factory=dict)
    side: Side = "both"
    duration_ms: int | None = Field(default=None, ge=100, le=1000)
    rate_hz: float | None = Field(default=None, ge=0, le=300)
    seed: int | None = Field(default=None, ge=0, le=4294967295)
    read_brain: bool = False
    text: str | None = Field(default=None, min_length=1, max_length=600)

    @model_validator(mode="after")
    def check_action(self):
        count = sum(x is not None for x in (self.target, self.compound, self.odor_text))
        if (self.action == "smell" and count != 1) or (self.action != "smell" and count):
            raise ValueError("smell requires exactly one target, compound or odor_text")
        if self.reinforcement != "none" and self.action != "smell":
            raise ValueError("reinforcement requires an odor")
        if (self.action == "stimuli") != bool(self.stimuli):
            raise ValueError("stimuli action requires 1–4 catalog inputs")
        if len({s.key for s in self.stimuli}) != len(self.stimuli):
            raise ValueError("duplicate stimuli are not supported")
        if self.action == "channel":
            self.params = validate_channel(self.channel, self.params)
        elif self.channel is not None or self.params:
            raise ValueError("channel parameters require channel action")
        if (self.action == "chat") != bool(self.text and self.text.strip()):
            raise ValueError("chat requires text")
        return self


class GardenReset(StrictModel):
    body: bool = True
    memory: bool = False
    history: bool = False


class ThoughtReq(StrictModel):
    trial_id: str = Field(pattern=r"^[0-9a-f]{32}$")


def make_plan(req: GardenReq, sugar_scale: float = 1.0) -> dict:
    plan = {
        "stimuli": [],
        "duration_ms": req.duration_ms or (200 if req.action == "smell" else 300),
        "reinforcement": req.reinforcement,
    }
    if req.action == "smell":
        plan["odor_scale"] = req.odor_scale
        if req.odor_text:
            plan["odor_text"] = req.odor_text.strip()
        else:
            plan["odor_compound"] = req.compound or ODORS[req.target]
    elif req.action in STIMULUS_KEYS or req.action == "stimuli":
        items = req.stimuli or [GardenStimulus(key=STIMULUS_KEYS[req.action], rate_hz=req.rate_hz)]
        for item in items:
            rate = STIMULI[item.key].default_rate_hz if item.rate_hz is None else item.rate_hz
            if item.key.startswith(("sugar", "water")):
                rate *= sugar_scale
            plan["stimuli"].append({"key": item.key, "side": req.side, "rate_hz": rate})
    elif req.action == "channel":
        params = dict(req.params)
        if req.channel in ("sound", "taste", "touch"):
            params["side"] = None if req.side == "both" else req.side
        if req.channel in ("touch", "light") and req.seed is not None:
            params["seed"] = req.seed
        plan["channels"] = [{"channel": req.channel, "params": params}]
    return plan


def bounded_llm_plan(plan):
    """Treat language-generated plans as untrusted input, not executable instructions."""
    duration = plan.get("duration_ms", 300)
    if not isinstance(duration, int) or isinstance(duration, bool) or not 100 <= duration <= 1000:
        raise ValueError("planner duration is outside 100–1000 ms")
    raw = plan.get("stimuli", [])
    if len(raw) > 4:
        raise ValueError("planner selected too many stimuli")
    out = {
        "stimuli": [],
        "channels": [],
        "duration_ms": duration,
        "reinforcement": plan.get("reinforcement", "none"),
    }
    if out["reinforcement"] not in ("none", "reward", "punish"):
        raise ValueError("invalid planner reinforcement")
    for item in raw:
        clean = GardenStimulus(key=item["key"], rate_hz=item.get("rate_hz"))
        side = item.get("side", "both")
        if side not in ("left", "right", "both"):
            raise ValueError("invalid planner side")
        out["stimuli"].append({**clean.model_dump(), "side": side})
    channels = plan.get("channels", []) or []
    if len(channels) > 2:
        raise ValueError("planner selected too many physical channels")
    for item in channels:
        name, params = item["channel"], dict(item.get("params") or {})
        side = params.pop("side", None)
        valid = validate_channel(name, params)
        if side is not None:
            if name not in ("sound", "taste", "touch") or side not in ("left", "right", "both"):
                raise ValueError("invalid planner channel side")
            valid["side"] = None if side == "both" else side
        out["channels"].append({"channel": name, "params": valid})
    odor = plan.get("odor_text") or ""
    if not isinstance(odor, str) or len(odor) > 120:
        raise ValueError("invalid planner odor")
    if odor.strip():
        out["odor_text"] = odor.strip()
    elif out["reinforcement"] != "none":
        raise ValueError("learning requires an odor")
    return out
