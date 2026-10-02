"""Grounded English descriptions and compact, cached optional LLM reflections."""

import json
import re
from functools import lru_cache

MOTOR_PHRASES = {
    "escape_takeoff": "I have a signal to escape",
    "proboscis_extension": "I have a signal to extend my proboscis",
    "antennal_grooming": "I have a signal to clean my antennae",
    "walk_backward": "I have a signal to move backward",
    "walk_forward": "I have a signal to move forward",
    "turn": "My turning channel is active",
    "head_neck_movement": "My head-movement channel is active",
}


def readout_thought(summary):
    b = summary.get("behaviours", {})
    sentences = [
        f"{phrase} ({b[key]['max_rate_hz']:g} Hz)."
        for key, phrase in MOTOR_PHRASES.items()
        if b.get(key, {}).get("max_rate_hz", 0) > 0
    ][:3]
    if not sentences:
        sentences.append(
            "My tracked motor outputs are silent."
            if summary.get("n_spikes_total", 0)
            else "I had no spikes in this trial, and no motor response."
        )
    v = summary.get("valence")
    if v:
        sentences.append(
            "My mushroom-body readout favors approaching this odor."
            if v["score"] > 0.1
            else "My mushroom-body readout favors avoiding this odor."
            if v["score"] < -0.1
            else "My odor-response index is close to neutral."
        )
        if abs(v["delta"]) > 0.1:
            sentences.append("My learned response differs from the naive comparison.")
    odor = summary.get("odor_source", {})
    if odor.get("status") == "unmapped":
        sentences.append(
            "No odor mapping was found in the data; this does not mean the object has no smell."
        )
    if odor.get("status") == "error":
        sentences.append("The odor input could not be obtained because of a technical error.")
    if summary.get("learning"):
        count = summary["learning"]["synapses_changed"]
        sentences.append(
            f"Reinforcement changed {count} of my memory synapses."
            if count
            else "Reinforcement was delivered, but my memory weights did not change in this trial."
        )
    return {"text": " ".join(sentences), "source": "readout", "status": "ready"}


SYSTEM = (
    "You voice a simulated fruit fly. Rephrase only the evidence in English, in first person, "
    "in one or two short sentences. Say I or my. Describe neural signals, not performed movements, "
    "consciousness or feelings. Silence is a valid response. Do not invent effects, numbers or causes. "
    'Return JSON {"reply":"..."}.'
)


def reflection_evidence(summary):
    # Wording depends only on this exact evidence. Never cache/replay a neural trial.
    text = readout_thought(summary)["text"]
    return re.sub(r" \([0-9.]+ Hz\)", "", text)


@lru_cache(maxsize=128)
def _render_cached(llm, evidence):
    schema = {
        "type": "object",
        "properties": {"reply": {"type": "string"}},
        "required": ["reply"],
        "additionalProperties": False,
    }
    result = llm.complete(SYSTEM, "Evidence: " + evidence, schema=schema, max_tokens=96)
    if not isinstance(result, dict) or not isinstance(result.get("reply"), str):
        raise ValueError("Invalid narrator reply")
    text = result["reply"].strip()
    if (
        not text
        or len(text) > 650
        or re.search(r"[\u0400-\u04ff]", text)
        or not re.search(r"\b(I|my|me)\b", text, re.I)
    ):
        raise ValueError("Narrator must describe its own response in English")
    return {
        "text": text,
        "source": "llm",
        "status": "ready",
        "model": getattr(llm, "last_model_ids", None) or [llm.name],
    }


def narrate_snapshot(llm, snapshot):
    before = _render_cached.cache_info().hits
    result = dict(_render_cached(llm, reflection_evidence(snapshot["summary"])))
    result["cached"] = _render_cached.cache_info().hits > before
    return result
