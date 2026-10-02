"""The pet: a persistent body state around the brain model, served over HTTP with a small web UI.

    python -m flypet.pet            # then open http://127.0.0.1:8765

The connectome model has no internal state between runs, so everything that persists
(satiety, position, the event log) is an explicit body model added here and saved to
data/pet_state.json. Learned KC→MBON weights persist separately in memory.npz.
"""

from __future__ import annotations
import asyncio, copy, json, logging, os, time, uuid
from collections import OrderedDict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .catalog import STIMULI
from .engine import Brain, StimInput
from . import analysis as A
from .chat import FlyChat
from .garden import (
    GardenReq,
    GardenReset,
    ThoughtReq,
    make_plan,
    bounded_llm_plan,
    CHANNEL_FIELDS,
    CHANNEL_LABELS,
)
from .thoughts import readout_thought, narrate_snapshot
from .llm import get_llm
from . import english as EN
from .sessions import SessionStore

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.environ.get("FLYPET_STATE_DIR", ROOT / "data"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "pet_state.json"
STATIC = ROOT / "flypet" / "static"

app = FastAPI(title="flypet")
_allowed_origins = [
    s.strip().rstrip("/")
    for s in os.environ.get("FLYPET_ALLOWED_ORIGINS", "").split(",")
    if s.strip()
]
if "*" in _allowed_origins:
    raise ValueError("FLYPET_ALLOWED_ORIGINS must name exact frontend origins")
if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Flypet-Session"],
        allow_credentials=False,
    )
app.mount(
    "/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static"
)
_exec = ThreadPoolExecutor(max_workers=1)  # Brian2 objects must always live on one thread
_thought_exec = ThreadPoolExecutor(max_workers=1)
_chat: FlyChat | None = None
_brain: Brain | None = None
_garden_task: asyncio.Task | None = None
_thought_task: asyncio.Task | None = None
_thought_trial: str | None = None
_trial_snapshots = OrderedDict()
_snapshot_lock = Lock()
_narrator = None
_sessions = SessionStore()
_current_session = None


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "satiety": 0.3,
        "born": datetime.now().isoformat(),
        "last_tick": time.time(),
        "events": [],
        "n_stimuli": 0,
        "name": "Дрозя",
    }


STATE = load_state()


def save_state():
    if _current_session is not None:
        return
    STATE_PATH.write_text(json.dumps(STATE, ensure_ascii=False, indent=1, default=str))


def tick():
    """Slow body dynamics: satiety decays ~0.02 per minute of wall-clock time."""
    now = time.time()
    dt_min = (now - STATE.get("last_tick", now)) / 60
    STATE["satiety"] = max(0.0, STATE["satiety"] - 0.02 * dt_min)
    STATE["last_tick"] = now


def apply_body(summary: dict, keys: list[str]) -> dict:
    """Update body state from what the brain did. Explicit rules, not connectome-derived."""
    bv = A.behaviour_vector(summary)
    fed = any(k.startswith(("sugar", "water")) for k in keys) and bv["proboscis"] > 0.3
    if fed:
        STATE["satiety"] = min(1.0, STATE["satiety"] + 0.15 * bv["proboscis"])
    return bv


def hunger_scale(key: str) -> float:
    """Hungry flies respond to sugar more strongly; here that is a plain multiplier on the input rate."""
    if key.startswith(("sugar", "water")):
        return 1.0 - 0.6 * STATE["satiety"]
    return 1.0


def _run_stimulus(key: str, side: str | None, rate_hz: float | None, duration_ms: int):
    st = STIMULI[key]
    ids = st.resolve(side=None if side in (None, "both") else side) or st.resolve()
    rate = (st.default_rate_hz if rate_hz is None else rate_hz) * hunger_scale(key)
    res = _brain.run([StimInput(ids, rate, key=f"{key}/{side or 'both'}")], duration_ms=duration_ms)
    summary = A.summarize(res)
    if _chat is not None and getattr(_chat, "vnc", None) is not None:
        try:
            summary["body"] = _chat.vnc.run_from_brain(
                res, duration_ms=min(duration_ms, 200)
            ).brief()
        except Exception as e:
            summary["body"] = {"ошибка": str(e)}
    if _chat is not None and _chat.reader is not None:
        try:
            summary["brain_reading"] = _chat.reader.describe(res)
        except Exception as e:
            summary["brain_reading"] = f"(ошибка чтения: {e})"
    return summary, rate


def behaviours_text(summary: dict) -> str:
    """Behaviour-associated neural readout; physical movement is not simulated."""
    txt = A.behaviours_text(summary)
    body = summary.get("body") or {}
    if body.get("мышцы"):
        txt += (
            "\n\nмоторный выход (MaleCNS; движение тела не симулируется): "
            + body["мышцы"]
            + "\n  мотонейроны: "
            + ", ".join(body.get("мотонейроны", []))
        )
    return txt


class StimReq(BaseModel):
    key: str
    side: str | None = None
    rate_hz: float | None = None
    duration_ms: int = 300


class ChatReq(BaseModel):
    text: str


class SmellReq(BaseModel):
    text: str
    reinforcement: str = "none"  # none | reward | punish
    duration_ms: int = 200


class ResetReq(BaseModel):
    # Default remains the existing body/UI reset. Memory and chat require explicit opt-in.
    memory: bool = False
    history: bool = False


def _log(kind: str, text: str, bv: dict | None = None):
    STATE["events"].append(
        {"t": datetime.now().isoformat(timespec="seconds"), "kind": kind, "text": text, "bv": bv}
    )
    STATE["events"] = STATE["events"][-200:]


@app.on_event("startup")
async def _startup():
    global _brain, _chat
    loop = asyncio.get_event_loop()

    def init():
        global _brain, _chat
        _chat = FlyChat(state_dir=STATE_DIR)
        _brain = _chat.brain

    await loop.run_in_executor(_exec, init)


@app.get("/")
async def index():
    return RedirectResponse("/garden")


@app.get("/garden")
async def garden():
    return FileResponse(Path(__file__).resolve().parent / "static" / "garden.html")


def _run_garden(req: GardenReq):
    """Run on the existing Brian2 worker, with the same animal and learned weights."""
    started = time.monotonic()
    tick()
    if (req.action == "chat" or req.odor_text) and not isinstance(_chat._llm, EN.EnglishLLM):
        _chat._llm = EN.EnglishLLM(_chat.llm)
    if req.read_brain and _chat.reader is None:
        raise ValueError("The Qwen reader is disabled on this server; uncheck the reader option.")
    if req.action == "chat":
        proposed = EN.plan(_chat.llm, req.text)
        if not proposed.get("is_stimulus"):
            reply = str(proposed.get("reply_if_no_stimulus") or "Describe a stimulus for me.")[
                :1200
            ]
            return {
                "source": "dialogue",
                "thought": {"text": reply, "source": "llm", "status": "ready"},
                "state": _garden_state(),
                "wall_s": round(time.monotonic() - started, 2),
            }
        plan = bounded_llm_plan(proposed)
        if req.duration_ms is not None:
            plan["duration_ms"] = req.duration_ms
        for stimulus in plan["stimuli"]:
            if stimulus["rate_hz"] is None:
                stimulus["rate_hz"] = STIMULI[stimulus["key"]].default_rate_hz
            stimulus["rate_hz"] *= hunger_scale(stimulus["key"])
    else:
        plan = make_plan(req, hunger_scale("sugar"))
    if plan.get("odor_compound"):
        if _chat.door is None:
            raise ValueError("DoOR is unavailable on this server.")
        try:
            _chat.door.glomerular(plan["odor_compound"])
        except KeyError as error:
            raise ValueError("Select a compound from the DoOR list.") from error
    # Garden object resolution must never silently fall back to random embedding.
    strict_before = _chat.strict_door
    try:
        _chat.strict_door = True
        _, summary = _chat.simulate(plan, seed=req.seed, read_brain=req.read_brain)
    finally:
        _chat.strict_door = strict_before
    bv = apply_body(summary, [s["key"] for s in plan["stimuli"]])
    STATE["n_stimuli"] += 1
    _log("garden", f"{req.action}: {req.target or ''}", bv)
    save_state()
    thought = readout_thought(summary)
    trial_id = uuid.uuid4().hex
    with _snapshot_lock:
        _trial_snapshots[trial_id] = {
            "created": time.monotonic(),
            "plan": copy.deepcopy(plan),
            "summary": copy.deepcopy(summary),
            "thought": thought,
            "owner": _current_session,
        }
        while len(_trial_snapshots) > 32:
            _trial_snapshots.popitem(last=False)
    if req.action == "chat":
        _chat.history.append(
            {
                "user": req.text,
                "plan": {**plan, "interpretation": proposed.get("interpretation", "")},
                "reply": thought["text"],
                "behaviours_short": A.behaviours_text(summary),
            }
        )
        _chat.history[:] = _chat.history[-20:]
    odor_info = EN.display(summary.get("odor_source"))
    if odor_info and odor_info.get("mapping") == "direct":
        odor_info["why"] = "Explicit chemical input from measured DoOR responses."
    return {
        "source": "brian2",
        "trial_id": trial_id,
        "thought": thought,
        "action": req.action,
        "target": req.target,
        "input": {
            "stimuli": plan["stimuli"],
            "compound": plan.get("odor_compound"),
            "odor_text": plan.get("odor_text"),
            "channels": plan.get("channels", []),
            "reinforcement": plan["reinforcement"],
            "duration_ms": plan["duration_ms"],
        },
        "behaviour": bv,
        "readouts": EN.readouts(summary["behaviours"]),
        "valence": summary.get("valence"),
        "odor_source": odor_info,
        "body": EN.display(summary.get("body")),
        "simulation": EN.display(summary["simulation"]),
        "brain_reading": summary.get("brain_reading"),
        "learning": summary.get("learning"),
        "n_active_downstream": summary["n_active_downstream"],
        "n_spikes_total": summary["n_spikes_total"],
        "wall_s": round(time.monotonic() - started, 2),
        "state": _garden_state(),
    }


def _garden_state():
    return {
        "satiety": round(STATE["satiety"], 3),
        "n_stimuli": STATE["n_stimuli"],
        "memory_strength": round(float(_chat.mb.memory_strength()), 4),
        "associations": _chat.associations()[-12:],
    }


def _session_token(request):
    token = request.headers.get("x-flypet-session") if request else None
    if token and not _sessions.contains(token):
        raise HTTPException(401, "Your temporary session expired. Reconnect to start a new pet.")
    if request and request.headers.get("x-flypet-public") == "1" and not token:
        raise HTTPException(401, "Start a garden session first.")
    return token


def _with_session(worker, req, token):
    global STATE, _current_session
    if token is None:
        return worker(req)
    entry = _sessions.get(token)
    old_state, old_owner = STATE, _current_session
    old = (
        _chat.mb.mem,
        _chat.mb.log,
        _chat.history,
        _chat._simulation_count,
        _chat.persist_memory,
        _chat.resolve_cache_path,
        _chat.log_path,
    )
    try:
        STATE, _current_session = entry["state"], token
        if entry["memory"] is None:
            import numpy as np

            entry["memory"] = np.ones_like(_chat.mb.mem)
        _chat.mb.mem, _chat.mb.log, _chat.history = entry["memory"], entry["log"], entry["history"]
        _chat._simulation_count, _chat.persist_memory = entry["counter"], False
        _chat.resolve_cache_path, _chat.log_path = None, None
        _chat.mb._push()
        return worker(req)
    finally:
        entry.update(
            state=STATE,
            memory=_chat.mb.mem,
            log=_chat.mb.log[-128:],
            history=_chat.history[-20:],
            counter=_chat._simulation_count,
        )
        (
            _chat.mb.mem,
            _chat.mb.log,
            _chat.history,
            _chat._simulation_count,
            _chat.persist_memory,
            _chat.resolve_cache_path,
            _chat.log_path,
        ) = old
        _chat.mb._push()
        STATE, _current_session = old_state, old_owner


async def _garden_submit(worker, req, token=None):
    global _garden_task
    if _chat is None:
        return JSONResponse({"error": "Brain is not ready."}, status_code=503)
    if _garden_task is not None and not _garden_task.done():
        return JSONResponse({"error": "The brain is busy. Try again shortly."}, status_code=429)

    async def run():
        return await asyncio.get_running_loop().run_in_executor(
            _exec, _with_session, worker, req, token
        )

    _garden_task = asyncio.create_task(run())
    # Client disconnects must not free the slot while Brian2 is still running.
    _garden_task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
    try:
        return await asyncio.shield(_garden_task)
    except ValueError as error:
        return JSONResponse({"error": str(error)[:240]}, status_code=422)
    except Exception:
        logging.getLogger(__name__).exception("Garden request failed")
        return JSONResponse(
            {"error": "Simulation failed. No new brain response is available."}, status_code=503
        )


@app.post("/api/garden")
async def garden_trial(req: GardenReq, request: Request = None):
    return await _garden_submit(_run_garden, req, _session_token(request))


@app.post("/api/garden/session")
async def garden_session():
    try:
        return {"session_id": _sessions.create(), "expires_after_seconds": _sessions.ttl}
    except RuntimeError:
        return JSONResponse(
            {"error": "Too many new sessions. Please try again in a minute."}, status_code=429
        )


@app.get("/api/garden/options")
async def garden_options(request: Request = None):
    token = _session_token(request)

    def options(_):
        return {
            "stimuli": [
                {"key": k, "label": EN.STIMULUS_LABELS[k], "rate_hz": s.default_rate_hz}
                for k, s in STIMULI.items()
            ],
            "channels": {
                k: {"label": CHANNEL_LABELS[k], "fields": fields}
                for k, fields in CHANNEL_FIELDS.items()
            },
            "odors": _chat.door.well_covered() if _chat and _chat.door else [],
            "components": _chat.component_status if _chat else {},
            "llm_thoughts": os.environ.get("FLYPET_THOUGHTS", "1") != "0",
            "state": _garden_state() if _chat else None,
        }

    return await asyncio.get_running_loop().run_in_executor(
        _exec, _with_session, options, None, token
    )


def _reset_garden(req):
    global STATE
    if req.memory or req.history:
        _chat.reset(
            memory=req.memory,
            history=req.history,
            persist_memory=req.memory and _current_session is None,
        )
    if req.body:
        STATE = {
            "satiety": 0.3,
            "born": datetime.now().isoformat(),
            "last_tick": time.time(),
            "events": [],
            "n_stimuli": 0,
            "name": STATE.get("name", "Дрозя"),
        }
    save_state()
    with _snapshot_lock:
        for key in list(_trial_snapshots):
            if _trial_snapshots[key].get("owner") == _current_session:
                del _trial_snapshots[key]
    return {"ok": True, "reset": req.model_dump(), "state": _garden_state()}


@app.post("/api/garden/reset")
async def garden_reset(req: GardenReset, request: Request = None):
    return await _garden_submit(_reset_garden, req, _session_token(request))


def _describe_snapshot(snapshot):
    global _narrator
    if _narrator is None:
        _narrator = get_llm(
            backend=os.environ.get("FLYPET_NARRATOR_BACKEND"),
            model=os.environ.get("FLYPET_NARRATOR_MODEL"),
        )
        _narrator.context_length = int(os.environ.get("FLYPET_NARRATOR_CONTEXT", "4096"))
    return narrate_snapshot(_narrator, snapshot)


@app.post("/api/garden/thought")
async def garden_thought(req: ThoughtReq, request: Request = None):
    global _thought_task, _thought_trial
    token = _session_token(request)
    with _snapshot_lock:
        snapshot = _trial_snapshots.get(req.trial_id)
        if (
            snapshot is None
            or snapshot.get("owner") != token
            or time.monotonic() - snapshot["created"] > 600
        ):
            return JSONResponse(
                {"error": "This trial has expired. Apply a new stimulus."}, status_code=404
            )
        if (
            snapshot["thought"].get("source") == "llm"
            or os.environ.get("FLYPET_THOUGHTS", "1") == "0"
        ):
            return {"trial_id": req.trial_id, "thought": snapshot["thought"]}
    if _thought_task is not None and not _thought_task.done():
        if _thought_trial != req.trial_id:
            return JSONResponse(
                {"trial_id": req.trial_id, "thought": {**snapshot["thought"], "status": "busy"}},
                status_code=429,
            )
    else:

        async def render():
            try:
                thought = await asyncio.get_running_loop().run_in_executor(
                    _thought_exec, _describe_snapshot, snapshot
                )
            except Exception:
                logging.getLogger(__name__).exception("Garden narrator failed")
                thought = {**snapshot["thought"], "status": "unavailable"}
            with _snapshot_lock:
                if req.trial_id in _trial_snapshots:
                    _trial_snapshots[req.trial_id]["thought"] = thought
            return {"trial_id": req.trial_id, "thought": thought}

        _thought_trial = req.trial_id
        _thought_task = asyncio.create_task(render())
    return await asyncio.shield(_thought_task)


@app.get("/healthz")
async def healthz():
    return {"ok": _brain is not None, "neurons": _brain.n if _brain else 0}


@app.get("/api/state")
async def state():
    tick()
    return {
        "satiety": round(STATE["satiety"], 3),
        "name": STATE["name"],
        "n_stimuli": STATE["n_stimuli"],
        "events": STATE["events"][-30:],
        "llm": os.environ.get("FLYPET_LLM", "auto"),
        "memory": {
            "associations": _chat.associations()[-12:] if _chat else [],
            "strength": round(_chat.mb.memory_strength(), 4) if _chat else 0,
        },
        "stimuli": {k: {"label": s.label_ru, "expected": s.expected} for k, s in STIMULI.items()},
    }


@app.post("/api/stimulate")
async def stimulate(req: StimReq):
    if req.key not in STIMULI:
        return JSONResponse({"error": f"unknown stimulus {req.key}"}, status_code=400)
    tick()
    loop = asyncio.get_event_loop()
    summary, rate = await loop.run_in_executor(
        _exec, _run_stimulus, req.key, req.side, req.rate_hz, req.duration_ms
    )
    bv = apply_body(summary, [req.key])
    STATE["n_stimuli"] += 1
    _log("stimulus", f"{STIMULI[req.key].label_ru} ({req.side or 'both'}, {rate:.0f} Гц)", bv)
    save_state()
    return {
        "behaviour": bv,
        "behaviours_text": behaviours_text(summary),
        "summary": {
            k: summary[k]
            for k in (
                "duration_ms",
                "wall_s",
                "n_active_downstream",
                "n_spikes_total",
                "active_by_super_class",
                "stimulated",
            )
        },
        "top": summary["top_downstream"][:8],
        "state": {"satiety": round(STATE["satiety"], 3)},
        "rate_used_hz": rate,
    }


@app.post("/api/smell")
async def smell(req: SmellReq):
    tick()
    loop = asyncio.get_event_loop()
    plan = {
        "is_stimulus": True,
        "interpretation": f"запах: {req.text}",
        "duration_ms": req.duration_ms,
        "stimuli": [],
        "odor_text": req.text.strip(),
        "reinforcement": req.reinforcement if req.reinforcement in ("reward", "punish") else "none",
    }
    res, summary = await loop.run_in_executor(_exec, _chat.simulate, plan)
    bv = apply_body(summary, [])
    STATE["n_stimuli"] += 1
    _log(
        "smell",
        f"запах {req.text} ({plan['reinforcement']}) → валентность {summary.get('valence', {}).get('score')}",
        bv,
    )
    save_state()
    return {
        "behaviour": bv,
        "behaviours_text": behaviours_text(summary),
        "valence": summary.get("valence"),
        "learning": summary.get("learning"),
        "associations": summary.get("associations", []),
        "state": {"satiety": round(STATE["satiety"], 3)},
        "wall_s": summary["wall_s"],
    }


@app.post("/api/chat")
async def chat(req: ChatReq):
    tick()
    loop = asyncio.get_event_loop()
    try:
        entry = await loop.run_in_executor(_exec, _chat.turn, req.text, False)
    except (
        Exception
    ) as e:  # LLM backend missing / unreachable: keep the pet alive, report the reason
        return JSONResponse(
            {"error": f"LLM недоступна: {type(e).__name__}: {str(e)[:200]}"}, status_code=503
        )
    bv = None
    if "summary" in entry:
        bv = apply_body(entry["summary"], [s["key"] for s in entry["plan"]["stimuli"]])
        STATE["n_stimuli"] += 1
    _log("chat", f"Ты: {req.text} / Муха: {entry['reply']}", bv)
    save_state()
    return {
        "reply": entry["reply"],
        "plan": entry["plan"],
        "behaviour": bv,
        "behaviours_text": behaviours_text(entry["summary"]) if "summary" in entry else "",
        "state": {"satiety": round(STATE["satiety"], 3)},
        "wall_s": entry["wall_s"],
    }


@app.post("/api/reset")
async def reset(req: ResetReq | None = None):
    """Reset body/UI; optional flags explicitly reset learned memory and/or chat history."""
    global STATE
    req = req or ResetReq()
    if _chat is not None and (req.memory or req.history):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _exec,
            lambda: _chat.reset(memory=req.memory, history=req.history, persist_memory=req.memory),
        )
    STATE = {
        "satiety": 0.3,
        "born": datetime.now().isoformat(),
        "last_tick": time.time(),
        "events": [],
        "n_stimuli": 0,
        "name": STATE.get("name", "Дрозя"),
    }
    save_state()
    return {
        "ok": True,
        "reset": {
            "body": True,
            "memory": req.memory and _chat is not None,
            "history": req.history and _chat is not None,
        },
    }


def main():
    import uvicorn, argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
