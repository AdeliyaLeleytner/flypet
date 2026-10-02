"""Local/private live demo of the learned latent bridges and continuing brain."""

from __future__ import annotations
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, AsyncExitStack
import json
import os
from pathlib import Path
import time
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from .latent_sessions import BrainSession
from .latent_questions import SUMMARY as SUMMARY_QUESTION, CHANGE_QUALITATIVE


class SessionRequest(BaseModel):
    seed: int | None = Field(default=None, ge=0, lt=2**32)


class TurnRequest(BaseModel):
    session: str = Field(min_length=20, max_length=80)
    text: str = Field(min_length=1, max_length=600)
    mode: str = "interact"
    learning: bool = True

    @field_validator("text")
    @classmethod
    def meaningful_text(cls, value):
        if not value.strip():
            raise ValueError("Write an experience or a question.")
        return value.strip()


class TokenRequest(BaseModel):
    session: str = Field(min_length=20, max_length=80)


class SwapRequest(TokenRequest):
    donor: str = Field(min_length=20, max_length=80)


class TwinsRequest(BaseModel):
    first: str = Field(min_length=20, max_length=80)
    second: str = Field(min_length=20, max_length=80)
    odor: str = "acetic acid"


def create_app(bundle, device=None):
    sessions = {}
    locks = {}
    calls = {}
    model_executor = ThreadPoolExecutor(max_workers=1)
    max_sessions = int(os.environ.get("FLYPET_LATENT_MAX_SESSIONS", "4"))
    ttl = int(os.environ.get("FLYPET_LATENT_TTL_SECONDS", "1800"))
    if not 2 <= max_sessions <= 16 or not 60 <= ttl <= 3600:
        raise ValueError("Session limits outside the supported range")
    state = {"models": None, "created": [], "progress": {}}

    async def model_call(method, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(
            model_executor, lambda: getattr(state["models"], method)(*args, **kwargs)
        )

    def session_for(token):
        session = sessions.get(token)
        if session is None or time.monotonic() - session.touched > ttl:
            raise HTTPException(401, "This session expired. Start a new fly.")
        return session

    async def close_session(token):
        session = sessions.pop(token, None)
        locks.pop(token, None)
        calls.pop(token, None)
        state["progress"].pop(token, None)
        if session:
            await asyncio.to_thread(session.close)

    async def expire_sessions():
        while True:
            await asyncio.sleep(30)
            for token in list(sessions):
                if (
                    token in sessions
                    and not locks[token].locked()
                    and time.monotonic() - sessions[token].touched > ttl
                ):
                    await close_session(token)

    async def ensure_room(number):
        now = time.monotonic()
        state["created"] = [x for x in state["created"] if now - x < 60]
        if len(state["created"]) + number > 12:
            raise HTTPException(429, "Please wait before creating another fly.")
        for token in list(sessions):
            session = sessions.get(token)
            if session and not locks[token].locked() and now - session.touched > ttl:
                await close_session(token)
        if len(sessions) + number > max_sessions:
            raise HTTPException(429, "All live flies are in use. Please try again shortly.")

    def allocate(seed):
        session = BrainSession(bundle, seed)
        sessions[session.token] = session
        locks[session.token] = asyncio.Lock()
        calls[session.token] = 0
        state["created"].append(time.monotonic())
        return {
            "session": session.token,
            "seed": session.seed,
            "reader": state["models"].reader_config["kind"],
        }

    @asynccontextmanager
    async def lifespan(app):
        def load_models():
            from .latent_inference import LatentModels

            return LatentModels(bundle, device)

        state["models"] = await asyncio.get_running_loop().run_in_executor(
            model_executor, load_models
        )
        expiry = asyncio.create_task(expire_sessions())
        try:
            yield
        finally:
            expiry.cancel()
            try:
                await expiry
            except asyncio.CancelledError:
                pass
            for token in list(sessions):
                await close_session(token)
            model_executor.shutdown(wait=True)

    app = FastAPI(title="Flypet Neural Link", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.get("/")
    async def index():
        return FileResponse(Path(__file__).parent / "static/latent.html")

    @app.get("/healthz")
    async def health():
        return {"ready": state["models"] is not None, "sessions": len(sessions)}

    @app.post("/api/session")
    async def new_session(req: SessionRequest):
        await ensure_room(1)
        return allocate(req.seed)

    @app.post("/api/pair")
    async def new_pair(req: SessionRequest):
        await ensure_room(2)
        first = allocate(req.seed)
        second = allocate(first["seed"])
        return {"first": first, "second": second}

    @app.post("/api/turn")
    async def turn(req: TurnRequest):
        if req.mode not in ("interact", "ask", "audit", "compare"):
            raise HTTPException(422, "Unknown interaction mode")
        session = session_for(req.session)
        lock = locks[req.session]
        if lock.locked():
            raise HTTPException(429, "This fly is already responding.")
        if calls[req.session] >= 60:
            raise HTTPException(429, "This session reached its turn limit. Start a new fly.")
        async with lock:
            calls[req.session] += 1
            t = time.monotonic()
            try:
                drive = None
                initially_observed = session.has_observation
                if req.mode == "interact":
                    drive = await model_call("write", req.text)
                    snapshot = await asyncio.to_thread(
                        session.step, drive["drive_hz"], req.learning
                    )
                    question = SUMMARY_QUESTION
                else:
                    snapshot = await asyncio.to_thread(session.observe)
                    question = req.text
                    if req.mode == "audit":
                        question = "Report the current approach_hz, avoid_hz and valence as JSON. Valence is (approach_hz-avoid_hz)/(approach_hz+avoid_hz+1), positive when approach exceeds avoidance. Use one decimal for rates and three for valence."
                    if req.mode == "compare":
                        question = CHANGE_QUALITATIVE
                reply = await model_call(
                    "read",
                    snapshot["current"],
                    snapshot["reference"],
                    question=question,
                    **({"mode": "comparison"} if req.mode == "compare" else {}),
                )
                result = {
                    "reply": reply,
                    "telemetry": snapshot["telemetry"],
                    "wall_s": time.monotonic() - t,
                    "source": "generated from neural tokens",
                    "mode": req.mode,
                }
                if drive is not None:
                    names = state["models"].writer_config["channels"]
                    rates = drive["channel_rates_hz"]
                    result["drive"] = [
                        {"channel": name, "rate_hz": float(value)}
                        for name, value in sorted(zip(names, rates), key=lambda x: -x[1])[:8]
                    ]
                replay = {"initial_zero_probe": not initially_observed and req.mode != "interact"}
                if drive is not None:
                    replay["channel_rates_hz"] = drive["channel_rates_hz"].tolist()
                session.history.append(
                    {
                        "input": req.text,
                        "mode": req.mode,
                        "learning": req.learning,
                        "replay": replay,
                        **result,
                    }
                )
                return result
            except HTTPException:
                raise
            except Exception:
                import logging

                logging.getLogger(__name__).exception("Latent interaction failed")
                raise HTTPException(
                    503, "This interaction failed. Start a new session if the problem persists."
                )

    @app.post("/api/reference")
    async def reference(req: TokenRequest):
        session = session_for(req.session)
        if locks[req.session].locked():
            raise HTTPException(429, "Wait for the current response.")
        async with locks[req.session]:
            initial = not session.has_observation
            snapshot = await asyncio.to_thread(session.pin_reference)
            session.history.append(
                {
                    "mode": "pin_reference",
                    "replay": {"initial_zero_probe": initial},
                    "telemetry": snapshot["telemetry"],
                }
            )
        return {"ok": True, "telemetry": snapshot["telemetry"]}

    @app.post("/api/export")
    async def export(req: TokenRequest):
        session = session_for(req.session)
        return {
            "schema_version": 1,
            "seed": session.seed,
            "model_manifest": state["models"].manifest,
            "initial_memory": "naive",
            "step_ms": 250,
            "events": session.history,
        }

    @app.post("/api/close")
    async def close(req: TokenRequest):
        session_for(req.session)
        if locks[req.session].locked():
            raise HTTPException(429, "Wait for the current response.")
        await close_session(req.session)
        return {"ok": True}

    @app.post("/api/progress")
    async def progress(req: TokenRequest):
        session_for(req.session)
        return {"phase": state["progress"].get(req.session, "")}

    @app.post("/api/swap")
    async def swap(req: SwapRequest):
        if req.session == req.donor:
            raise HTTPException(422, "Choose two different flies.")
        recipient = session_for(req.session)
        donor = session_for(req.donor)
        if not recipient.has_observation or not donor.has_observation:
            raise HTTPException(
                409, "Give both flies an experience before comparing their neural states."
            )
        tokens = sorted([req.session, req.donor])
        if any(locks[t].locked() for t in tokens):
            raise HTTPException(429, "Wait for both flies to finish.")
        if calls[req.session] > 58:
            raise HTTPException(429, "This session reached its turn limit.")
        async with AsyncExitStack() as stack:
            for token in tokens:
                await stack.enter_async_context(locks[token])
            calls[req.session] += 2
            t = time.monotonic()
            own = await asyncio.to_thread(recipient.observe)
            other = await asyncio.to_thread(donor.observe)
            question = SUMMARY_QUESTION
            if state["models"].reader_config["kind"] == "numeric":
                question = state["models"].reader_config["question"]
            original = await model_call("read", own["current"], own["reference"], question=question)
            swapped = await model_call(
                "read", other["current"], own["reference"], question=question
            )
            result = {
                "question": question,
                "original": original,
                "swapped": swapped,
                "own_telemetry": own["telemetry"],
                "donor_telemetry": other["telemetry"],
                "changed_input": "current neural observation; reference and question held fixed; derived neural differences recomputed when used",
                "history_text_supplied_to_reader": False,
                "brain_state_modified": False,
                "wall_s": time.monotonic() - t,
            }
            recipient.history.append({"mode": "state_swap", **result})
            return result

    @app.post("/api/experiment")
    async def experiment(req: TwinsRequest):
        if req.first == req.second:
            raise HTTPException(422, "Choose two different flies.")
        if req.odor not in (
            "ethyl acetate",
            "methyl acetate",
            "1-hexanol",
            "2-heptanone",
            "hexanal",
            "acetic acid",
            "linalool",
            "geosmin",
        ):
            raise HTTPException(422, "Choose one of the supported scents for this experiment.")
        pair = [session_for(req.first), session_for(req.second)]
        if pair[0].seed != pair[1].seed:
            raise HTTPException(422, "The twins experiment requires a shared starting seed.")
        if any(s.has_observation for s in pair):
            raise HTTPException(409, "This experiment starts with a fresh pair.")
        tokens = sorted([req.first, req.second])
        if any(locks[t].locked() for t in tokens):
            raise HTTPException(429, "Wait for both flies to finish.")
        async with AsyncExitStack() as stack:
            for token in tokens:
                await stack.enter_async_context(locks[token])
            for token in tokens:
                calls[token] += 8
            started = time.monotonic()

            def phase(text):
                for token in tokens:
                    state["progress"][token] = text

            try:
                phase("Preparing the experience…")
                texts = [
                    f"Give the fly moderate exposure to {req.odor}. No reward or punishment.",
                    f"Give the fly moderate exposure to {req.odor}. Pair it with a sugar reward.",
                    f"Give the fly moderate exposure to {req.odor}. Pair it with an aversive punishment.",
                ]
                drives = [await model_call("write", text) for text in texts]

                async def step_log(session, drive, text, learning, source="learned_writer"):
                    snapshot = await asyncio.to_thread(session.step, drive["drive_hz"], learning)
                    session.history.append(
                        {
                            "mode": "interact",
                            "input": text,
                            "learning": learning,
                            "reply": "",
                            "drive_source": source,
                            "telemetry": snapshot["telemetry"],
                            "replay": {
                                "channel_rates_hz": drive["channel_rates_hz"].tolist(),
                                "initial_zero_probe": False,
                            },
                        }
                    )
                    return snapshot

                phase("Measuring the shared starting response…")
                baseline = await asyncio.gather(
                    *(step_log(s, drives[0], texts[0], False) for s in pair)
                )
                same_baseline = all(
                    baseline[0]["telemetry"][k] == baseline[1]["telemetry"][k]
                    for k in ("spikes", "active_neurons", "approach_hz", "avoid_hz", "valence")
                )
                if not same_baseline:
                    raise RuntimeError("Twin baseline replay differs")
                for session in pair:
                    snap = await asyncio.to_thread(session.pin_reference)
                    session.history.append(
                        {
                            "mode": "pin_reference",
                            "telemetry": snap["telemetry"],
                            "replay": {"initial_zero_probe": False},
                        }
                    )
                for trial in range(3):
                    phase(f"Teaching the twins · {trial + 1} of 3…")
                    await asyncio.gather(
                        *(
                            step_log(s, drive, text, True)
                            for s, drive, text in zip(pair, drives[1:], texts[1:])
                        )
                    )
                phase("Letting activity settle…")
                zero = {
                    k: np.zeros_like(v)
                    for k, v in drives[0].items()
                    if k in ("drive_hz", "channel_rates_hz")
                }
                await asyncio.gather(
                    *(
                        step_log(
                            s,
                            zero,
                            "A 250 ms pause with no external drive.",
                            False,
                            "scripted_zero_drive",
                        )
                        for s in pair
                    )
                )
                phase("Showing both flies the same scent…")
                snapshots = await asyncio.gather(
                    *(step_log(s, drives[0], texts[0], False) for s in pair)
                )
                phase("Reading their responses…")
                replies = []
                question = SUMMARY_QUESTION
                for session, snapshot in zip(pair, snapshots):
                    reply = await model_call(
                        "read", snapshot["current"], snapshot["reference"], question=question
                    )
                    result = {
                        "reply": reply,
                        "telemetry": snapshot["telemetry"],
                        "source": "generated from neural tokens",
                        "mode": "ask",
                    }
                    session.history.append(
                        {"input": question, "learning": False, "replay": {}, **result}
                    )
                    replies.append(result)
                return {
                    "first": replies[0],
                    "second": replies[1],
                    "wall_s": time.monotonic() - started,
                    "same_seed": True,
                    "same_baseline_readout": same_baseline,
                    "same_final_drive": True,
                    "description": "Three learned reward/punishment exposures, a zero-input rest, then the same learned scent probe. The reader receives neural tokens and a fixed question.",
                }
            except Exception:
                import logging

                logging.getLogger(__name__).exception("Twins experiment failed")
                raise HTTPException(
                    503, "The experiment did not finish. Start a new pair to retry."
                )
            finally:
                for token in tokens:
                    state["progress"].pop(token, None)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device")
    a = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(a.bundle, a.device), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
