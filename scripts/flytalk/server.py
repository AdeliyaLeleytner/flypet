#!/usr/bin/env python3
"""FlyTalk HTTP server (bind to localhost; reach it through an SSH tunnel).

POST /reset {"task": "diagnose"|"erase", "seed": int}      -> {"eid", "prompt", "budget"}
POST /step  {"eid": str, "action": str}                     -> {"observation", "done", "steps_left", "transcript"}
POST /result {"eid": str}                                   -> the episode result (after it is done)
POST /reset_many {"specs": [[task, seed, eid|null], ...]}   -> {"episodes": [{"eid", "prompt", "budget"}]}
POST /step_many {"items": [[eid, action], ...]}             -> {"observations": {eid: str}, "done": {eid: bool}}
GET  /info                                                  -> panel, reference valences, budget, delta
"""

from __future__ import annotations
import argparse, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet.flytalk import PANEL, Env, describe, render_transcript
from flypet.teach import FlyPool

ENV: Env | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/reset":
                ep = ENV.reset(req.get("task", "diagnose"), int(req.get("seed", 0)))
                return self._send(
                    200, {"eid": ep.eid, "prompt": describe(ep, ENV), "budget": ep.budget}
                )
            if self.path == "/reset_many":
                eps = ENV.reset_many([tuple(x) for x in req["specs"]])
                return self._send(
                    200,
                    {
                        "episodes": [
                            {"eid": e.eid, "prompt": describe(e, ENV), "budget": e.budget}
                            for e in eps
                        ]
                    },
                )
            if self.path == "/step_many":
                obs = ENV.step_many([tuple(x) for x in req["items"]])
                return self._send(
                    200,
                    {
                        "observations": obs,
                        "done": {e: ENV.episodes[e].done for e in obs},
                        "results": {e: ENV.episodes[e].result for e in obs if ENV.episodes[e].done},
                    },
                )
            if self.path == "/step":
                obs = ENV.step(req["eid"], req["action"])
                ep = ENV.episodes[req["eid"]]
                return self._send(
                    200,
                    {
                        "observation": obs,
                        "done": ep.done,
                        "steps_left": ep.steps_left(),
                        "transcript": render_transcript(ep),
                    },
                )
            if self.path == "/result":
                ep = ENV.episodes[req["eid"]]
                return self._send(
                    200,
                    {
                        "done": ep.done,
                        "result": ep.result,
                        "history": ep.history,
                        "transcript": ep.transcript,
                        "task": ep.task,
                    },
                )
            self._send(404, {"error": "unknown path"})
        except Exception as exc:  # report, keep serving
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self):
        if self.path == "/info":
            return self._send(
                200,
                {"panel": ENV.panel, "naive": ENV.naive, "budget": ENV.budget, "delta": ENV.delta},
            )
        self._send(404, {"error": "unknown path"})


def main(argv=None):
    global ENV
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--host", default="127.0.0.1", help="keep 127.0.0.1 unless the network is trusted"
    )
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--delta", type=float, default=0.15)
    a = ap.parse_args(argv)
    ENV = Env(FlyPool(a.workers), PANEL, None, budget=a.budget, delta=a.delta)
    print(f"FlyTalk on {a.host}:{a.port}; panel {PANEL}; untrained fly {ENV.naive}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
