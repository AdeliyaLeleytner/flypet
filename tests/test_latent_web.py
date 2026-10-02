"""API contract checks with instrumented bridges, without a second model load."""

import sys, time, types, unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from flypet.latent_web import create_app


class FakeModels:
    reads = []
    read_modes = []

    def __init__(self, *args):
        self.reader_config = {"kind": "paired_dialogue"}
        self.writer_config = {"channels": ["test"]}
        self.manifest = {"revision": "test"}

    def write(self, text):
        return {"drive_hz": np.array([float(len(text))]), "channel_rates_hz": np.array([1.0])}

    def read(self, current, reference, question, mode="current"):
        self.reads.append((current, reference, question))
        self.read_modes.append(mode)
        return str(current["value"])


class FakeSession:
    number = 0

    def __init__(self, bundle, seed=None):
        type(self).number += 1
        self.token = str(type(self).number).zfill(24)
        self.seed = seed or 7
        self.touched = time.monotonic()
        self.history = []
        self.has_observation = False
        self.steps = 0

    def step(self, drive, learning):
        self.steps += 1
        self.has_observation = True
        self.current = {"value": float(drive[0])}
        self.reference = {"value": 0}
        return self.observe()

    def observe(self):
        return {
            "current": self.current,
            "reference": self.reference,
            "telemetry": {"steps": self.steps},
        }

    def close(self):
        pass


class WebTests(unittest.TestCase):
    def test_swap_holds_question_and_reference_without_advancing_brains(self):
        FakeModels.reads = []
        with (
            patch.dict(
                sys.modules,
                {"flypet.latent_inference": types.SimpleNamespace(LatentModels=FakeModels)},
            ),
            patch("flypet.latent_web.BrainSession", FakeSession),
        ):
            with TestClient(create_app("unused")) as client:
                a = client.post("/api/session", json={}).json()["session"]
                b = client.post("/api/session", json={}).json()["session"]
                self.assertEqual(
                    client.post("/api/swap", json={"session": a, "donor": b}).status_code, 409
                )
                for token, text in [(a, "a short scent"), (b, "a different longer scent")]:
                    self.assertEqual(
                        client.post("/api/turn", json={"session": token, "text": text}).status_code,
                        200,
                    )
                result = client.post("/api/swap", json={"session": a, "donor": b}).json()
                self.assertNotEqual(result["original"], result["swapped"])
                own, donor = FakeModels.reads[-2:]
                self.assertEqual(own[1:], donor[1:])
                self.assertNotEqual(own[0], donor[0])
                self.assertEqual(result["own_telemetry"]["steps"], 1)
                self.assertEqual(result["donor_telemetry"]["steps"], 1)
                self.assertFalse(result["history_text_supplied_to_reader"])
                self.assertFalse(result["brain_state_modified"])
                self.assertEqual(
                    client.post(
                        "/api/turn", json={"session": a, "text": "Compare", "mode": "compare"}
                    ).status_code,
                    200,
                )
                self.assertEqual(FakeModels.read_modes[-1], "comparison")
                self.assertEqual(
                    client.post("/api/turn", json={"session": a, "text": "  "}).status_code, 422
                )
                client.post("/api/session", json={}).raise_for_status()
                self.assertEqual(client.post("/api/pair", json={}).status_code, 429)
                # A new visitor must not evict an existing fly's live memory.
                self.assertEqual(client.post("/api/export", json={"session": a}).status_code, 200)
                self.assertEqual(client.post("/api/close", json={"session": a}).status_code, 200)
                self.assertEqual(client.post("/api/export", json={"session": a}).status_code, 401)
                pair = client.post("/api/pair", json={"seed": 123}).json()
                self.assertEqual(pair["first"]["seed"], 123)
                self.assertEqual(pair["second"]["seed"], 123)


if __name__ == "__main__":
    unittest.main()
