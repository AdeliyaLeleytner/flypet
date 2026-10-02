"""Garden contract, isolated state, and cancellation/concurrency regressions."""

import asyncio
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from flypet import pet
from flypet.garden import GardenReq, ODORS, make_plan


class GardenTests(TestCase):
    def test_only_bounded_actions_and_explicit_chemicals(self):
        for target, compound in ODORS.items():
            plan = make_plan(GardenReq(action="smell", target=target))
            self.assertEqual(plan["odor_compound"], compound)
            self.assertNotIn("odor_text", plan)
        self.assertEqual(make_plan(GardenReq(action="baseline"))["stimuli"], [])
        for body in [
            {"action": "smell"},
            {"action": "unknown"},
            {"action": "sugar", "duration_ms": 1000000},
            {"action": "smell", "target": "<script>"},
            {"action": "looming", "target": "apple"},
        ]:
            with self.assertRaises(ValidationError):
                GardenReq(**body)

    def test_routes_and_trial_use_shared_simulator_without_language_model(self):
        summary = {
            "behaviours": {},
            "n_active_downstream": 0,
            "n_spikes_total": 0,
            "simulation": {"seed": 17},
            "wall_s": 0,
        }
        fake = Mock()
        fake.simulate.return_value = (None, summary)
        fake.mb.memory_strength.return_value = 0.0
        fake.associations.return_value = []
        with ExitStack() as stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(patch.object(pet, "_chat", fake))
            stack.enter_context(patch.object(pet, "_garden_task", None))
            stack.enter_context(patch.object(pet, "STATE_PATH", Path(directory) / "state.json"))
            stack.enter_context(
                patch.object(pet, "STATE", {"satiety": 0.3, "n_stimuli": 0, "events": []})
            )
            client = TestClient(pet.app)  # no lifespan: fake brain only
            page = client.get("/garden")
            self.assertEqual(page.status_code, 200)
            self.assertIn("/static/garden-brain.js", page.text)
            self.assertEqual(client.get("/static/garden-brain.js").status_code, 200)
            response = client.post("/api/garden", json={"action": "smell", "target": "apple"})
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["source"], "brian2")
            self.assertEqual(body["input"]["compound"], "hexyl acetate")
            self.assertEqual(body["behaviour"]["escape"], 0)
            fake.simulate.assert_called_once_with(
                make_plan(GardenReq(action="smell", target="apple")), seed=None, read_brain=False
            )
            fake.turn.assert_not_called()
            self.assertEqual(pet.STATE["n_stimuli"], 1)
            self.assertTrue((Path(directory) / "state.json").exists())

    def test_disconnect_does_not_release_running_brain_slot(self):
        async def scenario():
            started, release = threading.Event(), threading.Event()

            def blocking(_):
                started.set()
                release.wait(timeout=3)
                return {"source": "brian2"}

            with (
                ThreadPoolExecutor(max_workers=1) as executor,
                patch.object(pet, "_exec", executor),
                patch.object(pet, "_chat", Mock()),
                patch.object(pet, "_garden_task", None),
                patch.object(pet, "_run_garden", blocking),
            ):
                request = asyncio.create_task(pet.garden_trial(GardenReq(action="baseline")))
                try:
                    await asyncio.wait_for(asyncio.to_thread(started.wait, 1), 2)
                    self.assertTrue(started.is_set())
                    request.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await request
                    busy = await pet.garden_trial(GardenReq(action="baseline"))
                    self.assertEqual(busy.status_code, 429)
                finally:
                    release.set()
                    await pet._garden_task
                result = await pet.garden_trial(GardenReq(action="baseline"))
                self.assertEqual(result["source"], "brian2")

        asyncio.run(scenario())

    def test_simulation_failure_is_not_a_fake_response(self):
        async def scenario():
            with (
                patch.object(pet, "_chat", Mock()),
                patch.object(pet, "_garden_task", None),
                patch.object(pet, "_run_garden", side_effect=RuntimeError("private local details")),
            ):
                response = await pet.garden_trial(GardenReq(action="baseline"))
                self.assertEqual(response.status_code, 503)
                self.assertNotIn(b"private local details", response.body)

        asyncio.run(scenario())
