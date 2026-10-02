import asyncio
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
import tempfile
import numpy as np
from fastapi.testclient import TestClient
from flypet import pet, english
from flypet.sessions import SessionStore
from flypet.garden import ThoughtReq
from flypet.thoughts import narrate_snapshot, _render_cached


class PublicGardenTests(TestCase):
    def test_sessions_switch_memory_and_restore_private_state_even_after_error(self):
        store = SessionStore()
        a, b = store.create(), store.create()
        fake = SimpleNamespace(
            mb=SimpleNamespace(mem=np.array([0.4, 0.8]), log=["private"], _push=Mock()),
            history=["private"],
            _simulation_count=7,
            persist_memory=True,
            resolve_cache_path=Path("private"),
            log_path=Path("private.log"),
        )
        original = fake.mb.mem
        private = {"satiety": 0.7}
        with (
            patch.object(pet, "_sessions", store),
            patch.object(pet, "_chat", fake),
            patch.object(pet, "STATE", private),
            patch.object(pet, "_current_session", None),
        ):

            def train(_):
                fake.mb.mem *= 0.5
                fake.mb.log.append("learned")
                fake.history.append("hello")
                fake._simulation_count += 1
                pet.STATE["satiety"] = 0.9
                self.assertFalse(fake.persist_memory)

            pet._with_session(train, None, a)
            np.testing.assert_equal(store.get(a)["memory"], [0.5, 0.5])

            def inspect(_):
                np.testing.assert_equal(fake.mb.mem, [1, 1])
                self.assertEqual(fake.history, [])
                raise RuntimeError("interrupted")

            with self.assertRaises(RuntimeError):
                pet._with_session(inspect, None, b)
            self.assertIs(fake.mb.mem, original)
            self.assertIs(pet.STATE, private)
            self.assertEqual(fake.history, ["private"])
            self.assertEqual(fake._simulation_count, 7)
            self.assertIsNone(pet._current_session)

    def test_public_marker_requires_valid_session_and_trials_are_owner_bound(self):
        store = SessionStore()
        owner = store.create()
        other = store.create()
        trial = "c" * 32
        snap = {
            "created": __import__("time").monotonic(),
            "owner": owner,
            "thought": {"source": "llm", "text": "My response."},
        }
        with (
            patch.object(pet, "_sessions", store),
            patch.object(pet, "_trial_snapshots", OrderedDict({trial: snap})),
        ):
            c = TestClient(pet.app)
            self.assertEqual(
                c.post(
                    "/api/garden", json={"action": "baseline"}, headers={"X-Flypet-Public": "1"}
                ).status_code,
                401,
            )
            self.assertEqual(
                c.post(
                    "/api/garden/thought",
                    json={"trial_id": trial},
                    headers={"X-Flypet-Session": other},
                ).status_code,
                404,
            )
            self.assertEqual(
                c.post(
                    "/api/garden/thought",
                    json={"trial_id": trial},
                    headers={"X-Flypet-Session": owner},
                ).status_code,
                200,
            )

    def test_state_is_not_written_for_public_sessions(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(pet, "_current_session", "public"),
            patch.object(pet, "STATE_PATH", Path(tmp) / "state.json"),
        ):
            pet.save_state()
            self.assertFalse((Path(tmp) / "state.json").exists())

    def test_wording_cache_never_invents_a_neural_trial(self):
        _render_cached.cache_clear()
        llm = Mock()
        llm.name = "fixture"
        llm.last_model_ids = ["fixture"]
        llm.complete.return_value = {"reply": "My proboscis output is active."}
        summary = {
            "behaviours": {"proboscis_extension": {"max_rate_hz": 60}},
            "n_spikes_total": 100,
        }
        one = narrate_snapshot(llm, {"summary": summary, "plan": {}})
        summary["behaviours"]["proboscis_extension"]["max_rate_hz"] = 65
        two = narrate_snapshot(llm, {"summary": summary, "plan": {}})
        self.assertFalse(one["cached"])
        self.assertTrue(two["cached"])
        self.assertEqual(llm.complete.call_count, 1)
        summary["behaviours"] = {}
        llm.complete.return_value = {"reply": "My tracked motor outputs are silent."}
        three = narrate_snapshot(llm, {"summary": summary, "plan": {}})
        self.assertFalse(three["cached"])
        self.assertEqual(llm.complete.call_count, 2)

    def test_frontend_and_option_labels_are_english(self):
        import re

        root = Path(__file__).resolve().parents[1]
        for name in [
            "flypet/static/garden.html",
            "flypet/static/garden-controls.js",
            "flypet/garden.py",
        ]:
            self.assertIsNone(re.search("[\u0400-\u04ff]", (root / name).read_text()), name)
        self.assertEqual(len(english.STIMULUS_LABELS), 27)
        self.assertNotIn(
            "Гц", english.display({"мышцы": "крыло L: 1/2 нейронов, до 50 Гц"})["muscles"]
        )
