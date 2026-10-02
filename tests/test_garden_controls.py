import asyncio
import json
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from pydantic import ValidationError
from flypet.garden import (
    GardenReq,
    GardenReset,
    ThoughtReq,
    make_plan,
    bounded_llm_plan,
    CHANNEL_FIELDS,
    STIMULI,
)
from flypet.thoughts import readout_thought, narrate_snapshot
from flypet import pet
from deploy.export_hf_garden import export


class ControlsTests(TestCase):
    def test_every_catalog_and_physical_channel_has_a_valid_plan(self):
        for key in STIMULI:
            req = GardenReq(
                action="stimuli",
                stimuli=[{"key": key, "rate_hz": 0}],
                side="left",
                duration_ms=100,
                seed=0,
            )
            plan = make_plan(req)
            self.assertEqual(plan["stimuli"], [{"key": key, "rate_hz": 0, "side": "left"}])
        for key in CHANNEL_FIELDS:
            req = GardenReq(action="channel", channel=key, params={}, seed=17)
            self.assertEqual(make_plan(req)["channels"][0]["channel"], key)

    def test_mixtures_and_reinforcement_are_explicit_and_bounded(self):
        plan = make_plan(
            GardenReq(action="stimuli", stimuli=[{"key": "sugar"}, {"key": "bitter"}]), 0.5
        )
        self.assertEqual([x["rate_hz"] for x in plan["stimuli"]], [75, 150])
        for reinforcement in ("reward", "punish"):
            plan = make_plan(
                GardenReq(action="smell", compound="ethyl acetate", reinforcement=reinforcement)
            )
            self.assertEqual(plan["reinforcement"], reinforcement)
        for payload in [
            {"action": "channel", "channel": "light", "params": {"subsample": 1000000}},
            {"action": "channel", "channel": "sound", "params": {"frequency_hz": float("nan")}},
            {"action": "channel", "channel": "taste", "params": {"site": "made-up"}},
            {"action": "sugar", "reinforcement": "reward"},
            {"action": "smell", "target": "apple", "compound": "ethyl acetate"},
            {"action": "stimuli", "stimuli": [{"key": "sugar"}] * 5},
            {"action": "stimuli", "stimuli": [{"key": "sugar", "rate_hz": 301}]},
            {"action": "sugar", "seed": -1},
        ]:
            with self.assertRaises(ValidationError):
                GardenReq(**payload)

    def test_language_plans_obey_browser_limits(self):
        valid = {"duration_ms": 300, "stimuli": [{"key": "sugar", "rate_hz": 0, "side": "left"}]}
        self.assertEqual(bounded_llm_plan(valid)["stimuli"][0]["rate_hz"], 0)
        for plan in [
            dict(valid, duration_ms=100000),
            dict(valid, channels=[{"channel": "light", "params": {"subsample": 100000}}]),
            dict(valid, odor_text="x" * 121),
            dict(valid, reinforcement="reward"),
        ]:
            with self.assertRaises(ValueError):
                bounded_llm_plan(plan)

    def test_body_reset_preserves_brain_memory_history_and_seed_counter(self):
        fake = Mock()
        fake.mb.memory_strength.return_value = 0.2
        fake.associations.return_value = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(pet, "_chat", fake),
            patch.object(
                pet, "STATE", {"satiety": 0.9, "events": [], "n_stimuli": 5, "name": "fixture"}
            ),
            patch.object(pet, "STATE_PATH", Path(tmp) / "state.json"),
            patch.object(pet, "_trial_snapshots", OrderedDict()),
        ):
            result = pet._reset_garden(GardenReset(body=True))
            fake.reset.assert_not_called()
            self.assertEqual(result["state"]["satiety"], 0.3)
            pet._reset_garden(GardenReset(body=False, memory=True))
            fake.reset.assert_called_once_with(memory=True, history=False, persist_memory=True)

    def test_static_export_is_minimal_and_contains_no_private_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "site"
            manifest = export(out, "https://brain.example.org")
            self.assertEqual(len(manifest["files"]), 5)
            self.assertIn("sdk: static", (out / "README.md").read_text())
            self.assertIn(
                "https://brain.example.org", (out / "static/garden-config.js").read_text()
            )
            self.assertIn('src="./static/garden-controls.js"', (out / "index.html").read_text())
            with self.assertRaises(FileExistsError):
                export(out, "https://brain.example.org")
            for address in [
                "http://brain.example.org",
                "https://user:secret@example.org",
                "https://example.org?token=secret",
            ]:
                with self.assertRaises(ValueError):
                    export(Path(tmp) / "invalid", address)


class ThoughtTests(TestCase):
    def test_null_and_sensory_only_thoughts_do_not_invent_movement(self):
        null = readout_thought({"behaviours": {}, "n_spikes_total": 0})
        self.assertIn("no spikes", null["text"])
        sensory = readout_thought({"behaviours": {}, "n_spikes_total": 100})
        self.assertIn("motor outputs are silent", sensory["text"])
        unmapped = readout_thought({"odor_source": {"status": "unmapped"}})
        self.assertIn("does not mean the object has no smell", unmapped["text"])

    def test_thoughts_use_observed_outputs_not_input_label(self):
        summary = {
            "behaviours": {"antennal_grooming": {"max_rate_hz": 80}},
            "n_spikes_total": 10,
            "learning": {"synapses_changed": 0},
        }
        thought = readout_thought(summary)
        self.assertIn("clean my antennae", thought["text"])
        self.assertIn("did not change", thought["text"])
        llm = Mock(name="test")
        llm.name = "fake"
        llm.last_model_ids = ["fake-model"]
        llm.complete.return_value = {"reply": "I have a signal to clean my antennae."}
        reply = narrate_snapshot(llm, {"plan": {"stimuli": [{"key": "sugar"}]}, "summary": summary})
        self.assertEqual(reply["source"], "llm")
        evidence = llm.complete.call_args.args[1]
        self.assertIn("clean my antennae", evidence)
        self.assertNotIn("sugar", evidence)
        llm.complete.return_value = {}
        with self.assertRaises(ValueError):
            narrate_snapshot(llm, {"plan": {}, "summary": {"n_spikes_total": 0}})

    def test_narrator_failure_keeps_grounded_reply_and_unknown_trials_are_rejected(self):
        async def scenario():
            trial = "a" * 32
            snapshot = {
                "created": time.monotonic(),
                "plan": {},
                "summary": {},
                "thought": {"text": "Тишина.", "source": "readout"},
            }
            with (
                patch.object(pet, "_trial_snapshots", OrderedDict({trial: snapshot})),
                patch.object(pet, "_thought_task", None),
                patch.object(pet, "_describe_snapshot", side_effect=RuntimeError("offline")),
            ):
                reply = await pet.garden_thought(ThoughtReq(trial_id=trial))
                self.assertEqual(reply["thought"]["text"], "Тишина.")
                self.assertEqual(reply["thought"]["status"], "unavailable")
                missing = await pet.garden_thought(ThoughtReq(trial_id="b" * 32))
                self.assertEqual(missing.status_code, 404)

        asyncio.run(scenario())
