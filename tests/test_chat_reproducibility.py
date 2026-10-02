"""Lightweight regression tests: no connectome, language model, or user-state writes."""

from contextlib import ExitStack
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from flypet import chat
from flypet.engine import StimInput
from flypet.odor import resolve_object


class FakeBrain:
    def __init__(self):
        self.calls = []

    def run(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return SimpleNamespace(inputs=inputs, **kwargs)


class FakeMB:
    def __init__(self, brain):
        self.mem = np.array([0.4, 0.8])
        self.log = []
        self.saved = []

    def _push(self):
        pass

    def load(self, path):
        self.mem = np.load(path)["mem"].copy()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.saved.append(path)
        np.savez(path, mem=self.mem)

    def reset(self):
        self.mem[:] = 1
        self.log.clear()

    def valence(self, result):
        return SimpleNamespace(score=0.2, approach_hz=10, avoid_hz=5, top=[])

    def reward(self):
        return StimInput([90], 60, "dopamine:PAM(reward)")

    def punishment(self):
        return StimInput([91], 60, "dopamine:PPL1(punishment)")

    def learn(self, result, note):
        self.mem *= 0.9
        entry = {"n_synapses_changed": 2, "note": note}
        self.log.append(entry)
        return entry

    def memory_strength(self):
        return float(1 - self.mem.mean())


class FakeDoor:
    name2key = {"ethyl acetate": "key"}

    def __init__(self, **kwargs):
        pass

    def well_covered(self, n):
        return ["ethyl acetate"]

    def glomerular(self, compound):
        if compound.lower() not in self.name2key:
            raise KeyError(compound)
        return {"DM1": 0.8, "DM2": 0.0}

    def stim(self, compound, scale=1):
        self.glomerular(compound)
        return [StimInput([11, 12], 120 * scale, f"{compound}:DM1")]


def summary(result):
    return {
        "top_downstream": [],
        "stimulated": {},
        "duration_ms": result.duration_ms,
        "n_active_downstream": 1,
        "active_by_super_class": {},
        "wall_s": 0,
        "n_spikes_total": 2,
    }


class ChatReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(chat, "Brain", FakeBrain))
        self.stack.enter_context(patch.object(chat.corrections, "apply", lambda brain: brain))
        self.stack.enter_context(patch.object(chat, "MushroomBody", FakeMB))
        self.stack.enter_context(patch.object(chat, "DoorOdor", FakeDoor))
        self.stack.enter_context(patch.object(chat, "Vnc", Mock))
        self.stack.enter_context(patch.object(chat.A, "summarize", summary))
        self.stack.enter_context(
            patch.object(chat.A, "behaviours_text", lambda result: "neural activity")
        )
        self.encoder = self.stack.enter_context(patch.object(chat, "OdorEncoder"))

    def make_chat(self, **kwargs):
        options = dict(
            state_dir=self.tmp / "case",
            enable_door=True,
            enable_vnc=False,
            enable_reader=False,
            log=False,
            strict_door=True,
        )
        options.update(kwargs)
        return chat.FlyChat(**options)

    def test_zero_rate_and_seed_are_preserved_in_replay_record(self):
        fly = self.make_chat(seed=30)
        stimulus = SimpleNamespace(default_rate_hz=150, resolve=lambda **kw: [21, 22])
        with patch.dict(chat.STIMULI, {"fixture": stimulus}):
            p = {"stimuli": [{"key": "fixture", "rate_hz": 0}], "duration_ms": 200}
            _, first = fly.simulate(p, seed=123)
            _, second = fly.simulate(p, seed=123)
        self.assertEqual(first["simulation"], second["simulation"])
        self.assertEqual(
            first["simulation"]["inputs"],
            [{"ids": [21, 22], "rate_hz": 0.0, "key": "fixture/both"}],
        )
        self.assertEqual([kw["seed"] for _, kw in fly.brain.calls], [123, 123])
        _, third = fly.simulate({"stimuli": []})
        self.assertEqual(third["simulation"]["seed"], 32)

    def test_direct_chemical_bypasses_llm_and_saves_only_in_case_directory(self):
        fly = self.make_chat()
        fly._llm = Mock()
        _, result = fly.simulate(
            {"odor_compound": "ethyl acetate", "reinforcement": "reward"}, seed=9
        )
        fly._llm.complete.assert_not_called()
        self.encoder.assert_not_called()
        self.assertEqual(result["odor_source"]["status"], "resolved")
        self.assertEqual(result["odor_source"]["mapping"], "direct")
        self.assertEqual(fly.mb.saved, [self.tmp / "case" / "memory.npz"])
        self.assertEqual([kw["seed"] for _, kw in fly.brain.calls], [9, 9])
        other = self.make_chat(state_dir=self.tmp / "other")
        self.assertEqual(other.history, [])
        np.testing.assert_array_equal(other.mb.mem, [0.4, 0.8])
        restarted = self.make_chat()
        np.testing.assert_allclose(restarted.mb.mem, fly.mb.mem)

    def test_garden_skips_only_optional_reader_and_never_calls_llm(self):
        fly = self.make_chat()
        fly.reader = Mock()
        fly._llm = Mock()
        _, result = fly.simulate({"odor_compound": "ethyl acetate"}, seed=9, read_brain=False)
        fly.reader.describe.assert_not_called()
        fly._llm.complete.assert_not_called()
        self.assertIn("valence", result)
        self.assertFalse(result["simulation"]["reader_requested"])
        self.assertEqual([kwargs["seed"] for _, kwargs in fly.brain.calls], [9, 9])

    def test_required_reader_cannot_be_skipped(self):
        fly = self.make_chat()
        fly.require_components.add("reader")
        with self.assertRaisesRegex(ValueError, "required brain reader"):
            fly.simulate({"stimuli": []}, read_brain=False)
        self.assertEqual(fly.brain.calls, [])

    def test_missing_mapping_never_becomes_absent_smell_or_training(self):
        fly = self.make_chat()
        fly._llm = Mock()
        fly._llm.complete.return_value = {"compound": "", "why": "no reliable match"}
        _, result = fly.simulate({"odor_text": "unknown item", "reinforcement": "reward"})
        self.assertEqual(result["odor_source"]["status"], "unmapped")
        self.assertNotIn("odor_unsmellable", result)
        self.assertNotIn("valence", result)
        self.assertNotIn("learning", result)
        self.assertEqual(result["simulation"]["inputs"], [])
        self.assertFalse(result["odor_input_present"])
        self.encoder.assert_not_called()
        self.assertTrue(fly.resolve_cache_path.exists())
        fly.narrate("unknown item", {}, result)
        narrator_system, narrator_prompt = fly._llm.complete.call_args.args
        self.assertNotIn("не пахнет ничем", narrator_prompt)
        self.assertIn("unmapped", narrator_prompt)
        self.assertIn("движение тела", narrator_system)

    def test_resolver_error_fails_strict_case_and_is_not_cached(self):
        fly = self.make_chat()
        fly._llm = Mock()
        fly._llm.complete.side_effect = TimeoutError("offline")
        with self.assertRaisesRegex(RuntimeError, "DoOR stimulus failed"):
            fly.simulate({"odor_text": "apple"})
        self.encoder.assert_not_called()
        self.assertFalse(fly.resolve_cache_path.exists())
        self.assertEqual(fly.brain.calls, [])

    def test_required_components_fail_instead_of_silent_degradation(self):
        with self.assertRaisesRegex(RuntimeError, "required component vnc is disabled"):
            self.make_chat(require_components=("vnc",))
        with patch.object(chat, "DoorOdor", side_effect=OSError("missing data")):
            with self.assertRaisesRegex(RuntimeError, "required component door is unavailable"):
                self.make_chat(require_components=("door",))
        fly = self.make_chat(require_components=("vnc",), enable_vnc=True)
        fly.vnc.run_from_brain.side_effect = RuntimeError("bridge failed")
        with self.assertRaisesRegex(RuntimeError, "required VNC simulation failed"):
            fly.simulate({"stimuli": []})

    def test_naive_counterfactual_failure_restores_learned_weights(self):
        fly = self.make_chat()
        memory_before = fly.mb.mem.copy()
        original_run = fly.brain.run

        def run(inputs, **kwargs):
            if fly.brain.calls:
                raise RuntimeError("counterfactual failed")
            return original_run(inputs, **kwargs)

        fly.brain.run = run
        with self.assertRaisesRegex(RuntimeError, "counterfactual failed"):
            fly.simulate({"odor_compound": "ethyl acetate"})
        np.testing.assert_array_equal(fly.mb.mem, memory_before)
        self.assertEqual(fly.mb.saved, [])

    def test_reset_is_explicit_and_does_not_overwrite_checkpoint_by_default(self):
        fly = self.make_chat()
        fly.mb.save(fly.memory_path)
        saved = fly.memory_path.read_bytes()
        fly.history = [{"reply": "fixture"}]
        fly.reset()
        self.assertEqual(fly.history, [])
        self.assertEqual(fly.memory_path.read_bytes(), saved)
        np.testing.assert_array_equal(fly.mb.mem, [0.4, 0.8])
        fly.reset(memory=True)
        self.assertEqual(fly.memory_path.read_bytes(), saved)
        np.testing.assert_array_equal(fly.mb.mem, [1, 1])
        fly.reset(memory=True, persist_memory=True)
        with np.load(fly.memory_path) as z:
            np.testing.assert_array_equal(z["mem"], [1, 1])


class ResolverTests(unittest.TestCase):
    def test_retryable_errors_and_invalid_chemical_never_cache(self):
        llm = Mock()
        llm.complete.side_effect = [
            TimeoutError("offline"),
            {"compound": "imaginary", "why": "guess"},
            {"compound": "ethyl acetate", "why": "proxy"},
        ]
        cache = {}
        for _ in range(2):
            result = resolve_object("apple", llm, FakeDoor(), cache, cache_path=None)
            self.assertEqual(result["status"], "error")
            self.assertEqual(cache, {})
        result = resolve_object("apple", llm, FakeDoor(), cache, cache_path=None)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(cache["apple"], result)

    def test_legacy_cached_exception_is_retried(self):
        llm = Mock()
        llm.complete.return_value = {"compound": "ethyl acetate", "why": "proxy"}
        cache = {"apple": {"compound": "", "why": "ошибка: service unavailable"}}
        result = resolve_object("apple", llm, FakeDoor(), cache, cache_path=None)
        self.assertEqual(result["status"], "resolved")
        llm.complete.assert_called_once()


class PetResetTests(unittest.TestCase):
    def test_default_reset_preserves_memory_and_explicit_reset_declares_scope(self):
        from flypet import pet

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(pet, "STATE_PATH", Path(directory) / "pet_state.json"),
            patch.object(pet, "STATE", {"satiety": 0.9, "name": "fixture"}),
            patch.object(pet, "_chat", Mock()) as fake_chat,
        ):
            default = asyncio.run(pet.reset())
            fake_chat.reset.assert_not_called()
            self.assertEqual(default["reset"], {"body": True, "memory": False, "history": False})
            explicit = asyncio.run(pet.reset(pet.ResetReq(memory=True, history=True)))
            fake_chat.reset.assert_called_once_with(memory=True, history=True, persist_memory=True)
            self.assertEqual(explicit["reset"], {"body": True, "memory": True, "history": True})
            saved = json.loads(pet.STATE_PATH.read_text())
            self.assertEqual(saved["name"], "fixture")

    def test_pet_zero_rate_does_not_get_defaulted(self):
        from flypet import pet

        brain = FakeBrain()
        stimulus = SimpleNamespace(default_rate_hz=150, resolve=lambda **kw: [21])
        with (
            patch.object(pet, "_brain", brain),
            patch.object(pet, "_chat", None),
            patch.object(pet, "STATE", {"satiety": 0.5}),
            patch.dict(pet.STIMULI, {"fixture": stimulus}),
            patch.object(pet.A, "summarize", summary),
        ):
            _, rate = pet._run_stimulus("fixture", None, 0, 200)
        self.assertEqual(rate, 0)
        self.assertEqual(brain.calls[0][0][0].rate_hz, 0)


if __name__ == "__main__":
    unittest.main()
