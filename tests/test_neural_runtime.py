import copy
from functools import lru_cache
import threading
import unittest
from unittest.mock import patch

import brian2 as b
import numpy as np

from flypet.engine import Brain, StimInput
from flypet.neural_runtime import NeuralRuntime


class TinyConnectome:
    @staticmethod
    def neuron_order():
        return np.array([101, 102, 103, 104]), [1, 2, 2]

    @staticmethod
    def id_maps():
        return {101: 0, 102: 1, 103: 2, 104: 3}, {0: 101, 1: 102, 2: 103, 3: 104}

    @staticmethod
    @lru_cache(maxsize=1)
    def connections():
        return (
            np.array([0, 1, 2, 3], dtype=np.int32),
            np.array([2, 2, 3, 2], dtype=np.int32),
            np.array([90, -20, 75, 20], dtype=np.int16),
        )


class WindowMemory:
    """A nonlinear weight update makes accidental per-call learning detectable."""

    def __init__(self, brain):
        self.b = brain
        self.mem = np.ones(brain.n_syn)
        self.log = []

    def learn(self, result, note=""):
        self.mem *= 0.95
        self.b.set_weights(np.arange(self.b.n_syn), self.mem)
        entry = {"n_synapses_changed": self.b.n_syn, "counts": result.counts, "note": note}
        self.log.append(entry)
        return entry

    def reset(self):
        self.mem.fill(1)
        self.b.set_weights(np.arange(self.b.n_syn), self.mem)
        self.log.clear()


def flatten(results):
    offset, rows = 0.0, []
    for result in results:
        for i, times in result.trains.items():
            rows.extend((i, t + offset) for t in times)
        offset += result.duration_s
    return np.array(sorted(rows)).reshape(-1, 2)


class NeuralRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = b.prefs.codegen.target
        cls.dt = b.defaultclock.dt
        b.prefs.codegen.target = "numpy"
        b.defaultclock.dt = 0.1 * b.ms

    @classmethod
    def tearDownClass(cls):
        b.prefs.codegen.target = cls.target
        b.defaultclock.dt = cls.dt

    def setUp(self):
        b.start_scope()
        self.brain = Brain(conn=TinyConnectome)

    def test_partition_invariance_including_private_noise_and_final_state(self):
        with NeuralRuntime(self.brain, seed=41, learning_window_ms=10) as runtime:
            full = runtime.advance([173.25, 91.75], 53.7)
            runtime.reset(seed=41)
            parts = []
            for duration in (0.1, 3.2, 17.4, 20.0, 13.0):
                # This must not perturb episode noise.
                np.random.random(100)
                parts.append(runtime.advance([173.25, 91.75], duration))
            np.testing.assert_allclose(
                flatten([full.result]), flatten([s.result for s in parts]), rtol=0, atol=1e-12
            )
            self.assertGreater(len(flatten([full.result])), 0)
            for key in ("voltage_mv", "synaptic_mv", "refractory_remaining_ms"):
                np.testing.assert_allclose(
                    full.observation[key], parts[-1].observation[key], atol=1e-6, rtol=0
                )
            self.assertAlmostEqual(runtime.observe()["episode_time_ms"], 53.7)

    def test_delay_survives_step_boundary_and_monitor_rotation(self):
        with NeuralRuntime(self.brain, seed=3, max_rate_hz=10000, monitor_limit=1) as runtime:
            a = runtime.advance([10000, 0], 0.1)
            self.assertEqual(a.result.counts, {0: 1})
            # Arrival is 1.8 ms later, beyond the first call. Replacing only the
            # monitor must preserve that pending synaptic event and its weight.
            network = runtime.brain.net
            syn = runtime.brain.syn
            before_arrival = runtime.advance([0, 0], 0.8)
            self.assertEqual(before_arrival.observation["synaptic_mv"][2], 0)
            z = runtime.advance([0, 0], 4.2)
            self.assertGreater(z.observation["synaptic_mv"][2], 0)
            self.assertGreater(z.observation["voltage_mv"][2], -52)
            self.assertIs(runtime.brain.net, network)
            self.assertIs(runtime.brain.syn, syn)
            self.assertGreaterEqual(runtime.monitor_rotations, 2)

    def test_rotation_matches_no_rotation(self):
        with NeuralRuntime(self.brain, seed=53, monitor_limit=1) as runtime:
            parts = [runtime.advance([240, 20], 20) for _ in range(5)]
            runtime.monitor_limit = 100000
            runtime.reset(seed=53)
            ref = runtime.advance([240, 20], 100)
            np.testing.assert_allclose(
                flatten([x.result for x in parts]), flatten([ref.result]), atol=1e-12, rtol=0
            )
            np.testing.assert_array_equal(
                parts[-1].observation["voltage_mv"], ref.observation["voltage_mv"]
            )

    def test_learning_window_independent_of_caller_chunking(self):
        memory = WindowMemory(self.brain)
        with NeuralRuntime(
            self.brain, seed=13, memory=memory, learning=True, learning_window_ms=10
        ) as runtime:
            full = runtime.advance([220, 80], 50)
            ref_mem, ref_log = memory.mem.copy(), copy.deepcopy(memory.log)
            runtime.reset(seed=13, reset_memory=True)
            steps = [runtime.advance([220, 80], d) for d in (3, 12, 6, 29)]
            np.testing.assert_array_equal(memory.mem, ref_mem)
            self.assertEqual(memory.log, ref_log)
            self.assertEqual(sum(len(s.learning_events) for s in steps), 5)
            np.testing.assert_allclose(
                flatten([full.result]), flatten([s.result for s in steps]), atol=1e-12, rtol=0
            )

    def test_changing_learning_mid_window_is_rejected(self):
        memory = WindowMemory(self.brain)
        with NeuralRuntime(self.brain, seed=1, memory=memory, learning_window_ms=10) as runtime:
            runtime.advance([0, 0], 3)
            with self.assertRaises(ValueError):
                runtime.set_learning(True)
            runtime.advance([0, 0], 7)
            runtime.set_learning(True)
            runtime.advance([0, 0], 10)
            self.assertEqual(len(memory.log), 1)

    def test_reference_input_mapping_and_invalid_calls_leave_state_untouched(self):
        with NeuralRuntime(self.brain, seed=7, input_ids=[102, 101]) as runtime:
            np.testing.assert_array_equal(runtime.root_ids, [101, 102])
            rates = runtime.drive_from_inputs([StimInput([101], 72.4), StimInput([102], 13.2)])
            np.testing.assert_array_equal(rates, [72.4, 13.2])
            state = copy.deepcopy(runtime.rng.bit_generator.state)
            for rates, duration in (
                ([301, 0], 1),
                ([np.nan, 0], 1),
                ([0], 1),
                ([0, 0], 0.15),
                ([0, 0], 1001),
            ):
                with self.assertRaises(ValueError):
                    runtime.advance(rates, duration)
            self.assertEqual(runtime.tick, 0)
            self.assertEqual(state, runtime.rng.bit_generator.state)
            with self.assertRaises(ValueError):
                runtime.drive_from_inputs([StimInput([104], 10)])

    def test_legacy_reset_blocked_while_active_and_available_after_close(self):
        with NeuralRuntime(self.brain, seed=4) as runtime:
            runtime.advance([100, 0], 1)
            with self.assertRaises(RuntimeError):
                self.brain.run([], duration_ms=1)
            with self.assertRaises(RuntimeError):
                self.brain._build()
            object_count = len(self.brain.net.objects)
            with self.assertRaises(RuntimeError):
                self.brain.run_timed([], 1)
            self.assertEqual(object_count, len(self.brain.net.objects))
            with self.assertRaises(RuntimeError):
                NeuralRuntime(self.brain, seed=3)
        self.brain.run([], duration_ms=1, seed=3)
        with self.assertRaises(RuntimeError):
            runtime.advance([0, 0], 1)

    def test_thread_affinity(self):
        errors = []
        with NeuralRuntime(self.brain, seed=0) as runtime:

            def other_thread():
                try:
                    runtime.advance([0, 0], 1)
                except RuntimeError as exc:
                    errors.append(str(exc))

            thread = threading.Thread(target=other_thread)
            thread.start()
            thread.join()
            self.assertEqual(len(errors), 1)
            self.assertEqual(runtime.tick, 0)

    def test_seeded_replay_and_live_seed_diversity(self):
        with NeuralRuntime(self.brain, seed=None) as runtime:
            seed = runtime.seed
            a = runtime.advance([250, 120], 80)
            runtime.reset(seed=seed)
            b_ = runtime.advance([250, 120], 80)
            np.testing.assert_allclose(
                flatten([a.result]), flatten([b_.result]), atol=1e-12, rtol=0
            )
            runtime.reset(seed=(seed + 1) % 2**32)
            c = runtime.advance([250, 120], 80)
            self.assertNotEqual(flatten([a.result]).tolist(), flatten([c.result]).tolist())

    def test_interrupted_step_and_post_run_learning_error_require_reset(self):
        memory = WindowMemory(self.brain)
        with NeuralRuntime(
            self.brain, seed=1, memory=memory, learning=True, learning_window_ms=10
        ) as runtime:
            with patch.object(self.brain.net, "run", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    runtime.advance([100, 0], 10)
            with self.assertRaises(RuntimeError):
                runtime.advance([100, 0], 10)
            with self.assertRaises(RuntimeError):
                runtime.observe()
            runtime.reset(seed=1, reset_memory=True)
            with patch.object(memory, "learn", side_effect=RuntimeError("simulated failure")):
                with self.assertRaises(RuntimeError):
                    runtime.advance([100, 0], 10)
            with self.assertRaises(RuntimeError):
                runtime.advance([100, 0], 10)
            runtime.reset(seed=1, reset_memory=True)
            runtime.advance([100, 0], 10)

    def test_failed_reset_and_initialization_do_not_leave_silent_corruption(self):
        with NeuralRuntime(self.brain, seed=1) as runtime:
            with patch.object(self.brain, "_flush", side_effect=RuntimeError("reset failed")):
                with self.assertRaises(RuntimeError):
                    runtime.reset(seed=2)
            with self.assertRaises(RuntimeError):
                runtime.advance([100, 0], 10)
            runtime.reset(seed=2)
            runtime.advance([100, 0], 10)
        objects = set(self.brain.net.objects)
        with patch.object(self.brain, "_flush", side_effect=RuntimeError("init failed")):
            with self.assertRaises(RuntimeError):
                NeuralRuntime(self.brain, seed=0)
        self.assertEqual(set(self.brain.net.objects), objects)
        self.assertFalse(self.brain._continuous_active)
        with NeuralRuntime(self.brain, seed=0):
            pass

    def test_advance_leaves_numpy_global_random_state_unchanged(self):
        with NeuralRuntime(self.brain, seed=9) as runtime:
            before = np.random.get_state()
            runtime.advance([150, 20], 20)
            after = np.random.get_state()
            self.assertEqual(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2:], after[2:])

    def test_drive_does_not_change_input_refractory_policy(self):
        with NeuralRuntime(self.brain, seed=9, input_refractory_ms=0.7) as runtime:
            before = np.asarray(self.brain.neu.rfc[:] / b.ms).copy()
            runtime.advance([0, 0], 2)
            runtime.advance([1e-9, 220], 2)
            runtime.advance([220, 0], 2)
            np.testing.assert_array_equal(before, np.asarray(self.brain.neu.rfc[:] / b.ms))
            np.testing.assert_allclose(before[:2], [0.7, 0.7])
            np.testing.assert_allclose(before[2:], [2.2, 2.2])


if __name__ == "__main__":
    unittest.main()
