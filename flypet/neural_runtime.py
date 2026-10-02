"""Continuing Brian2 episodes for learned, continuous neural interfaces.

One runtime owns one brain on one process/thread. Legacy Brain.run remains a
resetting trial. Input is Bernoulli per time step, like Brain.run, but generated
with a private PCG64 stream in time-major order. It therefore does not depend on
UI chunk boundaries or Brian2's process-global Cython random-number buffers.

Observations are reader features, NOT restart checkpoints: synaptic delay queues
and RNG stay in the live worker. Restart reproducibility uses the episode seed,
initial memory, input-port order and recorded actions on the same backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import secrets
import threading
import time
import weakref

import numpy as np
from brian2 import SpikeGeneratorGroup, SpikeMonitor, Synapses, Hz, ms, mV, second

from .engine import Brain, RunResult, StimInput, MAX_MONITOR_SPIKES


@dataclass
class NeuralStep:
    result: RunResult
    observation: dict
    learning_events: list[dict]


class NeuralRuntime:
    """A continuing episode. Use as a context manager; do not share across workers.

    Each advance replaces the previous drive. Learning is updated at fixed
    model-time windows, independently of how the caller partitions advances.
    """

    _active: weakref.ReferenceType | None = None

    def __init__(
        self,
        brain: Brain,
        *,
        seed: int | None = None,
        input_ids: list[int] | None = None,
        memory=None,
        learning: bool = False,
        learning_window_ms: float = 250.0,
        max_rate_hz: float = 300.0,
        max_step_ms: float = 1000.0,
        input_refractory_ms: float = 0.0,
        monitor_limit: int = MAX_MONITOR_SPIKES,
    ):
        active = self._active() if self._active is not None else None
        if active is not None and not active.closed:
            raise RuntimeError("Use a separate process for another active neural runtime")
        if getattr(brain, "_continuous_active", False):
            raise RuntimeError("This brain already has an active runtime")
        self.brain, self.memory = brain, memory
        self.closed = False
        self._faulted = False
        self._owner = (os.getpid(), threading.get_ident())
        self.dt_ms = float(brain.neu.clock.dt / ms)
        self.window_ticks = self._ticks(learning_window_ms)
        self.max_step_ticks = self._ticks(max_step_ms)
        if not np.isfinite(max_rate_hz) or not 0 < max_rate_hz <= 1000.0 / self.dt_ms:
            raise ValueError("Rate bound must be positive and at most one input per tick")
        if not isinstance(monitor_limit, int) or monitor_limit < 1:
            raise ValueError("monitor_limit must be a positive integer")
        self.max_rate_hz, self.monitor_limit = float(max_rate_hz), monitor_limit
        refractory_ticks = float(input_refractory_ms) / self.dt_ms
        if (
            not np.isfinite(refractory_ticks)
            or refractory_ticks < 0
            or abs(refractory_ticks - round(refractory_ticks)) > 1e-7
        ):
            raise ValueError("Input refractory period must be nonnegative and on the time grid")
        self.input_refractory_ms = float(input_refractory_ms)
        if memory is not None and memory.b is not brain:
            raise ValueError("Memory must belong to this brain")
        if learning and memory is None:
            raise ValueError("Learning requires a MushroomBody")
        if input_ids is None:
            ports = np.arange(brain.n_stim, dtype=np.int32)
        else:
            try:
                ports = np.array([brain.flyid2i[int(i)] for i in input_ids], dtype=np.int32)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Unknown input neuron") from exc
            if len(np.unique(ports)) != len(ports):
                raise ValueError("Input neurons must be unique")
            ports.sort()
        if not len(ports) or np.any(ports >= brain.n_stim):
            raise ValueError("Input ports must be a nonempty subset of stimulable neurons")
        self.ports = ports
        self.root_ids = np.array([brain.i2flyid[int(i)] for i in ports], dtype=np.int64)
        self._port_by_id = {int(root): j for j, root in enumerate(self.root_ids)}
        self.seed = self._seed(seed)  # validate before touching the network
        self.learning = bool(learning)
        self.monitor_rotations = 0
        self._legacy_sources = [
            (obj, obj.active)
            for obj in (getattr(brain, "_gen", None), getattr(brain, "_gen_syn", None))
            if obj is not None
        ]
        for obj, _ in self._legacy_sources:
            obj.active = False
        self.source = SpikeGeneratorGroup(
            len(ports),
            [],
            [] * ms,
            clock=brain.neu.clock,
            when="start",
            order=-2,
            name="latent_input",
        )
        self.input_syn = Synapses(
            self.source,
            brain.neu,
            on_pre="v_post += w_drive",
            namespace={"w_drive": brain.params["w_stim"]},
            name="latent_input_syn",
        )
        self.input_syn.connect(i=np.arange(len(ports)), j=ports)
        self.input_syn.pre.when = "start"
        self.input_syn.pre.order = -1
        self.source.active = self.input_syn.active = False
        brain.net.add(self.source, self.input_syn)
        try:
            self.reset(seed=self.seed)
        except BaseException:
            self.source.active = self.input_syn.active = False
            brain.net.remove(self.source, self.input_syn)
            for obj, was_active in self._legacy_sources:
                obj.active = was_active
            brain._continuous_active = False
            self.closed = True
            raise
        NeuralRuntime._active = weakref.ref(self)

    @staticmethod
    def _seed(seed):
        if seed is None:
            return secrets.randbits(32)
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or not 0 <= seed < 2**32
        ):
            raise ValueError("seed must be an integer in [0, 2**32)")
        return int(seed)

    def _ticks(self, duration_ms):
        value = float(duration_ms) / self.dt_ms
        if not np.isfinite(value) or value <= 0 or abs(value - round(value)) > 1e-7:
            raise ValueError("Duration must be positive and an integer number of simulation ticks")
        return int(round(value))

    def _check(self):
        if self.closed:
            raise RuntimeError("NeuralRuntime is closed")
        if self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("NeuralRuntime must stay on its owning process/thread")
        if float(self.brain.neu.clock.dt / ms) != self.dt_ms:
            raise RuntimeError("Cannot change the simulation clock during an episode")

    def _rotate_monitor(self):
        old = self.brain.mon
        self.brain.net.remove(old)
        old.active = False
        self.brain.mon = SpikeMonitor(self.brain.neu, name=old.name)
        self.brain.net.add(self.brain.mon)
        self.monitor_rotations += 1

    def reset(self, *, seed: int | None = None, reset_memory: bool = False):
        """Explicit episode boundary. Preserve learned memory unless requested."""
        self._check()
        value = self._seed(seed)
        if reset_memory and self.memory is None:
            raise ValueError("No memory attached")
        self._faulted = True
        self.source.active = self.input_syn.active = False
        self.brain._continuous_active = False
        try:
            self.brain._flush()
            # Reset refractory history as well as visible voltages. This is
            # local to the new mode; published legacy trial semantics stay put.
            self.brain.neu.lastspike = -1e9 * second
            self.brain.neu.not_refractory = True
            # Fixed write-port physiology: a tiny positive rate must not also
            # turn off refractoriness and offer an unpriced control channel.
            self.brain.neu.rfc[self.ports] = self.input_refractory_ms * ms
            if reset_memory:
                self.memory.reset()
            self._rotate_monitor()
        finally:
            self.brain._continuous_active = True
        self.seed = value
        self.rng = np.random.Generator(np.random.PCG64(value))
        self.tick = 0
        self._window_count = np.zeros(self.brain.n, dtype=np.int64)
        self._window_ticks = 0
        self.source.set_spikes([], [] * ms)
        self.source.active = self.input_syn.active = True
        self._faulted = False

    def set_learning(self, enabled: bool):
        self._check()
        if self._faulted:
            raise RuntimeError("Reset the episode after a failed simulation step")
        if self._window_ticks:
            raise ValueError("Change learning only at a fixed-window boundary")
        if enabled and self.memory is None:
            raise ValueError("Learning requires a MushroomBody")
        self.learning = bool(enabled)

    def drive_from_inputs(self, inputs: list[StimInput]):
        """Reference-data helper; inference may pass drive_hz directly."""
        rates = np.zeros(len(self.ports), dtype=np.float64)
        for inp in inputs:
            if not np.isfinite(inp.rate_hz) or not 0 <= inp.rate_hz <= self.max_rate_hz:
                raise ValueError("Input rate outside the runtime bound")
            try:
                idx = [self._port_by_id[int(i)] for i in inp.ids]
            except KeyError as exc:
                raise ValueError("Stimulus includes neurons outside the declared ports") from exc
            # Same overlap rule as Brain.run: the last group wins.
            rates[idx] = inp.rate_hz
        return rates

    def _input_spikes(self, rates, n_ticks, start_s):
        indices, ticks = [], []
        probabilities = rates * (self.dt_ms / 1000.0)
        # Draw every declared port even if its drive is zero: a fixed stream per
        # time/port makes noise independent of action sparsity and API slicing.
        for offset in range(0, n_ticks, 32):
            n = min(32, n_ticks - offset)
            rows, cols = np.nonzero(self.rng.random((n, len(rates))) < probabilities)
            indices.append(cols.astype(np.int32))
            ticks.append(rows + offset)
        idx = np.concatenate(indices)
        when = start_s + np.concatenate(ticks) * (self.dt_ms / 1000.0)
        self.source.set_spikes(idx, when * second, sorted=True)

    def advance(self, drive_hz, duration_ms: float = 250.0) -> NeuralStep:
        self._check()
        if self._faulted:
            raise RuntimeError("Reset the episode after a failed simulation step")
        ticks = self._ticks(duration_ms)
        if ticks > self.max_step_ticks:
            raise ValueError("Requested advance exceeds max_step_ms")
        rates = np.asarray(drive_hz, dtype=np.float64)
        if rates.shape != self.ports.shape or not np.isfinite(rates).all():
            raise ValueError("drive_hz must be a finite vector in declared input-port order")
        if np.any(rates < 0) or np.any(rates > self.max_rate_hz):
            raise ValueError("drive_hz is outside the runtime bound")
        # Any interruption or post-run error leaves partially advanced state.
        # Keep it faulted until the entire operation succeeds, including learning.
        self._faulted = True
        t0 = time.perf_counter()
        brain = self.brain
        for op in brain.stim_ops:
            op.active = False
        brain.neu.r_stim = 0 * Hz
        accumulated, events = {}, []
        elapsed = 0
        while elapsed < ticks:
            n = min(ticks - elapsed, self.window_ticks - self._window_ticks)
            if int(brain.mon.num_spikes) >= self.monitor_limit:
                self._rotate_monitor()
            start_s = float(brain.net.t / second)
            self._input_spikes(rates, n, start_s)
            before = int(brain.mon.num_spikes)
            brain.net.run(n * self.dt_ms * ms)
            trains = brain._trains_since(before, start_s)
            for i, values in trains.items():
                accumulated.setdefault(i, []).append(values + elapsed * self.dt_ms / 1000.0)
                self._window_count[i] += len(values)
            self.tick += n
            self._window_ticks += n
            elapsed += n
            if self._window_ticks == self.window_ticks:
                if self.learning:
                    counts = {
                        int(i): int(self._window_count[i])
                        for i in np.flatnonzero(self._window_count)
                    }
                    window = RunResult(self.window_ticks * self.dt_ms / 1000.0, {}, {}, 0.0, counts)
                    event = self.memory.learn(window, note=f"neural-runtime tick {self.tick}")
                    events.append({**event, "episode_time_ms": self.tick * self.dt_ms})
                self._window_count.fill(0)
                self._window_ticks = 0
        merged = {i: np.concatenate(parts) for i, parts in accumulated.items()}
        active_ids = self.root_ids[rates > 0].tolist()
        result = RunResult(
            ticks * self.dt_ms / 1000.0,
            merged,
            {"continuous": active_ids},
            time.perf_counter() - t0,
            {i: len(ts) for i, ts in merged.items()},
        )
        brain.n_runs += 1
        observation = self._observation()
        self._faulted = False
        return NeuralStep(result, observation, events)

    def observe(self):
        """Full-neuron, float32 reader observation; not a restart checkpoint."""
        self._check()
        if self._faulted:
            raise RuntimeError("Reset the episode before observing a failed step")
        return self._observation()

    def _observation(self):
        neu = self.brain.neu
        remaining = np.maximum(
            0.0,
            np.asarray(neu.rfc[:] / ms)
            - (float(self.brain.net.t / ms) - np.asarray(neu.lastspike[:] / ms)),
        )
        return {
            "schema_version": 1,
            "episode_time_ms": self.tick * self.dt_ms,
            "voltage_mv": np.asarray(neu.v[:] / mV, dtype=np.float32).copy(),
            "synaptic_mv": np.asarray(neu.g[:] / mV, dtype=np.float32).copy(),
            "refractory_remaining_ms": remaining.astype(np.float32),
        }

    def close(self):
        if self.closed:
            return
        self._check()
        self.source.active = self.input_syn.active = False
        self.brain.net.remove(self.source, self.input_syn)
        for obj, was_active in self._legacy_sources:
            obj.active = was_active
        self.brain._continuous_active = False
        self.closed = True
        NeuralRuntime._active = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
