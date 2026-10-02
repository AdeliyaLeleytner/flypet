"""Experimental differentiable LIF core for resource profiling, not the live pet.

Forward dynamics retain the uncorrected engine's LIF constants, signs, reset,
refractory interval and transmission delay. Input is a continuous voltage-equivalent
drive, NOT the engine's Poisson input. Spike derivatives are surrogate gradients;
reset and refractory scheduling are detached. Dopamine plasticity is not included.
Only positive presynaptic gains are trained, not independent synaptic weights.
"""

from dataclasses import dataclass
import math
from functools import partial
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale=1.0):
        ctx.save_for_backward(x)
        ctx.scale = scale
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        value = ctx.scale * grad / (1 + x.abs()).square()
        return (value, None) if len(ctx.needs_input_grad) == 2 else value


class FixedSparsePropagation(torch.autograd.Function):
    """Exact linear backward with a cached transpose of the FIXED sparse matrix."""

    @staticmethod
    def forward(ctx, x, w, wt):
        ctx.save_for_backward(wt)
        return torch.sparse.mm(w, x.T).T

    @staticmethod
    def backward(ctx, grad):
        (wt,) = ctx.saved_tensors
        return torch.sparse.mm(wt, grad.T.contiguous()).T, None, None


@dataclass(frozen=True)
class LIFParameters:
    dt_ms: float = 0.1
    membrane_tau_ms: float = 20.0
    synapse_tau_ms: float = 5.0
    threshold_above_rest_mv: float = 7.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    synapse_mv_per_contact: float = 0.275

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite physiological parameter")
        if (
            min(
                self.dt_ms,
                self.membrane_tau_ms,
                self.synapse_tau_ms,
                self.threshold_above_rest_mv,
                self.synapse_mv_per_contact,
            )
            <= 0
        ):
            raise ValueError("Time constants, step and scale must be positive")
        if min(self.refractory_ms, self.delay_ms) < 0:
            raise ValueError("Delay and refractory interval cannot be negative")
        for duration in (self.refractory_ms, self.delay_ms):
            if abs(duration / self.dt_ms - round(duration / self.dt_ms)) > 1e-6:
                raise ValueError("Delay and refractory interval must be whole ticks")


class PhysiologicalCore(nn.Module):
    def __init__(self, n, pre, post, signed_contacts, parameters=None, *, surrogate_scale=1.0):
        super().__init__()
        self.parameters_spec = p = parameters or LIFParameters()
        self.n = int(n)
        if not math.isfinite(surrogate_scale) or surrogate_scale <= 0:
            raise ValueError("Invalid surrogate scale")
        self.surrogate_scale = float(surrogate_scale)
        pre = torch.as_tensor(pre, dtype=torch.long)
        post = torch.as_tensor(post, dtype=torch.long)
        values = torch.as_tensor(signed_contacts, dtype=torch.float32)
        if self.n < 1 or pre.ndim != 1 or pre.shape != post.shape or pre.shape != values.shape:
            raise ValueError("Invalid graph shape")
        if not torch.isfinite(values).all() or (
            len(pre) and (min(pre.min(), post.min()) < 0 or max(pre.max(), post.max()) >= self.n)
        ):
            raise ValueError("Invalid graph indices or weights")
        # W[post, pre], same orientation as the simulator; no dense N-by-N matrix.
        with torch.sparse.check_sparse_tensor_invariants():
            matrix = torch.sparse_coo_tensor(
                torch.stack((post, pre)), values * p.synapse_mv_per_contact, (self.n, self.n)
            ).coalesce()
            self.register_buffer("w", matrix.to_sparse_csr())
            self.register_buffer("wt", matrix.transpose(0, 1).coalesce().to_sparse_csr())
        # Gain range [0.25,2], initialized to exactly one. Never changes edge sign.
        start = math.log((1 - 0.25) / (2 - 1))
        self.raw_gain = nn.Parameter(torch.full((self.n,), start))
        self.delay_ticks = round(p.delay_ms / p.dt_ms)
        self.refractory_ticks = round(p.refractory_ms / p.dt_ms)
        self.av = math.exp(-p.dt_ms / p.membrane_tau_ms)
        self.ag = math.exp(-p.dt_ms / p.synapse_tau_ms)
        self.coupling = (
            p.dt_ms / p.membrane_tau_ms * self.av
            if p.membrane_tau_ms == p.synapse_tau_ms
            else p.synapse_tau_ms / (p.membrane_tau_ms - p.synapse_tau_ms) * (self.av - self.ag)
        )

    def gain(self):
        return 0.25 + 1.75 * torch.sigmoid(self.raw_gain)

    def initial_state(self, batch):
        v = self.raw_gain.new_zeros(batch, self.n)
        return (
            v,
            torch.zeros_like(v),
            torch.zeros_like(v, dtype=torch.long),
            torch.ones_like(v, dtype=torch.bool),
            *(torch.zeros_like(v) for _ in range(self.delay_ticks)),
        )

    def _chunk(
        self,
        drive,
        impulses,
        *state,
        ins,
        outs,
        refractory_duration,
        offset,
        readout_start,
        record=False,
    ):
        v, g, refractory, write_allowed, *pending = state
        gain = self.gain()
        count = drive.new_zeros(drive.shape[1], len(outs))
        all_counts = torch.zeros_like(v)
        traces = []
        for local_tick in range(len(drive)):
            # Voltage impulses are an explicit start-of-tick input, matching
            # Brian2's run_regularly(..., when='start'). Continuous drive remains
            # separately available for differentiable, engineered text inputs.
            vv0 = v + torch.zeros_like(v).index_copy(1, ins, impulses[local_tick]) * write_allowed
            current = torch.zeros_like(v).index_copy(1, ins, drive[local_tick])
            active = (refractory == 0).to(v.dtype)
            vv = (
                active * (self.av * vv0 + self.coupling * g + (1 - self.av) * current)
                + (1 - active) * vv0
            )
            gg = active * self.ag * g + (1 - active) * g
            spike = (
                SurrogateSpike.apply(
                    vv - self.parameters_spec.threshold_above_rest_mv, self.surrogate_scale
                )
                * active
            )
            emitted = FixedSparsePropagation.apply(spike * gain, self.w, self.wt)
            if self.delay_ticks:
                arrivals = pending[0]
                pending = [*pending[1:], emitted]
            else:
                arrivals = emitted
            event = spike.detach()
            v = vv * (1 - event)
            # Brian2's '(unless refractory)' also protects synaptic writes.
            g = (gg + arrivals * active) * (1 - event)
            refractory = torch.where(
                event.bool(), (refractory_duration - 1).clamp_min(0), (refractory - 1).clamp_min(0)
            )
            # Start-of-next-tick input runs BEFORE Brian2 updates not_refractory.
            # Even refractory=0 therefore blocks a directly consecutive jump.
            write_allowed = active.bool() & ~event.bool()
            if offset + local_tick >= readout_start:
                count = count + spike[:, outs]
            all_counts = all_counts + event
            if record:
                traces.append(
                    {
                        "v": v.detach().clone(),
                        "g": g.detach().clone(),
                        "spike": event.clone(),
                        "refractory": refractory.clone(),
                    }
                )
        values = (v, g, refractory, write_allowed, *pending, count, all_counts)
        return (values, traces) if record else values

    def forward(
        self,
        drive,
        input_indices,
        output_indices,
        record=False,
        *,
        checkpoint_steps=0,
        impulses_mv=None,
        input_refractory_ms=None,
        initial_state=None,
        readout_ticks=None,
    ):
        """Full BPTT, with optional recomputation rather than state detachment.

        State includes the complete chronological delay queue. Episodes reset by
        default; passing initial_state explicitly continues a sequence. Inputs are
        T,B,input continuous voltage-equivalent drives plus optional voltage jumps.
        """
        if (
            drive.ndim != 3
            or drive.device != self.raw_gain.device
            or drive.dtype != self.raw_gain.dtype
        ):
            raise ValueError("Expected T,B,input on the core device/dtype")
        ins = torch.as_tensor(input_indices, dtype=torch.long, device=drive.device)
        outs = torch.as_tensor(output_indices, dtype=torch.long, device=drive.device)
        if (
            ins.ndim != 1
            or outs.ndim != 1
            or drive.shape[2] != len(ins)
            or len(ins.unique()) != len(ins)
            or len(outs.unique()) != len(outs)
        ):
            raise ValueError("Input/output port shape or uniqueness mismatch")
        for ports in (ins, outs):
            if not len(ports) or ports.min() < 0 or ports.max() >= self.n:
                raise ValueError("Invalid ports")
        if not torch.isfinite(drive).all() or drive.shape[0] < 1 or checkpoint_steps < 0:
            raise ValueError("Invalid drive/checkpoint length")
        if record and checkpoint_steps:
            raise ValueError("Trace recording and recomputation must be separate runs")
        impulses = torch.zeros_like(drive) if impulses_mv is None else impulses_mv
        if (
            impulses.shape != drive.shape
            or impulses.device != drive.device
            or not torch.isfinite(impulses).all()
        ):
            raise ValueError("Invalid voltage impulses")
        duration = torch.full(
            (1, self.n), self.refractory_ticks, device=drive.device, dtype=torch.long
        )
        if input_refractory_ms is not None:
            ticks = input_refractory_ms / self.parameters_spec.dt_ms
            if ticks < 0 or abs(ticks - round(ticks)) > 1e-6:
                raise ValueError("Invalid input refractory time")
            duration[:, ins] = round(ticks)
        state = (
            self.initial_state(drive.shape[1]) if initial_state is None else tuple(initial_state)
        )
        if len(state) != 4 + self.delay_ticks or any(
            t.shape != (drive.shape[1], self.n) or t.device != drive.device for t in state
        ):
            raise ValueError("State shape/device does not match core")
        window = len(drive) if readout_ticks is None else int(readout_ticks)
        if not 1 <= window <= len(drive):
            raise ValueError("Invalid readout window")
        count = drive.new_zeros(drive.shape[1], len(outs))
        all_counts = torch.zeros_like(state[0])
        traces = []
        size = checkpoint_steps or len(drive)
        for start in range(0, len(drive), size):
            end = min(start + size, len(drive))
            fn = partial(
                self._chunk,
                ins=ins,
                outs=outs,
                refractory_duration=duration,
                offset=start,
                readout_start=len(drive) - window,
                record=record,
            )
            if checkpoint_steps and torch.is_grad_enabled():
                values = checkpoint(
                    fn,
                    drive[start:end],
                    impulses[start:end],
                    *state,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                values = fn(drive[start:end], impulses[start:end], *state)
            if record:
                values, tr = values
                traces.extend(tr)
            state = tuple(values[:-2])
            count = count + values[-2]
            all_counts = all_counts + values[-1]
        return {
            "rate_hz": count / (window * self.parameters_spec.dt_ms / 1000),
            "voltage_mv": state[0][:, outs],
            "synaptic_mv": state[1][:, outs],
            "spike_count": all_counts.sum(),
            "neuron_spike_counts": all_counts,
            "state": state,
            "traces": traces,
        }
