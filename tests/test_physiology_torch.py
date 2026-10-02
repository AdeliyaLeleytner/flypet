import math
import unittest

try:
    import torch
except ImportError:  # language-model extras are optional, see requirements-llm.txt
    raise unittest.SkipTest("torch is not installed")
from flypet.physiology_torch import LIFParameters, PhysiologicalCore, FixedSparsePropagation


class PhysiologyCoreTests(unittest.TestCase):
    def test_checkpoint_outputs_and_gradients_across_delay_boundaries(self):
        torch.manual_seed(71)
        params = LIFParameters(delay_ms=0.7, refractory_ms=0.3)
        direct = PhysiologicalCore(4, [0, 1, 2, 2], [1, 2, 3, 1], [80, 70, 90, -8], params)
        checked = PhysiologicalCore(4, [0, 1, 2, 2], [1, 2, 3, 1], [80, 70, 90, -8], params)
        x = (torch.rand(57, 2, 1) * 150).requires_grad_()
        xx = x.detach().clone().requires_grad_()
        a = direct(x, [0], [1, 2, 3])
        b = checked(xx, [0], [1, 2, 3], checkpoint_steps=11)
        for key in ("rate_hz", "voltage_mv", "synaptic_mv", "neuron_spike_counts"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        for r in (a, b):
            (r["voltage_mv"].square().sum() + 1e-5 * r["rate_hz"].sum()).backward()
        torch.testing.assert_close(
            direct.raw_gain.grad, checked.raw_gain.grad, rtol=2e-5, atol=2e-5
        )
        torch.testing.assert_close(x.grad, xx.grad, rtol=2e-5, atol=2e-5)
        self.assertGreater(x.grad[:11].abs().sum().item(), 0)

    def test_explicit_state_continuation_keeps_delay_queue(self):
        m = PhysiologicalCore(2, [0], [1], [100], LIFParameters(delay_ms=0.7))
        x = torch.zeros(20, 1, 1)
        x[0] = 2000
        full = m(x, [0], [1])
        first = m(x[:3], [0], [1])
        second = m(x[3:], [0], [1], initial_state=first["state"])
        torch.testing.assert_close(full["voltage_mv"], second["voltage_mv"], rtol=0, atol=0)
        self.assertGreater(second["voltage_mv"].item(), 0)

    def test_checkpoint_carries_refractory_write_guard_for_pulse_input(self):
        a = PhysiologicalCore(2, [0], [1], [100], surrogate_scale=0.1)
        b = PhysiologicalCore(2, [0], [1], [100], surrogate_scale=0.1)
        p = torch.full((83, 1, 1), 68.75)
        x = torch.zeros_like(p)
        aa = a(x, [0], [0, 1], impulses_mv=p, input_refractory_ms=0)
        bb = b(x, [0], [0, 1], impulses_mv=p, input_refractory_ms=0, checkpoint_steps=13)
        for k in ["voltage_mv", "synaptic_mv", "neuron_spike_counts"]:
            torch.testing.assert_close(aa[k], bb[k], rtol=0, atol=0)
        aa["voltage_mv"].sum().backward()
        bb["voltage_mv"].sum().backward()
        torch.testing.assert_close(a.raw_gain.grad, b.raw_gain.grad, rtol=2e-5, atol=2e-5)

    def test_impulse_input_matches_brian2_scheduling_and_input_refractory(self):
        import numpy as np
        import brian2 as b

        b.start_scope()
        b.prefs.codegen.target = "numpy"
        b.defaultclock.dt = 0.1 * b.ms
        pulse = np.zeros((180, 2))
        pulse[:20, 0] = 68.75
        pulse[20::3, 0] = 68.75
        inp = b.TimedArray(pulse * b.mV, dt=0.1 * b.ms)
        ng = b.NeuronGroup(
            2,
            "dv/dt=(-v+g)/(20*ms):volt (unless refractory)\n"
            "dg/dt=-g/(5*ms):volt (unless refractory)\nrfc:second",
            threshold="v>7*mV",
            reset="v=0*mV;g=0*mV",
            refractory="rfc",
            method="linear",
            namespace={"inp": inp},
        )
        ng.rfc = [0, 2.2] * b.ms
        ng.run_regularly("v += inp(t,i)", when="start")
        syn = b.Synapses(ng, ng, on_pre="g_post += 27.5*mV", delay=1.8 * b.ms)
        syn.connect(i=[0], j=[1])
        mon = b.StateMonitor(ng, ["v", "g"], record=True, when="end")
        sm = b.SpikeMonitor(ng)
        b.Network(ng, syn, mon, sm).run(18 * b.ms)
        m = PhysiologicalCore(2, [0], [1], [100])
        p = torch.tensor(pulse[:, :1], dtype=torch.float32)[:, None, :]
        y = m(torch.zeros_like(p), [0], [0, 1], impulses_mv=p, input_refractory_ms=0, record=True)
        for k, ref in [("v", np.asarray(mon.v / b.mV).T), ("g", np.asarray(mon.g / b.mV).T)]:
            arr = torch.stack([r[k][0] for r in y["traces"]]).numpy()
            np.testing.assert_allclose(arr, ref, rtol=2e-5, atol=2e-5)
        actual = np.argwhere(torch.stack([r["spike"][0] for r in y["traces"]]).numpy() > 0)
        expected = np.c_[np.rint(np.asarray(sm.t / b.ms) / 0.1).astype(int), np.asarray(sm.i)]
        np.testing.assert_array_equal(actual, expected)

    def test_cached_sparse_backward_equals_dense_autograd(self):
        m = PhysiologicalCore(3, [0, 1, 2], [1, 2, 0], [2, -3, 4])
        torch.manual_seed(12)
        x = torch.randn(2, 3, requires_grad=True)
        xx = x.detach().clone().requires_grad_()
        y = FixedSparsePropagation.apply(x, m.w, m.wt)
        expected = xx @ m.w.to_dense().T
        torch.testing.assert_close(y, expected)
        y.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, xx.grad)

    def test_passive_solution(self):
        p = LIFParameters(threshold_above_rest_mv=100)
        m = PhysiologicalCore(1, [], [], [], p)
        y = m(torch.full((50, 1, 1), 10.0), [0], [0])
        self.assertAlmostEqual(y["voltage_mv"].item(), 10 * (1 - math.exp(-5 / 20)), places=5)
        self.assertEqual(y["spike_count"].item(), 0)

    def test_sign_direction_and_delay(self):
        p = LIFParameters(delay_ms=0.2, refractory_ms=0.3)
        for sign in (1, -1):
            m = PhysiologicalCore(2, [0], [1], [100 * sign], p)
            drive = torch.zeros(5, 1, 1)
            drive[0] = 2000
            y = m(drive, [0], [0, 1], record=True)
            tr = y["traces"]
            self.assertEqual(tr[0]["spike"][0, 0].item(), 1)
            self.assertEqual(tr[1]["g"][0, 1].item(), 0)
            self.assertAlmostEqual(tr[2]["g"][0, 1].item(), 27.5 * sign, places=5)
            self.assertEqual(tr[2]["v"][0, 1].item(), 0)
            self.assertTrue(tr[3]["v"][0, 1].item() * sign > 0)

    def test_refractory_and_reset(self):
        p = LIFParameters(delay_ms=0, refractory_ms=0.3)
        m = PhysiologicalCore(1, [], [], [], p)
        y = m(torch.full((7, 1, 1), 2000.0), [0], [0], record=True)
        self.assertEqual([int(t["spike"].item()) for t in y["traces"]], [1, 0, 0, 1, 0, 0, 1])
        self.assertTrue(all(t["v"].item() == 0 for t in y["traces"]))

    def test_gradient_reaches_brain_gain_and_input(self):
        p = LIFParameters(delay_ms=0.2)
        m = PhysiologicalCore(2, [0], [1], [100], p)
        drive = torch.zeros(6, 1, 1)
        drive[0] = 2000
        drive.requires_grad_()
        y = m(drive, [0], [1])
        loss = y["voltage_mv"].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(m.raw_gain.grad).all())
        self.assertGreater(m.raw_gain.grad[0].abs().item(), 0)
        self.assertGreater(drive.grad.abs().sum().item(), 0)
        before = m.raw_gain.detach().clone()
        torch.optim.SGD(m.parameters(), lr=0.01).step()
        self.assertFalse(torch.equal(before, m.raw_gain.detach()))
        self.assertTrue(((m.gain() > 0.25) & (m.gain() < 2)).all())

    def test_disconnected_output_and_batch_independence(self):
        m = PhysiologicalCore(2, [], [], [])
        drive = torch.full((30, 2, 1), 100.0)
        drive[:, 1] = 0
        y = m(drive, [0], [0, 1])
        self.assertTrue((y["voltage_mv"][:, 1] == 0).all())
        one = m(drive[:, :1], [0], [0, 1])
        self.assertTrue(torch.equal(y["rate_hz"][:1], one["rate_hz"]))
        self.assertEqual(y["rate_hz"][1, 0].item(), 0)

    def test_time_grid_and_ports_rejected(self):
        with self.assertRaises(ValueError):
            LIFParameters(delay_ms=0.25)
        m = PhysiologicalCore(2, [0], [1], [1])
        with self.assertRaises(ValueError):
            m(torch.zeros(2, 1, 2), [0, 0], [1])

    def test_forward_matches_independent_brian2_microcircuit(self):
        import numpy as np
        import brian2 as b

        b.start_scope()
        b.prefs.codegen.target = "numpy"
        b.defaultclock.dt = 0.1 * b.ms
        stimulus = np.zeros((12, 2))
        stimulus[0, 0] = 2000
        stimulus[4, 0] = 2000
        applied = b.TimedArray(stimulus * b.mV, dt=0.1 * b.ms)
        neurons = b.NeuronGroup(
            2,
            "dv/dt=(-v+g+applied(t,i))/(20*ms):volt (unless refractory)\n"
            "dg/dt=-g/(5*ms):volt (unless refractory)",
            threshold="v>7*mV",
            reset="v=0*mV;g=0*mV",
            refractory=0.3 * b.ms,
            method="linear",
            namespace={"applied": applied},
        )
        synapses = b.Synapses(neurons, neurons, on_pre="g_post += 27.5*mV", delay=0.2 * b.ms)
        synapses.connect(i=[0], j=[1])
        monitor = b.StateMonitor(neurons, ["v", "g"], record=True, when="end")
        spikes = b.SpikeMonitor(neurons)
        b.Network(neurons, synapses, monitor, spikes).run(1.2 * b.ms)
        core = PhysiologicalCore(2, [0], [1], [100], LIFParameters(delay_ms=0.2, refractory_ms=0.3))
        y = core(
            torch.tensor(stimulus[:, :1], dtype=torch.float32)[:, None, :], [0], [0, 1], record=True
        )
        for key, expected in [
            ("v", np.asarray(monitor.v / b.mV).T),
            ("g", np.asarray(monitor.g / b.mV).T),
        ]:
            actual = torch.stack([t[key][0] for t in y["traces"]]).numpy()
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        actual_spikes = np.argwhere(torch.stack([t["spike"][0] for t in y["traces"]]).numpy() > 0)
        expected_spikes = np.c_[
            np.rint(np.asarray(spikes.t / b.ms) / 0.1).astype(int), np.asarray(spikes.i)
        ]
        np.testing.assert_array_equal(actual_spikes, expected_spikes)


if __name__ == "__main__":
    unittest.main()
