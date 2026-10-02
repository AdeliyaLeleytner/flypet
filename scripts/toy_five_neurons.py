"""Five-neuron walk-through of the exact update rules used in flypet.engine (same constants),
with dt = 1 ms for readability (the real engine uses 0.1 ms) and delay 2 steps (1.8 ms)."""

import math, numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=160)
names = ["ORN", "PN", "LN", "KC", "MBON"]
v0, vth, tm, tau, w_syn = -52.0, -45.0, 20.0, 5.0, 0.275
delay, rfc, dt = 2, 2, 1.0  # steps; refractory 2.2 ms -> 2 steps
# synapse counts from the connectome (toy), sign by transmitter, corrections: PN->KC x2, KC->MBON x3
count = np.zeros((5, 5))
count[0, 1] = 40
count[0, 2] = 20
count[2, 1] = 20
count[1, 3] = 30
count[3, 4] = 8  # ORN node stands for a whole glomerulus
sign = np.ones((5, 5))
sign[2, :] = -1  # LN is GABAergic
scale = np.ones((5, 5))
scale[1, 3] = 2
scale[3, 4] = 3  # corrections (base_scale); mem = 1 (naive)
W = count * sign * scale * w_syn  # mV, W[pre, post]
print("W[pre, post] в мВ (строка = кто спайкует, столбец = кому прибавляется g):")
print("        " + "  ".join(f"{n:>6s}" for n in names))
for i, n in enumerate(names):
    print(f"{n:>6s}  " + "  ".join(f"{W[i, j]:6.2f}" for j in range(5)))
input_spikes = {
    0,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
}  # ORN population fires every 2 ms (r_stim path: v += 68.75 -> spike)
v = np.full(5, v0)
g = np.zeros(5)
last = np.full(5, -99)
queue = {}
A_coef = -tau / (tm - tau)
print(
    f"\nправило шага (dt=1 мс): g_new = g·e^(−1/5) ; v_new = v0 + ((v−v0) − A)·e^(−1/20) + A·e^(−1/5), A = −g/3 ; спайк если v > −45 → v=−52, g=0, пауза 2 шага"
)
print(
    f"\n{'t':>3} | {'вход':>4} | {'доставлено (g +=)':<34} | {'g после':<32} | {'v после':<36} | спайки"
)
for t in range(30):
    arrivals = queue.pop(t, [])
    deliv = []
    for pre, post in arrivals:
        if t - last[post] > rfc:  # (unless refractory): вход в рефрактерности игнорируется
            g[post] += W[pre, post]
            deliv.append(f"{names[pre]}→{names[post]} {W[pre, post]:+.2f}")
        else:
            deliv.append(f"{names[pre]}→{names[post]} (рефр.)")
    # exact update over one step
    A = A_coef * g
    refr = (t - last) <= rfc
    v_new = v0 + ((v - v0) - A) * math.exp(-dt / tm) + A * math.exp(-dt / tau)
    v = np.where(refr, v, v_new)
    g = np.where(refr, g, g * math.exp(-dt / tau))
    if t in input_spikes:
        v[0] = v0 + 68.75  # внешний стимул: прямо в v, минуя g
    spk = [i for i in range(5) if v[i] > vth]
    for i in spk:
        v[i] = v0
        g[i] = 0.0
        last[i] = t
        for j in np.flatnonzero(count[i]):
            queue.setdefault(t + delay, []).append((i, j))
    print(
        f"{t:>3} | {'ORN' if t in input_spikes else '':>4} | {', '.join(deliv):<34} | {np.array2string(g, precision=2):<32} | {np.array2string(v, precision=1):<36} | {' '.join(names[i] for i in spk)}"
    )
# --- learning step: KC->MBON
rate_kc, rate_dan = (
    40.0,
    60.0,
)  # Hz measured in the probe; DAN = PAM driven (reward), MBON in a PAM compartment
a = min(1, rate_kc / 15)
d = min(1, rate_dan / 20)
d = d if d >= 0.3 else 0
mem = max(0.05, 1.0 * (1 - 0.6 * a * d))
print(
    f"\nобучение: KC {rate_kc} Гц → a = min(1, 40/15) = {a}; дофамин на MBON {rate_dan} Гц → d = min(1, 60/20) = {d} (порог 0.3 пройден)"
)
print(
    f"mem_new = max(0.05, 1 × (1 − 0.6·{a}·{d})) = {mem:.2f};  W[KC→MBON] = 8 × 0.275 × 3 × {mem:.2f} = {8 * 0.275 * 3 * mem:.2f} мВ (было {W[3, 4]:.2f})"
)
