"""
Optimal Control Pulse Shaping — Expected PnL Objective
========================================================
Parameterizes Omega(t) and Delta(t) with a PHYSICS-MOTIVATED ansatz
(6 parameters total).  The optimization target is E[PnL] — expected
portfolio return estimated via Monte Carlo sampling of the quantum
output distribution.  This is the "quantum‑useful" regime: we directly
optimise the financial metric we care about.

Design notes
------------
- Each function evaluation runs N_SHOTS_OPT = 500 circuit samples.
  A full Nelder‑Mead run costs ~190 000 simulations.

- The objective is noisy (Monte Carlo error ≈ σ_PnL / √N_shots).
  Countermeasures:
      • fatol relaxed to 0.005 (noise floor)
      • maxfev increased to 500
      • larger initial simplex (20 % perturbation per dimension)
      • evaluation cache to avoid re‑sampling visited points

Bloqade version: 0.34.0
"""

import sys
import io
import os
import base64
import numpy as np
from scipy.optimize import minimize
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- UTF-8 output & font setup -------------------------------------------
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

from bloqade import analog
from bloqade.analog.emulate.ir.state_vector import AnalogGate

# =========================================================================
# 0.  Shot budget — the central tuning knobs
# =========================================================================
N_SHOTS_SCAN  = 200      # coarse scan: cheap, rough
N_SHOTS_OPT   = 500      # Nelder‑Mead:  balance cost vs noise
N_SHOTS_FINAL = 2000     # final verification: high precision

# =========================================================================
# 1.  Problem Data
# =========================================================================
prices = np.array([
    98.50, 98.62, 98.91, 99.05, 99.12, 99.45,
    99.90, 100.02, 100.40, 100.55, 101.10
], dtype=float)

v_weights = np.array([
    0.04, 0.11, 0.08, 0.03, 0.12, 0.06,
    0.09, 0.05, 0.10, 0.13, 0.03
], dtype=float)

delta_min = 0.30
N = len(prices)
scale = 30.0
atom_coords = [(float((p - prices.min()) * scale), 0.0) for p in prices]

Omega_max   = 2.0 * np.pi * 0.8    # 5.03 rad/μs
Delta_amp   = 2.0 * np.pi * 0.8    # 5.03 rad/μs
alpha       = 2.0 * np.pi * 3.0    # weight scale
R_b_target  = delta_min * scale    # 9.0 μm
T_total     = 10.0                 # μs

# Optimal solution (ground truth from exhaustive search)
OPTIMAL_BITSTRING = (0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1)
OPTIMAL_PNL = sum(v_weights[i] for i, b in enumerate(OPTIMAL_BITSTRING) if b == 1)

# Weight array for fast dot‑product PnL computation (vectorised)
W = v_weights.astype(np.float64)

print("=" * 70)
print("Optimal Control — Expected PnL Objective (Monte Carlo)")
print("=" * 70)
print(f"N={N}, T={T_total} μs, optimal PnL*={OPTIMAL_PNL:.4f}")
print(f"Shot budget: scan={N_SHOTS_SCAN}, opt={N_SHOTS_OPT}, final={N_SHOTS_FINAL}")
print(f"Est. total cost: ~{380 * N_SHOTS_OPT // 1000}k simulations")

# =========================================================================
# 2.  Parameterisation — 6‑parameter physics ansatz
# =========================================================================
# Control vector  x = [f_omega, f_delta, f_alpha, frac_neg, frac_cross, frac_hold]
#   f_omega   ∈ [0.3, 1.5]   scale factor for Omega_max
#   f_delta   ∈ [0.3, 1.5]   scale factor for Delta_amp
#   f_alpha   ∈ [0.3, 3.0]   scale factor for alpha (site‑dependent detuning)
#   frac_neg  ∈ [0.2, 0.6]   T_neg  / T_total
#   frac_cross∈ [0.1, 0.4]   T_cross/ T_total
#   frac_hold ∈ [0.1, 0.4]   T_hold / T_total
#   remainder: T_final = T_total - T_neg - T_cross - T_hold

BOUNDS = [
    (0.3, 1.5),   # f_omega
    (0.3, 1.5),   # f_delta
    (0.3, 3.0),   # f_alpha
    (0.2, 0.6),   # frac_neg
    (0.1, 0.4),   # frac_cross
    (0.1, 0.4),   # frac_hold
]

# =========================================================================
# 3.  Core: build a Bloqade program and sample E[PnL]
# =========================================================================
def _build_program(x):
    """Build a Bloqade program from the 6‑parameter vector x.

    Returns (program, T_final).  Returns (None, 0) if the timing is invalid.
    """
    f_omega, f_delta, f_alpha, frac_neg, frac_cross, frac_hold = x

    T_neg   = frac_neg  * T_total
    T_cross = frac_cross * T_total
    T_hold  = frac_hold * T_total
    T_final = T_total - T_neg - T_cross - T_hold
    if T_final < 0.1 * T_total:
        return None, 0.0

    Oy = Omega_max * f_omega
    Dy = Delta_amp * f_delta
    Ay = alpha     * f_alpha

    durations  = [T_neg, T_cross, T_hold, T_final]
    omega_vals = [0.0, Oy, Oy, Oy, 0.0]
    delta_vals = [-Dy, 0.0, Dy, Dy, 0.0]
    s_vals     = [0.0, 0.0, 1.0, 1.0, 1.0]

    try:
        program = (
            analog.start.add_position(atom_coords)
            .rydberg.rabi.amplitude.uniform
            .piecewise_linear(durations=durations, values=omega_vals)
            .detuning.uniform
            .piecewise_linear(durations=durations, values=delta_vals)
            .detuning.location(labels=list(range(N)),
                               scales=(Ay * v_weights).tolist())
            .piecewise_linear(durations=durations, values=s_vals)
        )
        return program, T_final
    except (ValueError, RuntimeError):
        return None, 0.0


def sample_epnl(x, n_shots=N_SHOTS_OPT):
    """Run *n_shots* samples and return (E[PnL], std_error, raw_PnLs).

    E[PnL] = (1/n_shots) Σ_k PnL(bitstring_k)

    Returns (0.0, 0.0, array) on failure.
    """
    program, T_final = _build_program(x)
    if program is None:
        return 0.0, 0.0, np.array([0.0])

    try:
        emu = program.bloqade.python().hamiltonian(blockade_radius=R_b_target)[0]
        gate = AnalogGate(emu.hamiltonian)
        samples = gate.run(shots=n_shots)

        # Each sample is a string of '0'/'1' (or tuple of ints).
        # Compute PnL for every shot.
        pnls = np.array([
            sum(v_weights[i] for i, b in enumerate(s) if int(b) == 1)
            for s in samples
        ], dtype=np.float64)

        mean_pnl = float(np.mean(pnls))
        se_pnl   = float(np.std(pnls, ddof=1) / np.sqrt(n_shots)) if n_shots > 1 else 0.0
        return mean_pnl, se_pnl, pnls

    except (ValueError, RuntimeError) as e:
        return 0.0, 0.0, np.array([0.0])


def sample_epnl_fast(x, n_shots=N_SHOTS_OPT):
    """Thin wrapper: return only E[PnL] (for the optimiser)."""
    mu, _se, _ = sample_epnl(x, n_shots)
    return mu


# =========================================================================
# 4.  Baseline — evaluate the PWL (piecewise‑linear) ansatz
# =========================================================================
print("\n--- Baseline E[PnL] ---")
x_pwl = np.array([1.0, 1.0, 1.0, 0.4, 0.25, 0.25])
mu_pwl, se_pwl, pnls_pwl = sample_epnl(x_pwl, n_shots=N_SHOTS_FINAL)
print(f"PWL:  E[PnL] = {mu_pwl:.4f} ± {se_pwl:.4f}  ({mu_pwl/OPTIMAL_PNL*100:.1f}% of optimal)")

# =========================================================================
# 5.  Coarse scan — find a good starting simplex
# =========================================================================
print(f"\n--- Coarse Scan (n_shots={N_SHOTS_SCAN}) ---")

# Grid: 3×3×4 = 36 points — enough to locate a decent basin
f_omega_list = [0.6, 0.9, 1.2]
f_delta_list = [0.6, 0.9, 1.2]
f_alpha_list = [0.5, 1.2, 2.0, 2.8]

best_mu  = mu_pwl   # fall back to PWL baseline
best_x   = x_pwl.copy()
scan_hist = []       # (x, E[PnL]) for reference

scan_count = 0
t_scan_start = time.time()
for fo in f_omega_list:
    for fd in f_delta_list:
        for fa in f_alpha_list:
            x_test = np.array([fo, fd, fa, 0.4, 0.25, 0.25])
            mu, se, _ = sample_epnl(x_test, n_shots=N_SHOTS_SCAN)
            scan_count += 1
            scan_hist.append((x_test.copy(), mu))
            if mu > best_mu:
                best_mu = mu
                best_x = x_test.copy()
                print(f"  [{scan_count:2d}] f_ω={fo:.1f} f_Δ={fd:.1f} f_α={fa:.1f}  "
                      f"→ E[PnL]={mu:.4f}±{se:.4f} ✓")
            elif scan_count % 12 == 0:
                print(f"  [{scan_count:2d}] … scanning, best so far E[PnL]={best_mu:.4f}")

t_scan = time.time() - t_scan_start
print(f"Scanned {scan_count} points in {t_scan:.1f}s, "
      f"best E[PnL] = {best_mu:.4f}  ({best_mu/OPTIMAL_PNL*100:.1f}% of optimal)")
print(f"Best x = [{', '.join(f'{v:.3f}' for v in best_x)}]")

# =========================================================================
# 6.  Nelder‑Mead optimisation on E[PnL]
# =========================================================================
print(f"\n--- Nelder‑Mead on E[PnL] (n_shots={N_SHOTS_OPT}) ---")
print("Noisy objective — expect slower convergence.  Patience!")

eval_count   = [scan_count]
best_ever    = [best_mu]
best_ever_x  = [best_x.copy()]
best_ever_se = [0.0]
history       = []   # (eval#, E[PnL], SE) for convergence plot

# Cache to avoid re‑evaluating identical x (Nelder‑Mead sometimes re‑visits)
_eval_cache = {}

def objective(x):
    # Round to avoid cache misses from floating‑point noise
    key = tuple(np.round(v, 8) for v in x)
    if key in _eval_cache:
        mu, se = _eval_cache[key]
    else:
        mu, se, _ = sample_epnl(np.array(x), n_shots=N_SHOTS_OPT)
        _eval_cache[key] = (mu, se)

    eval_count[0] += 1
    history.append((eval_count[0], mu, se))

    if mu > best_ever[0]:
        best_ever[0]   = mu
        best_ever_x[0] = np.array(x).copy()
        best_ever_se[0]= se
        print(f"  eval {eval_count[0]:3d}: E[PnL]={mu:.4f}±{se:.4f}  "
              f"({mu/OPTIMAL_PNL*100:.1f}%)  "
              f"x=[{x[0]:.3f},{x[1]:.3f},{x[2]:.3f},"
              f"{x[3]:.3f},{x[4]:.3f},{x[5]:.3f}] ✓")
    elif eval_count[0] % 50 == 0:
        print(f"  eval {eval_count[0]:3d}: E[PnL]={mu:.4f}±{se:.4f},  "
              f"best={best_ever[0]:.4f} ({best_ever[0]/OPTIMAL_PNL*100:.1f}%)")

    return -mu   # minimise negative E[PnL] = maximise E[PnL]


# Initial simplex: perturb each dimension by ~20 % of its range
def initial_simplex(x0, bounds, scale=0.20):
    """Build (n+1) vertices; vertex i perturbs dimension i-1."""
    n = len(x0)
    simp = [x0.copy()]
    for i in range(n):
        lo, hi = bounds[i]
        delta = (hi - lo) * scale
        v = x0.copy()
        v[i] = np.clip(x0[i] + delta, lo, hi)
        # Avoid degenerate vertex
        if np.allclose(v, x0):
            v[i] = np.clip(x0[i] - delta, lo, hi)
        simp.append(v)
    return simp

simplex = initial_simplex(best_x, BOUNDS, scale=0.20)

t_opt_start = time.time()

result = minimize(
    objective, best_x,
    method='Nelder-Mead',
    options={
        'maxiter': 300,
        'maxfev':  500,
        'xatol':   0.015,    # relaxed — noisy objective needs looser tolerance
        'fatol':   0.005,    # stop when ΔE[PnL] < 0.005 (≈ noise floor)
        'adaptive': True,
        'initial_simplex': simplex,
    },
)

t_opt = time.time() - t_opt_start
total_evals = eval_count[0]
est_sims = scan_count * N_SHOTS_SCAN + (total_evals - scan_count) * N_SHOTS_OPT

print(f"\nOptimisation finished in {t_opt:.1f}s")
print(f"Total evaluations: {total_evals}  (scan {scan_count} + NM {total_evals - scan_count})")
print(f"Est. total simulations: ~{est_sims // 1000}k")
print(f"PWL baseline:   E[PnL] = {mu_pwl:.4f} ± {se_pwl:.4f}")
print(f"Best found:     E[PnL] = {best_ever[0]:.4f} ± {best_ever_se[0]:.4f}")
print(f"Gain:           ΔE[PnL] = {best_ever[0] - mu_pwl:.4f}  "
      f"({(best_ever[0]/mu_pwl - 1)*100:.1f}% vs PWL)")

x_opt = best_ever_x[0]

# =========================================================================
# 7.  Final high‑precision verification  (many shots)
# =========================================================================
print(f"\n--- Final Verification (n_shots={N_SHOTS_FINAL}) ---")

mu_pwl_final, se_pwl_final, pnls_pwl_final = sample_epnl(x_pwl, n_shots=N_SHOTS_FINAL)
mu_opt_final, se_opt_final, pnls_opt_final = sample_epnl(x_opt, n_shots=N_SHOTS_FINAL)

# Also collect bitstring samples for excitation plot
def sample_bitstrings(x, n_shots):
    program, _ = _build_program(x)
    if program is None:
        return np.zeros((n_shots, N), dtype=int)
    emu = program.bloqade.python().hamiltonian(blockade_radius=R_b_target)[0]
    gate = AnalogGate(emu.hamiltonian)
    samples = gate.run(shots=n_shots)
    arr = np.array([[int(b) for b in s] for s in samples], dtype=int)
    return arr

print("Collecting bitstring samples for PWL …")
bs_pwl = sample_bitstrings(x_pwl, N_SHOTS_FINAL)
print("Collecting bitstring samples for Optimised …")
bs_opt = sample_bitstrings(x_opt, N_SHOTS_FINAL)

p_opt_pwl = np.mean(np.all(bs_pwl == OPTIMAL_BITSTRING, axis=1))
p_opt_opt = np.mean(np.all(bs_opt == OPTIMAL_BITSTRING, axis=1))

print(f"\n{'':>12}  {'PWL':>12}  {'Optimised':>12}")
print(f"{'E[PnL]':>12}  {mu_pwl_final:>12.4f}  {mu_opt_final:>12.4f}")
print(f"{'SE':>12}  {se_pwl_final:>12.4f}  {se_opt_final:>12.4f}")
print(f"{'P(optimal)':>12}  {p_opt_pwl:>11.2%}  {p_opt_opt:>11.2%}")
print(f"{'vs optimal':>12}  {mu_pwl_final/OPTIMAL_PNL:>11.1%}  {mu_opt_final/OPTIMAL_PNL:>11.1%}")

# -----------------------------------------------------------------------
# 7b.  Bitstring frequency table — what the quantum sampler actually outputs
# -----------------------------------------------------------------------
def bitstring_table(bs, label, top_n=20):
    """Print top-N bitstrings by occurrence count with their PnL."""
    # Convert each row (0/1 array) → string key + PnL
    records = {}
    for row in bs:
        key = ''.join(str(b) for b in row)
        if key not in records:
            records[key] = {'count': 0, 'pnl': sum(v_weights[i] for i, b in enumerate(row) if b == 1)}
        records[key]['count'] += 1

    total = len(bs)
    sorted_recs = sorted(records.items(), key=lambda kv: -kv[1]['count'])

    sep = '-' * 62
    print(f"\n{sep}")
    print(f"Bitstring frequency — {label}  (top {top_n} of {len(records)} unique)")
    print(sep)
    print(f"{'bitstring':>12s}  {'PnL':>8s}  {'count':>6s}  {'freq':>8s}  {'bar'}")
    print(sep)

    for key, rec in sorted_recs[:top_n]:
        freq = rec['count'] / total
        bar_len = int(freq * 50)
        marker = ' ★' if key == ''.join(str(b) for b in OPTIMAL_BITSTRING) else ''
        print(f"{key:>12s}  {rec['pnl']:8.4f}  {rec['count']:6d}  {freq:7.2%}  "
              f"{'█' * bar_len}{marker}")

    if len(sorted_recs) > top_n:
        other_count = sum(r['count'] for _, r in sorted_recs[top_n:])
        print(f"{'… others':>12s}  {'—':>8s}  {other_count:6d}  {other_count/total:7.2%}")
    print(sep)

bitstring_table(bs_pwl, "PWL baseline", top_n=20)
bitstring_table(bs_opt, "Optimised", top_n=20)

# =========================================================================
# 8.  Visualisation
# =========================================================================
print("\nGenerating plots …")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Optimal Control — Expected PnL Objective', fontsize=14, fontweight='bold')

t_fine = np.linspace(0, T_total, 200)

# --- helper: evaluate waveforms for plotting ---
def eval_waveform(x, t):
    f_omega, f_delta, f_alpha, frac_neg, frac_cross, frac_hold = x
    T_n = frac_neg  * T_total
    T_c = frac_cross * T_total
    T_h = frac_hold  * T_total
    T_f = T_total - T_n - T_c - T_h
    Oy = Omega_max * f_omega
    Dy = Delta_amp * f_delta
    t_cum  = np.array([0, T_n, T_n+T_c, T_n+T_c+T_h, T_n+T_c+T_h+T_f])
    o_vals = np.array([0, Oy, Oy, Oy, 0])
    d_vals = np.array([-Dy, 0, Dy, Dy, 0])
    omega = np.interp(t, t_cum, o_vals)
    delta = np.interp(t, t_cum, d_vals)
    return omega, delta

omega_pwl, delta_pwl = eval_waveform(x_pwl, t_fine)
omega_opt, delta_opt = eval_waveform(x_opt, t_fine)

# (a) Pulses
ax = axes[0, 0]
ax.plot(t_fine, omega_pwl/(2*np.pi), 'b--', lw=1.5, alpha=0.6, label='Ω PWL')
ax.plot(t_fine, omega_opt/(2*np.pi), 'b-', lw=2.5, label='Ω Opt')
ax.plot(t_fine, delta_pwl/(2*np.pi), 'r--', lw=1.5, alpha=0.6, label='Δ PWL')
ax.plot(t_fine, delta_opt/(2*np.pi), 'r-', lw=2.5, label='Δ Opt')
ax.axhline(0, color='gray', lw=0.5, ls='--')
ax.set_xlabel('Time (μs)', fontsize=10)
ax.set_ylabel('Freq (MHz)', fontsize=10)
ax.set_title(f'Pulses (Ω×{x_opt[0]:.2f}, Δ×{x_opt[1]:.2f}, α×{x_opt[2]:.2f})', fontsize=10)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# (b) Parameter bar chart
ax = axes[0, 1]
labels = ['f_omega', 'f_delta', 'f_alpha', 'T_neg/T', 'T_cross/T', 'T_hold/T']
x_pos = np.arange(len(labels))
ax.bar(x_pos - 0.2, x_pwl, 0.35, color='#FF9800', edgecolor='#333', label='PWL')
ax.bar(x_pos + 0.2, x_opt, 0.35, color='#4CAF50', edgecolor='#333', label='Optimised')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Value', fontsize=10)
ax.set_title('Parameter Comparison', fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# (c) PnL distribution  (core panel — this IS the objective)
ax = axes[0, 2]
bins = np.linspace(0, OPTIMAL_PNL * 1.1, 35)
ax.hist(pnls_pwl_final, bins=bins, alpha=0.5, color='#FF9800', edgecolor='#E65100',
        label=f'PWL  E[PnL]={mu_pwl_final:.3f}',
        weights=np.ones(N_SHOTS_FINAL)/N_SHOTS_FINAL)
ax.hist(pnls_opt_final, bins=bins, alpha=0.5, color='#4CAF50', edgecolor='#1B5E20',
        label=f'Opt  E[PnL]={mu_opt_final:.3f}',
        weights=np.ones(N_SHOTS_FINAL)/N_SHOTS_FINAL)
ax.axvline(OPTIMAL_PNL, color='#E53935', lw=2, ls='--', label=f'Opt*={OPTIMAL_PNL:.3f}')
ax.axvline(mu_pwl_final, color='#FF9800', lw=1.5, ls=':', alpha=0.7)
ax.axvline(mu_opt_final, color='#4CAF50', lw=1.5, ls=':', alpha=0.7)
ax.set_xlabel('Total PnL', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title(f'PnL Distribution (n={N_SHOTS_FINAL})', fontsize=10)
ax.legend(fontsize=7)

# (d) Convergence trace
ax = axes[1, 0]
if history:
    evals, mus, ses = zip(*history)
    evals = np.array(evals); mus = np.array(mus); ses = np.array(ses)
    # Running best
    running_best = np.maximum.accumulate(mus)
    ax.fill_between(evals, mus - ses, mus + ses, alpha=0.2, color='#2196F3')
    ax.plot(evals, mus, 'o', ms=2, alpha=0.3, color='#2196F3', label='eval')
    ax.plot(evals, running_best, '-', lw=2, color='#E53935', label='running best')
    ax.axhline(OPTIMAL_PNL, color='#4CAF50', lw=1.5, ls='--', label=f'Opt*={OPTIMAL_PNL:.3f}')
    ax.axhline(mu_pwl, color='#FF9800', lw=1, ls=':', label=f'PWL={mu_pwl:.3f}')
ax.set_xlabel('Evaluation', fontsize=10)
ax.set_ylabel('E[PnL]', fontsize=10)
ax.set_title('Convergence (noisy objective)', fontsize=10)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# (e) Excitation probabilities
ax = axes[1, 1]
x_idx = np.arange(N)
bar_w = 0.35
p_ex_pwl = np.mean(bs_pwl, axis=0)
p_ex_opt = np.mean(bs_opt, axis=0)
ax.bar(x_idx - bar_w/2, p_ex_pwl, bar_w, color='#FF9800', edgecolor='#333', label='PWL')
ax.bar(x_idx + bar_w/2, p_ex_opt, bar_w, color='#4CAF50', edgecolor='#333', label='Optimised')
for i in range(N):
    if OPTIMAL_BITSTRING[i] == 1:
        ax.annotate('★', (i, 1.05), ha='center', fontsize=12, color='#E53935')
ax.axhline(0.5, color='gray', lw=1, ls='--')
ax.set_xticks(x_idx)
ax.set_xticklabels([f'{p:.2f}' for p in prices], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('P(|r⟩)', fontsize=10)
ax.set_title('Excitation Probabilities', fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# (f) E[PnL] bar comparison
ax = axes[1, 2]
categories = ['PWL', 'Optimised', 'Optimal*']
means  = [mu_pwl_final, mu_opt_final, OPTIMAL_PNL]
errors = [se_pwl_final, se_opt_final, 0]
colors = ['#FF9800', '#4CAF50', '#E53935']
bars = ax.bar(categories, means, yerr=errors, color=colors, edgecolor='#333',
              capsize=8, width=0.5)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('E[PnL]', fontsize=10)
ax.set_title('Expected Portfolio Return', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = 'optimal_control_pulse_epnl.png'
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")
plt.close(fig)

# =========================================================================
# 8b.  HTML Report — self‑contained, embeds PNG + all data tables
# =========================================================================
print("Generating HTML report …")

with open(plot_path, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

# --- helper: build bitstring frequency HTML rows ---
def bitstring_html_rows(bs, top_n=20):
    records = {}
    for row in bs:
        key = ''.join(str(b) for b in row)
        if key not in records:
            records[key] = {'count': 0, 'pnl': sum(v_weights[i] for i, b in enumerate(row) if b == 1)}
        records[key]['count'] += 1
    total = len(bs)
    sorted_recs = sorted(records.items(), key=lambda kv: -kv[1]['count'])
    rows = []
    for key, rec in sorted_recs[:top_n]:
        freq = rec['count'] / total
        is_opt = key == ''.join(str(b) for b in OPTIMAL_BITSTRING)
        cls = ' class="optimal"' if is_opt else ''
        bar_w = int(freq * 200)
        rows.append(
            f'<tr{cls}><td><code>{key}</code></td>'
            f'<td>{rec["pnl"]:.4f}</td><td>{rec["count"]}</td>'
            f'<td>{freq:.2%}</td>'
            f'<td><div class="bar"><div class="fill" style="width:{bar_w}px"></div></div></td></tr>'
        )
    if len(sorted_recs) > top_n:
        other_count = sum(r['count'] for _, r in sorted_recs[top_n:])
        rows.append(
            f'<tr><td colspan="2"><em>… {len(sorted_recs) - top_n} others</em></td>'
            f'<td>{other_count}</td><td>{other_count/total:.2%}</td><td></td></tr>'
        )
    return ''.join(rows), len(sorted_recs)

bs_rows_pwl, uniq_pwl = bitstring_html_rows(bs_pwl, top_n=20)
bs_rows_opt, uniq_opt = bitstring_html_rows(bs_opt, top_n=20)

# Parameter rows
param_rows = ''.join(
    f'<tr><td>{name}</td><td>{pwl_v:.3f}</td><td>{opt_v:.3f}</td></tr>'
    for name, pwl_v, opt_v in zip(labels, x_pwl, x_opt)
)

# Convergence data (JSON for inline use)
conv_json = []
if history:
    for eid, mu, se in history:
        conv_json.append({'eval': eid, 'mu': round(mu, 5), 'se': round(se, 5)})

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Optimal Control — E[PnL] Report</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #f5f6fa; color: #2d3436; line-height: 1.6; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; }}
h2 {{ font-size: 1.15rem; margin: 28px 0 12px; padding-bottom: 6px; border-bottom: 2px solid #4CAF50; }}
.header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #fff; padding: 28px 0 24px; margin-bottom: 20px; }}
.header .sub {{ opacity: .75; font-size: .9rem; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 24px; }}
.card {{ background: #fff; border-radius: 8px; padding: 16px 18px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
.card .label {{ font-size: .78rem; text-transform: uppercase; color: #636e72; letter-spacing: .5px; }}
.card .value {{ font-size: 1.5rem; font-weight: 700; color: #2d3436; }}
.card .sub {{ font-size: .78rem; color: #b2bec3; margin-top: 2px; }}
.card.green  .value {{ color: #27ae60; }}
.card.orange .value {{ color: #e67e22; }}
.card.red    .value {{ color: #e74c3c; }}
.plot-box {{ background: #fff; border-radius: 8px; padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.06); margin-bottom: 24px; }}
.plot-box img {{ width: 100%; border-radius: 4px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.06); margin-bottom: 20px; }}
th, td {{ padding: 8px 14px; text-align: left; font-size: .88rem; }}
th {{ background: #2d3436; color: #fff; font-weight: 600; text-transform: uppercase; letter-spacing: .4px; font-size: .76rem; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
tr.optimal {{ background: #e8f5e9 !important; font-weight: 600; }}
tr.optimal td:first-child::after {{ content: " ★"; color: #e53935; }}
td code {{ background: #eee; padding: 1px 6px; border-radius: 3px; font-size: .82rem; }}
.bar {{ width: 200px; height: 14px; background: #ecf0f1; border-radius: 7px; overflow: hidden; display: inline-block; }}
.bar .fill {{ height: 100%; background: linear-gradient(90deg, #4CAF50, #66BB6A); border-radius: 7px; }}
.footer {{ text-align: center; font-size: .78rem; color: #b2bec3; margin-top: 30px; padding: 16px 0; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 700px) {{ .cols {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="header">
  <div class="container">
    <h1>Optimal Control &mdash; Expected PnL Objective</h1>
    <div class="sub">Physics ansatz (6 params) + Nelder‑Mead &middot; Bloqade {sys.modules.get('bloqade', '').__version__ if hasattr(sys.modules.get('bloqade', ''), '__version__') else '0.34.0'} &middot; {time.strftime('%Y-%m-%d %H:%M')}</div>
  </div>
</div>
<div class="container">

<!-- Key Metrics -->
<div class="cards">
  <div class="card green">
    <div class="label">Optimised E[PnL]</div>
    <div class="value">{mu_opt_final:.4f}</div>
    <div class="sub">± {se_opt_final:.4f} &middot; {mu_opt_final/OPTIMAL_PNL*100:.1f}% of optimal</div>
  </div>
  <div class="card orange">
    <div class="label">PWL Baseline E[PnL]</div>
    <div class="value">{mu_pwl_final:.4f}</div>
    <div class="sub">± {se_pwl_final:.4f} &middot; {mu_pwl_final/OPTIMAL_PNL*100:.1f}% of optimal</div>
  </div>
  <div class="card">
    <div class="label">Δ vs Baseline</div>
    <div class="value">{mu_opt_final - mu_pwl_final:+.4f}</div>
    <div class="sub">{(mu_opt_final/mu_pwl_final - 1)*100:+.1f}% improvement</div>
  </div>
  <div class="card red">
    <div class="label">Optimal PnL*</div>
    <div class="value">{OPTIMAL_PNL:.4f}</div>
    <div class="sub">Ground truth (exhaustive search)</div>
  </div>
  <div class="card">
    <div class="label">Total Simulations</div>
    <div class="value">~{est_sims // 1000}k</div>
    <div class="sub">{total_evals} evals &middot; {t_scan + t_opt:.0f}s wall time</div>
  </div>
  <div class="card">
    <div class="label">P(optimal bitstring)</div>
    <div class="value">{p_opt_opt:.2%}</div>
    <div class="sub">PWL: {p_opt_pwl:.2%}</div>
  </div>
</div>

<!-- Plot -->
<div class="plot-box">
  <img src="data:image/png;base64,{img_b64}" alt="Analysis Plots">
</div>

<!-- Parameters -->
<h2>Pulse Parameters</h2>
<table>
  <thead><tr><th>Parameter</th><th>PWL</th><th>Optimised</th></tr></thead>
  <tbody>{param_rows}</tbody>
</table>

<!-- Bitstring Tables -->
<h2>Bitstring Frequency — Optimised  (top 20 of {uniq_opt} unique)</h2>
<table>
  <thead><tr><th>Bitstring</th><th>PnL</th><th>Count</th><th>Frequency</th><th>Distribution</th></tr></thead>
  <tbody>{bs_rows_opt}</tbody>
</table>

<h2>Bitstring Frequency — PWL Baseline  (top 20 of {uniq_pwl} unique)</h2>
<table>
  <thead><tr><th>Bitstring</th><th>PnL</th><th>Count</th><th>Frequency</th><th>Distribution</th></tr></thead>
  <tbody>{bs_rows_pwl}</tbody>
</table>

<div class="footer">
  Shot budget: scan={N_SHOTS_SCAN} | opt={N_SHOTS_OPT} | final={N_SHOTS_FINAL} &middot;
  Nelder‑Mead: maxfev=500, fatol=0.005 &middot;
  Optimal bitstring = {OPTIMAL_BITSTRING}
</div>

</div>
</body>
</html>'''

html_path = 'optimal_control_pulse_epnl.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"HTML report saved to: {html_path}")

# =========================================================================
# 9.  Summary
# =========================================================================
sep = '=' * 70
print(f"\n{sep}\nSUMMARY — Expected PnL Objective\n{sep}")
print(f"Method:       Physics ansatz (6 params) + Nelder‑Mead on E[PnL]")
print(f"Shot budget:  scan={N_SHOTS_SCAN} | opt={N_SHOTS_OPT} | final={N_SHOTS_FINAL}")
print(f"Total evals:  {total_evals}  (~{est_sims // 1000}k simulations)")
print(f"Wall time:    scan {t_scan:.0f}s + NM {t_opt:.0f}s = {t_scan+t_opt:.0f}s")
print()
print(f"{'':>16}  {'PWL':>12}  {'Optimised':>12}  {'Δ':>10}")
print(f"{'E[PnL]':>16}  {mu_pwl_final:>12.4f}  {mu_opt_final:>12.4f}  "
      f"{mu_opt_final-mu_pwl_final:>+10.4f}")
print(f"{'P(optimal)':>16}  {p_opt_pwl:>11.2%}  {p_opt_opt:>11.2%}")
print(f"{'vs optimal*':>16}  {mu_pwl_final/OPTIMAL_PNL:>11.1%}  "
      f"{mu_opt_final/OPTIMAL_PNL:>11.1%}")
print()
print("Parameters (PWL → Optimised):")
for name, pwl_v, opt_v in zip(labels, x_pwl, x_opt):
    print(f"  {name:>12s}: {pwl_v:.3f} → {opt_v:.3f}")
print()
print("Note: E[PnL] is the expectation over the quantum output distribution.")
print("A higher E[PnL] means the quantum sampler produces better portfolios")
print("ON AVERAGE — this is what matters for a quantum‑enhanced trading strategy.")
print(sep)
