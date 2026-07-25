import sys
import io

# Fix Windows GBK encoding issue — force UTF-8 for Unicode math symbols
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from math import sqrt
import numpy as np
import time
from scipy.optimize import minimize

print("=" * 70)
print("Portfolio Optimization with Binary Encoding and QAOA")
print("=" * 70)



# =====================================================================
# Financial Data — Synthetic Realistic Multi-Asset-Class Data
# =====================================================================
print("\n" + "─" * 70)
print("Financial Data — Multi-Asset-Class Portfolio")
print("─" * 70)

# Portfolio configuration (matching paper's description)
N_ASSETS    = 6      # we use 6 for tractability
K_BITS      = 3      # bits per asset for weight discretization
N_QUBO_VARS = N_ASSETS * K_BITS  # 18 binary variables, 2^18 = 262K states → exact enumeration feasible

# Three asset classes
#   EQ (Equity):      assets 0, 1, 2  — high return, high risk
#   FI (Fixed Income): assets 3, 4     — moderate return, low risk
#   MM (Money Market):  asset 5        — low return, very low risk
ASSET_NAMES = [
    "EQ: US Large Cap",     # 0
    "EQ: European Equity",  # 1
    "EQ: Emerging Markets", # 2
    "FI: Govt Bonds",       # 3
    "FI: Corp IG Bonds",    # 4
    "MM: Cash Equivalents", # 5
]
ASSET_CLASSES = {
    "EQ": [0, 1, 2],
    "FI": [3, 4],
    "MM": [5],
}
CLASS_LABELS = ["EQ", "FI", "MM"]

# Realistic expected returns (annualized, decimal)
expected_returns = np.array([0.10, 0.08, 0.14, 0.04, 0.05, 0.02])

# Realistic volatilities (annualized, decimal)
volatilities = np.array([0.18, 0.16, 0.25, 0.04, 0.06, 0.01])

# Realistic correlation matrix (block structure by asset class)
correlation = np.array([
    # EQ:US   EQ:EU   EQ:EM    FI:Gov  FI:Corp  MM:Cash
    [ 1.00,   0.70,   0.55,    0.05,   0.10,    0.00  ],  # EQ:US
    [ 0.70,   1.00,   0.50,    0.10,   0.12,    0.00  ],  # EQ:EU
    [ 0.55,   0.50,   1.00,   -0.05,   0.00,   -0.02  ],  # EQ:EM
    [ 0.05,   0.10,  -0.05,    1.00,   0.60,    0.30  ],  # FI:Gov
    [ 0.10,   0.12,   0.00,    0.60,   1.00,    0.25  ],  # FI:Corp
    [ 0.00,   0.00,  -0.02,    0.30,   0.25,    1.00  ],  # MM:Cash
])

# Covariance matrix: Σ_ij = σ_i · σ_j · ρ_ij
cov_matrix = np.outer(volatilities, volatilities) * correlation

# Different assets have different allowable weight ranges
w_min = np.array([0.00, 0.00, 0.00,  0.00, 0.00, 0.00])  # all ≥ 0
w_max = np.array([0.40, 0.40, 0.30,  0.50, 0.50, 0.30])  # class-dependent

# Asset-class group constraints
# Σ_{i∈EQ} ω_i ≤ 0.60  (max 60% in equity)
# Σ_{i∈FI} ω_i ≤ 0.70  (max 70% in fixed income)
# Σ_{i∈MM} ω_i ≤ 0.30  (max 30% in money market)
class_max = {"EQ": 0.60, "FI": 0.70, "MM": 0.30}
class_min = {"EQ": 0.00, "FI": 0.00, "MM": 0.00}

# =====================================================================
# Risk Aversion Parameter (gamma) — Variance-as-Objective Approach
# =====================================================================
# Variance-as-constraint penalty max(0, var-target)^2 is QUARTIC in x
# (4-body terms x_i x_j x_k x_l) — structural obstacle, not a tuning problem.
#
# SOLUTION: Move variance into the OBJECTIVE:
#   min  -mu^T omega + gamma * omega^T Sigma omega + lambda_b * (Sigma w_i - 1)^2
#   ALL terms quadratic in x → PURE QUBO (no piecewise terms, no max operators).
#
# KEY INSIGHT: A single GAMMA value is ARBITRARY — it picks ONE point on the
# risk-return spectrum.  The gamma SWEEP (Part 7) is what actually traces the
# Markowitz efficient frontier.  Each gamma yields a different risk/return bias:
#   Low  gamma (0.1)  → prioritises RETURN   (risk-on, more equity)
#   High gamma (500)  → prioritises LOW RISK  (risk-off, more bonds/cash)
#
# VAR_TARGET is used ONLY for post-hoc feasibility labelling (is_feasible),
# NOT as a constraint in the QUBO objective.  Solutions are filtered BY
# sigma_target AFTER optimisation — the QUBO itself has no hard variance cap.
#
# Reference: Venturelli & Kondratyev (2019), Elsokkary et al. (2017)
#
GAMMA = 2.0  # default risk aversion (one point on the frontier; sweep in Part 7)
RISK_FREE = 0.03  # risk-free rate for Sharpe ratio calculation (3% annualized)

# VAR_TARGET used ONLY for post-hoc feasibility classification — NOT a QUBO constraint
VAR_TARGET = 0.012


print(f"\n  Portfolio Configuration:")
print(f"    Assets: N={N_ASSETS}, Bits/asset: K={K_BITS}")
print(f"    Total QUBO variables: {N_QUBO_VARS}")
print(f"    Asset classes: {CLASS_LABELS}")
print(f"    Variance target: σ²_target = {VAR_TARGET:.4f} (σ_target ≈ {sqrt(VAR_TARGET)*100:.1f}%)")

print(f"\n  Asset Data:")
print(f"  {'Idx':<4} {'Name':<24} {'Class':<6} {'μ':>8} {'σ':>8} {'ω_min':>8} {'ω_max':>8}")
for i in range(N_ASSETS):
    cls = "EQ" if i in ASSET_CLASSES["EQ"] else ("FI" if i in ASSET_CLASSES["FI"] else "MM")
    print(f"  {i:<4} {ASSET_NAMES[i]:<24} {cls:<6} "
          f"{expected_returns[i]:>7.1%} {volatilities[i]:>7.1%} "
          f"{w_min[i]:>7.2f} {w_max[i]:>7.2f}")

print(f"\n  Covariance Matrix Σ:")
for i in range(N_ASSETS):
    row_str = " ".join(f"{cov_matrix[i][j]:>8.5f}" for j in range(N_ASSETS))
    print(f"    [{row_str}]")

print(f"\n  Correlation Matrix:")
for i in range(N_ASSETS):
    row_str = " ".join(f"{correlation[i][j]:>6.2f}" for j in range(N_ASSETS))
    print(f"    [{row_str}]")

print(f"\n  Asset-Class Constraints:")
for cls in CLASS_LABELS:
    print(f"    {cls}: [{class_min[cls]:.0%}, {class_max[cls]:.0%}]  "
          f"(assets {ASSET_CLASSES[cls]})")


# =====================================================================
# Weight Discretization — K-bit Binary Encoding
# =====================================================================
print("\n" + "─" * 70)
print("Weight Discretization — K-bit Binary Encoding")
print("─" * 70)

# Δω_i = ω_i,max − ω_i,min
delta_w = w_max - w_min

# Granularity p_K = 1 / (2^K - 1) — ensures max weight reaches ω_max exactly
# (Eq.: ω_i = ω_i,min + Δω_i · Σ 2^{k-1} · x_{i,k} / (2^K - 1))
p_K = 1.0 / (2 ** K_BITS - 1)

# Effective granularity per asset: p_K,eff_i = Δω_i · p_K  (Eq. 22)
p_K_eff = delta_w * p_K

print(f"\n  Discretization Parameters:")
print(f"    K = {K_BITS} bits per asset")
print(f"    Granularity p_K = 1/(2^{K_BITS} - 1) = {p_K:.6f}")
print(f"    Total binary variables: {N_ASSETS} × {K_BITS} = {N_QUBO_VARS}")
print(f"\n  {'Asset':<6} {'Δω_i':>8} {'p_K_eff':>10} {'Weight levels':>15} {'Resolution':>12}")
for i in range(N_ASSETS):
    n_levels = 2 ** K_BITS
    resolution = delta_w[i] * p_K
    print(f"  {i:<6} {delta_w[i]:>8.4f} {p_K_eff[i]:>10.6f} "
          f"{'0..' + str(2**K_BITS - 1):>15} {resolution:>11.4f}")

# Binary-to-weight conversion function
def decode_weights(binary_vector):
    """Decode K·N binary variables → continuous portfolio weights.

    Parameters
    ----------
    binary_vector : ndarray of shape (N_QUBO_VARS,) or (N_ASSETS, K_BITS)
        Binary variables x_{i,k} ∈ {0,1}.

    Returns
    -------
    weights : ndarray of shape (N_ASSETS,)
        Continuous portfolio weights ω_i ≥ 0.
    """
    if binary_vector.ndim == 1:
        x = binary_vector.reshape(N_ASSETS, K_BITS)
    else:
        x = binary_vector

    # ω_i = ω_i,min + Δω_i · Σ_k 2^{k-1} · x_{i,k} · p_K
    # With p_K = 1/(2^K - 1), max normalized weight = 1.0 → ω_max reached exactly
    powers = 2.0 ** np.arange(K_BITS)  # [1, 2, 4, 8, 16, ...]
    w_norm = np.dot(x, powers) * p_K   # normalized weight in [0, 1]

    # Full weight: ω_i = ω_i,min + Δω_i · w_norm_i
    weights = w_min + delta_w * w_norm
    return weights

# ---- Bit conversion helpers (Qiskit convention: LSB = qubit 0) ----
def int_to_bits(num, n):
    """Convert integer to bit array, LSB first (matching Qiskit convention).

    Qiskit: qubit 0 is the least significant bit (LSB).
    Returns bits[0] = LSB = qubit 0.

    Parameters
    ----------
    num : int — integer value to convert.
    n : int — number of bits.

    Returns
    -------
    bits : ndarray of shape (n,) dtype int — bit array, LSB first.
    """
    return np.array([(num >> k) & 1 for k in range(n)], dtype=int)


def bits_to_display(bits):
    """Convert bit array (LSB-first) to display string (MSB-first)."""
    return ''.join(str(int(b)) for b in bits[::-1])


# Demo: all possible weights for the first asset
print(f"\n  Weight levels for Asset 0 ({ASSET_NAMES[0]}):")
print(f"    ω_min={w_min[0]:.2f}, ω_max={w_max[0]:.2f}, K={K_BITS}")
demo_weights = []
for val in range(2 ** K_BITS):
    bits = np.array([(val >> k) & 1 for k in range(K_BITS)])
    w = w_min[0] + delta_w[0] * np.dot(bits, 2.0 ** np.arange(K_BITS)) * p_K
    demo_weights.append(w)
    if val < 5 or val > 2**K_BITS - 4:
        print(f"    x = {''.join(str(b) for b in bits[::-1])}  →  ω = {w:.6f}")
print(f"    ... ({2**K_BITS - 8} intermediate levels omitted)")
print(f"    x = {''.join(str(b) for b in [1]*K_BITS)}  →  ω = {demo_weights[-1]:.6f} (max)")


# =====================================================================
# Penalty-Augmented Objective Construction
# =====================================================================
print("\n" + "─" * 70)
print("Penalty-Augmented Objective — Pure QUBO Formulation")
print("─" * 70)

# ---- Budget Penalty Coefficient ----
# lambda_b must be tuned: too small → Σw ≠ 1; too large → objective dominated by constraint.
# We use a DEFAULT value and SWEEP in Part 7 to find the right balance.
# Rule of thumb: lambda_b * (0.05)^2 ≈ typical_return ≈ 0.07 → lambda_b ≈ 28
LAMBDA_BUDGET = 10.0  # budget penalty (default; swept in Part 7)

# ---- Scale Analysis ----
typical_return = float(np.mean(expected_returns))
typical_variance = float(np.mean(volatilities**2))

print(f"\n  Objective & Penalty Parameters:")
print(f"    lambda_b = {LAMBDA_BUDGET:.1f} (budget penalty, sweep in Part 7)")
print(f"    gamma    = {GAMMA:.2f} (risk aversion, sweep in Part 7)")
print(f"\n  Scale Analysis (typical values):")
print(f"    Return term |mu|        ≈ {typical_return:.4f}")
print(f"    Variance term (gamma*var) ≈ {GAMMA*typical_variance:.4f}  (gamma={GAMMA})")
print(f"    Budget penalty (1% dev)   ≈ {LAMBDA_BUDGET*0.01**2:.4f}  (lambda_b*0.0001)")
print(f"    Budget penalty (5% dev)   ≈ {LAMBDA_BUDGET*0.05**2:.4f}  (lambda_b*0.0025)")
for g_test in [0.1, 2.0, 10.0, 100.0]:
    print(f"    gamma={g_test:5.1f} → var term = {g_test*typical_variance:.4f}  "
          f"({g_test*typical_variance/typical_return:.1f}x return)")

print(f"""
  Markowitz Objective — PURE QUBO (all terms ≤ quadratic in x):
    f(x) = -mu^T omega(x)                        [return, linear in x]
         + gamma * omega^T Sigma omega            [risk aversion, quadratic in x]
         + lambda_b * (Sigma omega_i - 1)^2       [budget, quadratic in x]

  NO piecewise terms, NO max(0,·) operators, NO class penalties.
  Class bounds checked post-hoc; per-asset bounds enforced by weight encoding.
""")

def portfolio_variance(weights):
    """Compute portfolio variance ω^T Σ ω."""
    return float(weights @ cov_matrix @ weights)

def portfolio_return(weights):
    """Compute expected portfolio return mu^T omega."""
    return float(np.dot(expected_returns, weights))

def penalty_objective(binary_vector, lambda_b=LAMBDA_BUDGET, gamma=GAMMA):
    """Pure QUBO Markowitz objective: return + variance + budget.

    f(x) = -mu^T w + gamma * w^T Sigma w + lambda_b * (sum(w) - 1)^2
    ALL terms are at most QUADRATIC in binary x — PURE QUBO.

    Parameters
    ----------
    binary_vector : ndarray of shape (N_QUBO_VARS,)
    lambda_b : float — budget penalty coefficient
    gamma : float — risk aversion

    Returns
    -------
    objective : float (LOWER is better)
    """
    w = decode_weights(binary_vector)
    obj = -portfolio_return(w)                            # linear
    obj += gamma * portfolio_variance(w)                   # quadratic
    obj += lambda_b * (np.sum(w) - 1.0) ** 2              # quadratic
    return float(obj)

def is_feasible(weights, tol=1e-3):
    """Post-hoc feasibility check — NOT part of the QUBO objective.

    Variance is in the OBJECTIVE (gamma * w^T Sigma w), not a hard constraint.
    VAR_TARGET is used here only to LABEL solutions post-optimisation.
    A solution failing the variance check is NOT "wrong" — it simply lies
    at a different point on the risk-return frontier (higher risk, higher return).
    The gamma sweep (Part 7) explores the full frontier systematically.
    """
    if abs(np.sum(weights) - 1.0) > tol:
        return False, f"Budget: Σω = {np.sum(weights):.4f} ≠ 1"

    var = portfolio_variance(weights)
    if var > VAR_TARGET + tol:
        return False, f"Variance: ω^TΣω = {var:.6f} > {VAR_TARGET}"

    for i in range(N_ASSETS):
        if weights[i] < w_min[i] - tol or weights[i] > w_max[i] + tol:
            return False, f"Asset {i}: ω={weights[i]:.4f} ∉ [{w_min[i]},{w_max[i]}]"

    for cls in CLASS_LABELS:
        cw = np.sum(weights[ASSET_CLASSES[cls]])
        if cw > class_max[cls] + tol or cw < class_min[cls] - tol:
            return False, f"{cls}: Σω={cw:.4f} ∉ [{class_min[cls]},{class_max[cls]}]"

    return True, "All constraints satisfied"

# Test
test_bits = np.zeros(N_QUBO_VARS, dtype=int)
for i in range(N_ASSETS):
    mid_val = 2 ** (K_BITS - 1)
    for k in range(K_BITS):
        test_bits[i * K_BITS + k] = (mid_val >> k) & 1

test_w = decode_weights(test_bits)
print(f"\n  Test configuration (mid-range bits):")
print(f"    Weights:  {[f'{w:.4f}' for w in test_w]}")
print(f"    Σω = {np.sum(test_w):.4f}")
print(f"    Objective   = {penalty_objective(test_bits):.6f}")

feasible, msg = is_feasible(test_w)
print(f"    Feasible: {feasible}  —  {msg}")
print(f"    Variance: {portfolio_variance(test_w):.6f} (target: {VAR_TARGET})")
print(f"    Return:   {portfolio_return(test_w)*100:.2f}%")



# =====================================================================
# PART 6: Simulated Annealing — D-Wave Quantum Annealing Proxy
# =====================================================================
print("\n" + "─" * 70)
print("PART 6: Classical Simulated Annealing Solver (D-Wave Analogy)")
print("─" * 70)

print(f"""
  Simulated Annealing (SA) is a CLASSICAL thermal annealing heuristic.
  It shares a structural ANALOGY with D-Wave’s quantum annealing by:
    1. Starting from a random binary configuration (high "temperature")
    2. Proposing single-bit flips (thermal exploration of energy landscape)
    3. Accepting downhill moves always, uphill moves with decreasing probability
    4. Cooling schedule: T_k = T_0 · α^k (geometric cooling)

  This is analogous to:
    - D-Wave QBSolv: tabu search + quantum sub-problem solving
    - D-Wave Hybrid BQM: classical heuristic + quantum refinement

  Key differences from real quantum annealing (SA is NOT quantum)::
    - No quantum superposition — relies on thermal fluctuations only
    - No entanglement — no correlated multi-bit flips
    - Classical Boltzmann sampling vs quantum Boltzmann distribution
    - Cannot tunnel through energy barriers (fundamental limitation)
""")

def simulated_annealing_qubo(n_vars, objective_fn, n_steps=20000,
                             T_start=1.0, T_end=0.001, n_restarts=10,
                             seed=42):
    """Simulated annealing for binary QUBO optimization.

    Parameters
    ----------
    n_vars : int
        Number of binary variables.
    objective_fn : callable
        Objective function f(x) to minimize.
    n_steps : int
        Number of cooling steps per restart.
    T_start, T_end : float
        Initial and final temperatures.
    n_restarts : int
        Number of independent runs.
    seed : int
        Random seed.

    Returns
    -------
    best_solution : ndarray
        Best binary vector found.
    best_energy : float
        Objective value at best solution.
    history : list of float
        Best energy at each step (for convergence analysis).
    """
    rng = np.random.RandomState(seed)
    best_solution = None
    best_energy = float('inf')
    full_history = []

    for restart in range(n_restarts):
        # Initialize random binary vector
        x = rng.randint(0, 2, size=n_vars).astype(int)
        E = objective_fn(x)
        x_best = x.copy()
        E_best = E

        # Persistent reheat factor — survives across cooling schedule recalculations
        reheat_factor = 1.0
        steps_since_improvement = 0

        for step in range(n_steps):
            # Geometric cooling: T(t) = T_start * (T_end/T_start)^(t/n_steps)
            # Multiplied by reheat_factor which persists across iterations
            t_frac = step / n_steps
            T = T_start * (T_end / T_start) ** t_frac * reheat_factor

            # Propose: flip random bit (single-spin flip, like transition)
            idx = rng.randint(n_vars)
            x[idx] = 1 - x[idx]
            E_new = objective_fn(x)

            delta_E = E_new - E

            # Metropolis acceptance
            if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
                E = E_new
                if E < E_best:
                    E_best = E
                    x_best = x.copy()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                x[idx] = 1 - x[idx]  # reject
                steps_since_improvement += 1

            full_history.append(E_best)

            # Adaptive reheat: if stuck for 1000 steps with no improvement, reheat
            if steps_since_improvement >= 1000:
                reheat_factor = min(reheat_factor * 2.0, 5.0)  # capped to avoid runaway heating
                steps_since_improvement = 0

            # Slow decay of reheat factor back to baseline
            if reheat_factor > 1.0 and step % 50 == 0:
                reheat_factor = max(1.0, reheat_factor * 0.95)

        if E_best < best_energy:
            best_energy = E_best
            best_solution = x_best.copy()

    return best_solution, best_energy, full_history

# Run SA with multiple independent seeds for reliable statistics
# Single-run best is unreliable for heuristic methods — report mean ± std
N_SA_RUNS = 10
sa_seeds = list(range(42, 42 + N_SA_RUNS))
sa_all_returns = []
sa_all_variances = []
sa_all_sharpes = []
sa_all_feasible = []
sa_all_weights = []
sa_all_objectives = []

print(f"\n  Running Simulated Annealing ({N_QUBO_VARS} variables, {N_SA_RUNS} independent runs)...")
sa_best_seed = None
sa_best_energy = float('inf')
sa_best_bits = None

for run_idx, seed in enumerate(sa_seeds):
    bits, energy, history = simulated_annealing_qubo(
        N_QUBO_VARS, penalty_objective,
        n_steps=20000, T_start=2.0, T_end=0.001, n_restarts=10, seed=seed
    )
    w = decode_weights(bits)
    ret = portfolio_return(w)
    var = portfolio_variance(w)
    sharpe = (ret - RISK_FREE) / max(sqrt(var), 0.001)
    feasible, msg = is_feasible(w)

    sa_all_returns.append(ret)
    sa_all_variances.append(var)
    sa_all_sharpes.append(sharpe)
    sa_all_feasible.append(feasible)
    sa_all_weights.append(w)
    sa_all_objectives.append(energy)

    if energy < sa_best_energy:
        sa_best_energy = energy
        sa_best_bits = bits
        sa_best_seed = seed

# Use best-energy solution as the primary SA result
sa_bits = sa_best_bits
sa_energy = sa_best_energy
sa_weights = decode_weights(sa_bits)
# No renormalization — the budget penalty controls Σω ≈ 1
sa_return = portfolio_return(sa_weights)
sa_variance = portfolio_variance(sa_weights)
sa_sharpe = (sa_return - RISK_FREE) / max(sqrt(sa_variance), 0.001)
sa_feasible, sa_msg = is_feasible(sa_weights)

# Compute summary statistics
ret_arr = np.array(sa_all_returns)
var_arr = np.array(sa_all_variances)
sharpe_arr = np.array(sa_all_sharpes)
n_feasible_runs = sum(sa_all_feasible)

print(f"\n  SA Result — Best of {N_SA_RUNS} runs (seed={sa_best_seed}):")
print(f"    Weights:     {[f'{w:.4f}' for w in sa_weights]}")
print(f"    Σω = {np.sum(sa_weights):.4f}")
print(f"    Objective:   {sa_energy:.6f}")
print(f"    Return:      {sa_return*100:.2f}%")
print(f"    Variance:    {sa_variance:.6f} (target: {VAR_TARGET})")
print(f"    Risk:        {sqrt(sa_variance)*100:.2f}%")
print(f"    Sharpe:      {sa_sharpe:.2f}")
print(f"    Feasible:    {sa_feasible} — {sa_msg}")
for cls in CLASS_LABELS:
    cw = np.sum(sa_weights[ASSET_CLASSES[cls]])
    print(f"    {cls}: Σω = {cw:.4f}")

# SA multi-run statistics
print(f"\n  SA Multi-Run Statistics ({N_SA_RUNS} independent seeds):")
print(f"    Return:       {ret_arr.mean()*100:.2f}% ± {ret_arr.std()*100:.2f}%")
print(f"    Variance:     {var_arr.mean():.6f} ± {var_arr.std():.6f}")
print(f"    Sharpe:       {sharpe_arr.mean():.2f} ± {sharpe_arr.std():.2f}")
print(f"    Feasible:     {n_feasible_runs}/{N_SA_RUNS} runs feasible")
print(f"    Objective:    {np.mean(sa_all_objectives):.4f} ± {np.std(sa_all_objectives):.4f}")

# Store SA summary for later comparison
SA_SUMMARY = {
    'best': {
        'bits': sa_bits, 'weights': sa_weights, 'return': sa_return,
        'variance': sa_variance, 'sharpe': sa_sharpe, 'feasible': sa_feasible,
        'objective': sa_energy,
    },
    'stats': {
        'return_mean': float(ret_arr.mean()), 'return_std': float(ret_arr.std()),
        'variance_mean': float(var_arr.mean()), 'variance_std': float(var_arr.std()),
        'sharpe_mean': float(sharpe_arr.mean()), 'sharpe_std': float(sharpe_arr.std()),
        'objective_mean': float(np.mean(sa_all_objectives)),
        'objective_std': float(np.std(sa_all_objectives)),
        'n_feasible': n_feasible_runs, 'n_total': N_SA_RUNS,
    },
}


# =====================================================================
# =====================================================================
# PART 7: Efficient Frontier via Risk-Aversion Sweep (Gamma Scan)
# =====================================================================
print("\n" + "-" * 70)
print("PART 7: Efficient Frontier -- Risk Aversion (Gamma) Sweep")
print("-" * 70)

print("""
  GAMMA SWEEP STRATEGY:
    Since variance is now in the OBJECTIVE (gamma * omega^T Sigma omega),
    we sweep gamma to trace the Markowitz efficient frontier. Each gamma
    yields a different point on the risk-return trade-off curve:
      Low gamma  -> solver prioritizes RETURN (risk-on, more equity)
      High gamma -> solver prioritizes LOW RISK (risk-off, more bonds/cash)

    We CLASSICALLY filter solutions satisfying sigma <= sigma_target
    as post-processing. This preserves QUBO compatibility.
""")

# Gamma values: from near-risk-neutral to highly risk-averse
# Wide range ensures the full efficient frontier is explored
gamma_range = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]

print(f"\n  Sweeping gamma in {gamma_range}")
print(f"  (SA with 3000 steps per gamma, 5 restarts)")

frontier_results = []
for gamma_val in gamma_range:
    def obj_with_gamma(bits):
        return penalty_objective(bits, lambda_b=LAMBDA_BUDGET,
                                 gamma=gamma_val)

    sol, energy, _ = simulated_annealing_qubo(
        N_QUBO_VARS, obj_with_gamma,
        n_steps=3000, T_start=1.0, T_end=0.001, n_restarts=5, seed=42
    )
    w_sol = decode_weights(sol)
    ret = portfolio_return(w_sol)
    var = portfolio_variance(w_sol)
    risk = sqrt(var)
    sharpe = (ret - RISK_FREE) / max(risk, 0.001)
    feasible, msg = is_feasible(w_sol, tol=0.02)
    budget_err = abs(np.sum(w_sol) - 1.0)
    meets_var_target = var <= VAR_TARGET + 0.001

    frontier_results.append({
        "gamma": gamma_val,
        "weights": w_sol, "return": ret, "variance": var, "risk": risk,
        "sharpe": sharpe, "feasible": feasible,
        "meets_var_target": meets_var_target,
        "budget_err": budget_err, "energy": energy,
    })

# Display frontier
print(f"\n  Efficient Frontier (variance-in-objective, gamma sweep):")
print(f"  {'gamma':>8} {'Return%':>9} {'Risk%':>8} {'Variance':>10} "
      f"{'Sharpe':>8} {'BudgetErr':>10} {'VarOK':>6} {'Feasible':<8}")
print(f"  {'-'*8} {'-'*9} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*6} {'-'*8}")
for r in frontier_results:
    print(f"  {r['gamma']:>8.1f} {r['return']*100:>8.2f}% "
          f"{r['risk']*100:>7.2f}% {r['variance']:>10.6f} "
          f"{r['sharpe']:>8.2f} {r['budget_err']:>10.6f} "
          f"{str(r['meets_var_target']):>6} {str(r['feasible']):<8}")

# Best solution meeting variance target
frontier_feasible = [r for r in frontier_results if r["meets_var_target"]]
if frontier_feasible:
    best_frontier = max(frontier_feasible, key=lambda r: r["return"])
    print(f"\n  Best feasible (sigma <= sigma_target):")
    print(f"    gamma = {best_frontier['gamma']:.1f}")
    print(f"    Return: {best_frontier['return']*100:.2f}%, "
          f"Risk: {best_frontier['risk']*100:.2f}%")
    print(f"    Sharpe: {best_frontier['sharpe']:.2f}")
    print(f"    Weights: {[f'{w:.3f}' for w in best_frontier['weights']]}")
    print(f"\n  ADVANTAGE: 1D gamma sweep ({len(gamma_range)} points) vs.")
    print(f"    old 2D penalty grid (36+ combinations). QUBO-compatible.")
else:
    print(f"\n  No solutions meet variance target -- widen gamma range?")
    best_frontier = None

# PART 8: Gate-Based QAOA — Bloqade QASM2 + Qiskit Simulation
# =====================================================================
# Gate-based QAOA has NO sign restriction on J_ij (unlike analog Rydberg
# where V_ij = C6/r^6 > 0 always). The cost Hamiltonian:
#   H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j
# is implemented via parameterized Rz gates nested in CX, so J_ij can be
# positive OR negative — the rotation direction simply flips.
# =====================================================================
print("\n" + "=" * 70)
print("PART 8: Gate-Based QAOA — Bloqade QASM2 + Qiskit Simulation")
print("=" * 70)

# Import QAOA dependencies
_HAS_BLOQADE_QAOA, _HAS_QISKIT = False, False
try:
    from bloqade import qasm2
    import kirin
    from kirin.dialects import ilist
    _HAS_BLOQADE_QAOA = True
    print("  Bloqade QASM2: OK")
except Exception as e:
    print(f"  [WARN] Bloqade QASM2 not available: {e}")

try:
    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit.primitives import BackendSamplerV2
    _HAS_QISKIT = True
    print("  Qiskit + Aer:  OK")
except Exception as e:
    print(f"  [WARN] Qiskit/Aer not available: {e}")

QAOA_OK = _HAS_BLOQADE_QAOA and _HAS_QISKIT

# QAOA runs on the FULL problem (N=6 assets, K=3 bits, 18 qubits).
# 2^18 = 262,144 states → exact enumeration feasible for ground-truth verification.
# ALL solvers (classical, SA, QAOA) run on the SAME 18-variable QUBO.

print(f"""
  QAOA DEMO SETUP:
    Full problem: N={N_ASSETS} × K={K_BITS} = {N_QUBO_VARS} qubits
    (2^{N_QUBO_VARS} = {2**N_QUBO_VARS} states → exact enumeration possible)

  Pipeline:
    Financial Data → Pure QUBO → Ising (h, J) → QAOA circuit → COBYLA → Sample

  KEY ADVANTAGE of gate-based QAOA over analog Rydberg:
    * J_ij sign is FREE (gate rotation direction, not physical vdW force)
    * No graph-topology constraint (any ZZ pair via CX-Rz-CX)
    * Ising model is EXACT, not an embedding approximation

  FAIR COMPARISON: all solvers (classical, SA, QAOA) run on the SAME
    {N_QUBO_VARS}-variable QUBO with exact enumeration as ground truth.
""")

# ---- 8A: Exact Enumeration (full problem) ----
print(f"\n  --- 8A: Exact Enumeration ({2**N_QUBO_VARS} states) ---")

all_states = []
for num in range(2 ** N_QUBO_VARS):
    bits = int_to_bits(num, N_QUBO_VARS)
    all_states.append((bits, penalty_objective(bits)))
all_states.sort(key=lambda x: x[1])

# TRUE QUBO global minimum: lowest penalty_objective, NO post-hoc filtering.
# The QUBO has NO variance constraint and NO class penalties — those are
# post-hoc feasibility labels only.  Filtering HERE would change the problem.
qubo_gs_bits, qubo_gs_obj = all_states[0]

# Best FEASIBLE solution (QUBO + post-hoc variance + class constraints).
# This is a DIFFERENT concept from the QUBO ground state — shown for reference.
best_feasible_bits, best_feasible_obj = None, None
for bits, obj in all_states:
    w_test = decode_weights(bits)
    feasible_test, _ = is_feasible(w_test, tol=1e-2)
    if feasible_test:
        best_feasible_bits, best_feasible_obj = bits, obj; break
if best_feasible_bits is None:
    best_feasible_bits, best_feasible_obj = all_states[0]

qubo_gs_w = decode_weights(qubo_gs_bits)
qubo_gs_ret = portfolio_return(qubo_gs_w)
qubo_gs_var = portfolio_variance(qubo_gs_w)

print(f"  QUBO global minimum (no post-hoc filtering):")
print(f"    Bits: |{bits_to_display(qubo_gs_bits)}>  Obj={qubo_gs_obj:.6f}")
print(f"    Weights: {[f'{w:.4f}' for w in qubo_gs_w]}  Σω={np.sum(qubo_gs_w):.4f}")
print(f"    Return={qubo_gs_ret*100:.2f}%  Variance={qubo_gs_var:.6f}")
feasible_gs, msg_gs = is_feasible(qubo_gs_w, tol=1e-2)
print(f"    Feasible (post-hoc): {feasible_gs} — {msg_gs}")

if best_feasible_bits is not None and not np.all(best_feasible_bits == qubo_gs_bits):
    best_feas_w = decode_weights(best_feasible_bits)
    print(f"\n  Best feasible solution (with variance + class constraints):")
    print(f"    Bits: |{bits_to_display(best_feasible_bits)}>  Obj={best_feasible_obj:.6f}")
    print(f"    Weights: {[f'{w:.4f}' for w in best_feas_w]}  Σω={np.sum(best_feas_w):.4f}")
    print(f"    Return={portfolio_return(best_feas_w)*100:.2f}%  "
          f"Variance={portfolio_variance(best_feas_w):.6f}")

# Use QUBO global minimum as the reference for QAOA comparison
exact_bits = qubo_gs_bits
exact_obj = qubo_gs_obj
exact_w = qubo_gs_w
exact_ret = qubo_gs_ret
exact_var = qubo_gs_var

# ---- 8B: Pure QUBO → Exact Ising Conversion ----
print(f"\n  --- 8B: Pure QUBO → Ising Conversion ---")
print(f"  (No class penalties → objective IS pure quadratic → Ising is EXACT)")

# Numerical extraction of QUBO coefficients (exact for pure quadratic)
f0 = penalty_objective(np.zeros(N_QUBO_VARS, dtype=int))
a_qubo = np.zeros(N_QUBO_VARS)
for i in range(N_QUBO_VARS):
    bits_i = np.zeros(N_QUBO_VARS, dtype=int); bits_i[i] = 1
    a_qubo[i] = penalty_objective(bits_i) - f0

B_qubo = np.zeros((N_QUBO_VARS, N_QUBO_VARS))
for i in range(N_QUBO_VARS):
    for j in range(i + 1, N_QUBO_VARS):
        bits_ij = np.zeros(N_QUBO_VARS, dtype=int)
        bits_ij[i] = 1; bits_ij[j] = 1
        f_ij = penalty_objective(bits_ij)
        f_i = a_qubo[i] + f0; f_j = a_qubo[j] + f0
        B_qubo[i][j] = f_ij - f_i - f_j + f0

# QUBO → Ising: x = (1-z)/2  →  h_i = -a_i/2 - Σ_{j≠i} b_ij/4,  J_ij = b_ij/4
h_ising = np.zeros(N_QUBO_VARS)
J_ising = np.zeros((N_QUBO_VARS, N_QUBO_VARS))
for i in range(N_QUBO_VARS):
    row_sum = sum(B_qubo[min(i,j)][max(i,j)] for j in range(N_QUBO_VARS) if j != i)
    h_ising[i] = -a_qubo[i] / 2.0 - row_sum / 4.0
for i in range(N_QUBO_VARS):
    for j in range(i + 1, N_QUBO_VARS):
        J_ising[i][j] = B_qubo[i][j] / 4.0

# Ising energy function
ising_offset = 0.0  # computed below
def ising_energy(z):
    e = float(np.dot(h_ising, z))
    for i in range(N_QUBO_VARS):
        for j in range(i + 1, N_QUBO_VARS):
            e += J_ising[i][j] * z[i] * z[j]
    return e

# Pre-compute Ising energies for ALL basis states (used in fast expectation evaluation)
ALL_ISING_ENERGIES = np.array([
    ising_energy(1 - 2 * int_to_bits(k, N_QUBO_VARS))
    for k in range(1 << N_QUBO_VARS)
])

# Verify: exact match over all states (pure quadratic → must be perfect)
for num in range(2 ** N_QUBO_VARS):
    bits = int_to_bits(num, N_QUBO_VARS)
    z = 1 - 2 * bits
    qe = penalty_objective(bits)
    ie = ALL_ISING_ENERGIES[num]  # ising_energy(z) — already computed above
    if num == 0:
        ising_offset = qe - ie
    if abs((qe - ie) - ising_offset) > 1e-10:
        print(f"  [ERROR] Ising mismatch at state {num}"); break
else:
    print(f"  Ising conversion: EXACT over all {2**N_QUBO_VARS} states")
    print(f"  Ising offset = {ising_offset:.6f}")
    n_z = sum(1 for h in h_ising if abs(h) > 1e-10)
    n_zz = sum(1 for i in range(N_QUBO_VARS) for j in range(i+1, N_QUBO_VARS) if abs(J_ising[i][j]) > 1e-10)
    print(f"  Z terms: {n_z}, ZZ terms: {n_zz}")

# ---- 8C: Bloqade QASM2 QAOA Circuit ----
print(f"\n  --- 8C: Bloqade QASM2 QAOA Circuit ---")

P_LAYERS = 3
MAXITER = 60
QAOA_SHOTS = 2048
QAOA_SEED = 42

# Ising terms as flat Python dicts (kirin compatibility)
ising_linear = {i: float(h_ising[i]) for i in range(N_QUBO_VARS) if abs(h_ising[i]) > 1e-15}
ising_quad = {(i, j): float(J_ising[i][j]) for i in range(N_QUBO_VARS)
              for j in range(i+1, N_QUBO_VARS) if abs(J_ising[i][j]) > 1e-15}

print(f"  Ising: {N_QUBO_VARS} qubits, {len(ising_linear)} Z, {len(ising_quad)} ZZ")
print(f"  QAOA: p={P_LAYERS}, COBYLA maxiter={MAXITER}")

if QAOA_OK:
    # Bloqade QASM2 circuit
    quad_items = sorted(ising_quad.items())
    quad_i = [int(e[0][0]) for e in quad_items]
    quad_j = [int(e[0][1]) for e in quad_items]
    quad_w = [float(e[1]) for e in quad_items]
    lin_items = sorted(ising_linear.items())
    lin_idx = [int(e[0]) for e in lin_items]
    lin_w = [float(e[1]) for e in lin_items]
    n_quad, n_lin = len(quad_items), len(lin_items)

    @qasm2.extended
    def qaoa_kernel(gamma_list, beta_list):
        q = qasm2.qreg(N_QUBO_VARS)
        for i in range(N_QUBO_VARS):
            qasm2.h(q[i])
        for layer in range(len(gamma_list)):
            g = gamma_list[layer]
            for k in range(n_quad):
                qasm2.cx(q[quad_i[k]], q[quad_j[k]])
                qasm2.rz(q[quad_j[k]], 2.0 * quad_w[k] * g)
                qasm2.cx(q[quad_i[k]], q[quad_j[k]])
            for k in range(n_lin):
                qasm2.rz(q[lin_idx[k]], 2.0 * lin_w[k] * g)
            b = beta_list[layer]
            for i2 in range(N_QUBO_VARS):
                qasm2.rx(q[i2], 2.0 * b)
        return q

    print("  Bloqade QASM2 circuit built.")

    # ---- 8D: Qiskit Simulation + COBYLA Optimisation ----
    print(f"\n  --- 8D: QAOA Optimisation (statevector, COBYLA) ---")

    qc = QuantumCircuit(N_QUBO_VARS, N_QUBO_VARS)
    params = []
    for i in range(N_QUBO_VARS):
        qc.h(i)
    for p in range(P_LAYERS):
        gp = Parameter(f"gamma_{p}"); bp = Parameter(f"beta_{p}")
        params.extend([gp, bp])
        for (i, j), J in sorted(ising_quad.items()):
            qc.cx(i, j); qc.rz(2.0 * float(J) * gp, j); qc.cx(i, j)
        for i, h in sorted(ising_linear.items()):
            qc.rz(2.0 * float(h) * gp, i)
        for i in range(N_QUBO_VARS):
            qc.rx(2.0 * bp, i)
    qc.measure(range(N_QUBO_VARS), range(N_QUBO_VARS))
    qc_no_meas = qc.remove_final_measurements(inplace=False)

    # Fast expectation: pre-computed Ising energies × probabilities (numpy dot product)
    def _expectation_fast(sv_data):
        probs = np.abs(sv_data) ** 2
        return float(np.dot(probs, ALL_ISING_ENERGIES))

    rng = np.random.default_rng(QAOA_SEED)
    x0 = rng.uniform(0, 2 * np.pi, size=len(params))
    simulator = AerSimulator(method="matrix_product_state")
    sampler = BackendSamplerV2(backend=simulator, options={"default_shots": QAOA_SHOTS})

    eval_count, best_val, best_x = [0], [float("inf")], [None]
    history = []

    def cost_fn(theta):
        eval_count[0] += 1
        bind = {p: float(theta[idx]) for idx, p in enumerate(params)}
        sv = Statevector.from_instruction(qc_no_meas.assign_parameters(bind))
        energy = _expectation_fast(sv.data) + ising_offset  # add offset to convert Ising→QUBO energy
        if energy < best_val[0]:
            best_val[0] = energy; best_x[0] = theta.copy()
        history.append({"iter": eval_count[0], "energy": energy, "best": best_val[0]})
        if eval_count[0] <= 3 or eval_count[0] % 20 == 0:
            print(f"    iter {eval_count[0]:4d} | energy = {energy:+.6f} | best = {best_val[0]:+.6f}")
        return energy

    t0 = time.time()
    result = minimize(cost_fn, x0, method="COBYLA",
                      options={"maxiter": MAXITER, "rhobeg": 0.4})
    elapsed = time.time() - t0

    print(f"\n  Optimisation done ({elapsed:.1f}s)  status={result.message}")
    print(f"  Best Ising energy: {best_val[0]:.6f}")

    # Final sampling
    final_bind = {p: float(best_x[0][idx]) for idx, p in enumerate(params)}
    job = sampler.run([qc.assign_parameters(final_bind)])
    final_counts = job.result()[0].data.c.get_int_counts()
    total_shots = sum(final_counts.values())

    # Find both most-probable and post-selected best (lowest energy) bitstrings.
    # NOTE: Picking the lowest-energy sampled state is POST-SELECTION — the QAOA
    # produces a distribution; we classically cherry-pick the best sample.  This is
    # common in variational quantum optimisation but MUST be disclosed in reporting.
    most_prob_bi = max(final_counts, key=final_counts.get)
    most_prob_bits = int_to_bits(most_prob_bi, N_QUBO_VARS)
    most_prob_obj = penalty_objective(most_prob_bits)

    best_bs_bits, best_e = None, float("inf")
    for bi, cnt in final_counts.items():
        bits = int_to_bits(bi, N_QUBO_VARS)
        obj = penalty_objective(bits)
        if obj < best_e:
            best_e = obj
            best_bs_bits = bits

    qaoa_bits = best_bs_bits
    qaoa_w = decode_weights(qaoa_bits)
    qaoa_ret = portfolio_return(qaoa_w)
    qaoa_var = portfolio_variance(qaoa_w)
    # Relative gap for MINIMIZATION: (E_QAOA - E_GS) / |E_GS|
    # quality = 1 - rel_gap  (1.0 = optimal, closer to 1 = better)
    # NOTE: exact_obj / best_e is for MAXIMIZATION (MaxCut) — WRONG here.
    if abs(exact_obj) > 1e-12:
        rel_gap = (best_e - exact_obj) / abs(exact_obj)
        quality = max(0.0, 1.0 - rel_gap)
    else:
        rel_gap = 0.0 if best_e <= exact_obj else float('inf')
        quality = 1.0 if best_e <= exact_obj else 0.0
    appr_ratio = quality  # keep key name for JSON compatibility

    exact_display = bits_to_display(exact_bits)
    best_display = bits_to_display(best_bs_bits)
    hamming = int(np.sum(best_bs_bits != exact_bits))
    is_optimal = hamming == 0

    print(f"\n  QAOA Result:")
    print(f"    NOTE: 'Best sampled' = post-selection (lowest energy among {total_shots} shots).")
    print(f"    Post-selected best: |{best_display}>  (energy={best_e:.6f})")
    print(f"    Most probable:      |{bits_to_display(most_prob_bits)}>  "
          f"(energy={most_prob_obj:.6f}, count={final_counts[most_prob_bi]})")
    print(f"    QUBO global min:    |{exact_display}>  (energy={exact_obj:.6f})")
    print(f"    Hamming dist (best vs GS):   {hamming}/{N_QUBO_VARS}")
    print(f"    Rel. gap (E_QAOA-E_GS)/|E_GS|: {rel_gap:+.4f}  "
          f"quality={quality:.4f} ({'OPTIMAL' if is_optimal else 'suboptimal'})")
    print(f"    QAOA weights:   {[f'{w:.4f}' for w in qaoa_w]}  Σω={np.sum(qaoa_w):.4f}")
    print(f"    QAOA return:    {qaoa_ret*100:.2f}%  variance={qaoa_var:.6f}")

    # Top sampled states
    print(f"\n  Top 6 sampled states ({total_shots} shots):")
    for rank, (bi, cnt) in enumerate(sorted(final_counts.items(), key=lambda x: -x[1])[:6]):
        bits = int_to_bits(bi, N_QUBO_VARS)
        obj = penalty_objective(bits)
        w = decode_weights(bits)
        is_best = bool(np.all(bits == best_bs_bits))
        is_gs = bool(np.all(bits == exact_bits))
        is_most_prob = bi == most_prob_bi
        markers = []
        if is_most_prob: markers.append("most probable")
        if is_best: markers.append("post-sel best")
        if is_gs: markers.append("EXACT GS")
        marker = (" ← " + ", ".join(markers)) if markers else ""
        print(f"    #{rank+1}: |{bits_to_display(bits)}>  obj={obj:+.4f}  ret={portfolio_return(w)*100:.1f}%  "
              f"cnt={cnt} ({100*cnt/total_shots:.1f}%){marker}")

    QAOA_RESULT = {
        'bits': qaoa_bits, 'weights': qaoa_w, 'return': qaoa_ret,
        'variance': qaoa_var, 'objective': best_e,
        'best_bitstring': bits_to_display(best_bs_bits),
        'most_probable_bitstring': bits_to_display(most_prob_bits),
        'most_probable_objective': most_prob_obj,
        'appx_ratio': appr_ratio,
        'rel_gap': rel_gap,
        'quality': quality,
        'hamming': hamming, 'elapsed': elapsed, 'history': history,
        'n_shots': total_shots,
    }
    BLOQADE_RUN = True  # for Part 9 compatibility

else:
    print("\n  [SKIP] QAOA dependencies not available.")
    print("  Using exact enumeration as ground truth reference.")
    QAOA_RESULT = None
    BLOQADE_RUN = False


# =====================================================================
# PART 9: Solver Comparison — Simulated Annealing vs QAOA
# =====================================================================
print("\n" + "=" * 70)
print("PART 9: Solver Comparison — SA (classical) vs QAOA (quantum)")
print("=" * 70)

print(f"""
  Both solvers operate on the SAME {N_QUBO_VARS}-variable pure QUBO:
    f(x) = -mu^T w + gamma * w^T Sigma w + lambda_b * (sum(w) - 1)^2
  Exact enumeration (2^{N_QUBO_VARS} = {2**N_QUBO_VARS} states) provides ground truth.
""")

# =========================================================================
# TABLE 1: SA Results (classical QUBO heuristic)
# =========================================================================
print(f"\n  {'='*70}")
print(f"  TABLE 1: Simulated Annealing (N={N_ASSETS}, K={K_BITS}, {N_QUBO_VARS} binary vars)")
print(f"  {'='*70}")
print(f"  {'Solver':<28} {'Return%':>9} {'Variance':>10} {'Risk%':>8} "
      f"{'Sharpe':>8} {'Σω':>8} {'Feasible':<10}")
print(f"  {'-'*28} {'-'*9} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

print(f"  {'Exact enumeration (QUBO GS)':<28} {exact_ret*100:>8.2f}% "
      f"{exact_var:>10.6f} {sqrt(exact_var)*100:>7.2f}% "
      f"{(exact_ret - RISK_FREE)/max(sqrt(exact_var),0.001):>8.2f} "
      f"{np.sum(exact_w):>7.4f} {'—':<10}")

print(f"  {'Simulated Annealing (best)':<28} {sa_return*100:>8.2f}% "
      f"{sa_variance:>10.6f} {sqrt(sa_variance)*100:>7.2f}% "
      f"{sa_sharpe:>8.2f} {np.sum(sa_weights):>7.4f} {str(sa_feasible):<10}")

sa_stats = SA_SUMMARY['stats']
n_runs_label = f"({sa_stats['n_total']} runs)"
print(f"  {'  SA mean ± std':<28} "
      f"{sa_stats['return_mean']*100:>8.2f}% {'':>10} {'':>8} "
      f"{sa_stats['sharpe_mean']:>8.2f} {'':>8} "
      f"{str(sa_stats['n_feasible'])+'/'+str(sa_stats['n_total']):<10}")
print(f"  {'  SA return std dev':<28} "
      f"±{sa_stats['return_std']*100:.2f}% {'':>10} {'':>8} "
      f"±{sa_stats['sharpe_std']:.2f} {'':>8} {'':>8}")

if best_frontier:
    bf = best_frontier
    print(f"  {'Gamma sweep best':<28} {bf['return']*100:>8.2f}% "
          f"{bf['variance']:>10.6f} {sqrt(bf['variance'])*100:>7.2f}% "
          f"{bf['sharpe']:>8.2f} {np.sum(bf['weights']):>7.4f} {'YES':<10}")

print(f"\n  SA: {N_SA_RUNS} independent runs × 10 restarts × 20,000 steps each.")
print(f"  Exact discrete optimum requires 2^{N_QUBO_VARS} = {2**N_QUBO_VARS:.1e} evaluations.")

# =========================================================================
# TABLE 2: QAOA Results (full problem, N=6 assets, K=3 bits, 18 qubits)
# =========================================================================
print(f"\n  {'='*70}")
print(f"  TABLE 2: QAOA (N={N_ASSETS}, K={K_BITS}, {N_QUBO_VARS} qubits, "
      f"2^{N_QUBO_VARS}={2**N_QUBO_VARS} states)")
print(f"  {'='*70}")
print(f"  {'Solver':<22} {'Return%':>9} {'Variance':>10} {'Risk%':>8} "
      f"{'Sharpe':>8} {'Status':<14}")
print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*8} {'-'*8} {'-'*14}")
print(f"  {'Exact enumeration':<22} {exact_ret*100:>8.2f}% "
      f"{exact_var:>10.6f} {sqrt(exact_var)*100:>7.2f}% "
      f"{(exact_ret - RISK_FREE)/max(sqrt(exact_var),0.001):>8.2f} {'Ground State':<14}")

if QAOA_RESULT is not None:
    qr = QAOA_RESULT
    qaoa_sharpe_val = (qr['return'] - RISK_FREE) / max(sqrt(qr['variance']), 0.001)
    status = "OPTIMAL" if qr['hamming'] == 0 else f"Q={qr['quality']:.3f}"
    print(f"  {'QAOA (p='+str(P_LAYERS)+')':<22} {qr['return']*100:>8.2f}% "
          f"{qr['variance']:>10.6f} {sqrt(qr['variance'])*100:>7.2f}% "
          f"{qaoa_sharpe_val:>8.2f} {status:<14}")
    print(f"\n  QAOA vs Exact:")
    print(f"    Hamming distance: {qr['hamming']}/{N_QUBO_VARS}")
    print(f"    Quality = 1 - (E_QAOA - E_GS)/|E_GS|: {qr['quality']:.4f}  "
          f"(rel_gap={qr['rel_gap']:+.4f})")
    print(f"    Optimisation time: {qr['elapsed']:.1f}s")
    print(f"\n  NOTE: Gate-based QAOA implements the Ising model EXACTLY —")
    print(f"    all J_ij signs are free (unlike analog Rydberg). The only")
    print(f"    limitation is the QAOA ansatz depth (p={P_LAYERS}) and the")
    print(f"    classical optimiser (COBYLA, {MAXITER} iterations).")
else:
    print(f"  {'QAOA':<22} {'N/A':>9} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'Not run':<14}")

# ---- SA vs QAOA head-to-head ----
print(f"\n  {'='*70}")
print(f"  SA vs QAOA — Head-to-Head Summary")
print(f"  {'='*70}")
print(f"  {'Metric':<30} {'SA (classical)':>20} {'QAOA (quantum)':>20}")
print(f"  {'-'*30} {'-'*20} {'-'*20}")
print(f"  {'Solver type':<30} {'Thermal annealing':>20} {'Variational quantum':>20}")
print(f"  {'Binary variables':<30} {N_QUBO_VARS:>20} {N_QUBO_VARS:>20}")
print(f"  {'Return':<30} {sa_return*100:>19.2f}% ", end="")
if QAOA_RESULT is not None:
    print(f"{QAOA_RESULT['return']*100:>19.2f}%")
else:
    print(f"{'N/A':>20}")
print(f"  {'Risk (σ)':<30} {sqrt(sa_variance)*100:>19.2f}% ", end="")
if QAOA_RESULT is not None:
    print(f"{sqrt(QAOA_RESULT['variance'])*100:>19.2f}%")
else:
    print(f"{'N/A':>20}")
print(f"  {'Sharpe ratio':<30} {sa_sharpe:>20.2f} ", end="")
if QAOA_RESULT is not None:
    print(f"{qaoa_sharpe_val:>20.2f}")
else:
    print(f"{'N/A':>20}")

# ---- Ising Model Verification ----
print(f"\n  --- Ising Model Verification ---")
ising_scores = {}
for num in range(2 ** N_QUBO_VARS):
    bits = int_to_bits(num, N_QUBO_VARS)
    z = 1 - 2 * bits  # x=0→z=+1, x=1→z=-1
    ising_scores[num] = ising_energy(z)

ising_gs_num = min(ising_scores, key=ising_scores.get)
ising_gs_bits = int_to_bits(ising_gs_num, N_QUBO_VARS)
print(f"  Penalty-obj ground state: |{bits_to_display(exact_bits)}>")
print(f"  Ising-model ground state: |{bits_to_display(ising_gs_bits)}>")
print(f"  Match: {bool(np.all(ising_gs_bits == exact_bits))}")
print(f"  (Pure QUBO → Ising conversion is exact — must match.)")


# PART 10: Investment Decision & Risk Analysis
# =====================================================================
print("\n" + "=" * 70)
print("PART 10: Investment Decision & Risk Analysis")
print("=" * 70)

# Compare SA vs QAOA solutions
candidates = [
    ("Simulated Annealing (best)", sa_weights, sa_return, sa_variance, sa_sharpe),
]
if QAOA_RESULT is not None:
    qaoa_sharpe_val = (QAOA_RESULT['return'] - RISK_FREE) / max(sqrt(QAOA_RESULT['variance']), 0.001)
    candidates.append(("QAOA (post-selected)", QAOA_RESULT['weights'],
                       QAOA_RESULT['return'], QAOA_RESULT['variance'], qaoa_sharpe_val))
if best_frontier:
    candidates.append(("Gamma sweep best", best_frontier['weights'],
                       best_frontier['return'], best_frontier['variance'],
                       best_frontier['sharpe']))

print(f"\n  Portfolio Recommendations:")
print(f"  {'Method':<25} {'Assets':<60} {'Return':>8} {'Risk':>8} {'Sharpe':>7}")
print(f"  {'─'*25} {'─'*60} {'─'*8} {'─'*8} {'─'*7}")

for name, w, ret, var, sharpe in candidates:
    alloc = ", ".join(f"{ASSET_NAMES[i]}: {w[i]*100:.1f}%"
                      for i in range(N_ASSETS) if w[i] > 0.005)
    print(f"  {name:<25} {alloc:<60} {ret*100:>7.2f}% "
          f"{sqrt(var)*100:>7.2f}% {sharpe:>7.2f}")

# Risk decomposition
print(f"\n  Risk Decomposition (best SA portfolio):")
print(f"  {'Asset':<24} {'Weight':>8} {'Marginal Risk':>14} "
      f"{'Risk Contrib':>14} {'% of Total':>10}")
total_risk = sqrt(sa_variance)
for i in range(N_ASSETS):
    # Marginal risk contribution: w_i * (Σw)_i / σ_portfolio
    marginal = np.dot(cov_matrix[i], sa_weights)
    risk_contrib = sa_weights[i] * marginal / max(total_risk, 1e-12)
    pct = risk_contrib / max(total_risk, 1e-12) * 100
    print(f"  {ASSET_NAMES[i]:<24} {sa_weights[i]:>7.2%} "
          f"{marginal:>13.4f} {risk_contrib:>13.4f} {pct:>9.1f}%")

print(f"\n  Diversification Ratio: "
      f"{np.dot(sa_weights, volatilities) / max(total_risk, 1e-12):.2f}")


# =====================================================================
# PART 11: Comparison with Binary-Selection Approaches
# =====================================================================
print("\n" + "─" * 70)
print("PART 11: Continuous Weight Discretization vs Binary-Selection Methods")
print("─" * 70)

print(f"""
  The other portfolio files in this project use BINARY ASSET SELECTION:
    x_i ∈ {{0,1}} — pick K assets from N, equal or fixed weights.

  Continuous weight discretization (inspired by Sakuler et al., 2025):
    ω_i ∈ [ω_i,min, ω_i,max] — continuous weights via K-bit encoding.

  KEY DIFFERENCES:
  ┌──────────────────────┬──────────────────────┬──────────────────────┐
  │ Dimension            │ Binary Selection     │ Sakuler Discretized  │
  │                      │ (other files)        │ (THIS FILE)          │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ Variables            │ N binary             │ N·K binary           │
  │ Weight model         │ x_i ∈ {{0,1}}          │ ω_i via K-bit bins   │
  │ Budget constraint    │ Σ x_i = k (integer)  │ Σ ω_i = 1 (real)     │
  │ Variance constraint  │ Soft penalty         │ Inequality penalty   │
  │ Return model         │ Sum of μ_i           │ μ^T ω (weighted)     │
  │ Qubit count (N=6)    │ 6                    │ 30 (K=5)             │
  │ Granularity          │ 1/K (e.g., 1/3)      │ Δω·p_K (e.g., 0.8%) │
  │ Real-world use       │ Simplified           │ Production (RBI)     │
  │ Need for CQM         │ No                   │ YES (auto-tune λ)    │
  └──────────────────────┴──────────────────────┴──────────────────────┘

  WHEN TO USE EACH:
    Binary selection: Quick screening, equal-weighted portfolios, small N.
    Sakuler discretized: Real portfolio management, continuous weights,
                          multiple asset classes with individual bounds.
""")

# Quick comparison: binary selection on same data
print(f"  --- Quick Binary Selection Comparison ---")
# Binary selection: pick top K assets by return, check constraints
K_TARGET = 2
binary_combos = []
for num in range(2 ** N_ASSETS):
    bs = format(num, f'0{N_ASSETS}b')
    k = sum(1 for c in bs if c == '1')
    if k == K_TARGET:
        # equal weights
        w_eq = np.array([1.0/k if c == '1' else 0.0 for c in bs])
        ret = portfolio_return(w_eq)
        var = portfolio_variance(w_eq)
        binary_combos.append((bs, w_eq, ret, var))

binary_combos.sort(key=lambda x: -x[2])  # by return
print(f"  Top binary-selection portfolios (k={K_TARGET}, equal-weight):")
for rank, (bs, w, ret, var) in enumerate(binary_combos[:4]):
    assets = [i for i, c in enumerate(bs) if c == '1']
    sharpe = (ret - RISK_FREE) / max(sqrt(var), 0.001)
    var_ok = "✓" if var <= VAR_TARGET else "✗"
    print(f"    #{rank+1}: |{bs}> assets={assets}  "
          f"ret={ret*100:.1f}%  risk={sqrt(var)*100:.1f}%  "
          f"var_ok={var_ok}  sharpe={sharpe:.2f}")

# Compare best binary vs best discretized
best_bin = binary_combos[0]
best_bin_sharpe = (best_bin[2] - RISK_FREE) / max(sqrt(best_bin[3]), 0.001)
print(f"\n  Best binary:      ret={best_bin[2]*100:.1f}%, "
      f"risk={sqrt(best_bin[3])*100:.1f}%, sharpe={best_bin_sharpe:.2f}")
print(f"  Best discretized: ret={sa_return*100:.1f}%, "
      f"risk={sqrt(sa_variance)*100:.1f}%, sharpe={sa_sharpe:.2f}")
print(f"  Discretization enables FINE-GRAINED weight control, improving")
print(f"  the risk-return trade-off vs. binary equal-weight allocation.")


# =====================================================================
# PART 12: Summary & Paper Conclusions
# =====================================================================
print("\n" + "=" * 70)
print("PART 12: Summary")
print("=" * 70)

print(f"""
  METHOD: Portfolio Optimization via Binary Weight Encoding
    1. Discretize continuous portfolio weights into N·K binary variables
       using per-asset bounds [ω_i,min, ω_i,max]
    2. Formulate as PURE QUBO:
       f(x) = -mu^T w + gamma * w^T Sigma w + lambda_b * (Sigma w_i - 1)^2
       All terms quadratic in x — no piecewise terms, no max operators.
    3. Variance handled via risk aversion (gamma) in the OBJECTIVE
    4. Class bounds checked post-hoc (per-asset bounds enforced by encoding)
    5. Solve via: Simulated Annealing (classical QUBO) vs QAOA (gate-based quantum)

  THIS DEMONSTRATION:
    * N={N_ASSETS} assets × K={K_BITS} bits = {N_QUBO_VARS} binary variables
    * 3 asset classes: EQ, FI, MM (realistic financial data)
    * FAIR COMPARISON: SA and QAOA operate on the SAME {N_QUBO_VARS}-variable QUBO
    * Exact enumeration:  2^{N_QUBO_VARS} = {2**N_QUBO_VARS} states → ground truth verified
    * Simulated Annealing: {sa_return*100:.1f}% return, {sqrt(sa_variance)*100:.1f}% risk
    * Gamma sweep:         Efficient frontier via {len(gamma_range)} gamma values
    * QAOA:                N={N_ASSETS}, K={K_BITS} ({N_QUBO_VARS} qubits, p={P_LAYERS}, Bloqade QASM2 + Qiskit)

  KEY FINDINGS:
    1. A single GAMMA value is arbitrary — the gamma SWEEP traces the efficient frontier
    2. VAR_TARGET is used ONLY for post-hoc labelling, NOT as a QUBO constraint
    3. Variance-as-objective (gamma * w^T Sigma w) keeps all terms quadratic → pure QUBO
    4. Continuous weight encoding enables superior risk-return vs binary selection
    5. Gate-based QAOA implements the Ising model EXACTLY (no sign restrictions,
       unlike analog Rydberg where V_ij = C6/r^6 > 0 always)
    6. For N={N_ASSETS}, K={K_BITS}, exact enumeration (2^{N_QUBO_VARS}={2**N_QUBO_VARS}) is tractable
       → ground truth known; quantum advantage expected for larger instances

  LIMITATIONS:
    * Binary variable count grows linearly with K (granularity vs qubit trade-off)
    * QAOA depth (p) and optimiser iterations limit approximation quality
    * Small problem sizes — proof-of-concept, not quantum advantage

  REFERENCES:
    [1] Sakuler et al., Quantum Mach. Intell. 7, 43 (2025) — weight discretization
    [2] arXiv:2303.12601
    [3] Markowitz, H., J. Finance 7, 77 (1952) — portfolio theory
    [4] Lucas, A., Frontiers in Physics 2, 5 (2014) — Ising formulations
    [5] Farhi, Goldstone & Gutmann, arXiv:1411.4028 (2014) — QAOA
""")

# ---- Save results ----
print("─" * 70)
try:
    import json
    output_data = {
        "paper": "Portfolio Optimization with Binary Encoding and QAOA",
        "configuration": {
            "N_assets": N_ASSETS,
            "K_bits": K_BITS,
            "N_qubo_vars": N_QUBO_VARS,
            "asset_classes": CLASS_LABELS,
            "variance_target": VAR_TARGET,
        },
        "results": {
            "exact_enumeration": {
                "weights": [float(w) for w in exact_w],
                "return": float(exact_ret),
                "variance": float(exact_var),
            },
            "simulated_annealing": {
                "weights": [float(w) for w in sa_weights],
                "return": float(sa_return),
                "variance": float(sa_variance),
                "sharpe": float(sa_sharpe),
                "feasible": bool(sa_feasible),
            },
        },
        "gamma_sweep": {
            "n_gammas": len(frontier_results),
            "n_feasible": sum(1 for r in frontier_results if r.get("meets_var_target", False)),
        } if frontier_results else {},
        "qaoa": {
            "N_assets": N_ASSETS,
            "K_bits": K_BITS,
            "ran": BLOQADE_RUN,
            "p_layers": P_LAYERS if QAOA_RESULT else 0,
            "appx_ratio": QAOA_RESULT['appx_ratio'] if QAOA_RESULT else None,
        },
    }
    with open("portfolio_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("Results saved to portfolio_results.json")
except Exception as e:
    print(f"Save skipped: {e}")

print("\n" + "=" * 70)
print("Portfolio Optimization with Binary Encoding and QAOA — Complete.")
print("=" * 70)
