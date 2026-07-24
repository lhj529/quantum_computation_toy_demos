#!/usr/bin/env python3
"""
CVRP QAOA Solver — Bloqade QASM2 + Qiskit Simulation
=====================================================================

Strict QAOA implementation for Capacitated Vehicle Routing Problem (CVRP).

Pipeline:
  1. Parse .vrp file (VRPLIB format)
  2. Enumerate feasible routes → compact set-partitioning QUBO
  3. QUBO → Ising Hamiltonian (Z + ZZ terms)
  4. Build QAOA circuit (Bloqade QASM2 sequential + SIMD parallel)
  5. Optimise (γ, β) via COBYLA + Qiskit Aer statevector simulation
  6. Decode bitstring → CVRP routes with greedy infeasibility repair

Dimensionality reduction:
  - Compact route-enumeration (avoids O(n^2) MTZ integer variables)
  - Route count capping (max_routes)
  - Smart filtering: "top_efficient" / "single_pair" / "all"
  - Auto-switch to sampling mode for >28 qubits


Usage:
  python cvrp_qaoa_bloqade.py                  # run with default settings below
  python cvrp_qaoa_bloqade.py --help           # show all CLI options
"""

from __future__ import annotations

import itertools
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

# ======================================================================
#  0. Configuration
# ======================================================================
# penalty ≈ （1.5~2） × max(route_cost)
# ── VRP file  ──
VRP_FILE = "P-n16-k8-micro.vrp"

# ── QAOA parameters ──
P_LAYERS = 6              # QAOA depth
MAXITER = 150              # COBYLA max iterations
SEED = 42                 # random seed
SHOTS = 4096              # shots for final sampling

# ── Problem encoding ──
PENALTY = None              # QUBO constraint penalty
MAX_ROUTES = 256          # max feasible routes (dimensionality reduction)
ROUTE_FILTER = "top_efficient"  # "top_efficient" | "single_pair" | "all"

# ── Display ──
SHOW_BLOQADE = True       # print Bloqade QASM2 circuit
SHOW_PLOTS = True         # save convergence + distribution plots

# ======================================================================
#  1. Environment checks
# ======================================================================

_HAS_BLOQADE, _HAS_QISKIT = False, False

try:
    from bloqade import qasm2  # noqa: F401
    import kirin
    from kirin.dialects import ilist
    _HAS_BLOQADE = True
except Exception:
    print("[WARN] bloqade not available – Bloqade circuit display disabled.")

try:
    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit.primitives import BackendSamplerV2
    _HAS_QISKIT = True
except Exception as exc:
    print(f"[WARN] qiskit/aer not available – simulation disabled: {exc}")
    sys.exit(1)


# ======================================================================
#  2. CVRP data structures & VRP file parser
# ======================================================================

@dataclass
class CVRPInstance:
    name: str
    dimension: int
    capacity: int
    num_vehicles: int
    edge_weight_type: str = "EUC_2D"
    node_coords: list[tuple[int, int]] = field(default_factory=list)
    demands: list[int] = field(default_factory=list)
    depot: int = 0
    optimal_value: float | None = None


def parse_vrp_file(path: str | Path) -> CVRPInstance:
    """Parse a standard VRPLIB/CVRPLIB .vrp file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CVRP instance not found: {path}")

    content = path.read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    name, dimension, capacity, edge_weight_type, optimal_value = "", 0, 0, "EUC_2D", None
    in_node_coord, in_demand, in_depot = False, False, False
    node_coords: dict[int, tuple[int, int]] = {}
    demands: dict[int, int] = {}
    depots: list[int] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("NODE_COORD_SECTION"):
            in_node_coord, in_demand, in_depot = True, False, False; continue
        elif upper.startswith("DEMAND_SECTION"):
            in_demand, in_node_coord, in_depot = True, False, False; continue
        elif upper.startswith("DEPOT_SECTION"):
            in_depot, in_node_coord, in_demand = True, False, False; continue
        elif upper.startswith("EOF"):
            break
        if ":" in line and not (in_node_coord or in_demand or in_depot):
            key, _, val = line.partition(":")
            key, val = key.strip().upper(), val.strip()
            if key == "NAME":       name = val
            elif key == "DIMENSION": dimension = int(val)
            elif key == "CAPACITY":  capacity = int(val)
            elif key == "EDGE_WEIGHT_TYPE": edge_weight_type = val
            elif key == "COMMENT":
                m = re.search(r"(?:Optimal value|Best known)[:\s]*(\d+\.?\d*)", val, re.IGNORECASE)
                if m: optimal_value = float(m.group(1))
            continue
        if in_node_coord:
            parts = line.split()
            if len(parts) >= 3: node_coords[int(parts[0])] = (int(parts[1]), int(parts[2]))
        elif in_demand:
            parts = line.split()
            if len(parts) >= 2: demands[int(parts[0])] = int(parts[1])
        elif in_depot:
            for p in line.split():
                try:
                    v = int(p)
                    if v == -1: break
                    depots.append(v - 1 if v > 0 else v)
                except ValueError: pass

    n = dimension or len(node_coords) or len(demands)
    coords_list = [node_coords.get(i + 1, (0, 0)) for i in range(n)]
    demands_list = [demands.get(i + 1, 0) for i in range(n)]
    depot = depots[0] if depots else 0
    total_demand = sum(demands_list)
    estimated_k = max(1, (total_demand + capacity - 1) // capacity) if capacity > 0 else 1
    kv = re.search(r'k(\d+)', name, re.IGNORECASE)
    num_vehicles = int(kv.group(1)) if kv else estimated_k

    return CVRPInstance(name=name, dimension=n, capacity=capacity,
                        num_vehicles=num_vehicles, edge_weight_type=edge_weight_type,
                        node_coords=coords_list, demands=demands_list,
                        depot=depot, optimal_value=optimal_value)


# ======================================================================
#  3. Distance & TSP tour cost
# ======================================================================

def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1]) # Euclid Distance


def _tsp_tour_cost(customers: tuple[int, ...], depot: int,
                   coords: list[tuple[int, int]]) -> float:
    """Optimal TSP tour cost for visiting *customers* (exhaustive)."""
    if len(customers) == 0: return 0.0
    if len(customers) == 1: return 2.0 * _dist(coords[depot], coords[customers[0]])
    best = float("inf")
    for perm in itertools.permutations(customers):
        cost = 0.0; prev = depot
        for c in perm:
            cost += _dist(coords[prev], coords[c]); prev = c
        cost += _dist(coords[prev], coords[depot])
        if cost < best: best = cost
    return best


# ======================================================================
#  4. Route enumeration with smart filtering (dimensionality reduction)
# ======================================================================

@dataclass
class Route:
    route_id: int
    customers: tuple[int, ...]
    demand: float
    cost: float
    efficiency: float = 0.0


def enumerate_feasible_routes(instance: CVRPInstance, max_routes: int = 256,
                               filter_mode: str = "top_efficient") -> list[Route]:
    """
    Enumerate all feasible customer subsets with total demand ≤ capacity.
    filter_mode: "all" | "top_efficient" | "single_pair"
    """
    customers = [i for i in range(instance.dimension) if i != instance.depot]
    Q, depot = instance.capacity, instance.depot
    routes: list[Route] = []

    if filter_mode == "single_pair":
        for c in customers:
            if instance.demands[c] <= Q:
                cost = _tsp_tour_cost((c,), depot, instance.node_coords)
                routes.append(Route(route_id=len(routes), customers=(c,),
                                    demand=instance.demands[c], cost=cost,
                                    efficiency=cost / instance.demands[c] if instance.demands[c] > 0 else 1e9))
        for c1, c2 in itertools.combinations(customers, 2):
            d = instance.demands[c1] + instance.demands[c2]
            if d <= Q:
                cost = _tsp_tour_cost((c1, c2), depot, instance.node_coords)
                routes.append(Route(route_id=len(routes), customers=(c1, c2),
                                    demand=d, cost=cost, efficiency=cost / d if d > 0 else 1e9))
        if len(routes) > max_routes:
            routes.sort(key=lambda r: r.efficiency)
            routes = routes[:max_routes]
        return routes

    # Enumerate ALL feasible subsets first (no early break)
    total_subsets = sum(1 for r in range(1, len(customers) + 1)
                        for _ in itertools.combinations(customers, r))
    if total_subsets > 200_000:
        print(f"  [WARN] {total_subsets} subsets to check; may be slow. "
              f"Consider 'single_pair' filter or reducing customer count.")
    checked = 0
    for r in range(1, len(customers) + 1):
        for subset in itertools.combinations(customers, r):
            checked += 1
            total_demand = sum(instance.demands[c] for c in subset)
            if total_demand <= Q:
                cost = _tsp_tour_cost(subset, depot, instance.node_coords)
                eff = cost / total_demand if total_demand > 0 else 1e9
                routes.append(Route(route_id=len(routes), customers=subset,
                                    demand=total_demand, cost=cost, efficiency=eff))

    # Sort by efficiency and truncate AFTER full enumeration
    if filter_mode == "top_efficient" and len(routes) > max_routes:
        routes.sort(key=lambda r: r.efficiency)
        routes = routes[:max_routes]

    # Sanity check: every customer must be coverable by at least one route
    covered = set()
    for r in routes:
        covered.update(r.customers)
    missing = [c for c in customers if c not in covered]
    if missing:
        raise ValueError(
            f"Route enumeration cannot cover customer(s) {missing}. "
            f"Try increasing max_routes (currently {max_routes}) or using 'all' filter."
        )

    return routes


# ======================================================================
#  5. Direct QUBO construction (compact set-partitioning)
# ======================================================================

@dataclass
class QUBO:
    matrix: np.ndarray
    offset: float = 0.0
    var_names: list[str] = field(default_factory=list)

    @property
    def num_vars(self) -> int: return self.matrix.shape[0]

    def evaluate(self, x: np.ndarray | list[int]) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.matrix @ x) + self.offset


@dataclass
class IsingTerms:
    num_qubits: int
    linear: dict[int, float]
    quadratic: dict[tuple[int, int], float]
    offset: float = 0.0


def _auto_penalty(routes: list[Route], factor: float = 4.0) -> float:
    """Compute penalty as factor * max_route_cost.

    Rationale: the largest penalty term from violating a single customer
    constraint is P * 1² = P.  The most you could save by dropping that
    customer is max_route_cost.  So P > max_route_cost is sufficient,
    and factor=2.0 gives a safety margin without drowning the objective.
    """
    if not routes: return 100.0
    max_cost = max(r.cost for r in routes)
    return max(factor * max_cost, 1.0)


def build_cvrp_qubo(instance: CVRPInstance, routes: list[Route],
                     penalty: float | None = None,
                     num_vehicles: int | None = None) -> QUBO:
    """
    Build QUBO for CVRP compact set-partitioning.

    Variables: y_0…y_{R-1} (route select), s_0…s_{S-1} (slack for ≤K).
    Objective: min Σ cost_r · y_r
    Constraints → quadratic penalties:
      (C1) Σ_{r∋c} y_r = 1  (each customer covered exactly once)
      (C2) Σ y_r + Σ 2^s · s_s = K  (vehicle count, slack-encoded ≤)

    If penalty is None,then autompute the penalty.
    """
    K = num_vehicles if num_vehicles is not None else instance.num_vehicles
    customers = [i for i in range(instance.dimension) if i != instance.depot]
    R = len(routes)

    if penalty is None:
        penalty = _auto_penalty(routes)
    penalty = float(penalty)

    # Slack bits for ≤K constraint
    max_slack = max(0, K)
    S = max(1, int(math.ceil(math.log2(max_slack + 1)))) if max_slack > 0 else 1
    N = R + S
    var_names = [f"y_{routes[r].route_id}" for r in range(R)] + [f"slack_{s}" for s in range(S)]

    Q = np.zeros((N, N), dtype=float)
    offset = 0.0
    K_f = float(K)

    # Objective: min Σ cost_r · y_r
    for r, route in enumerate(routes):
        Q[r, r] += route.cost

    # Constraint (C1): each customer exactly once
    for c in customers:
        covering = [r for r, route in enumerate(routes) if c in route.customers]
        for r in covering:
            Q[r, r] += penalty * (1.0 - 2.0)
        for i in range(len(covering)):
            for j in range(i + 1, len(covering)):
                ri, rj = covering[i], covering[j]
                if ri < rj: Q[ri, rj] += 2.0 * penalty
                else:       Q[rj, ri] += 2.0 * penalty
        offset += penalty

    # Constraint (C2): Σ y_r + Σ 2^s s_s = K
    for r in range(R):
        Q[r, r] += penalty * (1.0 - 2.0 * K_f)
    for r in range(R):
        for rp in range(r + 1, R):
            Q[r, rp] += penalty * 2.0
    for s in range(S):
        idx = R + s; coeff = 2.0 ** s
        Q[idx, idx] += penalty * coeff * coeff
        Q[idx, idx] += penalty * (-2.0 * K_f * coeff)
    for s1 in range(S):
        for s2 in range(s1 + 1, S):
            Q[R + s1, R + s2] += penalty * 2.0 * (2.0 ** s1) * (2.0 ** s2)
    for r in range(R):
        for s in range(S):
            idx_s, coeff = R + s, 2.0 ** s
            if r < idx_s: Q[r, idx_s] += penalty * 2.0 * coeff
            else:          Q[idx_s, r] += penalty * 2.0 * coeff
    offset += penalty * K_f * K_f

    return QUBO(matrix=Q, offset=offset, var_names=var_names)


# ======================================================================
#  6. QUBO → Ising conversion
# ======================================================================

def qubo_to_ising(qubo: QUBO) -> IsingTerms:
    """Convert QUBO to Ising: x = (1 - s)/2,  s ∈ {+1, -1}."""
    n, Q = qubo.num_vars, qubo.matrix
    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}
    offset = 0.0
    for i in range(n):
        linear[i] = linear.get(i, 0.0) - 0.5 * Q[i, i]
        offset += 0.5 * Q[i, i]
        for j in range(i + 1, n):
            w = Q[i, j]
            if abs(w) < 1e-15: continue
            c_zz = w / 4.0
            quadratic[(i, j)] = quadratic.get((i, j), 0.0) + c_zz
            linear[i] = linear.get(i, 0.0) - c_zz
            linear[j] = linear.get(j, 0.0) - c_zz
            offset += c_zz
    return IsingTerms(num_qubits=n,
                      linear={k: v for k, v in linear.items() if abs(v) > 1e-15},
                      quadratic={k: v for k, v in quadratic.items() if abs(v) > 1e-15},
                      offset=offset + qubo.offset)


# ======================================================================
#  7. Bloqade QAOA circuits (QASM2 extended dialect)
# ======================================================================

def build_bloqade_qaoa_sequential(ising: IsingTerms, p_layers: int = 2):
    """Build sequential QAOA circuit with Bloqade QASM2."""
    if not _HAS_BLOQADE: raise ImportError("bloqade required.")

    n = ising.num_qubits
    quad_items = sorted(ising.quadratic.items(), key=lambda x: (x[0][0], x[0][1]))
    linear_items = sorted(ising.linear.items(), key=lambda x: x[0])

    # Pre-build flat lists (native Python int/float for kirin compatibility)
    quad_i = [int(e[0][0]) for e in quad_items]
    quad_j = [int(e[0][1]) for e in quad_items]
    quad_w = [float(e[1]) for e in quad_items]
    lin_idx = [int(e[0]) for e in linear_items]
    lin_w = [float(e[1]) for e in linear_items]
    n_quad, n_lin = len(quad_items), len(linear_items)

    @qasm2.extended
    def kernel(gamma: ilist.IList[float, Any], beta: ilist.IList[float, Any]):
        q = qasm2.qreg(n)
        for i in range(n):
            qasm2.h(q[i])
        for layer in range(len(gamma)):
            g_val = gamma[layer]
            # Cost: ZZ terms
            for idx in range(n_quad):
                qi = quad_i[idx]
                qj = quad_j[idx]
                w = quad_w[idx]
                qasm2.cx(q[qi], q[qj])
                qasm2.rz(q[qj], 2.0 * w * g_val)
                qasm2.cx(q[qi], q[qj])
            # Cost: Z terms
            for idx2 in range(n_lin):
                qi2 = lin_idx[idx2]
                w2 = lin_w[idx2]
                qasm2.rz(q[qi2], 2.0 * w2 * g_val)
            # Mixer
            b_val = beta[layer]
            for i2 in range(n):
                qasm2.rx(q[i2], 2.0 * b_val)
        return q
    return kernel


def display_bloqade_circuit(ising: IsingTerms, p_layers: int = 1):
    """Print and emit Bloqade QASM2 circuit."""
    if not _HAS_BLOQADE:
        print("[SKIP] Bloqade not available.")
        return
    print(f"\n{'='*60}")
    print(f"  Bloqade QAOA Circuit (QASM2 Extended, p={p_layers})")
    print(f"  Qubits: {ising.num_qubits}")
    print(f"{'='*60}")
    try:
        kernel_seq = build_bloqade_qaoa_sequential(ising, p_layers)
        print("\n--- Sequential (kirin IR) ---")
        kernel_seq.code.print()
    except Exception as exc:
        print(f"  Sequential circuit error: {exc}")
    try:
        kernel_seq = build_bloqade_qaoa_sequential(ising, p_layers)
        g_list = ilist.IList([0.5])
        b_list = ilist.IList([0.3])

        @qasm2.extended
        def main():
            kernel_seq(g_list, b_list)
        target = qasm2.emit.QASM2()
        ast = target.emit(main)
        print("\n--- Emitted QASM2 (gamma=0.5, beta=0.3) ---")
        qasm2.parse.pprint(ast)
    except Exception as exc:
        print(f"  QASM2 emission error: {exc}")


# ======================================================================
#  8. Qiskit QAOA circuit & simulation
# ======================================================================

def build_qiskit_qaoa_circuit(ising: IsingTerms,
                               p_layers: int = 2) -> tuple[QuantumCircuit, list[Parameter]]:
    """Build standard QAOA circuit with Qiskit (with measurements)."""
    n = ising.num_qubits
    qc = QuantumCircuit(n, n)
    params: list[Parameter] = []
    for i in range(n): qc.h(i)
    for p in range(p_layers):
        gamma_p = Parameter(f"gamma_{p}"); beta_p = Parameter(f"beta_{p}")
        params.extend([gamma_p, beta_p])
        for (i, j), J in sorted(ising.quadratic.items()):
            qc.cx(i, j); qc.rz(2.0 * float(J) * gamma_p, j); qc.cx(i, j)
        for i, h in sorted(ising.linear.items()):
            qc.rz(2.0 * float(h) * gamma_p, i)
        for i in range(n): qc.rx(2.0 * beta_p, i)
    qc.measure(range(n), range(n))
    return qc, params


def _expectation_from_statevector(sv: np.ndarray, ising: IsingTerms) -> float:
    """⟨ψ| H_Ising |ψ⟩ from statevector."""
    probs = np.abs(sv) ** 2; n, total = ising.num_qubits, 0.0
    for bits in range(1 << n):
        p = probs[bits]
        if p < 1e-15: continue
        contrib = sum(h * (-1.0 if (bits >> i) & 1 else 1.0) for i, h in ising.linear.items())
        for (i, j), J in ising.quadratic.items():
            contrib += J * (-1.0 if (bits >> i) & 1 else 1.0) * (-1.0 if (bits >> j) & 1 else 1.0)
        total += p * contrib
    return total + ising.offset


def _expectation_from_samples(counts: dict[str, int], ising: IsingTerms) -> float:
    """⟨H_Ising⟩ from measurement counts."""
    total_shots = sum(counts.values())
    if total_shots == 0: return float("inf")
    n, total = ising.num_qubits, 0.0
    for bitstring, count in counts.items():
        bits = int(bitstring, 2)
        contrib = sum(h * (-1.0 if (bits >> (n - 1 - i)) & 1 else 1.0)
                      for i, h in ising.linear.items())
        for (i, j), J in ising.quadratic.items():
            contrib += J * (-1.0 if (bits >> (n - 1 - i)) & 1 else 1.0) * (-1.0 if (bits >> (n - 1 - j)) & 1 else 1.0)
        total += contrib * count
    return total / total_shots + ising.offset


def optimize_qaoa(ising: IsingTerms, p_layers: int = 2, maxiter: int = 100,
                  shots: int = 4096, seed: int = 42, use_statevector: bool = True,
                  verbose: bool = True) -> dict[str, Any]:
    """Run QAOA optimisation with COBYLA."""
    n = ising.num_qubits
    qc, params = build_qiskit_qaoa_circuit(ising, p_layers)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  QAOA Optimisation  |  Qubits: {n}  Layers: p={p_layers}")
        print(f"  Linear Z: {len(ising.linear)}  Quadratic ZZ: {len(ising.quadratic)}")
        print(f"  COBYLA  maxiter={maxiter}  {'statevector' if use_statevector else f'sampler({shots})'}")
        print(f"{'='*60}")

    qc_no_meas = qc.remove_final_measurements(inplace=False)
    history: list[dict] = []
    eval_count, best_val, best_x = [0], [float("inf")], [None]
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 2 * np.pi, size=len(params))
    simulator = AerSimulator(method="matrix_product_state")
    sampler = BackendSamplerV2(backend=simulator, options={"default_shots": shots})

    def cost_fn(theta: np.ndarray) -> float:
        eval_count[0] += 1
        bind = {p: float(theta[idx]) for idx, p in enumerate(params)}
        if use_statevector:
            bound = qc_no_meas.assign_parameters(bind)
            sv = Statevector.from_instruction(bound)
            energy = _expectation_from_statevector(sv.data, ising)
        else:
            job = sampler.run([qc.assign_parameters(bind)])
            counts = job.result()[0].data.c.get_int_counts()
            energy = _expectation_from_samples({format(k, f"0{n}b"): v for k, v in counts.items()}, ising)
        if energy < best_val[0]:
            best_val[0] = energy; best_x[0] = theta.copy()
        history.append({"iteration": eval_count[0], "energy": energy, "best_energy": best_val[0]})
        if verbose and (eval_count[0] <= 3 or eval_count[0] % 20 == 0):
            print(f"  iter {eval_count[0]:4d} | energy = {energy:.6f} | best = {best_val[0]:.6f}")
        return energy

    t0 = time.time()
    result = minimize(cost_fn, x0, method="COBYLA",
                      options={"maxiter": maxiter, "rhobeg": 0.4})
    elapsed = time.time() - t0

    if verbose:
        print(f"\n  Optimisation done ({elapsed:.1f}s)  status={result.message}")
        print(f"  Best energy: {best_val[0]:.6f}")

    # Final sampling
    final_bind = {p: float(best_x[0][idx]) for idx, p in enumerate(params)}
    job = sampler.run([qc.assign_parameters(final_bind)])
    final_counts = job.result()[0].data.c.get_int_counts()
    total_shots = sum(final_counts.values())
    best_bits, best_e = None, float("inf")
    for bi, cnt in final_counts.items():
        s = format(bi, f"0{n}b")
        e = _expectation_from_samples({s: 1}, ising)
        if e < best_e: best_e = e; best_bits = s

    if verbose:
        print(f"  Best bitstring: {best_bits}  (energy={best_e:.6f})")
        print(f"  Top 10 counts:")
        for bi, cnt in sorted(final_counts.items(), key=lambda x: -x[1])[:10]:
            s = format(bi, f"0{n}b")
            print(f"    {s} : {cnt:5d} ({100.0*cnt/total_shots:5.1f}%)  E={_expectation_from_samples({s:1}, ising):.4f}")

    return {"best_bitstring": best_bits, "best_energy": best_e,
            "best_theta": [float(x) for x in best_x[0]],
            "optimizer_result": result, "history": history,
            "final_counts": final_counts, "elapsed_sec": elapsed,
            "num_qubits": n, "p_layers": p_layers}


# ======================================================================
#  9. Solution decoding + greedy repair
# ======================================================================

def decode_cvrp_solution(bitstring: str, routes: list[Route], customers: list[int],
                          num_vehicles: int, num_slack_bits: int, depot: int = 0,
                          instance: CVRPInstance | None = None) -> dict[str, Any]:
    """Decode a QAOA bitstring into CVRP routes."""
    R = len(routes); bits = [int(ch) for ch in bitstring]
    y_bits, slack_bits = bits[:R], bits[R:R + num_slack_bits]
    selected = [r for r, b in enumerate(y_bits) if b == 1]
    selected_routes = [routes[r] for r in selected]
    coverage: dict[int, int] = {}
    for r_idx in selected:
        for c in routes[r_idx].customers:
            coverage[c] = coverage.get(c, 0) + 1
    uncovered = [c for c in customers if coverage.get(c, 0) == 0]
    overcovered = [c for c in customers if coverage.get(c, 0) > 1]
    feasible = (len(uncovered) == 0 and len(overcovered) == 0
                and len(selected_routes) <= num_vehicles)
    total_cost = sum(r.cost for r in selected_routes)
    route_details = []
    for r in selected_routes:
        nodes = [depot] + list(r.customers) + [depot]
        route_details.append({"route_one_based": [x + 1 for x in nodes],
                              "customers": list(r.customers),
                              "demand": r.demand, "cost": r.cost})
    return {"feasible": feasible, "total_cost": total_cost,
            "num_routes_used": len(selected_routes),
            "num_vehicles_available": num_vehicles, "routes": route_details,
            "uncovered_customers": uncovered, "overcovered_customers": overcovered,
            "selected_route_indices": selected, "raw_bitstring": bitstring,
            "y_bits": y_bits, "slack_bits": slack_bits}


def repair_cvrp_solution(decoded: dict, routes: list[Route], customers: list[int],
                          num_vehicles: int, num_slack_bits: int,
                          instance: CVRPInstance) -> dict[str, Any]:
    """Greedy repair: remove overcoverage, add uncovered, merge if too many routes."""
    if decoded["feasible"]: return decoded
    selected = set(decoded["selected_route_indices"])
    all_routes = {r.route_id: r for r in routes}

    # Coverage map
    def rebuild_coverage(sel):
        cov: dict[int, list[int]] = {}
        for rid in sel:
            for c in all_routes[rid].customers:
                cov.setdefault(c, []).append(rid)
        return cov

    cov = rebuild_coverage(selected)
    # Step 1: Resolve overcoverage (keep cheapest per customer)
    for c, rids in list(cov.items()):
        if len(rids) > 1:
            best_rid = min(rids, key=lambda rid: all_routes[rid].cost)
            for rid in rids:
                if rid != best_rid and rid in selected: selected.discard(rid)
    cov = rebuild_coverage(selected)

    # Step 2: Cover uncovered customers
    for c in customers:
        if c not in cov:
            cands = [r for r in routes if c in r.customers and r.route_id not in selected]
            if cands:
                cands.sort(key=lambda r: r.cost)
                for cand in cands:
                    already = {cc for rid in selected for cc in all_routes[rid].customers}
                    if c not in already:
                        selected.add(cand.route_id); break
    cov = rebuild_coverage(selected)

    # Step 3: Merge if too many routes (simple: best pair merge)
    while len(selected) > num_vehicles:
        sl = list(selected); merged = False
        best_saving = 0.0; best_merge = None
        for i in range(len(sl)):
            for j in range(i + 1, len(sl)):
                r1, r2 = all_routes[sl[i]], all_routes[sl[j]]
                combined = tuple(sorted(set(r1.customers + r2.customers)))
                dem = sum(instance.demands[c] for c in combined)
                if dem <= instance.capacity:
                    new_cost = _tsp_tour_cost(combined, instance.depot, instance.node_coords)
                    saving = (r1.cost + r2.cost) - new_cost
                    if saving > best_saving:
                        best_saving = saving; best_merge = (sl[i], sl[j], combined, dem, new_cost)
        if best_merge:
            selected.discard(best_merge[0]); selected.discard(best_merge[1])
            # Append new route to the original routes list so its ID stays valid
            new_id = len(routes)
            new_route = Route(route_id=new_id, customers=best_merge[2],
                              demand=best_merge[3], cost=best_merge[4],
                              efficiency=best_merge[4]/best_merge[3] if best_merge[3] > 0 else 1e9)
            routes.append(new_route)
            all_routes[new_id] = new_route
            selected.add(new_id); merged = True
        if not merged: break

    # Final assembly
    sel_routes = [all_routes[rid] for rid in selected]
    cov_final = rebuild_coverage(selected)
    uncovered_f = [c for c in customers if c not in cov_final]
    overcovered_f = [c for c, rs in cov_final.items() if len(rs) > 1]
    rd = []
    for r in sel_routes:
        nodes = [instance.depot] + list(r.customers) + [instance.depot]
        rd.append({"route_one_based": [x + 1 for x in nodes],
                   "customers": list(r.customers), "demand": r.demand, "cost": r.cost})
    return {"feasible": len(uncovered_f) == 0 and len(overcovered_f) == 0
                         and len(sel_routes) <= num_vehicles,
            "total_cost": sum(r.cost for r in sel_routes),
            "total_demand": sum(r.demand for r in sel_routes),
            "num_routes_used": len(sel_routes), "num_vehicles_available": num_vehicles,
            "routes": rd, "uncovered_customers": uncovered_f,
            "overcovered_customers": overcovered_f,
            "selected_route_indices": [r.route_id for r in sel_routes],
            "repaired": True, "raw_bitstring": decoded["raw_bitstring"]}


# ======================================================================
# 10. Brute-force exhaustive baseline
# ======================================================================

def solve_cvrp_brute_force(instance: CVRPInstance, routes: list[Route],
                            num_slack_bits: int, verbose: bool = True
                            ) -> dict[str, Any]:
    """Enumerate all 2^R route subsets to find the global CVRP optimum.

    Returns the optimal solution and a ranked list of all feasible solutions.
    This is the ground truth for QAOA comparison.
    """
    customers = [i for i in range(instance.dimension) if i != instance.depot]
    K, R, S = instance.num_vehicles, len(routes), num_slack_bits
    best_cost, best_sol = float("inf"), None
    all_feasible: list[dict] = []
    total_combos = 1 << R

    for mask in range(1, total_combos):
        selected = [r for r in range(R) if (mask >> r) & 1]
        if len(selected) > K:
            continue

        # Check exact coverage: each customer exactly once
        covered: dict[int, int] = {}
        total_cost, ok = 0.0, True
        for r_idx in selected:
            for c in routes[r_idx].customers:
                covered[c] = covered.get(c, 0) + 1
            total_cost += routes[r_idx].cost

        if len(covered) != len(customers):
            continue
        if any(cnt != 1 for cnt in covered.values()):
            continue

        # Build corresponding bitstring for QUBO evaluation
        bits = [1 if r in selected else 0 for r in range(R)]
        slack_val = K - len(selected)
        for sb in range(S):
            bits.append((slack_val >> sb) & 1)

        all_feasible.append({
            "selected_routes": selected,
            "num_routes": len(selected),
            "total_cost": total_cost,
            "bitstring": "".join(str(b) for b in bits),
        })

        if total_cost < best_cost:
            best_cost = total_cost
            best_sol = all_feasible[-1]

    # Sort by cost ascending
    all_feasible.sort(key=lambda s: s["total_cost"])

    if verbose and all_feasible:
        print(f"\n{'='*60}")
        print(f"  Brute-Force Baseline  ({len(all_feasible)} feasible solutions "
              f"out of {total_combos} combos)")
        print(f"{'='*60}")
        print(f"  Optimal cost: {best_cost:.2f}  "
              f"Routes used: {best_sol['num_routes']}/{K}")
        print(f"  Bitstring: {best_sol['bitstring']}")
        top_n = min(5, len(all_feasible))
        print(f"  Top {top_n} feasible solutions:")
        for i, sol in enumerate(all_feasible[:top_n]):
            marker = " ← OPTIMAL" if i == 0 else ""
            sel_names = [f"r{ri}" for ri in sol["selected_routes"]]
            print(f"    #{i+1}: cost={sol['total_cost']:.2f}  "
                  f"routes=[{', '.join(sel_names)}]  "
                  f"bits={sol['bitstring']}{marker}")

    return {"optimal": best_sol, "all_feasible": all_feasible,
            "num_feasible": len(all_feasible), "total_combos": total_combos}


# ======================================================================
# 11. Plotting
# ======================================================================

def plot_convergence(history: list[dict], output_path: str):
    iters = [h["iteration"] for h in history]
    energies = [h["energy"] for h in history]
    bests = [h["best_energy"] for h in history]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, energies, "o", alpha=0.3, markersize=3, label="Function eval")
    ax.plot(iters, bests, "-", color="tab:red", linewidth=1.5, label="Best so far")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Energy (Ising expectation)")
    ax.set_title("QAOA Optimisation Convergence (COBYLA)")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"  Convergence plot saved to: {output_path}")


def plot_result_distribution(counts: dict[int, int], ising: IsingTerms,
                              top_k: int = 20, output_path: str | None = None):
    n = ising.num_qubits
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])[:top_k]
    total = sum(counts.values())
    bitstrings, probs, energies = [], [], []
    best_e, best_idx = float("inf"), -1
    for idx, (bi, cnt) in enumerate(sorted_items):
        s = format(bi, f"0{n}b")
        e = _expectation_from_samples({s: 1}, ising)
        bitstrings.append(s); probs.append(100.0 * cnt / total); energies.append(e)
        if e < best_e: best_e = e; best_idx = idx
    colors = ["tab:purple" if i == best_idx else "tab:grey" for i in range(len(sorted_items))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(range(len(bitstrings)), probs, color=colors)
    ax1.set_xticks(range(len(bitstrings)))
    ax1.set_xticklabels(bitstrings, rotation=60, ha="right", fontsize=7)
    ax1.set_ylabel("Probability (%)"); ax1.set_title(f"Top {top_k} Bitstrings")
    ax1.grid(True, alpha=0.3, axis="y")
    ax2.bar(range(len(energies)), energies, color=colors)
    ax2.set_xticks(range(len(bitstrings)))
    ax2.set_xticklabels(bitstrings, rotation=60, ha="right", fontsize=7)
    ax2.set_ylabel("Ising Energy"); ax2.set_title("Corresponding Ising Energies")
    ax2.axhline(y=best_e, color="tab:red", linestyle="--", alpha=0.5,
                label=f"Best: {best_e:.4f}")
    ax2.legend(); ax2.grid(True, alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f"  Distribution plot saved to: {output_path}")


# ======================================================================
# 12. Main solver
# ======================================================================

def solve_cvrp_qaoa(vrp_path: str, p_layers: int = 2, maxiter: int = 60,
                     penalty: float | None = None, max_routes: int = 256,
                     route_filter: str = "top_efficient",
                     use_statevector: bool = True, shots: int = 4096,
                     seed: int = 42, show_bloqade: bool = True,
                     show_plots: bool = True, verbose: bool = True) -> dict[str, Any]:
    """End-to-end CVRP QAOA solver."""

    # 1. Load instance
    instance = parse_vrp_file(vrp_path)
    customers = [i for i in range(instance.dimension) if i != instance.depot]
    if verbose:
        print(f"\n{'#'*60}")
        print(f"  CVRP QAOA Solver — Bloqade QASM2 + Qiskit Simulation")
        print(f"  Instance: {instance.name}  ({len(customers)} customers + 1 depot)")
        print(f"  Vehicles: {instance.num_vehicles}  Capacity: {instance.capacity}")
        if instance.optimal_value: print(f"  Known optimum: {instance.optimal_value}")
        print(f"{'#'*60}")

    # 2. Enumerate routes
    routes = enumerate_feasible_routes(instance, max_routes=max_routes,
                                        filter_mode=route_filter)
    R = len(routes)
    if verbose:
        print(f"\n  Feasible routes: {R}  (filter={route_filter}, cap={max_routes})")

    # 3. Build QUBO
    qubo = build_cvrp_qubo(instance, routes, penalty=penalty,
                            num_vehicles=instance.num_vehicles)
    K = instance.num_vehicles
    S = max(1, int(math.ceil(math.log2(K + 1)))) if K > 0 else 1
    N = qubo.num_vars
    if verbose:
        max_cost = max(r.cost for r in routes) if routes else 0
        eff_penalty = _auto_penalty(routes) if penalty is None else penalty
        print(f"  QUBO vars: {N} ({R} route + {S} slack)  "
              f"penalty={eff_penalty:.1f}  (max route cost={max_cost:.1f})")
        if N > 28:
            print(f"  [WARN] {N} qubits > 28; consider reducing max_routes or using single_pair filter.")

    # 4. QUBO → Ising
    ising = qubo_to_ising(qubo)
    if verbose:
        print(f"  Ising: {ising.num_qubits} qubits, "
              f"{len(ising.linear)} Z + {len(ising.quadratic)} ZZ, "
              f"offset={ising.offset:.2f}")

    # 5. Display Bloqade circuit
    if show_bloqade and N <= 16:
        display_bloqade_circuit(ising, p_layers=min(p_layers, 1))

    # 6. Optimise QAOA
    effective_sv = use_statevector
    if effective_sv and ising.num_qubits > 28:
        if verbose: print(f"\n  [WARN] {ising.num_qubits} qubits > 28: forcing sampling mode.")
        effective_sv = False

    opt_result = optimize_qaoa(ising, p_layers=p_layers, maxiter=maxiter,
                                shots=shots, seed=seed, use_statevector=effective_sv,
                                verbose=verbose)

    # 7. Decode solution
    solution = decode_cvrp_solution(opt_result["best_bitstring"], routes, customers,
                                     num_vehicles=instance.num_vehicles,
                                     num_slack_bits=S, depot=instance.depot,
                                     instance=instance)
    # 7b. Repair if infeasible
    repaired = False
    if not solution["feasible"]:
        repaired = True
        if verbose:
            print(f"\n  ⚠ QAOA bitstring is INFEASIBLE — activating greedy repair.")
            print(f"    Uncovered customers: {solution.get('uncovered_customers', [])}")
            print(f"    Overcovered customers: {solution.get('overcovered_customers', [])}")
        solution_before = solution
        solution = repair_cvrp_solution(solution, routes, customers,
                                         num_vehicles=instance.num_vehicles,
                                         num_slack_bits=S, instance=instance)
        # Re-evaluate Ising energy of repaired solution
        if solution["feasible"]:
            rep_bits = [0] * R
            for rid in solution["selected_route_indices"]:
                if rid < R:
                    rep_bits[rid] = 1
            rep_bits.extend([0] * S)
            used = solution["num_routes_used"]
            slack_val = instance.num_vehicles - used
            for sb in range(S):
                rep_bits[R + sb] = (slack_val >> sb) & 1
            rep_bitstr = "".join(str(b) for b in rep_bits)
            rep_ising_e = _expectation_from_samples({rep_bitstr: 1}, ising)
            if verbose:
                print(f"  Repaired Ising energy: {rep_ising_e:.4f} "
                      f"(QAOA best: {opt_result['best_energy']:.4f})")
                print(f"  Cost before repair: {solution_before['total_cost']:.2f} → "
                      f"after: {solution['total_cost']:.2f}")
        else:
            if verbose:
                print(f"  ✗ Repair FAILED — solution remains infeasible.")

    if verbose:
        tag = " (REPAIRED)" if repaired else ""
        print(f"\n{'='*60}")
        print(f"  CVRP Solution{tag}")
        print(f"{'='*60}")
        print(f"  Feasible: {solution['feasible']}  |  "
              f"Cost: {solution['total_cost']:.2f}  |  "
              f"Routes: {solution['num_routes_used']}/{instance.num_vehicles}")
        if solution.get("uncovered_customers"):
            print(f"  [WARN] Uncovered: {solution['uncovered_customers']}")
        if solution.get("overcovered_customers"):
            print(f"  [WARN] Overcovered: {solution['overcovered_customers']}")
        if instance.optimal_value:
            print(f"  Approx ratio: {solution['total_cost'] / instance.optimal_value:.4f}")
        print(f"\n  Routes:")
        for rd in solution["routes"]:
            print(f"    {rd['route_one_based']}  demand={rd['demand']:.0f}  cost={rd['cost']:.2f}")

    # 8. Brute-force baseline comparison
    baseline = solve_cvrp_brute_force(instance, routes, num_slack_bits=S, verbose=verbose)

    if verbose and baseline["optimal"]:
        opt = baseline["optimal"]
        qaoa_cost = solution["total_cost"]
        gap = (qaoa_cost - opt["total_cost"]) / opt["total_cost"] * 100.0 if opt["total_cost"] > 0 else 0.0
        qaoa_rank = next((i + 1 for i, s in enumerate(baseline["all_feasible"])
                          if abs(s["total_cost"] - qaoa_cost) < 1e-6), None)

        print(f"\n{'='*60}")
        print(f"  QAOA vs Brute-Force")
        print(f"{'='*60}")
        print(f"  QAOA cost    : {qaoa_cost:.2f}")
        print(f"  Optimal cost : {opt['total_cost']:.2f}")
        print(f"  Gap          : {gap:+.2f}%")
        print(f"  QAOA rank    : #{qaoa_rank}/{baseline['num_feasible']}"
              if qaoa_rank else f"  QAOA rank    : not in feasible set")

        if repaired:
            print(f"\n  ⚠  RESULT CAME FROM GREEDY REPAIR, NOT QAOA.")
            print(f"     The QAOA bitstring was infeasible. The solution above was")
            print(f"     constructed by a classical heuristic, not by the quantum algorithm.")
            print(f"     → Try: increase penalty, add p_layers, or increase maxiter.")
        elif gap < 1e-6:
            print(f"  ✓ QAOA found the global optimum.")
        elif gap < 5.0:
            print(f"  ~ QAOA near-optimal (within 5%).")
            print(f"    → Try: increase p_layers or maxiter to close the gap.")
        else:
            print(f"  ✗ QAOA significantly suboptimal (gap > 5%).")
            print(f"    → Try: increase p_layers, tune --penalty, or use more shots.")

    # 9. Plots
    if show_plots:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base = os.path.splitext(os.path.basename(vrp_path))[0]
        plot_convergence(opt_result["history"],
                         os.path.join(script_dir, f"{base}_convergence_p{p_layers}.png"))
        plot_result_distribution(opt_result["final_counts"], ising, top_k=20,
                                  output_path=os.path.join(script_dir, f"{base}_distribution_p{p_layers}.png"))

    return {"instance": instance, "routes": routes, "qubo": qubo, "ising": ising,
            "optimization": opt_result, "solution": solution,
            "baseline": baseline, "num_qubits": N, "p_layers": p_layers}


# ======================================================================
# 13. Main
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CVRP QAOA Solver — Bloqade QASM2 + Qiskit Simulation")
    parser.add_argument("--vrp", default=None,
                        help=f"Path to .vrp file (default: {VRP_FILE} in script directory)")
    parser.add_argument("--p", type=int, default=P_LAYERS,
                        help=f"QAOA layers (default: {P_LAYERS})")
    parser.add_argument("--maxiter", type=int, default=MAXITER,
                        help=f"COBYLA max iterations (default: {MAXITER})")
    parser.add_argument("--penalty", type=float, default=PENALTY,
                        help=f"QUBO penalty (default: auto = 2*max_cost*N)")
    parser.add_argument("--max-routes", type=int, default=MAX_ROUTES,
                        help=f"Max routes (default: {MAX_ROUTES})")
    parser.add_argument("--route-filter", default=ROUTE_FILTER,
                        choices=["top_efficient", "single_pair", "all"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--no-statevector", action="store_true")
    parser.add_argument("--no-bloqade", action="store_true")
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    # Resolve VRP file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vrp_path = args.vrp
    if vrp_path is None:
        vrp_path = os.path.join(script_dir, VRP_FILE)
    elif not os.path.isabs(vrp_path):
        vrp_path = os.path.join(script_dir, vrp_path)
    if not os.path.exists(vrp_path):
        print(f"Error: VRP file not found: {vrp_path}")
        sys.exit(1)

    result = solve_cvrp_qaoa(
        vrp_path=vrp_path, p_layers=args.p, maxiter=args.maxiter,
        penalty=args.penalty, max_routes=args.max_routes,
        route_filter=args.route_filter,
        use_statevector=not args.no_statevector, shots=args.shots,
        seed=args.seed, show_bloqade=not args.no_bloqade,
        show_plots=not args.no_plots, verbose=True)

    print(f"\n{'='*60}")
    print(f"  Done.  Time: {result['optimization']['elapsed_sec']:.1f}s")
    print(f"  Best bitstring: {result['optimization']['best_bitstring']}")
    print(f"  Best energy   : {result['optimization']['best_energy']:.6f}")
    print(f"{'='*60}")
