# Portfolio Optimization with Binary Encoding, SA &amp; QAOA

A proof-of-concept demonstration of solving the Markowitz portfolio optimization problem
using **QUBO (Quadratic Unconstrained Binary Optimization)** formulation, solved by both
**Simulated Annealing (SA)** and **QAOA (Quantum Approximate Optimization Algorithm)**.

## Overview

Continuous portfolio weights are discretized into binary variables via K-bit encoding,
producing a pure QUBO that maps exactly to an Ising Hamiltonian without any embedding
approximation. Both a classical heuristic (SA) and a gate-based quantum variational
algorithm (QAOA) solve the **same 18-variable QUBO**, with exact enumeration providing
ground truth.

## Problem

| Parameter | Value |
|-----------|-------|
| Assets (N) | 6 (3 equity, 2 fixed-income, 1 money-market) |
| Bits per asset (K) | 3 |
| QUBO variables | 18 binary |
| Search space | 2<sup>18</sup> = 262,144 states |
| Objective | minimize -return + gamma * variance + lambda * (budget - 1)<sup>2</sup> |

## Pipeline

```
Financial Data → Weight Discretization (K-bit) → Pure QUBO → Ising Hamiltonian
                                                      ↓                ↓
                                               Simulated Annealing   QAOA (p=3)
                                                      ↓                ↓
                                               Exact Ground State   Approx. (Q≈0.97)
```

## Key Results

| Metric | SA (classical) | QAOA (p=3) | Exact GS |
|--------|---------------|------------|----------|
| Annual Return | 8.10% | 9.06% | 8.10% |
| Annual Risk (sigma) | 9.19% | 11.45% | 9.19% |
| Sharpe Ratio | 0.55 | 0.53 | 0.55 |
| QUBO Objective | -0.064093 | -0.062291 | -0.064093 |
| Hamming to GS | 0 / 18 | 7 / 18 | 0 |
| Approx. Quality | 1.000 | 0.972 | 1.000 |
| Feasible (sigma <= target) | YES | NO | YES |
| Runtime | <1 s | 339 s | ~0.5 s (enum.) |

- **SA found the exact global optimum** across all 10 independent runs.
- **QAOA achieved 0.972 quality** but did not reach the ground state; sampling distribution was
  nearly uniform (top state appeared only 2/2048 shots), indicating limited expressivity at p=3.
- **Gamma sweep** (7 values) traced the Markowitz efficient frontier; best feasible solution
  (gamma=5.0) achieved Sharpe 0.57 with 6.83% return and 6.67% risk.

## Hardware / Solver

- **SA**: Classical simulated annealing (geometric cooling, Metropolis acceptance, 10 runs * 10 restarts * 20,000 steps)
- **QAOA**: Bloqade QASM2 circuit construction + Qiskit Aer statevector simulation, COBYLA optimizer (60 iterations, 6 parameters)
- **Exact enumeration**: 262,144 states brute-force as ground truth

## Files

| File | Description |
|------|-------------|
| `financial_portfolio_sakuler.py` | Main pipeline: data setup, QUBO, SA, gamma sweep, QAOA, comparison |
| `portfolio_qaoa_sa_report.html` | Detailed business-report-style analysis (Chinese, 13 sections) |
| `portfolio_results.json` | Summary results (weights, returns, variances, Sharpe, QAOA quality) |

## Honest Limitations

- **SA outperforms QAOA** on this 18-variable problem by every metric.
- QAOA at p=3 lacks expressivity for 18 qubits; COBYLA struggles with the non-convex landscape.
- Post-selection (lowest-energy sample among 2048 shots) inflates QAOA performance.
- Toy scale: 2<sup>18</sup> states are enumerable; true quantum advantage expected only at N*K >= 30.
- Single-period static model; no transaction costs or multi-period dynamics.
- Synthetic data only; real-market validation needed.

## Dependencies

```
numpy, scipy, qiskit, qiskit-aer, bloqade-qasm2
```

## References

- Sakuler et al., Quantum Mach. Intell. 7, 43 (2025) — weight discretization
- Markowitz, H., J. Finance 7, 77 (1952) — portfolio theory
- Lucas, A., Frontiers in Physics 2, 5 (2014) — Ising formulations
- Farhi, Goldstone & Gutmann, arXiv:1411.4028 (2014) — QAOA
