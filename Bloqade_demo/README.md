# Bloqade QAOA Demo

A code demo of QuEra's Bloqade SDK for analog quantum simulation and digital QAOA.

## Contents

| Section | Description |
|---------|-------------|
| **Atom geometries** | Construct & visualise Honeycomb, Square, Chain, Kagome lattices with defect insertion |
| **Analog Rydberg** | Rabi waveforms (linear, constant, piecewise, poly, custom), detuning sweeps, blockade-radius simulation on Python backend |
| **Braket backend** | Full smooth C₆/r⁶ vdW interaction via `braket.local_emulator()` — no hard cutoff |
| **QAOA + edge-colouring** | SIMD-parallel QAOA circuit compilation using Bloqade QASM2, reducing depth from O(E) to O(Δ+1) |
| **QAOA optimisation (Qiskit)** | End-to-end MaxCut QAOA (COBYLA + Aer) for 3-regular graphs, N=8 & N=24, with brute-force ground-truth verification |

## Key Results

| Graph | Edges | QAOA(p=2) | Global Optimum |
|-------|-------|-----------|----------------|
| N=8, d=3 | 12 | 10 | 10 (brute force ✓) |
| N=24, d=3 | 36 | 31 | 32 (brute force, 97%) |

QAOA matches the optimum on N=8 and beats classical local search (`one_exchange`: 30) on N=24.

## Environment

Python 3.10, `bloqade`, `qiskit`, `qiskit-aer`, `networkx`, `scipy`.
