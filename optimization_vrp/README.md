# optimization_vrp

A toy demo that solves the **Capacitated Vehicle Routing Problem (CVRP)** using the **Quantum Approximate Optimization Algorithm (QAOA)**.

The pipeline parses a VRPLIB instance, enumerates feasible routes into a compact set-partitioning QUBO, maps it to an Ising Hamiltonian, builds a QAOA circuit with [Bloqade](https://github.com/QuEraComputing/bloqade) QASM2, and optimizes the parameters via COBYLA + Qiskit Aer statevector simulation.

> **Created:** 2026-07-23

## Disclaimer

This is merely **one modeling and optimization approach** — an experimental exploration of quantum-inspired combinatorial optimization. There is still much work to be done: deeper circuit depth, better classical benchmarks, larger instances, and real-hardware validation all remain open.
