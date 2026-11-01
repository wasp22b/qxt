#!/usr/bin/env python
"""
run_all.py - one command to reproduce the whole project.

    python run_all.py            # full run, 6 carbon weights (a few minutes)
    python run_all.py --fast     # 30-second smoke test: one weight, tiny settings

Outputs (folder ./results):  results.csv, tradeoff.png, convergence.png, circuit.png (or .txt)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import microgrid as mg  # noqa: E402


def versions() -> str:
    parts = []
    for mod in ("qiskit", "qiskit_aer", "qiskit_optimization"):
        try:
            m = __import__(mod)
            parts.append(f"{mod} {m.__version__}")
        except Exception:  # noqa: BLE001
            parts.append(f"{mod} MISSING")
    return " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Microgrid Dial: QAOA energy allocation")
    ap.add_argument("--weights", type=float, nargs="+", default=mg.DEFAULT_WEIGHTS)
    ap.add_argument("--reps", type=int, default=2, help="QAOA depth p")
    ap.add_argument("--restarts", type=int, default=4, help="random restarts of the classical optimiser")
    ap.add_argument("--maxiter", type=int, default=100, help="COBYLA iterations per restart")
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--penalty", type=float, default=mg.PENALTY_FACTOR, help="blackout penalty lambda")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results")
    ap.add_argument("--fast", action="store_true", help="quick smoke test")
    a = ap.parse_args()
    if a.fast:
        a.weights, a.reps, a.restarts, a.maxiter, a.shots = [0.05], 1, 1, 30, 2048

    try:
        import qiskit  # noqa: F401
    except ImportError:
        print("Qiskit is not installed in this environment. Activate your venv and run:\n"
              "  pip install -r requirements.txt")
        return 1

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    plants = mg.load_plants()
    print("Environment:", versions())
    print("Plants:", [(p.name, p.cost, p.co2) for p in plants], "| demand", mg.DEMAND_MW, "MW")
    print(f"QAOA p={a.reps}, restarts={a.restarts}, maxiter={a.maxiter}, shots={a.shots}, penalty={a.penalty}\n")

    rows, t0 = [], time.time()
    for w in a.weights:
        t = time.time()
        r = mg.solve_weight(plants, w, penalty_factor=a.penalty, reps=a.reps, restarts=a.restarts,
                            maxiter=a.maxiter, shots=a.shots, seed=a.seed)
        rows.append(r)
        print(f"   done in {time.time() - t:.0f}s | QUBO matches hand-written QUBO: {r['qubo_matches_manual']}")

    print("\n" + mg.format_table(rows, plants))
    print(f"\nTotal time: {time.time() - t0:.0f}s | backend: {rows[0]['backend']}")

    with open(out / "results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["carbon_weight", "qaoa_levels", "qaoa_cost", "qaoa_co2", "exact_levels", "exact_cost",
                     "exact_co2", "match_exact", "gap_pct", "p_optimal", "p_optimal_random", "p_feasible",
                     "p_feasible_random", "merit_order_cost", "qubits"])
        for r in rows:
            b = r["qaoa_best"]
            wr.writerow([r["w"], b["levels"] if b else "", b["cost"] if b else "", b["co2"] if b else "",
                         r["exact"]["levels"], r["exact"]["cost"], r["exact"]["co2"], r["match"],
                         r["gap_pct"], r["p_optimal"], r["p_optimal_random"], r["p_feasible"],
                         r["p_feasible_random"], r["greedy"]["cost"], r["n_qubits"]])

    mg.plot_tradeoff(rows, out / "tradeoff.png")
    mg.plot_convergence(rows[0]["histories"], out / "convergence.png")
    rows[0]["solver"].draw(out / "circuit.png")
    print(f"Saved results to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
