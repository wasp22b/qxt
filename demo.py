"""
Runnable demo script -- walks through all 12 features in presentation order.

    python demo.py                # full demo including the quantum comparison
    python demo.py --no-quantum   # skip QAOA (fast, no Qiskit needed)
    python demo.py --step 3       # jump straight to one step
"""

from __future__ import annotations

import argparse
import time
from typing import List

from energy_allocation import (
    SCENARIOS,
    SLOTS,
    SOURCES,
    W_CRIT,
    W_FAIR,
    ZONES,
    default_sources,
    report,
    solve,
    validate,
    with_backup_enabled,
    with_grid_restriction,
)

BAR = "=" * 78


def header(step: int, title: str, features: str) -> None:
    print(f"\n{BAR}\nSTEP {step} -- {title}\n{features}\n{BAR}")


def step1_normal() -> None:
    header(1, "A normal day", "Features 1, 2, 3, 4, 5, 8")
    print("Model:")
    for name, spec in SOURCES.items():
        caps = f", daily cap {spec['storage']} kWh" if "storage" in spec else ""
        print(f"  {name:9s} ${spec['cost']:.2f}/kWh  avail {spec['avail']}{caps}")
    for name, zone in ZONES.items():
        flag = "  <-- CRITICAL" if zone.get("critical") else ""
        print(f"  {name:12s} demand {zone['demand']}{flag}")
    print()
    sol = solve()
    report(sol)
    print("\nTalking points:")
    print("  * cheapest sources are dispatched first (Solar $0.05, Wind $0.08)")
    print("  * availability is a '<=' bound, so renewable surplus is reported, never forced")
    thermal = sol["generation"][sol["src"].index("Thermal")]
    print(f"  * Thermal daily cap {SOURCES['Thermal']['storage']} kWh is binding: "
          f"used {thermal.tolist()} (total {thermal.sum():.0f})")
    for rec in sol["recommendations"]:
        for msg in rec["messages"]:
            print(f"  * [{rec['slot']}] {msg}")


def step2_peak() -> None:
    header(2, "Demand scenarios", "Features 2, 9")
    rows = []
    for name, factor in SCENARIOS.items():
        sol = solve(scenario=name)
        rows.append(
            (name, factor, sol["cost"], sol["total_unmet"], sol["critical_ok"], sol["backup_used"])
        )
    print(f"{'scenario':10s} {'factor':>7s} {'cost $':>9s} {'unmet kWh':>10s} "
          f"{'critical':>9s} {'backup':>7s}")
    print("-" * 58)
    for name, factor, cost, unmet, crit, backup in rows:
        print(f"{name:10s} {factor:7.2f} {cost:9.2f} {unmet:10.1f} "
              f"{'OK' if crit else 'AT RISK':>9s} {'yes' if backup else 'no':>7s}")
    print("\nTalking point: solve() is a pure function of its inputs -- changing the")
    print("scenario just re-solves. No special-casing, no separate code path.")


def step3_shortage() -> None:
    header(3, "Shortage: grid restriction + no backup", "Features 6, 7, 9")
    sources = with_grid_restriction(default_sources(), slot=3, limit=10.0)
    sources = with_backup_enabled(sources, enabled=False)
    avail = sum(sources[s]["avail"][3] for s in sources)
    demand = sum(ZONES[z]["demand"][3] for z in ZONES)
    print(f"Evening: availability {avail:.0f} kWh vs demand {demand:.0f} kWh "
          f"-> shortfall {demand - avail:.0f} kWh\n")
    sol = solve(sources=sources)
    report(sol)

    print("\nFairness check (Feature 7) -- every non-critical zone loses the SAME %:")
    crit_names = [z for z in ZONES if ZONES[z].get("critical")]
    for i, name in enumerate(sol["zon"]):
        t = 3
        d, sd = sol["demand"][i, t], sol["served"][i, t]
        pct = 0.0 if d == 0 else (d - sd) / d * 100
        tag = "CRITICAL (protected)" if name in crit_names else ""
        print(f"  {name:12s} curtailed {pct:5.1f}%   {tag}")
    print(f"\n  fairness rate r_evening = {sol['curtailment_rate'][3] * 100:.1f}%")
    print("\nTalking point: a naive model with one unmet variable per zone lets the LP")
    print("dump the whole shortage on a single zone. r_t makes that structurally")
    print("impossible -- the curtailment is proportional by construction.")


def step4_quantum(no_quantum: bool = False) -> None:
    header(4, "Quantum module: QUBO -> QAOA -> LP repair", "Feature 10")
    from quantum_module import compare

    print("Architecture: quantum optimisation picks the DISCRETE decisions (source")
    print("output levels, which zones to shed); a classical LP then computes exact,")
    print("feasible flows. Hybrid quantum-classical, and the qubit count stays tiny.")
    if no_quantum:
        print("\n(--no-quantum: skipping QAOA, running the classical QUBO solvers only)")
    t0 = time.perf_counter()
    result = compare(t=3, use_quantum=not no_quantum)
    print(f"\nTotal comparison runtime: {time.perf_counter() - t0:.1f}s")
    best = next(
        (r for r in result["rows"] if r["solver"].startswith("QAOA best")), None
    )
    if best:
        verdict = "MATCHED" if best["gap"] < 1e-6 else f"gap {best['gap']:.2f}"
        print(f"QAOA vs exact optimum: {verdict}   P(optimum) = {result['p_opt']:.4f}")

    print("\nTalking points:")
    print("  * QAOA agreeing with brute force validates the QUBO formulation")
    print("  * classical preprocessing (variable fixing) shrinks the problem and")
    print("    raises P(optimum) by orders of magnitude -- run --no-reduce to see")
    print("    the difference")
    print("  * be honest with judges: a statevector simulator IS classical")
    print("    computation. It validates the formulation; IBM Quantum / D-Wave")
    print("    hardware is the next step (--export writes those formats)")
    print("  * the classical LP already solves this size optimally in milliseconds.")
    print("    Quantum's value appears when discrete decisions explode: hundreds of")
    print("    generators, multi-day unit commitment, network switching.")
    print("  * zone shedding (quantum, discrete) vs proportional curtailment (LP,")
    print("    continuous) are two legitimate shortage policies -- showing both is a")
    print("    feature, not a contradiction")


def step5_validate() -> bool:
    header(5, "Independent validation", "Feature 11")
    scenarios: List[tuple] = [("normal", default_sources()), ("peak", default_sources())]
    restricted = with_backup_enabled(with_grid_restriction(default_sources()), enabled=False)
    scenarios.append(("grid outage", restricted))

    all_ok = True
    for label, sources in scenarios:
        scenario = "peak" if label == "peak" else "normal"
        sol = solve(sources=sources, scenario=scenario)
        print(f"\n--- {label} ---")
        ok = validate(sol)
        all_ok = all_ok and ok
    print(f"\n{BAR}\nOverall: {'ALL SCENARIOS PASS' if all_ok else 'FAILURES PRESENT'}\n{BAR}")
    return all_ok


def step6_dashboard() -> None:
    header(6, "Dashboard", "Feature 12")
    print("  streamlit run app.py")
    print("\nTabs: Dispatch | Zones | Quantum module | Validation | Model.")
    print("Every control in the sidebar re-solves the LP, so the dashboard is the")
    print("live interface to Feature 9 (dynamic re-optimisation).")


STEPS = {
    1: step1_normal,
    2: step2_peak,
    3: step3_shortage,
    4: step4_quantum,
    5: step5_validate,
    6: step6_dashboard,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the full project demo")
    ap.add_argument("--no-quantum", action="store_true", help="skip the QAOA comparison")
    ap.add_argument("--step", type=int, choices=sorted(STEPS), help="run a single step")
    ap.add_argument("--penalty", type=float, default=5_000.0)
    args = ap.parse_args()

    print(BAR)
    print("QUANTUM-ASSISTED CITY ENERGY ALLOCATION -- DEMO")
    print(f"slots {list(SLOTS)}   W_CRIT {W_CRIT:,.0f}   W_FAIR {W_FAIR:,.0f}")
    print(BAR)

    if args.step:
        steps = [args.step]
    else:
        steps = sorted(STEPS)

    for n in steps:
        if n == 4:
            step4_quantum(no_quantum=args.no_quantum)
        else:
            STEPS[n]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
