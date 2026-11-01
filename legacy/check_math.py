"""
check_math.py - tests everything that does NOT need a quantum simulator.

Run this first (takes about a second):   python check_math.py

It proves, with exact enumeration, that
  1. the brute-force baseline finds the cheapest feasible allocation,
  2. our hand-written QUBO (slack bits + squared penalty) has its lowest-energy state at that same optimum,
  3. the Ising Hamiltonian that QAOA minimises reproduces the QUBO energies exactly,
  4. a too-small penalty weight lets blackouts win (the 'Goldilocks' effect).
If this passes, any later disagreement is about QAOA's search, not about the formulation.
"""
import numpy as np

import microgrid as mg

plants = mg.load_plants()
ok_all = True

print("Plants:", [(p.name, p.cost, p.co2) for p in plants], "| demand", mg.DEMAND_MW, "MW\n")
print(f"{'w':>5} | exact levels | cost $/h | CO2 kg/h | merit==exact | QUBO ground == exact | Ising == QUBO | qubits")
for w in mg.DEFAULT_WEIGHTS:
    ex = mg.brute_force(plants, w)
    gr = mg.merit_order(plants, w)
    terms = mg.manual_qubo_terms(plants, w, mg.DEMAND_MW, mg.PENALTY_FACTOR)
    g = mg.ground_state_check(terms, plants, w)
    n, const, lin, quad, _ = terms
    ising, offset = mg.qubo_to_ising(n, const, lin, quad)
    same = np.allclose(mg.ising_energies(n, ising, offset), mg.qubo_energies(n, const, lin, quad)[1])
    merit_ok = gr["levels"] in ex["all_optimal"]
    row_ok = g["ok"] and g["feasible"] and same and merit_ok
    ok_all &= row_ok
    print(f"{w:>5} | {str(ex['levels']):<12} | {ex['cost']:>8.0f} | {ex['co2']:>8.0f} | "
          f"{str(merit_ok):<12} | {str(g['ok'] and g['feasible']):<20} | {str(same):<13} | {n}")

print("\nPenalty study at w = 0.05 (what lambda does):")
for r in mg.penalty_study(plants, 0.05):
    state = "optimal & feasible" if r["optimal"] and r["feasible"] else ("FEASIBLE but not optimal" if r["feasible"] else "BLACKOUT wins")
    print(f"  lambda = {r['penalty']:>5}: ground state {r['ground_levels']} -> {state}")

print("\nRESULT:", "ALL CHECKS PASSED" if ok_all else "SOMETHING FAILED")
raise SystemExit(0 if ok_all else 1)
