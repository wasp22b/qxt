# Engineering guide: what runs, why it is valid, and how to demonstrate it

## 1. Recommended approach
**A classical safety-constrained multi-period dispatcher plus a gate-based, slack-free QAOA dispatch benchmark.**

The continuous source-to-zone allocation problem with linear costs is an LP unless additional discrete operating choices are introduced. It is not automatically NP-hard just because it is an energy problem. Our full controller becomes an MILP because storage has a binary charge/discharge mode. The quantum benchmark deliberately discretizes source dispatch into bounded integers.

For this hackathon, keep Qiskit + Aer, shallow depths p=1 and p=2, bounded encoding, proven equality penalties, CVaR ablation, multi-start angle search and explicit original-constraint validation. This is a practical implementation choice, not evidence that QAOA is the fastest optimization method. Do not assume the competition requires IBM hardware unless its actual rules say so.

## 2. Physical simulation and units
A slot has duration dt hours. Inputs `demand[t,z]` and `availability[t,s]` are mean electrical kW. The solver converts them into kWh per slot. All allocations, charging flows and shortages in the result are electrical kWh. Nameplate, feeder and route limits are kW, multiplied by dt. Default dt=1 hour.

For source s and zone z:
- x[t,s,z] ≥ 0: electrical energy delivered.
- q[t,s] ≥ 0: electrical energy consumed to charge the thermal store; only renewable sources may charge.
- u[t,z] ≥ 0: unmet demand.
- E[t]: stored heat, in kWh_th.
- v[t,z] ≥ 0: shortfall against that zone's minimum requirement.
- f[t] in [0,1]: common lower bound on noncritical served fractions.
- b[t] binary: charge mode (1) versus discharge mode (0).

Constraints:

```
Σ_s x[t,s,z] + u[t,z] = demand_kw[t,z] × dt
Σ_z x[t,s,z] + q[t,s] ≤ min(available_kw[t,s], nameplate_kw[s]) × dt
Σ_s x[t,s,z] ≤ feeder_kw[z] × dt
x[t,s,z] ≤ route_kw[s,z] × dt                # when specified
Σ_s x[t,s,z] + v[t,z] ≥ minimum_fraction[z] × demand_kwh[t,z]
Σ_s x[t,s,z] ≥ f[t] × demand_kwh[t,z]       # noncritical zones only

E[t+1] = E[t] + ηc Σ_s q[t,s] − Σ_z x[t,thermal,z] / ηd
reserve ≤ E[t] ≤ tank_capacity
E[0] = observed_initial_heat
Σ_s q[t,s] ≤ charge_limit_kw × dt × b[t]
Σ_z x[t,thermal,z] ≤ thermal_power_kw[t] × dt × (1 − b[t])
```

Thermal storage is **not free electricity**. For example, 500 kWh_th at ηd=0.4 gives only 200 kWh electrical before reserve constraints. Charging with 100 kWh electricity at ηc=0.9 adds 90 kWh_th and can later return 36 kWh electrical. This low round-trip efficiency is why surplus renewable charging may be useful but is not automatically economical. A thermal system serving heat demand directly needs a separate heat-demand model, not this electrical allocation model.

Source cost is charged on all generated electrical energy, including charging energy. The thermal source rate represents conversion/operating cost per electrical kWh; original renewable charging cost has already been paid on q. The initial stored heat is a sunk initial inventory with no acquisition cost in this horizon. Backup is consumed in real time; there is no backup SOC.

## 3. Priorities, fairness and feasibility
The optimizer runs six stages. Each prior stage's optimum is retained to an absolute 1e-6 objective tolerance:
1. Minimize sum of critical minimum deficits.
2. Minimize sum of household minimum deficits.
3. Minimize total unmet kWh.
4. Minimize emergency-generator kWh.
5. Maximize the sum over slots of f[t], the worst noncritical served/demand ratio per slot.
6. Retain each achieved f[t], then minimize cost + optional carbon price × emissions.

Thus “minimum cost” means **minimum cost within the safety, total-service, emergency and fairness priorities**. It does not mean that saving ₹1 justifies a blackout. Emergency backup can be used only when required to preserve the higher-priority service objectives, even if it were accidentally configured cheaper than grid supply.

This is one max-min floor per slot, not a full iterative lexicographic max-min ordering of every individual zone. Household floors can make ratios unequal. Feeder constraints can also limit attainable service. The reported Jain index uses noncritical served/demand ratios; it describes fairness, not physical grid adequacy. A common floor of zero when all noncritical service is zero is not a claim of satisfactory service.

Hard physics are never relaxed. Critical minima are service objectives with explicitly reported slack because no model can create electricity during zero supply. `validation.valid=True` means the physical plan is admissible; `critical_ok=False` means service protection was physically impossible under the configured limits. The UI distinguishes them.

## 4. Quantum encoding, with a hand calculation
Canonical example: electrical demand 50 kWh in one slot, Δ=10 kWh, four sources each with levels 0,1,2,3. Costs per kWh:

| Source | Solar | Wind | Grid | Thermal conversion |
|---|---:|---:|---:|---:|
| Cost INR/kWh | 1.8 | 2.4 | 7.2 | 3.2 |
| Maximum kWh | 30 | 30 | 30 | 30 |

Use two decision bits/source: g_s = x_s0 + 2x_s1. Capacity is built in. Demand is exact: Σ_s g_s = 5. Unused source availability is allowed; dispatched electrical energy must not exceed served demand.

A manual cheapest-first solution takes solar=3 blocks and wind=2 blocks, grid=0, thermal=0. Cost:

```
30 × 1.8 + 20 × 2.4 = ₹102
```

This is one optimal bitstring among 256 for these distinct costs, so uniform P(optimal)=1/256. Equal source costs produce many optimal encodings; the code counts them rather than hardcoding 1/256.

General bounded encoding uses weights whose sum equals the integer source cap. Cap=3 uses [1,2]; cap=2 uses [1,1]; cap=1 uses [1]; cap=0 uses []. Every represented integer is valid. Duplicate encodings (e.g. [1,1]) change uniform sampling probabilities and are included in all metrics.

Let a_i be the block weight of bit i, and d_i the monetary cost of turning it on. Normalize by M=max(1, Σ_s cost_s × cap_s × Δ), so feasible and infeasible original costs are all within [0,1]. Then:

```
E(x) = Σ_i (d_i/M)x_i + A(Σ_i a_i x_i − D)²,  A=2
constant = A D²
linear_i = d_i/M + A(a_i² − 2D a_i)
quadratic_ij = 2 A a_i a_j, i<j
```

If the target is a feasible integer, any invalid equality residual has magnitude at least 1 and incurs at least 2, while every feasible cost is ≤1. Therefore **every QUBO ground state is a true discrete optimum**. This is a small proof, not just a tuned penalty guess. The code also enumerates all bitstrings to check it.

Do not silently apply this equality model to the original legacy model's `generation ≥ demand` semantics if oversupply is genuinely necessary. Here it is appropriate because the subproblem dispatches exactly served electrical demand; at a coarse arbitrary resolution some instances become infeasible and are rejected.

## 5. Ising mapping and QAOA circuit
Substitute x_i=(1−Z_i)/2 into the QUBO:

```
H = offset I + Σ_i h_i Z_i + Σ_i<j J_ij Z_i Z_j
J_ij = quadratic_ij/4
h_i = −linear_i/2 − Σ_j≠i quadratic_min(i,j),max(i,j)/4
```

The implementation compares every diagonal Ising energy against its QUBO energy. A separate circuit test checks the complex amplitudes of one cost layer against the analytic phases, catching angle signs and endianness mistakes.

1. H gates prepare |+⟩^n.
2. For each layer, apply RZ(2γh_i) and RZZ(2γJ_ij), using one common Hamiltonian coefficient scale.
3. Apply RX(2β) to every qubit.
4. COBYLA tunes 2p parameters using random initial angles, never a classical solution warm start.
5. Finite-shot measurements produce bitstrings; Qiskit bit i is `(int(bitstring,2) >> i) & 1`.
6. Decode source levels, check exact target and bounds, rank valid sampled solutions by real monetary cost.

Global coefficient scaling changes the angle scale but not the minimizers. Dense demand coupling requires n(n−1)/2 RZZ terms per layer in this compact all-to-all description; actual hardware routing can add gates. Simulator circuit depth is not a hardware executable-depth guarantee.

Training is noiseless Aer statevector evaluation: the simulator returns exact circuit probabilities, and CVaR is computed classically. Final reporting uses shot-sampled circuit measurements. This is **not** hardware-noisy, shot-based training. A future hardware backend should estimate CVaR from counts during training and budget the resulting sampling noise and execution latency.

CVaR alpha=0.25 means average the lowest-energy 25% probability mass. Fractional mass at the boundary is handled correctly. CVaR can saturate or develop plateaus when sufficient low-energy probability is obtained; it is a heuristic objective choice, not a convergence guarantee. The optimizer chooses restarts by its training objective, not by using the known optimum probability as an oracle.

## 6. What the city-hour benchmark really represents
For selected slot t:
1. Read the full-city demand energy.
2. Set Δ=demand_energy/5.
3. Deduct scheduled charging from available electrical supply.
4. Conservatively limit thermal electrical output to the certified full-day scheduled amount for this hour.
5. For solar/wind/grid/thermal set cap_s=min(3, floor(available_s/Δ)).
6. If Σ cap_s < 5, reject this benchmark discretization rather than hiding infeasibility.
7. Build the same equality QUBO and compare QAOA, exact enumeration and integer MILP.
8. For each distinct measured best-valid p=1/p=2 proposal, fix the selected slot’s source totals in the full-city MILP. Complete all zone allocations, charging and thermal inventory across the horizon; reject a proposal if no physically valid completion exists.
9. Compare completed candidate plans with the unrestricted baseline using the first five priority-stage objectives before comparing monetary/carbon cost. Export the selected full plan without silently changing the active dashboard scenario.

In the provided weekday hour 13, wind and scheduled thermal are below one large block, so the benchmark contains four active qubits rather than eight. This is a **restriction**, not a claim that wind is physically unavailable or that all full-city routing was optimized quantum mechanically.

## 7. How to simulate changing conditions manually
Start with a normal plan. Treat only the first slot as executed. Read simulated or measured thermal SOC at its end. Update remaining source availability and demand; replan only the remaining horizon.

```python
from energy.scenarios import make_scenario
from energy.dispatch import solve
from energy.rolling import reoptimize

initial = solve(make_scenario())
# Assume 4 hours executed. A sensor reports 3,000 kWh_th remaining.
next_plan = reoptimize(initial, elapsed=4, observed_soc=3000.)

# Simulate reduced sunlight for the next optimization, not for executed history:
remaining = next_plan['config']['availability']
for row in remaining:
    row[0] *= 0.2
next_plan['config']['availability'] = remaining
updated = solve(next_plan['config'])
assert updated['validation']['valid']
```

Do not reset storage to its original initial inventory every hour. Do not independently optimize all hours without SOC links. Real deployment would use measured electrical power/energy, inverter and plant telemetry, and a calibrated forecast, execute the first dispatch setpoint, then repeat. This project does not send control signals to real equipment.

## 8. Efficiency roadmap, in order
1. Keep exact classical LP/MILP as the safety baseline; benchmark time including preprocessing, optimizer evaluations and quantum measurement overhead.
2. Spend qubits only on a clearly stated small discrete problem, not thousands of continuous routing variables.
3. Build bounds into encoding; use exact equality where justified; do not introduce heuristic inequality penalties without testing ground-state correspondence.
4. Use p=0,1,2, fixed restart/evaluation budgets, multiple seeds, CVaR ablations and both sampled and ideal-circuit probabilities.
5. Try better angle initialization/continuation, but label any classical assistance; warm starts remain valid hybrid methods.
6. Candidate-level classical recourse is implemented: freeze a sampled one-hour source dispatch, complete zone/storage constraints, reject impossible completions, compare valid plans under all priorities. A stronger future extension would jointly sample commitments over several hours or use a rolling decomposition rather than restrict a single hour.
7. Consider fixed-Hamming-weight encodings and excitation-preserving XY mixers only with a correct feasible initial state and capacity encoding. An XY mixer on ordinary binary-weight bits does NOT conserve dispatched energy automatically.
8. Add realistic noise/hardware experiments only after checking gate counts, routing, shot budgets and error bars. Hardware feasibility, annealing embedding and speedup are empirical questions.

There is no honest universal “most efficient quantum algorithm” for this application. For a working hackathon, the measured bounded QAOA pipeline is a defensible scope; for present-day operational efficiency, the classical dispatcher is the right engine.
