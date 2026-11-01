# Quantum City
## Resilient hourly energy allocation + an audited Qiskit QAOA laboratory

**Working research prototype, not a quantum-advantage or production-grid claim.** All input data is synthetic. Costs are illustrative INR; results are not estimates for a real Kerala utility.

### Start here
```bash
# Python 3.11–3.13 recommended; tested here on Python 3.13
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                            # 39 tests
python run_all.py                     # 5 city scenarios + 2 quantum benchmarks
python server.py --port 8000          # then open http://localhost:8000
```
`python run_all.py --fast` uses a smaller QAOA search budget. `python experiments.py` reproduces three-seed CVaR/expected-energy ablations. No IBM account, paid solver, cloud backend, or internet connection is needed after installing dependencies. The web interface has no external CDN assets. Use the **live server**, not a static HTML-file preview, for interactive functionality.

For the exact core package versions used here, see `requirements-tested.txt`. That file is an environment record; `requirements.txt` is the supported installation range. The optional browser test requires Playwright and Chromium separately (`pip install playwright`, `playwright install --with-deps chromium`, then `python scripts/browser_check.py` with the server running).

## What was actually built
- Source configuration: solar, wind, utility grid, thermal heat-to-power storage, emergency generator; up to 12 sources through JSON.
- Residential, hospital, water/emergency, commercial, office, industry profiles; up to 20 zones through JSON.
- 24 hourly slots by default, 1–48 supported; weekday/weekend/holiday/event profiles and five operating stress scenarios.
- Time-of-use grid tariffs: 1.5× base at 17–22h and 0.86× base at 00–06h; editable multiplier matrix.
- Multi-period source-to-zone allocation with supply, feeder, optional route, demand, reserve, charge/discharge and conversion-efficiency constraints.
- Explicit critical and household minima, physical bottleneck reporting, emergency-only backup, transparent noncritical shortage fairness.
- Cost/emission reporting, unused capacity separated from unused renewable potential, thermal inventory accounting, independent physical audit.
- Re-optimization after input edits; `energy.rolling.reoptimize` replans remaining hours using observed thermal inventory.
- Real Qiskit circuits, Aer statevector training, finite-shot measurement, CVaR or expected-energy objective, COBYLA multi-start, p=0/1/2 comparison, exact enumeration and discrete MILP baselines.
- Independent checks for every QUBO ground state and every Ising energy; measured feasible and optimal probabilities, confidence intervals, valid-sample cost gaps, runtime and gate/depth audits.
- Responsive browser dashboard, scenario comparisons, JSON exports, error handling, and session-local experiment jobs.

## Important scope distinction
**The city controller is classical. The quantum solver is a bounded dispatch benchmark.** Do not say that 8 qubits optimize an arbitrary 24-hour city MILP.

1. **Operational model:** SciPy/HiGHS solves a multi-period MILP, including binary storage charge/discharge modes. Critical/household protection and fairness have explicit precedence over monetary savings.
2. **Canonical quantum benchmark:** four sources with four discrete levels each, 8 decision qubits, exact 50 kWh served demand in 10 kWh blocks. Source rates come from the synthetic scenario defaults.
3. **City-hour quantum benchmark:** derive a restricted problem from the selected solved hour. Demand is five equal blocks; each of solar/wind/grid/thermal is capped at three blocks. Charge is deducted from availability; thermal uses no more than its certified scheduled output that hour. Zero-capacity sources use zero qubits. Some hours are infeasible at this coarse resolution and are explicitly rejected, not silently repaired.
4. QAOA and classical methods solve **the same discrete benchmark**. Quantum samples are decoded and checked against original bounded/equality constraints. The best valid sampled cost is compared with the certified classical optimum; ties are reported. In city-hour mode, measured QAOA source proposals are passed into a **full-horizon classical recourse solve** with that hour’s source totals fixed. This completes zone allocations, checks storage coupling and rejects proposals that cannot be extended. The selector retains the best valid full-city plan under the same lexicographic priorities. It exports the selected plan but does not silently overwrite the active dashboard plan.

This separation is deliberate: a truthful, reproducible quantum mechanism is stronger than a decorative circuit or an unsubstantiated city-scale speedup claim. Candidate-level full classical recourse is implemented in `energy/integration.py`. In the measured hour-13 run, the restricted quantum proposal extends to a feasible ₹85,680.55 day plan, but the unrestricted classical baseline costs ₹84,269.47 and is correctly retained. This is evidence of honest integration and selection, not quantum improvement.

## Measured results in this workspace
`python run_all.py` used seed 7, 3 random restarts, up to 100 COBYLA evaluations/restart, 8,192 measurement shots, CVaR alpha 0.25. Runtime depends on machine/package versions.

| Benchmark | Qubits | Uniform P(optimal) | Measured p=1 | Measured p=2 |
|---|---:|---:|---:|---:|
| Canonical 50 kWh | 8 | 0.3906% | 1.5259% | 1.3062% |
| Weekday hour 13 | 4 | 6.2500% | 13.7451% | 25.6592% |

All depths sampled the discrete optimum on these runs. This is not guaranteed on other seeds or instances. The canonical exact cost is ₹102: 30 kWh solar and 20 kWh wind. Classical enumeration/MILP is faster at this size. p=2 did **not** outperform p=1 for the canonical seed-7 run.

Three-seed ablation (seeds 7, 23, 41, same budgets; ideal circuit P(optimal), mean ± sample SD):

| Objective | p=1 | p=2 |
|---|---:|---:|
| CVaR 25% | 1.140% ± 0.384% | 1.449% ± 0.129% |
| Expected energy | 0.583% ± 0.558% | 0.617% ± 0.472% |

This small experiment supports CVaR **on this instance**, not a general superiority claim. Full per-seed outcomes, measurement probabilities, optimizer termination messages and histories are saved in `results/ablation.json`.

### City scenarios
| Scenario | Demand kWh | Unmet kWh | Backup kWh | Critical minima |
|---|---:|---:|---:|---|
| Normal | 18,637.3 | 0 | 0 | Met |
| Peak +45% | 27,024.1 | 306.0 | 0 | Violated: feeder bottleneck |
| Low solar/wind | 18,637.3 | 0 | 0 | Met |
| Evening grid outage | 18,637.3 | 785.8 | 1,796.7 | Met |
| Severe shortage | 18,637.3 | 11,573.1 | 1,560.0 | Met |

Normal time-of-use operating cost: approximately ₹84,269.47. The peak scenario deliberately exposes a real constraint: hospital demand becomes 159.5 kW but its feeder allows only 150 kW; water/emergency demand similarly exceeds its feeder. Extra generation cannot fix a feeder bottleneck. In the severe case, total supply can still protect the critical minima, while other zones necessarily lose service. Zero-supply tests verify that impossible critical minima are reported honestly.

## Files
| Path | Purpose |
|---|---|
| `server.py`, `web/index.html` | Local API + self-contained dashboard |
| `energy/scenarios.py` | Synthetic profiles and editable scenario schema |
| `energy/dispatch.py` | Lexicographic multi-period MILP + independent audit |
| `energy/integration.py` | Full-horizon recourse and lexicographic selection for QAOA source proposals |
| `energy/rolling.py` | Remaining-horizon replanning using observed SOC |
| `energy/quantum.py` | Bounded QUBO, Ising, explicit QAOA, Aer, CVaR, exact baselines |
| `run_all.py` | Reproduction CLI |
| `experiments.py` | Multi-seed CVaR ablation |
| `tests/` | 39 automated physical/math/circuit/integration tests |
| `scripts/browser_check.py` | Optional browser workflow smoke test |
| `docs/ENGINEERING_GUIDE.md` | Equations, hand calculation, simulation loop, technical decisions |
| `docs/CLAIMS_AUDIT.md` | Corrections to the supplied quantum explanation |
| `docs/SUBMISSION.md` | Rubric mapping, pitch, demo script, remaining work |
| `results/` | Actual measured JSON outputs and UI screenshots |
| `legacy/` | Uploaded project preserved; CSV placed at its expected `data/plants.csv` path |

The legacy single-period model includes a battery rather than thermal storage and a different currency/scale. It is provenance, **not** the current city implementation. Its `check_math.py` was run successfully; its old Qiskit primitive path and its README's nonexistent notebook were not certified as current deliverables. The legacy dependency file is separate.

## Boundaries
Synthetic scenario profiles are not trained forecasting. No AC/DC power flow, voltage/frequency/stability control, ramping, generator start-up costs, storage heat losses/degradation, stochastic robustness, distribution-network topology, live sensors, hardware noise, or real QPU execution. Export and demand-response actions are recommendations, not optimized control variables. Thermal store is modeled as a heat-to-electricity resource with explicit losses; a real plant's efficiencies and ratings must replace the illustrative values. End-of-horizon reserve is enforced, but no terminal energy value is included, so plans may draw down to reserve. Replan with real measured inventory rather than restarting storage each hour.

Native optimization calls are serialized on one stable worker to avoid conflicting native solver thread pools; HTTP status requests remain responsive. The HTTP server is for a local/hackathon demo, not an authenticated production service. Session jobs/history are not persistent databases. This is decision support, not safety-certified infrastructure control.
