# Hackathon submission and finishing plan

## One-line project description
**Quantum City is a resilient energy-allocation lab that protects essential services, plans hourly renewable and thermal-storage use, and tests a compact, mathematically verified QAOA dispatch mechanism against exact classical baselines.**

Do not call this a quantum-powered city controller. The full city controller is classical; the quantum component is a measured discrete dispatch benchmark.

## Phase 1: idea and feasibility
| Criterion | Weight | Evidence to present |
|---|---:|---|
| Problem understanding / SDG | 20% | Scarce supply, expensive emergency generation, feeder bottlenecks, hospitals and household minima. SDG 7: affordability/reliability; SDG 13: renewable integration and emissions visibility. |
| Innovation / creativity | 25% | Safety-first lexicographic dispatch, electrical/thermal accounting, transparent shortage fairness, bounded no-slack quantum encoding, explicit formulation proof and multi-seed quantum ablation. None is claimed to be a new research algorithm. |
| Feasibility | 20% | Local Python + open-source HiGHS + Qiskit Aer; no paid services or credentials; five working scenarios and a reproducible CLI; dependencies and tests included. |
| Quantum planning | 25% | Explicit decision variables, bounded weights, equality QUBO, penalty proof, Ising mapping, RZ/RZZ/RX circuit, CVaR objective, p=0/1/2 and exact comparators. |
| Presentation clarity | 10% | Architecture split, dashboard, hand calculation, engineering guide, limitations and a four-minute demo. |

## Phase 2: prototype and implementation
| Criterion | Weight | Evidence to present |
|---|---:|---|
| MVP functionality | 25% | Hourly zone allocation, storage conservation, emergency-only backup, source/feeder/route limits, configurable inputs, shortage and reserve reports, comparison/export. |
| Quantum implementation | 25% | Real circuits executed on Aer, counted measurement samples, matched QUBO/Ising energies, classical MILP + enumeration, CVaR ablation and optimizer histories. One-slot QAOA proposals are integrated via full-horizon classical recourse; the quantum circuit does not solve the full MILP. |
| Technical implementation | 20% | 39 tests, independent validation, malformed-input rejection, remaining-horizon replanning, browser-tested UI and asynchronous quantum jobs. |
| Innovation / real-world impact | 15% | Prioritized resilience and auditable fairness; report unmet kWh, backup kWh and emissions. Synthetic outputs are not evidence of realized real-city savings. |
| Demo / presentation | 15% | Live normal/outage/peak runs, quantum depth table, downloadable results and measured ablation. Explain one failure case clearly. |

These are evidence mappings, not promised scores. A judge expecting a quantum circuit to solve the entire operational MILP may score the bounded quantum-proposal scope differently. Explain the decomposition honestly: one quantum source-dispatch proposal, then classical full-horizon recourse and plan selection.

## Four-minute demo script
**0:00–0:35 — problem and impact**
“Energy allocation is not just cheapest-first generation. Hospitals need protection, households need minimum service, and electrical demand changes across the day. A feeder can be a bottleneck even when supply is abundant. We target SDG 7 and SDG 13.”

**0:35–1:15 — working city model**
Open Energy overview. Show demand, cost, zero unmet energy and protected critical services. Scrub the hour slider. Explain daylight solar, working-hour offices, evening households, thermal reserve and time-of-use grid rates. Explain that electrical and thermal kWh differ.

**1:15–1:55 — shortage and a real failure**
Choose Evening grid outage, optimize. Show required emergency backup and fair noncritical shortages while critical minima are met. Choose Peak demand, optimize. Explain the feeder-induced hospital/water shortfall: extra generators cannot fix a narrow feeder. The audit remains physically valid while the critical-service status turns red.

**1:55–3:10 — actual quantum mechanism**
Open Quantum benchmark. Run canonical 8-qubit p=0/1/2 with CVaR. Explain two bits/source, 50 kWh exact demand, no slack, phase separation, interference and classical angle tuning. Show measured P(optimal), feasibility, costs, runtime and exact baseline. Mention that p=2 can be worse; do not hide it. Open the circuit/status panel.

**3:10–3:40 — validation and comparison**
Show 39 passing tests and the three-seed CVaR/expected-energy ablation. Explain ground-state verification and why exact classical methods remain faster here. Switch to current city-hour benchmark to show real scenario-derived inputs, explicitly noting its restricted one-hour scope. Show the full-horizon recourse result and why the lower-cost classical baseline is retained.

**3:40–4:00 — close and next step**
“Our deliverable is a safety-constrained planning prototype and a reproducible quantum optimization experiment, not a quantum speedup claim. Next is broader multi-hour quantum proposals, then calibrated forecasts and real telemetry.”

## Manual questions judges may ask
**What is quantum here?** RZ/RZZ cost evolution and RX mixing implement a gate-based quantum variational circuit; the simulator computes its state and samples measurements. COBYLA tunes angles; it does not directly choose source levels.

**Why not just use cheapest-first?** Without network/storage coupling, cheapest-first can work. Feeder constraints, protected minima, emergency-only backup, fairness and intertemporal storage require explicit coupled optimization. Our classical controller handles those.

**Why no slack qubits?** Exact served-demand equality and bounded capacity encoding avoid that particular slack requirement. A normalized objective plus integer squared penalty yields a provable ground-state match.

**What if quantum samples are invalid?** Reject them. Report feasible probability and best feasible sample. Never pass a low-QUBO-energy but invalid dispatch to operations.

**Do deeper circuits help?** Not guaranteed under a finite training budget. Our measured seed-7 canonical result is a counterexample. Multi-seed results are reported.

**Do you prove entanglement or advantage?** No. We implement entangling interactions and observe probability shaping, but neither is an entanglement witness nor a speedup demonstration.

**Can you control an actual microgrid?** Not yet. This is aggregate energy planning, not electrical protection, power-flow validation, voltage regulation or real actuator control.

## Feature acceptance and remaining work
| Requirement | Status |
|---|---|
| Multiple configurable sources and zones | Implemented; arbitrary additions through advanced JSON up to MVP bounds |
| Hourly profiles and day modes | Implemented as synthetic rule-based profiles, not trained forecasting |
| Cost/capacity/source-to-zone allocation | Implemented classically |
| Critical/household protection and fair shortages | Implemented with explicit impossible-minimum reporting |
| Thermal inventory / reserve / efficiency / charging modes | Implemented |
| Emergency generators only for real-time supply | Implemented |
| Dynamic replanning with observed SOC | Implemented as library API; UI input edits re-solve the configured horizon |
| Surplus/unused capacity and recommended actions | Implemented; export and flexible-load shifting are recommendations, not scheduled variables |
| QUBO, QAOA, Qiskit Aer | Implemented for bounded aggregate dispatch benchmarks |
| Exact baseline and valid-solution selection | Implemented on the same discrete benchmark; classical optimum retained, QAOA ties reported |
| Quantum source proposals participate in full-city plan selection | Implemented through full-horizon classical recourse; baseline retained unless a candidate is no worse under all priorities |
| Real QPU or hardware-noise evidence | Not implemented |
| Full grid-network physical simulation / live forecasting | Not implemented |

### Priority remaining tasks if you have another day
1. Confirm actual submission rules, team details, video length and allowed backends; the scoring rubric alone does not establish them.
2. Have each team member reproduce install/tests/CLI on a clean laptop.
3. Extend the implemented one-slot QAOA + full-horizon recourse pipeline to multiple commitment slots only if qubit and experiment budgets permit. Preserve explicit evidence of which decision came from QAOA and which from classical completion.
4. Calibrate one real or publicly documented city/feeder dataset with licensing and units; retain synthetic fallback for the live demo.
5. Add hardware-noise experiments only if time permits; prioritize correctness and reproducibility over a noisy decorative QPU run.
6. Record an offline demo and keep measured JSON, screenshots and environment versions in the submission.

### Safe final pitch
“We built and tested hourly energy dispatch with service-priority constraints and a compact QAOA formulation. We reduced the canonical dispatch register from 11 decision-plus-slack qubits in the supplied inequality model to 8 bounded decision qubits using served-demand equality, verified the mapping exhaustively, and measured probability amplification over a uniform circuit on synthetic instances. We do not claim a quantum runtime advantage or full-city quantum optimization.”
