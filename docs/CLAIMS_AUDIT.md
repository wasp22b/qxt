# Audit of the supplied explanation and uploaded code

## What was verified from your files
- `legacy/check_math.py` passes for all six carbon weights: merit order agrees with exhaustive search, manual slack-QUBO ground states are feasible optima, and Ising energies equal QUBO energies.
- The old code expects `data/plants.csv`, but your upload placed `plants.csv` beside the Python files. The preserved legacy copy includes that expected data path.
- The old model is four equal-capacity sources, one 50 MW dispatch hour, and two bits/source plus three demand-slack bits: 11 qubits. It is not zone-aware or multi-period.
- Its Battery is not a thermal store; it has no temporal state-of-charge model. Merely renaming the source would not implement storage.
- The old README references a notebook that was not attached. The new submission does not depend on it.
- Old mathematical tests passed here. That does **not** mean the legacy quantum primitive compatibility path was run and certified; the new explicit Aer circuit path was run instead.

## Claim-by-claim corrections
| Claim | Verdict / replacement |
|---|---|
| QAOA is a hybrid gate-based quantum algorithm | Correct. On Aer it is a classical simulation of a quantum circuit, not real QPU execution. |
| Hadamards prepare all 2^n bitstrings in superposition | Correct. Measurements still return individual bitstrings; this does not reveal all costs or solutions at once. |
| Demand penalty always entangles all qubits | Too strong. Quadratic interactions provide entangling gates and generally can create entanglement, but special angles, states and circuit structure can yield no entanglement. |
| Mixer moves amplitude from high-cost states to low-cost states | Useful intuition, not a guarantee. Cost phases and mixers together can reshape probabilities; optimization may fail or amplify the wrong states. |
| Higher P(optimal) than uniform proves quantum advantage | False. It shows instance-specific distribution shaping versus a uniform baseline. Classical nonuniform heuristics can also do this. It does not establish speedup or a quantum-only benefit. |
| p=2 must beat p=1 | False. The variational optimum may improve in an ideal nested family, but finite-budget, local, noisy optimization can perform worse. Our canonical seed-7 experiment does perform worse at p=2. |
| Random optimum probability is always 1/256 | False. It is count(optimal encoded bitstrings)/2^n. With n=11 include the full decision/slack register or compute the appropriate decision marginal. Degeneracy matters. |
| Aer builds a 2048×2048 matrix for every gate | False in normal statevector simulation. It stores 2^n complex amplitudes and applies gates using structured updates, not generally dense full-register matrices. Eight versus eleven statevector qubits changes amplitude storage by 8×, not a necessary 64× matrix-storage reduction. |
| Reducing 11 to 8 qubits is critical to make a laptop demo run | Exaggerated for this small size. Both are small. It reduces work, but circuit depth, optimizer calls, Python overhead and backend configuration also matter. |
| Unbalanced penalization is a guaranteed safe replacement for slack | False. Its originating paper explicitly describes approximate inequality encoding and says the original optimum need not be the Hamiltonian ground state. [1](https://iopscience.iop.org/article/10.1088/2058-9565/ad35e4) |
| CVaR drastically improves convergence for every instance | False as a universal claim. It is a published alternative variational objective. [1](https://research.ibm.com/publications/improving-variational-quantum-optimization-using-cvar) In this implementation we evaluate its effect through controlled multi-seed ablations rather than assume superiority. |
| Warm starts weaken the quantum role | Not a technical validity criterion. A classical warm start is legitimate hybrid design. Report its role and ablate it if used; this implementation uses random angles for simpler attribution. |
| Quantum annealing is universally more efficient than QAOA | Not established. Logical problem size, connectivity/embedding, noise, objective quality and full wall-clock budget differ. Raw qubit counts are not a performance comparison. |
| Annealing naturally reaches the ground state | Not guaranteed in finite-time noisy optimization. |
| QAOA needs p≥3 to match classical solvers | No universal threshold. Instance structure, noise, optimizer and comparator matter. |
| Every allocation problem is NP-hard | Incorrect. Continuous linear allocation is an LP. Integer commitments and operating choices create MILPs; discretizing an LP for QAOA does not establish an operational advantage. |
| Commercial solvers always solve millions of MILP variables in seconds | Not a general guarantee. Difficulty depends strongly on formulation, integrality, coupling, required gap and instance data. |
| This hackathon strictly requires IBM QAOA | Not established from the supplied scoring rubric. Qiskit was chosen because you requested it; verify the actual competition rules separately. |

## Why the new no-slack formulation is different
It uses exact **served-demand equality**, not unbalanced penalization of the original overgeneration inequality. Bounded source levels preserve capacity. Integer demand residuals and a normalized objective give a simple exact penalty guarantee. All bitstrings are audited against the original discrete problem, and all resulting Ising energies are checked. This change is appropriate for dispatch that must not oversupply served demand; it is not a universal inequality-to-equality trick.

## What the measurements support
- Gate-based QAOA circuits ran on Aer with nontrivial phase and mixer layers.
- The ideal distributions and shot samples can outperform uniform P(optimal) on the tested synthetic instances.
- Some optimizer runs hit their evaluation budget; these termination messages are retained, not hidden.
- Multi-seed CVaR ablation is favorable on the canonical instance but small and not general proof.
- Exact classical search and MILP solve these tiny benchmark instances faster.
- No hardware speedup, scalability advantage, quantum entanglement witness, noise robustness or real-city energy saving was measured.

Research references above support the literature claims only; the numerical measurements come from the code and JSON artifacts in this workspace.
