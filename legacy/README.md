# Microgrid Dial

**Quantum-assisted energy allocation for a village microgrid** (Qiskit Hack Day, Challenge 03: Quantum Optimization, Energy and Sustainability)
SDG 7 (affordable and clean energy) and SDG 13 (climate action).

## What it does
A village needs 50 MW. Four sources (Solar, Battery, Diesel, Grid) each run at 0, 10, 20 or 30 MW. We choose the levels that meet demand at the lowest **cost + w x CO2** score, then sweep the carbon weight `w` to draw a cost-vs-emissions trade-off curve (the "carbon dial").

```
plants.csv -> QuadraticProgram -> QUBO (slack + penalty) -> Ising Hamiltonian
           -> QAOA circuit (Qiskit, Aer simulator) <-> COBYLA tunes angles
           -> measure -> decode bitstrings to MW -> check feasibility and cost
           -> compare with exact brute force -> trade-off plot
```

## Quick start
```bash
python -m venv qiskit-env
source qiskit-env/bin/activate          # Windows: qiskit-env\Scripts\activate
pip install -r requirements.txt

python check_math.py                     # 1 s: exact checks, no simulator needed
python run_all.py --fast                 # ~30 s smoke test of the quantum path
python run_all.py                        # full run, results in ./results
jupyter notebook microgrid_dial.ipynb    # the demo notebook
```
Python 3.10+ recommended. Tested logic: see "Verification status" below.

## Files
| File | Purpose |
|---|---|
| `microgrid.py` | Library: data, baselines, QUBO, Ising, QAOA, decoding, plots |
| `check_math.py` | Exact tests of baseline, QUBO and Ising (no quantum backend) |
| `run_all.py` | One-command reproduction; writes `results/` |
| `microgrid_dial.ipynb` | Demo notebook (restart kernel and run all before presenting) |
| `data/plants.csv` | **Synthetic** plant data (edit freely) |

## The quantum component
* **Algorithm:** QAOA (depth p=2) with a COBYLA classical optimiser.
* **Mapping:** 2 qubits per source (levels 0..3 in 10 MW steps) = 8 decision qubits. The demand inequality adds 3 slack qubits, so **11 qubits** in total. The capacity limit is built into the encoding.
* **Penalty:** `lambda x (sum - slack - demand)^2` with `lambda = 2` (relative to a normalised objective). `penalty_study` shows that `lambda < 1` lets blackouts win.
* **Circuit:** Hadamards, then p x [cost-Hamiltonian evolution `exp(-i gamma H)`, RX(2 beta) mixer].
* **Role:** QAOA reshapes the probability distribution toward low-score allocations; the classical side sets up, tunes angles, decodes and verifies.
* **Metrics reported:** best feasible sampled allocation, probability of measuring an optimal allocation versus random guessing (1/256), probability of feasibility, gap to exact optimum.

## Qiskit usage
Libraries: `qiskit` (circuits, `PauliEvolutionGate`, `SparsePauliOp`), `qiskit-optimization` (`QuadraticProgram`, `QuadraticProgramToQubo`), `qiskit-aer` (`AerSimulator`, V2 Estimator and Sampler). If the Aer primitives fail to load, the code falls back to Qiskit's reference statevector primitives and prints which backend ran. QAOA is built directly from core Qiskit rather than `qiskit-algorithms`.

## Limitations (please read)
* Brute force solves this size instantly. We do **not** claim quantum advantage; this is a working, verified formulation and hybrid pipeline.
* Synthetic data, one dispatch hour, 4 sources, 10 MW resolution, noiseless simulator.
* QAOA is heuristic. Results depend on `--reps`, `--restarts`, `--maxiter`, seed and penalty.
* The carbon sweep yields a 3-step staircase (3 distinct allocations) because only a few allocations are optimal at this scale; several weights share a step.

## Verification status
* **Run and passing (`check_math.py`):** brute-force optimum, merit-order equivalence, hand-written QUBO ground state equals the optimum for all weights, Ising energies equal QUBO energies, penalty study.
* **Run with synthetic counts:** bitstring decoding, table and plots.
* **Self-checked at runtime on your machine:** `solve_weight` raises an error if Qiskit's QUBO ground state is not the optimum or if the Hamiltonian does not match the QUBO.
* Fill in your own measured numbers here after `python run_all.py`: qubits, P(optimal) per weight, runtime, backend.

## Acknowledgements
Qiskit, Qiskit Aer, Qiskit Optimization, NumPy, SciPy, Matplotlib. Plant data is invented for illustration. Parts of the code and documentation were drafted with AI assistance (Claude) and reviewed by the team.
