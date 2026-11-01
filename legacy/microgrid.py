"""
microgrid.py  -  core library for "Microgrid Dial"
===================================================
Quantum-assisted energy allocation for a village microgrid (Qiskit Hack Day, Challenge 03).

What this file does, in order:
  1. Loads the (synthetic) plant data.
  2. Classical baselines: exact brute force and a "merit order" heuristic.
  3. Builds the allocation problem as a Qiskit QuadraticProgram and converts it to a QUBO.
  4. Turns the QUBO into an Ising Hamiltonian and runs QAOA (our own circuit, Qiskit primitives).
  5. Decodes the measured bitstrings back into megawatts and checks cost and feasibility.

Design choices (so you can defend them to the judges)
-----------------------------------------------------
* Each plant has 2 bits  ->  4 output levels: 0, 10, 20, 30 MW.
  (bit0 counts 1 step, bit1 counts 2 steps, one step = 10 MW). The capacity limit (30 MW)
  is therefore built into the encoding and needs no extra constraint.
* The demand rule  "total output >= demand"  is an inequality. QUBO cannot hold inequalities,
  so Qiskit's QuadraticProgramToQubo adds slack bits and a squared penalty. We also write the same
  construction by hand (manual_qubo_terms) so the maths is visible and cross-checkable.
* The objective is normalised so the largest per-step coefficient is 1. This keeps QAOA's
  angles in a sensible range and makes the penalty weight easy to reason about.
* QAOA is built directly from Qiskit core (PauliEvolutionGate + RX mixer). Evaluation uses
  Qiskit Aer V2 primitives, with the reference statevector primitives as an automatic fallback.
"""
from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
STEP_MW = 10            # one output "step" in MW
BITS = 2                # bits per plant  ->  levels 0..3
MAX_LEVEL = 2 ** BITS - 1
DEMAND_MW = 50          # village demand for the dispatch hour
PENALTY_FACTOR = 2.0    # blackout penalty lambda, relative to the largest normalised coefficient (=1)
DEFAULT_WEIGHTS = [0.0, 0.01, 0.03, 0.05, 0.07, 0.15]   # carbon weights ($ per kg CO2)
DATA_FILE = Path(__file__).resolve().parent / "data" / "plants.csv"


@dataclass(frozen=True)
class Plant:
    name: str
    cost: float   # $ per MWh
    co2: float    # kg CO2 per MWh


def load_plants(path: Path | str = DATA_FILE) -> list[Plant]:
    with open(path, newline="") as f:
        return [Plant(r["name"].strip(), float(r["cost_per_mwh"]), float(r["co2_kg_per_mwh"]))
                for r in csv.DictReader(f)]


def var_name(plant: Plant, bit: int) -> str:
    return f"{plant.name}_b{bit}"


# --------------------------------------------------------------------------------------
# 1. Classical side: evaluation and baselines
# --------------------------------------------------------------------------------------
def evaluate(levels, plants, w: float) -> dict:
    """Cost ($/h), emissions (kg/h) and combined score for a tuple of per-plant levels."""
    mw = [int(l) * STEP_MW for l in levels]
    cost = sum(m * p.cost for m, p in zip(mw, plants))
    co2 = sum(m * p.co2 for m, p in zip(mw, plants))
    return {"levels": tuple(int(l) for l in levels), "mw": mw, "total_mw": sum(mw),
            "cost": cost, "co2": co2, "score": cost + w * co2}


def is_feasible(levels, demand_mw: int = DEMAND_MW) -> bool:
    return sum(levels) * STEP_MW >= demand_mw


def brute_force(plants, w: float, demand_mw: int = DEMAND_MW) -> dict:
    """Exact answer by checking all 4^n level combinations. This is our ground truth."""
    n = len(plants)
    best_key, best, feasible, scores = None, None, 0, {}
    for lv in itertools.product(range(MAX_LEVEL + 1), repeat=n):
        if not is_feasible(lv, demand_mw):
            continue
        feasible += 1
        r = evaluate(lv, plants, w)
        scores[lv] = r["score"]
        key = (round(r["score"], 9), r["total_mw"], lv)
        if best_key is None or key < best_key:
            best_key, best = key, r
    best = dict(best)
    best["all_optimal"] = {lv for lv, s in scores.items() if s <= best["score"] + 1e-9}
    best["n_feasible"], best["n_total"] = feasible, (MAX_LEVEL + 1) ** n
    return best


def merit_order(plants, w: float, demand_mw: int = DEMAND_MW) -> dict:
    """What a real dispatcher does: fill the cheapest (cost + w*CO2) sources first."""
    assert demand_mw % STEP_MW == 0, "demand must be a multiple of the step size"
    remaining = demand_mw // STEP_MW
    levels = [0] * len(plants)
    for i in sorted(range(len(plants)), key=lambda k: plants[k].cost + w * plants[k].co2):
        take = min(remaining, MAX_LEVEL)
        levels[i], remaining = take, remaining - take
    return evaluate(levels, plants, w)


# --------------------------------------------------------------------------------------
# 2. Problem formulation: QuadraticProgram -> QUBO
# --------------------------------------------------------------------------------------
def objective_scale(plants, w: float) -> float:
    """Largest cost-per-step among plants; dividing by it normalises coefficients to <= 1."""
    return max((p.cost + w * p.co2) * STEP_MW for p in plants)


def build_quadratic_program(plants, w: float, demand_mw: int = DEMAND_MW):
    """
    minimise   sum_i unit_i * (b0_i + 2*b1_i)          (normalised cost + carbon score)
    subject to sum_i (b0_i + 2*b1_i) >= demand / 10    (demand must be met)
    """
    from qiskit_optimization import QuadraticProgram

    scale = objective_scale(plants, w)
    qp = QuadraticProgram("microgrid_dial")
    obj, con = {}, {}
    for p in plants:
        unit = (p.cost + w * p.co2) * STEP_MW / scale
        for b in range(BITS):
            name = var_name(p, b)
            qp.binary_var(name)
            obj[name] = unit * (2 ** b)
            con[name] = 2 ** b
    qp.minimize(linear=obj)
    qp.linear_constraint(linear=con, sense=">=", rhs=demand_mw // STEP_MW, name="demand")
    return qp


def to_qubo(qp, penalty: float):
    """Qiskit adds integer slack bits and the squared penalty  penalty*(lhs - slack - rhs)^2."""
    from qiskit_optimization.converters import QuadraticProgramToQubo
    return QuadraticProgramToQubo(penalty=penalty).convert(qp)


def qubo_terms_from_qiskit(qubo, plants):
    """Extract (n, constant, linear vector, quadratic dict, bit_index) from a Qiskit QUBO."""
    n = qubo.get_num_vars()
    names = [v.name for v in qubo.variables]
    lin = np.zeros(n)
    for i, c in qubo.objective.linear.to_dict().items():
        lin[int(i)] += c
    quad = {}
    for (i, j), c in qubo.objective.quadratic.to_dict().items():
        i, j = int(i), int(j)
        if i == j:                 # x^2 = x for a binary variable
            lin[i] += c
        else:
            key = (min(i, j), max(i, j))
            quad[key] = quad.get(key, 0.0) + c
    bit_index = [[names.index(var_name(p, b)) for b in range(BITS)] for p in plants]
    return n, float(qubo.objective.constant), lin, quad, bit_index


def _slack_weights(slack_max: int) -> list[int]:
    weights, remaining, w = [], slack_max, 1
    while remaining > 0:
        t = min(w, remaining)
        weights.append(t)
        remaining -= t
        w *= 2
    return weights


def manual_qubo_terms(plants, w_carbon: float, demand_mw: int, penalty: float):
    """
    The same QUBO written by hand, so the penalty formula is transparent:
        E(x) = objective(x) + penalty * (sum_k a_k x_k - sum_j s_j y_j - rhs)^2
    x = decision bits, y = slack bits. Same layout as Qiskit: decision bits first, slack after.
    """
    rhs = demand_mw // STEP_MW
    scale = objective_scale(plants, w_carbon)
    n_dec = len(plants) * BITS
    a, obj = [], []
    for p in plants:
        unit = (p.cost + w_carbon * p.co2) * STEP_MW / scale
        for b in range(BITS):
            a.append(2 ** b)
            obj.append(unit * 2 ** b)
    slack = _slack_weights(len(plants) * MAX_LEVEL - rhs)
    v = np.array(a + [-s for s in slack], dtype=float)       # coefficients of the squared expression
    n = len(v)
    lin = np.zeros(n)
    lin[:n_dec] += np.array(obj)
    quad = {}
    for i in range(n):
        lin[i] += penalty * (v[i] ** 2 - 2 * rhs * v[i])
        for j in range(i + 1, n):
            quad[(i, j)] = penalty * 2 * v[i] * v[j]
    const = penalty * rhs ** 2
    bit_index = [[i * BITS + b for b in range(BITS)] for i in range(len(plants))]
    return n, const, lin, quad, bit_index


def qubo_energies(n, const, lin, quad):
    """Energy of every one of the 2^n bitstrings (row k: bit i = (k >> i) & 1 = qubit i)."""
    X = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    E = const + X @ lin
    for (i, j), c in quad.items():
        E += c * X[:, i] * X[:, j]
    return X, E


def decode_levels(x, bit_index) -> tuple:
    return tuple(int(sum(int(x[idx]) * 2 ** b for b, idx in enumerate(idxs))) for idxs in bit_index)


def ground_state_check(terms, plants, w: float, demand_mw: int = DEMAND_MW) -> dict:
    """Is the QUBO's lowest-energy state the true optimum (and feasible)? Exact, no quantum needed."""
    n, const, lin, quad, bit_index = terms
    X, E = qubo_energies(n, const, lin, quad)
    emin = E.min()
    ground = np.flatnonzero(E <= emin + 1e-9)
    decoded = [decode_levels(X[k], bit_index) for k in ground]
    opt = brute_force(plants, w, demand_mw)
    return {"ok": all(d in opt["all_optimal"] for d in decoded),
            "feasible": all(is_feasible(d, demand_mw) for d in decoded),
            "ground_levels": decoded[0], "n_ground": len(ground), "emin": float(emin)}


def penalty_study(plants, w: float, factors=(0.25, 0.5, 1.0, 2.0, 4.0, 16.0), demand_mw: int = DEMAND_MW):
    """Shows the 'Goldilocks' effect: too small a penalty makes blackouts the cheapest answer."""
    rows = []
    for f in factors:
        g = ground_state_check(manual_qubo_terms(plants, w, demand_mw, f), plants, w, demand_mw)
        rows.append({"penalty": f, "ground_levels": g["ground_levels"],
                     "feasible": g["feasible"], "optimal": g["ok"]})
    return rows


# --------------------------------------------------------------------------------------
# 3. QUBO -> Ising Hamiltonian  (x = (1 - Z)/2)
# --------------------------------------------------------------------------------------
def qubo_to_ising(n, const, lin, quad):
    """Return ([(qubit_indices, coefficient), ...], offset) with  E = offset + sum coeff * Z..Z."""
    z = -lin / 2.0
    offset = const + lin.sum() / 2.0
    zz = {}
    z = z.copy()
    for (i, j), c in quad.items():
        offset += c / 4.0
        z[i] -= c / 4.0
        z[j] -= c / 4.0
        zz[(i, j)] = zz.get((i, j), 0.0) + c / 4.0
    terms = [((i,), float(z[i])) for i in range(n) if abs(z[i]) > 1e-12]
    terms += [(ij, float(c)) for ij, c in zz.items() if abs(c) > 1e-12]
    return terms, float(offset)


def ising_energies(n, terms, offset):
    """Diagonal of the Ising Hamiltonian in plain numpy (used to cross-check the QUBO)."""
    X = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1)
    sign = 1 - 2 * X
    E = np.full(2 ** n, offset, dtype=float)
    for idxs, c in terms:
        E += c * np.prod(sign[:, list(idxs)], axis=1)
    return E


def to_sparse_pauli(n, terms):
    from qiskit.quantum_info import SparsePauliOp
    labels = []
    for idxs, c in terms:
        s = ["I"] * n
        for k in idxs:
            s[n - 1 - k] = "Z"          # Qiskit labels are little-endian: qubit 0 is the rightmost character
        labels.append(("".join(s), c))
    return SparsePauliOp.from_list(labels)


# --------------------------------------------------------------------------------------
# 4. Quantum side: QAOA with Qiskit primitives
# --------------------------------------------------------------------------------------
class QAOASolver:
    """
    QAOA ansatz:   |+>^n  ->  [ exp(-i * gamma_k * H_cost)  then  RX(2*beta_k) on every qubit ] x p
    A classical optimiser (COBYLA) tunes the 2p angles to minimise <H_cost>.
    """

    def __init__(self, n, terms, offset, reps: int = 2, seed: int = 7):
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.circuit.library import PauliEvolutionGate

        self.n, self.reps, self.offset, self.seed = n, reps, offset, seed
        self.H = to_sparse_pauli(n, terms)
        gammas, betas = ParameterVector("gamma", reps), ParameterVector("beta", reps)
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for k in range(reps):
            qc.append(PauliEvolutionGate(self.H, time=gammas[k]), list(range(n)))
            qc.rx(2 * betas[k], range(n))
        self.circuit = qc
        meas = qc.copy()
        meas.measure_all()
        self._setup_backend(qc, meas)
        self.theta = None
        self.history = []

    # -- backend ---------------------------------------------------------------------
    def _setup_backend(self, qc, meas):
        try:
            from qiskit_aer import AerSimulator
            from qiskit_aer.primitives import EstimatorV2, SamplerV2
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

            backend = AerSimulator()
            pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
            self.isa, self.isa_meas = pm.run(qc), pm.run(meas)
            self.obs = self.H.apply_layout(self.isa.layout)
            self.estimator = EstimatorV2()
            try:
                self.sampler = SamplerV2(seed=self.seed)
            except TypeError:
                self.sampler = SamplerV2()
            self.backend_name = "Qiskit Aer AerSimulator (V2 Estimator/Sampler)"
            self.expectation(np.zeros(2 * self.reps))          # probe: fail early, fall back below
        except Exception as exc:                               # noqa: BLE001  (deliberate broad fallback)
            from qiskit.primitives import StatevectorEstimator, StatevectorSampler
            self.isa, self.isa_meas, self.obs = qc, meas, self.H
            self.estimator = StatevectorEstimator()
            self.sampler = StatevectorSampler(seed=self.seed)
            self.backend_name = (f"qiskit.primitives Statevector reference simulator "
                                 f"(Aer path unavailable: {type(exc).__name__}: {exc})")

    def _values(self, theta):
        """theta = [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}] -> values in the circuit's parameter order."""
        named = {f"gamma[{k}]": theta[k] for k in range(self.reps)}
        named.update({f"beta[{k}]": theta[self.reps + k] for k in range(self.reps)})
        return np.array([named[par.name] for par in self.isa.parameters], dtype=float)

    # -- evaluation ------------------------------------------------------------------
    def expectation(self, theta) -> float:
        """<H_cost> without the constant offset."""
        job = self.estimator.run([(self.isa, self.obs, self._values(theta))])
        return float(np.asarray(job.result()[0].data.evs).reshape(-1)[0])

    def train(self, restarts: int = 4, maxiter: int = 100) -> dict:
        from scipy.optimize import minimize

        rng = np.random.default_rng(self.seed)
        best_val, best_theta, histories = np.inf, None, []
        for _ in range(restarts):
            theta0 = np.concatenate([rng.uniform(0.0, 1.0, self.reps),
                                     rng.uniform(0.0, np.pi / 2, self.reps)])
            hist = []

            def objective(th, hist=hist):
                v = self.expectation(th) + self.offset
                hist.append(v)
                return v

            res = minimize(objective, theta0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.4})
            histories.append(hist)
            if res.fun < best_val:
                best_val, best_theta = float(res.fun), res.x
        self.theta, self.history = best_theta, histories
        return {"energy": best_val, "theta": best_theta, "histories": histories}

    def sample(self, shots: int = 8192) -> dict:
        """Measure the trained circuit. Returns {bitstring: count} (bitstring is little-endian)."""
        job = self.sampler.run([(self.isa_meas, self._values(self.theta))], shots=shots)
        return dict(job.result()[0].data.meas.get_counts())

    def draw(self, path=None):
        try:
            fig = self.circuit.draw("mpl", fold=-1)
            if path:
                fig.savefig(path, bbox_inches="tight", dpi=150)
            return fig
        except Exception:                                      # pylatexenc missing etc.
            txt = str(self.circuit.draw("text", fold=120))
            if path:
                Path(path).with_suffix(".txt").write_text(txt)
            return txt


# --------------------------------------------------------------------------------------
# 5. Decoding results
# --------------------------------------------------------------------------------------
def analyse_counts(counts, bit_index, plants, w, demand_mw, optimal_levels) -> dict:
    shots = sum(counts.values())
    agg = {}
    for bits, c in counts.items():
        x = [int(ch) for ch in reversed(bits.replace(" ", ""))]          # x[i] = value of qubit i
        lv = decode_levels(x, bit_index)
        agg[lv] = agg.get(lv, 0) + c
    best, p_feas, p_opt = None, 0, 0
    for lv, c in agg.items():
        if is_feasible(lv, demand_mw):
            p_feas += c
            r = evaluate(lv, plants, w)
            if best is None or r["score"] < best["score"] - 1e-9:
                best = r
        if lv in optimal_levels:
            p_opt += c
    top_lv, top_c = max(agg.items(), key=lambda kv: kv[1])
    return {"best": best, "p_feasible": p_feas / shots, "p_optimal": p_opt / shots,
            "most_frequent": evaluate(top_lv, plants, w), "p_most_frequent": top_c / shots,
            "n_distinct": len(agg), "shots": shots}


def solve_weight(plants, w: float, demand_mw: int = DEMAND_MW, penalty_factor: float = PENALTY_FACTOR,
                 reps: int = 2, restarts: int = 4, maxiter: int = 100, shots: int = 8192,
                 seed: int = 7, verbose: bool = True) -> dict:
    """Full pipeline for one carbon weight: exact answer -> QUBO -> QAOA -> decoded comparison."""
    exact = brute_force(plants, w, demand_mw)
    greedy = merit_order(plants, w, demand_mw)

    qp = build_quadratic_program(plants, w, demand_mw)
    qubo = to_qubo(qp, penalty_factor)
    terms = qubo_terms_from_qiskit(qubo, plants)
    n, const, lin, quad, bit_index = terms

    # Validation 1: the QUBO's lowest-energy state must be the true optimum.
    g = ground_state_check(terms, plants, w, demand_mw)
    if not (g["ok"] and g["feasible"]):
        raise RuntimeError(f"QUBO ground state {g['ground_levels']} is not the optimum for w={w}. "
                           f"Increase penalty_factor (currently {penalty_factor}).")
    # Informational: does Qiskit's QUBO match our hand-written one?
    man = manual_qubo_terms(plants, w, demand_mw, penalty_factor)
    same = (man[0] == n) and np.allclose(qubo_energies(*man[:4])[1], qubo_energies(n, const, lin, quad)[1])

    # Validation 2: the Ising Hamiltonian must reproduce the QUBO energies exactly.
    ising, offset = qubo_to_ising(n, const, lin, quad)
    if not np.allclose(ising_energies(n, ising, offset), qubo_energies(n, const, lin, quad)[1]):
        raise RuntimeError("Ising Hamiltonian does not match the QUBO. Check qubit ordering.")

    solver = QAOASolver(n, ising, offset, reps=reps, seed=seed)
    if verbose:
        print(f"w={w:<5} qubits={n} (decision {len(plants) * BITS} + slack {n - len(plants) * BITS})  "
              f"backend: {solver.backend_name}")
    tr = solver.train(restarts=restarts, maxiter=maxiter)
    counts = solver.sample(shots)
    an = analyse_counts(counts, bit_index, plants, w, demand_mw, exact["all_optimal"])

    best = an["best"]
    gap = None if best is None else 100.0 * (best["score"] - exact["score"]) / exact["score"]
    return {"w": w, "n_qubits": n, "exact": exact, "greedy": greedy, "qaoa_best": best,
            "match": bool(best and best["levels"] in exact["all_optimal"]), "gap_pct": gap,
            "p_optimal": an["p_optimal"], "p_feasible": an["p_feasible"],
            "p_optimal_random": len(exact["all_optimal"]) / exact["n_total"],
            "p_feasible_random": exact["n_feasible"] / exact["n_total"],
            "most_frequent": an["most_frequent"], "p_most_frequent": an["p_most_frequent"],
            "energy": tr["energy"], "histories": tr["histories"], "counts": counts,
            "qubo_matches_manual": bool(same), "backend": solver.backend_name,
            "solver": solver, "bit_index": bit_index}


# --------------------------------------------------------------------------------------
# 6. Reporting
# --------------------------------------------------------------------------------------
def format_table(rows, plants) -> str:
    names = "/".join(p.name for p in plants)
    head = (f"{'w':>5} | {'levels ('+names+')':<26} | {'cost $/h':>9} | {'CO2 kg/h':>9} | "
            f"{'exact levels':<14} | {'match':<5} | {'P(opt)':>7} | {'random':>7} | {'P(feas)':>7}")
    out = [head, "-" * len(head)]
    for r in rows:
        b = r["qaoa_best"]
        lv = str(b["levels"]) if b else "none feasible"
        out.append(f"{r['w']:>5} | {lv:<26} | {(b['cost'] if b else float('nan')):>9.0f} | "
                   f"{(b['co2'] if b else float('nan')):>9.0f} | {str(r['exact']['levels']):<14} | "
                   f"{'YES' if r['match'] else 'no':<5} | {r['p_optimal']:>7.1%} | "
                   f"{r['p_optimal_random']:>7.2%} | {r['p_feasible']:>7.1%}")
    return "\n".join(out)


def plot_tradeoff(rows, path=None, show: bool = False):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    rows = sorted(rows, key=lambda r: r["w"])

    groups = {}
    for r in rows:
        groups.setdefault((r["exact"]["cost"], r["exact"]["co2"]), []).append(r["w"])
    pts = sorted(groups)
    ax1.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color="#1F3A5F", mfc="white", ms=11,
             label="Exact (brute force)")
    q = [(r["qaoa_best"]["cost"], r["qaoa_best"]["co2"]) for r in rows if r["qaoa_best"]]
    ax1.scatter([a for a, _ in q], [b for _, b in q], marker="x", s=70, color="#D1495B", zorder=5,
                label="QAOA best feasible sample")
    for (c, e), ws in groups.items():
        ax1.annotate("w = " + ", ".join(f"{x:g}" for x in ws), (c, e), textcoords="offset points",
                     xytext=(10, 8), fontsize=8)
    ax1.margins(x=0.12, y=0.15)
    ax1.set_xlabel(r"Generation cost (\$ per hour)")
    ax1.set_ylabel("Carbon emissions (kg CO$_2$ per hour)")
    ax1.set_title("Carbon dial: cost vs emissions trade-off")
    ax1.grid(alpha=0.3)
    ax1.legend()

    xs = np.arange(len(rows))
    ax2.bar(xs, [r["p_optimal"] for r in rows], color="#1D7A74", label="QAOA: P(measure an optimal allocation)")
    ax2.plot(xs, [r["p_optimal_random"] for r in rows], "k--", marker="_", ms=18,
             label="Random guessing")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{r['w']:g}" for r in rows])
    ax2.set_xlabel(r"Carbon weight w (\$ per kg CO$_2$)")
    ax2.set_ylabel("Probability")
    ax2.set_title("How much probability QAOA puts on the optimum")
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
    if show:
        plt.show()
    return fig


def plot_convergence(histories, path=None, show: bool = False):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, h in enumerate(histories):
        ax.plot(h, label=f"restart {k + 1}")
    ax.set_xlabel("COBYLA iteration")
    ax.set_ylabel("QAOA energy <H> (lower is better)")
    ax.set_title("Classical optimiser tuning the quantum circuit")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
    if show:
        plt.show()
    return fig
