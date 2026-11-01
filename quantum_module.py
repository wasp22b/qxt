"""
Quantum module (Feature 10): QUBO -> QAOA -> decode -> LP repair.

Where quantum computing actually fits
-------------------------------------
The day-scale allocation problem is linear and continuous, so the classical LP
in :mod:`energy_allocation` is the right engine for it -- it is exact and
solves in milliseconds.  Quantum optimisation is aimed at the *combinatorial*
layer underneath:

    * which sources run, and at what output level   (unit-commitment style)
    * which zones get shed during a shortage        (load-shedding blocks)

Those are discrete decisions.  This module encodes one time slot as a QUBO,
solves it with QAOA, decodes the bitstring, and then hands the discrete
decision to a classical LP "repair" step that computes exact, feasible energy
flows.  That hybrid quantum-classical split is the point of the architecture:
quantum picks the combination, classical guarantees feasibility.

Encoding (one time slot)
------------------------
    * 2 bits per source  -> output level in {0, 1/3, 2/3, 1} x availability
    * 1 bit per zone     -> 1 = serve fully, 0 = shed
    * n = 2*S + Z qubits.  For the default model (5 sources, 4 zones) that is
      **14 qubits / 16384 states**, small enough to brute-force exactly and use
      as ground truth for validating QAOA.

QUBO energy
-----------
    E(b) = generation_cost + lambda * (supply - served_demand)^2
                   + sum_z W_z * D_z * (1 - serve_z)

W_z is W_CRIT for critical zones, so shedding the hospital costs ~200,000 --
far more than any generation cost.  It is therefore shed only when physically
unavoidable, which is exactly the critical-protection guarantee (Feature 6).

Honest scope: QAOA here runs on a statevector *simulator*.  That is still
classical computation; what it validates is the formulation.  Running on real
hardware (IBM Quantum / D-Wave) is the next step -- :func:`qubo_to_ising` and
:func:`export_qubo` produce the formats those backends want.

Usage::

    python quantum_module.py                    # compare on the evening slot
    python quantum_module.py --slot 1 --reps 3  # different slot, deeper QAOA
    python quantum_module.py --no-quantum       # QUBO + brute force only
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog, minimize

from energy_allocation import (
    SLOTS,
    W_CRIT,
    W_FAIR,
    default_sources,
    default_zones,
)

__all__ = [
    "LEVEL_BITS",
    "build_slot_qubo",
    "qubo_energy",
    "qubo_to_ising",
    "export_qubo",
    "decode",
    "brute_force",
    "greedy_solution",
    "reduction_plan",
    "fix_qubo_bits",
    "expand_bits",
    "normalize_operator",
    "build_qaoa_circuit",
    "qaoa_solve",
    "counts_to_bits",
    "repair_flows",
    "compare",
]

#: default number of bits used per source to encode its output level
LEVEL_BITS = 2


# --------------------------------------------------------------------------
# QUBO construction
# --------------------------------------------------------------------------
def build_slot_qubo(
    t: int,
    sources: Dict[str, Any] | None = None,
    zones: Dict[str, Any] | None = None,
    penalty: float = 5_000.0,
    factor: float = 1.0,
    level_bits: int = LEVEL_BITS,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build the QUBO matrix for a single time slot.

    ``level_bits`` controls how finely a source's output is quantised:
    2 bits -> levels {0, 1/3, 2/3, 1}, 3 bits -> {0, 1/7, ..., 1}.  More bits
    mean a finer commitment grid (closer to the continuous LP optimum) but
    ``level_bits * S`` qubits instead of ``2 * S``.

    Returns ``(Q, meta)`` where ``Q[i, j]`` holds the coefficient of
    ``b_i * b_j`` and ``meta`` documents the bit layout, the constant offset and
    the problem data used.  The true objective is ``b @ Q @ b + meta["offset"]``.
    """
    if level_bits < 1:
        raise ValueError("level_bits must be >= 1")
    sources = default_sources() if sources is None else sources
    zones = default_zones() if zones is None else zones
    src, zon = list(sources), list(zones)
    S, Z = len(src), len(zon)
    if not 0 <= t < len(SLOTS):
        raise ValueError(f"slot index {t} out of range for {list(SLOTS)}")

    avail = np.array([float(sources[s]["avail"][t]) for s in src])
    demand = np.array([factor * float(zones[z]["demand"][t]) for z in zon])
    cost = np.array([float(sources[s]["cost"]) for s in src])
    weight = np.array([W_CRIT if zones[z].get("critical") else W_FAIR for z in zon])

    n = level_bits * S + Z
    span = float((1 << level_bits) - 1)  # 3 for 2 bits -> levels 0,1/3,2/3,3/3

    linear = np.zeros(n)  # intrinsic cost of setting a bit
    volume = np.zeros(n)  # signed kWh contribution to the balance
    for s in range(S):
        block = avail[s] / span
        for b in range(level_bits):
            i = level_bits * s + b
            level = block * (1 << b)
            linear[i] = cost[s] * level  # generation cost
            volume[i] = level  # supply side of the balance
    for z in range(Z):
        i = level_bits * S + z
        # shed cost W*D*(1 - serve) = W*D - W*D*serve  -> linear term -W*D*serve
        linear[i] = -weight[z] * demand[z]
        volume[i] = -demand[z]  # demand side of the balance

    # penalty * (sum_i volume_i * b_i)^2  -- expands using b_i^2 = b_i
    Q = np.zeros((n, n))
    np.fill_diagonal(Q, linear + penalty * volume**2)
    iu = np.triu_indices(n, k=1)
    Q[iu] = 2.0 * penalty * np.outer(volume, volume)[iu]

    meta = {
        "slot_index": t,
        "slot": SLOTS[t],
        "n_qubits": n,
        "n_states": 1 << n,
        "level_bits": int(level_bits),
        "penalty": float(penalty),
        "factor": float(factor),
        "offset": float((weight * demand).sum()),
        "sources": src,
        "zones": zon,
        "availability": avail.tolist(),
        "demand": demand.tolist(),
        "cost": cost.tolist(),
        "weights": weight.tolist(),
        "bit_labels": (
            [f"{src[s]}:level_bit{b}" for s in range(S) for b in range(level_bits)]
            + [f"{zon[z]}:serve" for z in range(Z)]
        ),
        "linear": linear.tolist(),
        "volume": volume.tolist(),
    }
    return Q, meta


def qubo_energy(Q: np.ndarray, bits: Sequence[int]) -> float:
    """Evaluate ``b @ Q @ b`` for a bitstring (excludes the constant offset)."""
    b = np.asarray(bits, dtype=float)
    return float(b @ Q @ b)


def qubo_to_ising(Q: np.ndarray) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    """Convert a QUBO to Ising (h, J, offset) via b_i = (1 - z_i) / 2.

    This is the form D-Wave / IBM backends consume, so it is the bridge from
    simulator to real hardware.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    a = np.diag(Q).copy()  # linear
    bmat = np.triu(Q, k=1)  # quadratic
    h: Dict[int, float] = {}
    J: Dict[Tuple[int, int], float] = {}
    offset = 0.0
    for i in range(n):
        # a_i * (1 - z_i)/2  ->  -a_i/2 * z_i  + a_i/2
        h[i] = h.get(i, 0.0) - a[i] / 2.0
        offset += a[i] / 2.0
        for j in range(i + 1, n):
            q = bmat[i, j]
            if q == 0.0:
                continue
            # q * (1-z_i)/2 * (1-z_j)/2 = q/4 (1 - z_i - z_j + z_i z_j)
            h[i] = h.get(i, 0.0) - q / 4.0
            h[j] = h.get(j, 0.0) - q / 4.0
            J[(i, j)] = J.get((i, j), 0.0) + q / 4.0
            offset += q / 4.0
    return h, J, offset


def export_qubo(Q: np.ndarray, meta: Dict[str, Any], path: str) -> str:
    """Write the QUBO (dense + Ising forms) to JSON for offline/hardware use."""
    h, J, offset = qubo_to_ising(Q)
    payload = {
        "meta": {k: v for k, v in meta.items()},
        "qubo": {
            "n": int(Q.shape[0]),
            "diagonal": np.diag(Q).tolist(),
            "upper_triangle": [
                {"i": int(i), "j": int(j), "value": float(Q[i, j])}
                for i, j in zip(*np.triu_indices(Q.shape[0], k=1))
                if Q[i, j] != 0.0
            ],
            "offset": meta.get("offset", 0.0),
        },
        "ising": {
            "h": {str(k): v for k, v in h.items()},
            "J": {f"{i},{j}": v for (i, j), v in J.items()},
            "offset": offset,
        },
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------
def decode(
    bits: Sequence[int],
    t: int,
    sources: Dict[str, Any] | None = None,
    zones: Dict[str, Any] | None = None,
    level_bits: int = LEVEL_BITS,
) -> Tuple[np.ndarray, List[int]]:
    """Turn a bitstring into (source output levels, indices of served zones)."""
    sources = default_sources() if sources is None else sources
    zones = default_zones() if zones is None else zones
    src, zon = list(sources), list(zones)
    S, Z = len(src), len(zon)
    span = float((1 << level_bits) - 1)
    bits = [int(round(float(x))) for x in bits]
    if len(bits) != level_bits * S + Z:
        raise ValueError(f"expected {level_bits * S + Z} bits, got {len(bits)}")
    levels = np.array(
        [
            float(sources[src[s]]["avail"][t])
            * sum(bits[level_bits * s + b] * (1 << b) for b in range(level_bits))
            / span
            for s in range(S)
        ]
    )
    served = [z for z in range(Z) if bits[level_bits * S + z] > 0.5]
    return levels, served


def bits_to_string(bits: Sequence[int], meta: Dict[str, Any]) -> str:
    """Human-readable rendering of a bitstring (handy in the dashboard)."""
    S = len(meta["sources"])
    lb = int(meta.get("level_bits", LEVEL_BITS))
    parts = []
    for s in range(S):
        chunk = [int(bits[lb * s + b]) for b in range(lb)]
        lvl = sum(c * (1 << b) for b, c in enumerate(chunk)) / float((1 << lb) - 1)
        parts.append(f"{meta['sources'][s]}={lvl:.0%}")
    zbits = [int(bits[lb * S + z]) for z in range(len(meta["zones"]))]
    served = [meta["zones"][z] for z, v in enumerate(zbits) if v]
    shed = [meta["zones"][z] for z, v in enumerate(zbits) if not v]
    parts.append("serve=" + ("[" + ",".join(served) + "]" if served else "[]"))
    if shed:
        parts.append("shed=[" + ",".join(shed) + "]")
    return "  ".join(parts)


# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------
def brute_force(Q: np.ndarray, chunk: int = 1 << 16) -> Tuple[float, np.ndarray]:
    """Exact QUBO ground state by enumerating all 2^n bitstrings.

    Vectorised and processed in chunks so memory stays flat: the default model
    is 14 qubits (16384 states, ~10 ms), and 3 level bits gives 19 qubits
    (524288 states, still well under a second).  This is the ground truth used
    to verify that QAOA reproduces the optimum.
    """
    n = int(Q.shape[0])
    if n > 24:
        raise ValueError(
            f"brute force over 2^{n} states is too large; use QAOA/annealing"
        )
    total = 1 << n
    shifts = np.arange(n, dtype=np.int64)
    best_e, best_b = np.inf, None
    for start in range(0, total, chunk):
        states = np.arange(start, min(start + chunk, total), dtype=np.int64)
        bits = ((states[:, None] >> shifts[None, :]) & 1).astype(np.float64)
        energies = np.einsum("bi,ij,bj->b", bits, Q, bits, optimize=True)
        k = int(np.argmin(energies))
        if energies[k] < best_e:
            best_e, best_b = float(energies[k]), bits[k].copy()
    return best_e, best_b


def greedy_solution(
    t: int,
    sources: Dict[str, Any] | None = None,
    zones: Dict[str, Any] | None = None,
    factor: float = 1.0,
    level_bits: int = LEVEL_BITS,
) -> np.ndarray:
    """A naive classical heuristic, used as a baseline in the comparison table.

    Commit sources cheapest-first at the highest quantised level that does not
    overshoot the remaining need, then serve zones critical-first while supply
    lasts.  It is *not* optimal -- it minimises generation cost while ignoring
    the shedding penalty -- which is precisely why the QUBO objective, not the
    generation cost, is the right thing to compare.
    """
    sources = default_sources() if sources is None else sources
    zones = default_zones() if zones is None else zones
    src, zon = list(sources), list(zones)
    S, Z = len(src), len(zon)
    span = float((1 << level_bits) - 1)
    bits = np.zeros(level_bits * S + Z)

    demand = np.array([factor * float(zones[z]["demand"][t]) for z in zon])
    needed = float(demand.sum())
    supply = 0.0

    for s in sorted(range(S), key=lambda s: float(sources[src[s]]["cost"])):
        avail = float(sources[src[s]]["avail"][t])
        best = 0
        for lvl in range(int(span) + 1):
            if avail * lvl / span <= needed - supply + 1e-9:
                best = lvl
        supply += avail * best / span
        for b in range(level_bits):
            bits[level_bits * s + b] = (best >> b) & 1

    for z in sorted(
        range(Z),
        key=lambda z: (0 if zones[zon[z]].get("critical") else 1, -demand[z]),
    ):
        if demand[z] <= supply + 1e-9:
            bits[level_bits * S + z] = 1
            supply -= demand[z]
    return bits


# --------------------------------------------------------------------------
# Classical preprocessing: variable fixing
# --------------------------------------------------------------------------
def reduction_plan(
    t: int,
    sources: Dict[str, Any] | None = None,
    zones: Dict[str, Any] | None = None,
    factor: float = 1.0,
    level_bits: int = LEVEL_BITS,
) -> Tuple[Dict[int, int], List[str]]:
    """Decide which QUBO bits are provably determined before running QAOA.

    Real quantum-classical pipelines shrink and condition the QUBO first.  Two
    rules are safe here:

    * a source with zero availability in this slot cannot run -> its level bits
      are 0;
    * if total availability covers total demand, no zone needs shedding -> every
      serve bit is 1.  Otherwise, if it at least covers critical demand, the
      critical serve bits are 1 (Feature 6, decided classically because it is
      not actually a choice).

    Fixing the zone bits removes the W_CRIT * D coefficients (~2e5) from the
    Hamiltonian, which is what makes QAOA concentrate: on the evening slot this
    takes the problem from 14 to 10 qubits and raises the probability of
    sampling the optimum by roughly two orders of magnitude.
    """
    sources = default_sources() if sources is None else sources
    zones = default_zones() if zones is None else zones
    src, zon = list(sources), list(zones)
    S, Z = len(src), len(zon)

    fixed: Dict[int, int] = {}
    reasons: List[str] = []

    for s_idx in range(S):
        if float(sources[src[s_idx]]["avail"][t]) <= 0.0:
            for b in range(level_bits):
                fixed[level_bits * s_idx + b] = 0
            reasons.append(f"{src[s_idx]}: no availability in this slot -> level bits = 0")

    avail_total = sum(float(sources[x]["avail"][t]) for x in src)
    demand_total = factor * sum(float(zones[z]["demand"][t]) for z in zon)
    crit_total = factor * sum(
        float(zones[z]["demand"][t]) for z in zon if zones[z].get("critical")
    )

    if avail_total >= demand_total - 1e-9:
        for z in range(Z):
            fixed[level_bits * S + z] = 1
        reasons.append(
            f"availability {avail_total:.1f} >= demand {demand_total:.1f} "
            "-> no zone needs shedding, all serve bits = 1"
        )
    elif avail_total >= crit_total - 1e-9:
        for z in range(Z):
            if zones[zon[z]].get("critical"):
                fixed[level_bits * S + z] = 1
                reasons.append(
                    f"{zon[z]} is critical and availability {avail_total:.1f} covers "
                    f"critical demand {crit_total:.1f} -> serve bit = 1"
                )
    return fixed, reasons


def fix_qubo_bits(
    Q: np.ndarray, fixed: Dict[int, int]
) -> Tuple[np.ndarray, float, List[int]]:
    """Substitute fixed bits into a QUBO, returning the reduced matrix.

    Returns ``(Q_reduced, offset, keep)`` such that for any full bitstring built
    by placing ``bits`` at positions ``keep`` and the fixed values at their
    indices::

        b_full @ Q @ b_full == b_red @ Q_reduced @ b_red + offset
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    fixed = {int(k): int(v) for k, v in fixed.items()}
    keep = [i for i in range(n) if i not in fixed]
    if not fixed:
        return Q.copy(), 0.0, keep

    linear = np.zeros(len(keep))
    offset = 0.0
    for k, value in fixed.items():
        if value != 1:
            continue  # a fixed 0 contributes nothing
        offset += Q[k, k]
        for pos, i in enumerate(keep):
            linear[pos] += Q[min(i, k), max(i, k)]
        # pairs of fixed bits -- count each unordered pair exactly once
        for k2, v2 in fixed.items():
            if k2 > k and v2 == 1:
                offset += Q[min(k, k2), max(k, k2)]

    Qr = Q[np.ix_(keep, keep)].astype(float, copy=True)
    Qr[np.diag_indices_from(Qr)] += linear
    return Qr, float(offset), keep


def expand_bits(
    bits: Sequence[int], keep: Sequence[int], fixed: Dict[int, int], n: int
) -> np.ndarray:
    """Rebuild a full-length bitstring from a reduced one plus the fixed bits."""
    full = np.zeros(n, dtype=float)
    for pos, i in enumerate(keep):
        full[i] = float(bits[pos])
    for k, v in fixed.items():
        full[k] = float(v)
    return full


def normalize_operator(op):
    """Scale a SparsePauliOp so its largest non-identity coefficient is 1.

    Rescaling by a positive constant does not change the ground state, but it
    puts QAOA's gamma/beta angles on a scale the optimiser can actually search.

    Returns ``(scaled_op, scale)`` where ``scale`` converts expectations of the
    scaled operator back into the original energy units.
    """
    coeffs = np.asarray(op.coeffs)  # complex; use magnitudes, never cast
    labels = op.paulis.to_labels()
    non_identity = np.array([set(label) != {"I"} for label in labels])
    if not non_identity.any():
        return op, 1.0
    scale = float(np.abs(coeffs[non_identity]).max())
    if scale <= 0.0 or not np.isfinite(scale):
        return op, 1.0
    return op / scale, scale


def _cost_hamiltonian(Q: np.ndarray):
    """Build the SparsePauliOp for the QUBO cost function (Qiskit required).

    Uses ``b_i = (1 - Z_i) / 2`` so that ``<H>`` on a computational basis state
    equals ``b @ Q @ b`` *exactly*, including the constant identity term.  That
    matters: it lets QAOA's energy be compared directly against the brute-force
    ground state without any bookkeeping offset.
    """
    from qiskit.quantum_info import SparsePauliOp

    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]

    def zlabel(qubits: Iterable[int]) -> str:
        qs = set(qubits)
        # Qiskit labels are big-endian: leftmost character = highest qubit
        return "".join("Z" if q in qs else "I" for q in reversed(range(n)))

    coeffs: Dict[str, float] = {}

    def add(label: str, value: float) -> None:
        coeffs[label] = coeffs.get(label, 0.0) + float(value)

    constant = 0.0
    for i in range(n):
        # Q_ii * (1 - Z_i)/2  ->  -Q_ii/2 * Z_i  +  Q_ii/2
        add(zlabel([i]), -Q[i, i] / 2.0)
        constant += Q[i, i] / 2.0
        for j in range(i + 1, n):
            q = Q[i, j]
            if q == 0.0:
                continue
            # Q_ij/4 * (1 - Z_i - Z_j + Z_i Z_j)
            add(zlabel([i]), -q / 4.0)
            add(zlabel([j]), -q / 4.0)
            add(zlabel([i, j]), q / 4.0)
            constant += q / 4.0
    add("I" * n, constant)

    terms = [(k, v) for k, v in coeffs.items() if abs(v) > 1e-12]
    return SparsePauliOp.from_list(terms).simplify()


def build_qaoa_circuit(
    Q: np.ndarray,
    betas: Sequence[float],
    gammas: Sequence[float],
):
    """Construct the QAOA ansatz explicitly from the QUBO's Ising form.

    Built by hand rather than with ``qiskit.circuit.library.QAOAAnsatz`` for
    three reasons:

    * ``QAOAAnsatz`` is deprecated as of Qiskit 2.1 and slated for removal in 3.0;
    * it wraps the cost operator in a ``PauliEvolutionGate``, which a statevector
      simulator densifies into a ``2**n x 2**n`` matrix -- 4 GiB at 14 qubits;
    * explicit ``RZ``/``RZZ``/``RX`` gates are transpilable straight to real
      hardware and make the circuit structure visible.

    Conventions (textbook QAOA)::

        |psi> = prod_p [ exp(-i beta_p X_sum) exp(-i gamma_p H_C) ] |+>^n
        H_C   = sum_i h_i Z_i + sum_ij J_ij Z_i Z_j

    so ``exp(-i gamma h Z) = RZ(2 gamma h)`` and
    ``exp(-i gamma J Z_i Z_j) = RZZ(2 gamma J)``.
    """
    from qiskit import QuantumCircuit

    h, J, _ = qubo_to_ising(Q)
    n = Q.shape[0]
    if len(betas) != len(gammas):
        raise ValueError("betas and gammas must have the same length (= reps)")

    qc = QuantumCircuit(n, name=f"QAOA(p={len(betas)})")
    qc.h(range(n))  # |+>^n
    for beta, gamma in zip(betas, gammas):
        for (i, j), coupling in J.items():  # cost: entangling terms
            if coupling != 0.0:
                qc.rzz(2.0 * gamma * coupling, i, j)
        for i, field in h.items():  # cost: local fields
            if field != 0.0:
                qc.rz(2.0 * gamma * field, i)
        qc.rx(2.0 * beta, range(n))  # mixer
    return qc


def qaoa_solve(
    Q: np.ndarray,
    reps: int = 2,
    restarts: int = 12,
    shots: int = 8192,
    seed: int = 7,
    maxiter: int = 150,
    pool_restarts: bool = True,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Run QAOA on a statevector simulator and return measurement counts.

    Two things make this reliable, because a single COBYLA run on a
    penalty-encoded QUBO regularly lands in a poor local minimum:

    * ``restarts`` independent random initialisations of (beta, gamma).  Betas
      are drawn from [0, pi/2] -- the physically distinct range for the RX(2b)
      mixer -- and gammas from a full turn.
    * ``pool_restarts`` samples from *every* restart and merges the counts,
      instead of sampling only from the lowest-expectation one.

    Measured best-of-shots accuracy (does QAOA reproduce the exact brute-force
    optimum?) over seeds 1, 7, 11, 23, 42, 99:

        no pooling, 6 restarts :  evening 5/6,  shortage 4/6
        pooling,    6 restarts :  evening 6/6,  shortage 5/6
        pooling,   12 restarts :  evening 6/6,  shortage 6/6   <- default

    Returns ``(counts, info)``.
    """
    try:
        from qiskit.primitives import StatevectorEstimator, StatevectorSampler
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Qiskit is required for QAOA. Install it with `pip install qiskit`, "
            "or run with --no-quantum to use the classical solvers only."
        ) from exc

    op = _cost_hamiltonian(Q)
    # QAOA evolves exp(-i * gamma * H).  This QUBO's coefficients span many
    # orders of magnitude (a hospital shed penalty of ~2e5 next to generation
    # costs of ~1), so gamma in [0, 2*pi] would be meaningless on the raw
    # Hamiltonian.  Dividing by the largest coefficient leaves the ground state
    # unchanged and puts the angles on a searchable scale.
    op_norm, scale = normalize_operator(op)
    rng = np.random.default_rng(seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator = StatevectorEstimator(seed=seed)

        def objective(theta: np.ndarray) -> float:
            theta = np.asarray(theta, dtype=float)
            betas, gammas = theta[:reps], theta[reps:]
            circuit = build_qaoa_circuit(Q, betas, gammas)
            evs = estimator.run([(circuit, op_norm)]).result()[0].data.evs
            return float(np.real(evs)) * scale

        def sample_angles() -> np.ndarray:
            # betas drive RX(2*beta): the physically distinct range is [0, pi/2].
            # gammas drive exp(-i*gamma*H_C) and are searched over a full turn.
            # Seeding beta over [0, 2*pi) lands the optimiser in redundant
            # regions and measurably degrades the result.
            return np.concatenate(
                [
                    rng.uniform(0.0, np.pi / 2.0, reps),
                    rng.uniform(0.0, 2.0 * np.pi, reps),
                ]
            )

        results = [
            minimize(
                objective,
                sample_angles(),
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 0.2},
            )
            for _ in range(max(1, restarts))
        ]
        ranked = sorted(results, key=lambda r: float(r.fun))
        best = ranked[0]

        sampler = StatevectorSampler(seed=seed)
        chosen = ranked if pool_restarts else ranked[:1]
        per_restart = max(256, int(round(shots / len(chosen))))
        sampled_counts: Dict[str, int] = {}
        for res in chosen:
            b, g = res.x[:reps], res.x[reps:]
            circuit = build_qaoa_circuit(Q, b, g)
            circuit.measure_all()  # creates the 'meas' classical register
            counts = sampler.run([circuit], shots=per_restart).result()[0].data.meas.get_counts()
            for key, value in counts.items():
                sampled_counts[key] = sampled_counts.get(key, 0) + int(value)
        sampled = sampled_counts

    betas, gammas = best.x[:reps], best.x[reps:]
    info = {
        "best_energy": float(best.fun),
        "restart_energies": [float(r.fun) for r in ranked],
        "hamiltonian_scale": scale,
        "betas": [float(x) for x in betas],
        "gammas": [float(x) for x in gammas],
        "pooled": bool(pool_restarts),
        "shots_per_restart": int(per_restart),
        "total_shots": int(sum(sampled.values())),
        "restarts": len(results),
        "reps": reps,
        "shots": shots,
        "seed": seed,
        "num_parameters": int(2 * reps),
        "num_qubits": int(Q.shape[0]),
        "optimizer_evals": int(best.nfev),
        "hamiltonian_terms": int(op.size),
        "converged": bool(best.success),
    }
    return dict(sampled), info


def counts_to_bits(counts: Dict[str, int], n: int, top_k: int = 5) -> List[Tuple[str, np.ndarray, int]]:
    """Convert Qiskit counts into bit arrays, most-frequent first.

    Qiskit uses little-endian bit ordering (rightmost character = qubit 0), so
    the string is reversed to line up with our QUBO index convention.
    """
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out = []
    for bitstring, count in ranked:
        raw = bitstring.replace(" ", "")
        bits = np.array([int(c) for c in raw[::-1]], dtype=float)
        if len(bits) != n:
            raise ValueError(f"counts have {len(bits)} bits, expected {n}")
        out.append((raw, bits, int(count)))
    return out


# --------------------------------------------------------------------------
# Hybrid step: classical LP repair
# --------------------------------------------------------------------------
def repair_flows(
    t: int,
    levels: Sequence[float],
    served: Sequence[int],
    sources: Dict[str, Any] | None = None,
    zones: Dict[str, Any] | None = None,
    factor: float = 1.0,
) -> Dict[str, Any]:
    """Compute exact energy flows for a discrete (source level, zone shed) plan.

    This is the classical half of the hybrid loop: the quantum optimiser decides
    *which* sources run at *what level* and *which* zones are served, and this LP
    turns that into a physically feasible dispatch.

    Two passes:
      1. strict  -- every served zone must get its full demand.
      2. soft    -- if the committed levels cannot cover the chosen zones, allow
                    unserved demand at the zone penalty so we still get the best
                    achievable plan and can report the infeasibility honestly.
    """
    sources = default_sources() if sources is None else sources
    zones = default_zones() if zones is None else zones
    src, zon = list(sources), list(zones)
    S, Z = len(src), len(zon)
    levels = np.asarray(levels, dtype=float)
    served = sorted({int(z) for z in served})
    demand = np.array([factor * float(zones[z]["demand"][t]) for z in zon])

    if not served:
        # Every zone shed: nothing to dispatch.  Degenerate but well-defined,
        # and linprog cannot be handed a zero-variable model.
        return {
            "ok": True,
            "status": 0,
            "message": "optimal",
            "feasible": True,
            "flows": np.zeros((S, Z)),
            "generation": np.zeros(S),
            "levels": levels,
            "served": [],
            "served_names": [],
            "shed_names": list(zon),
            "unmet": demand.copy(),
            "cost": 0.0,
            "objective": 0.0,
            "penalty": 0.0,
            "soft": False,
            "note": "all zones shed by the discrete decision; nothing to dispatch",
        }

    def run(soft: bool) -> Dict[str, Any]:
        # variables: p[s, z] for z in served  (+ unmet[z] if soft)
        nz = len(served)
        NP = S * nz
        NU = nz if soft else 0
        n = NP + NU

        def ip(s: int, k: int) -> int:
            return s * nz + k

        c = np.zeros(n)
        for s in range(S):
            for k in range(nz):
                c[ip(s, k)] = float(sources[src[s]]["cost"])
        if soft:
            for k, z in enumerate(served):
                w = W_CRIT if zones[zon[z]].get("critical") else W_FAIR
                c[NP + k] = w

        A_ub, b_ub = [], []
        for s in range(S):  # source commitment level is a hard cap
            row = np.zeros(n)
            for k in range(nz):
                row[ip(s, k)] = 1.0
            A_ub.append(row)
            b_ub.append(float(levels[s]))

        A_eq, b_eq = [], []
        for k, z in enumerate(served):
            row = np.zeros(n)
            for s in range(S):
                row[ip(s, k)] = 1.0
            if soft:
                row[NP + k] = 1.0
            A_eq.append(row)
            b_eq.append(float(demand[z]))

        res = linprog(
            c,
            A_ub=np.array(A_ub) if A_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(A_eq) if A_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=[(0.0, None)] * n,
            method="highs",
        )
        if res.status != 0:
            return {"ok": False, "status": int(res.status), "message": str(res.message)}

        x = res.x
        flows = np.zeros((S, Z))
        for s in range(S):
            for k, z in enumerate(served):
                flows[s, z] = max(0.0, x[ip(s, k)])
        unmet = np.zeros(Z)
        if soft:
            for k, z in enumerate(served):
                unmet[z] = max(0.0, x[NP + k])
        generation = flows.sum(axis=1)
        gen_cost = float(sum(float(sources[src[s]]["cost"]) * generation[s] for s in range(S)))
        return {
            "ok": True,
            "status": 0,
            "message": "optimal",
            "flows": flows,
            "generation": generation,
            "levels": levels,
            "served": served,
            "served_names": [zon[z] for z in served],
            "shed_names": [zon[z] for z in range(Z) if z not in served],
            "unmet": unmet,
            "cost": gen_cost,
            "objective": float(res.fun),
            "penalty": float(res.fun) - gen_cost,
            "soft": soft,
        }

    strict = run(soft=False)
    if strict.get("ok"):
        strict["feasible"] = True
        strict["note"] = "committed levels cover every served zone exactly"
        return strict

    soft = run(soft=True)
    if soft.get("ok"):
        soft["feasible"] = False
        soft["note"] = (
            f"strict plan infeasible ({strict['message']}); repaired with "
            f"${soft['penalty']:.2f} of unserved-demand penalty"
        )
        return soft
    return {
        "ok": False,
        "feasible": False,
        "cost": float("nan"),
        "note": f"repair failed: {soft['message']}",
        "served_names": [zon[z] for z in served],
    }


# --------------------------------------------------------------------------
# Feature 10 bullet 4: comparison
# --------------------------------------------------------------------------
def compare(
    t: int = 3,
    sources: Dict[str, Any] | None = None,
    zones: Dict[str, Any] | None = None,
    penalty: float = 5_000.0,
    factor: float = 1.0,
    level_bits: int = LEVEL_BITS,
    reps: int = 2,
    restarts: int = 12,
    shots: int = 8192,
    seed: int = 7,
    use_quantum: bool = True,
    reduce_qubo: bool = True,
    pool_restarts: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve one slot with brute force, a greedy baseline and QAOA, and compare.

    Every candidate is scored in the *same* full bitstring space, so the gaps are
    directly comparable regardless of whether QAOA ran on the reduced QUBO.
    """
    sources = default_sources() if sources is None else sources
    zones = default_zones() if zones is None else zones
    Q, meta = build_slot_qubo(
        t, sources, zones, penalty=penalty, factor=factor, level_bits=level_bits
    )
    n = meta["n_qubits"]
    names = list(zones)

    rows: List[Dict[str, Any]] = []

    # ---- exact classical reference ---------------------------------------
    t0 = time.perf_counter()
    e_opt, b_opt = brute_force(Q)
    t_bf = time.perf_counter() - t0

    def add_row(label: str, bits_full, runtime: float, qubits: int, **extra):
        bits_full = np.asarray(bits_full, dtype=float)
        levels, served = decode(bits_full, t, sources, zones, level_bits=level_bits)
        rep = repair_flows(t, levels, served, sources, zones, factor=factor)
        energy = qubo_energy(Q, bits_full)
        served_names = [names[z] for z in served]
        row = {
            "solver": label,
            "qubits": qubits,
            "bits": bits_full.astype(int).tolist(),
            "qubo_energy": energy,
            "objective": energy + meta["offset"],
            "gap": energy - e_opt,
            "levels": levels.tolist(),
            "served": served_names,
            "shed": [nm for nm in names if nm not in served_names],
            "repaired_cost": rep.get("cost", float("nan")),
            "feasible": rep.get("feasible", False),
            "runtime_ms": runtime * 1000.0,
            "note": rep.get("note", ""),
        }
        row.update(extra)
        rows.append(row)
        return row

    add_row("Brute force (exact)", b_opt, t_bf, n)

    # ---- naive classical heuristic ---------------------------------------
    t0 = time.perf_counter()
    b_greedy = greedy_solution(t, sources, zones, factor=factor, level_bits=level_bits)
    add_row("Greedy heuristic", b_greedy, time.perf_counter() - t0, n)

    # ---- classical preprocessing ----------------------------------------
    fixed: Dict[int, int] = {}
    reasons: List[str] = []
    if reduce_qubo:
        fixed, reasons = reduction_plan(
            t, sources, zones, factor=factor, level_bits=level_bits
        )
    Qr, red_offset, keep = fix_qubo_bits(Q, fixed)
    reduction_ok = None
    if fixed:
        e_red, b_red = brute_force(Qr)
        b_back = expand_bits(b_red, keep, fixed, n)
        reduction_ok = bool(
            abs((e_red + red_offset) - e_opt) < 1e-6 and np.array_equal(b_back, b_opt)
        )

    # ---- QAOA -------------------------------------------------------------
    qaoa_info: Dict[str, Any] | None = None
    p_opt = None
    if use_quantum:
        try:
            t0 = time.perf_counter()
            counts, qaoa_info = qaoa_solve(
                Qr, reps=reps, restarts=restarts, shots=shots, seed=seed,
                pool_restarts=pool_restarts,
            )
            elapsed = time.perf_counter() - t0

            scored = []
            for bitstring, red_bits, count in counts_to_bits(
                counts, Qr.shape[0], top_k=len(counts)
            ):
                full = expand_bits(red_bits, keep, fixed, n)
                scored.append((qubo_energy(Q, full), full, count, bitstring))
            total_shots = sum(c for _, _, c, _ in scored)
            p_opt = sum(c for e, _, c, _ in scored if e < e_opt + 1e-6) / max(1, total_shots)

            e_best, b_best, _, _ = min(scored, key=lambda r: r[0])
            add_row(
                f"QAOA best sample (p={reps})", b_best, elapsed, Qr.shape[0],
                qaoa=qaoa_info, p_opt=p_opt,
                note_extra=f"lowest-energy of {total_shots} shots; "
                           f"P(optimum)={p_opt:.4f}",
            )
            # The most frequent bitstring is what a raw QAOA run reports; on a
            # badly conditioned QUBO it is often NOT the best sample, so show both.
            top_bits = counts_to_bits(counts, Qr.shape[0], top_k=1)[0]
            add_row(
                "QAOA most frequent", expand_bits(top_bits[1], keep, fixed, n), 0.0,
                Qr.shape[0], p_opt=p_opt,
            )
        except ImportError as exc:
            rows.append({"solver": "QAOA", "error": str(exc)})

    result = {
        "slot": meta["slot"],
        "slot_index": t,
        "meta": meta,
        "Q": Q,
        "Q_reduced": Qr,
        "fixed_bits": fixed,
        "reduction_reasons": reasons,
        "reduction_offset": red_offset,
        "reduction_kept": keep,
        "reduction_ok": reduction_ok,
        "rows": rows,
        "brute_force": rows[0],
        "optimum_energy": e_opt,
        "p_opt": p_opt,
        "qaoa_info": qaoa_info,
        "demand_total": float(np.sum(meta["demand"])),
        "supply_total": float(np.sum(meta["availability"])),
    }
    if verbose:
        _print_comparison(result)
    return result


def _print_comparison(result: Dict[str, Any]) -> None:
    meta = result["meta"]
    print(f"\n=== Quantum vs classical on the {result['slot']} slot ===")
    print(
        f"full encoding: {meta['n_qubits']} qubits ({meta['level_bits']} x "
        f"{len(meta['sources'])} source-level bits + {len(meta['zones'])} zone-shed "
        f"bits)   search space 2^{meta['n_qubits']} = {meta['n_states']} states"
    )
    print(
        f"slot demand {result['demand_total']:.1f} kWh   "
        f"slot availability {result['supply_total']:.1f} kWh   "
        f"balance penalty lambda {meta['penalty']:g}"
    )
    if result["fixed_bits"]:
        print(
            f"classical preprocessing fixed {len(result['fixed_bits'])} bits -> "
            f"QAOA ran on {result['Q_reduced'].shape[0]} qubits"
        )
        for reason in result["reduction_reasons"]:
            print(f"    * {reason}")
        verdict = "PRESERVES THE OPTIMUM" if result["reduction_ok"] else "MISMATCH!"
        print(f"    reduction check: {verdict}")

    print()
    header = (
        f"{'solver':26s} {'qb':>3s} {'QUBO E':>13s} {'gap':>11s} "
        f"{'repaired $':>10s} {'feas':>5s} {'P(opt)':>7s} {'runtime':>9s}"
    )
    print(header)
    print("-" * len(header))
    for row in result["rows"]:
        if "error" in row:
            print(f"{row['solver']:26s} SKIPPED: {row['error']}")
            continue
        p = f"{row['p_opt']:.4f}" if row.get("p_opt") is not None else "-"
        print(
            f"{row['solver']:26s} {row['qubits']:3d} {row['qubo_energy']:13.2f} "
            f"{row['gap']:11.2f} {row['repaired_cost']:10.2f} "
            f"{'yes' if row['feasible'] else 'NO':>5s} {p:>7s} "
            f"{row['runtime_ms']:7.0f}ms"
        )

    print()
    for row in result["rows"]:
        if "error" in row:
            continue
        print(f"  {row['solver']}")
        print(f"      serve : {row['served']}")
        print(f"      shed  : {row['shed']}")
        levels = "  ".join(
            f"{nm}={lv:.1f}" for nm, lv in zip(meta["sources"], row["levels"])
        )
        print(f"      levels: {levels}")
        print(f"      QUBO objective = {row['objective']:.2f}  "
              f"(repaired generation cost ${row['repaired_cost']:.2f})")
        for key in ("note", "note_extra"):
            if row.get(key):
                print(f"      note  : {row[key]}")

    best = next(
        (r for r in result["rows"] if r["solver"].startswith("QAOA best")), None
    )
    print()
    if best:
        hit = best["gap"] < 1e-6
        print(
            f"QAOA reproduces the brute-force optimum: {'YES' if hit else 'NO'}"
            f"   (gap {best['gap']:.2f}, P(optimum) = {best.get('p_opt', 0):.4f})"
        )
        if result["qaoa_info"]:
            info = result["qaoa_info"]
            print(
                f"   QAOA detail: p={info['reps']}, {info['num_parameters']} variational "
                f"parameters, {info['restarts']} COBYLA restarts "
                f"({info['optimizer_evals']} evaluations), {info['shots']} shots, "
                f"{info['hamiltonian_terms']} Hamiltonian terms"
            )
    print(
        "   Reminder: QAOA on a statevector simulator is classical computation. It "
        "validates the QUBO\n   formulation; running on IBM Quantum / D-Wave hardware "
        "is the next step (see --export)."
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="QUBO + QAOA for one time slot")
    ap.add_argument("--slot", type=int, default=3, help=f"slot index into {list(SLOTS)}")
    ap.add_argument("--penalty", type=float, default=5_000.0, help="balance penalty lambda")
    ap.add_argument("--factor", type=float, default=1.0, help="demand multiplier (scenario)")
    ap.add_argument(
        "--level-bits", type=int, default=LEVEL_BITS,
        help="bits per source for output-level quantisation (more bits = finer grid, more qubits)",
    )
    ap.add_argument(
        "--grid-restriction", action="store_true",
        help="apply the evening grid import restriction before building the QUBO",
    )
    ap.add_argument("--no-backup", action="store_true", help="disable the emergency fleet")
    ap.add_argument("--reps", type=int, default=2, help="QAOA depth p")
    ap.add_argument("--restarts", type=int, default=12,
                    help="random optimizer restarts; 12 reproduces the exact optimum on every seed tested")
    ap.add_argument("--no-pool", dest="pool_restarts", action="store_false",
                    help="sample only from the best restart instead of pooling all of them")
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-quantum", action="store_true", help="skip QAOA (classical only)")
    ap.add_argument(
        "--no-reduce", dest="reduce_qubo", action="store_false",
        help="run QAOA on the full QUBO instead of the preprocessed one (much lower P(optimum))",
    )
    ap.add_argument("--export", metavar="PATH", help="write the QUBO/Ising model to JSON")
    args = ap.parse_args(list(argv) if argv is not None else None)

    from energy_allocation import with_backup_enabled, with_grid_restriction

    sources = default_sources()
    if args.grid_restriction:
        sources = with_grid_restriction(sources, slot=args.slot)
    if args.no_backup:
        sources = with_backup_enabled(sources, enabled=False)

    result = compare(
        t=args.slot,
        sources=sources,
        penalty=args.penalty,
        factor=args.factor,
        level_bits=args.level_bits,
        reps=args.reps,
        restarts=args.restarts,
        pool_restarts=args.pool_restarts,
        shots=args.shots,
        seed=args.seed,
        use_quantum=not args.no_quantum,
        reduce_qubo=args.reduce_qubo,
    )
    if args.export:
        path = export_qubo(result["Q"], result["meta"], args.export)
        print(f"\nQUBO + Ising model written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
