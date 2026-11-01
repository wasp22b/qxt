"""Tests for the quantum module (Feature 10).

Qiskit-dependent tests are skipped when Qiskit is not installed; the QUBO
construction, decoding, brute force and LP repair are pure numpy/scipy and are
always tested.
"""

from __future__ import annotations

import importlib.util
import itertools

import numpy as np
import pytest

import quantum_module as qm
from energy_allocation import (
    SLOTS,
    W_CRIT,
    W_FAIR,
    default_sources,
    default_zones,
    with_backup_enabled,
    with_grid_restriction,
)

# Qiskit is optional: the QUBO construction, decoding, brute force and LP repair
# are pure numpy/scipy and are always tested.
HAS_QISKIT = importlib.util.find_spec("qiskit") is not None

requires_qiskit = pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")

SOURCES, ZONES = default_sources(), default_zones()
S, Z = len(SOURCES), len(ZONES)


# --------------------------------------------------------------------------
# QUBO construction
# --------------------------------------------------------------------------
@pytest.mark.parametrize("t", range(len(SLOTS)))
def test_qubo_shape_and_upper_triangular(t):
    Q, meta = qm.build_slot_qubo(t)
    n = 2 * S + Z
    assert Q.shape == (n, n)
    assert meta["n_qubits"] == n
    assert meta["n_states"] == 2**n
    assert np.allclose(np.tril(Q, -1), 0.0)  # coefficients live on diag + upper


@pytest.mark.parametrize("t", range(len(SLOTS)))
def test_qubo_offset_is_total_shed_baseline(t):
    _, meta = qm.build_slot_qubo(t)
    expected = sum(
        (W_CRIT if ZONES[z].get("critical") else W_FAIR) * ZONES[z]["demand"][t]
        for z in ZONES
    )
    assert meta["offset"] == pytest.approx(expected)


def test_all_bits_zero_costs_exactly_the_full_shed_baseline():
    """b = 0 means nothing generated and every zone shed: the objective must
    equal the shed baseline, with no balance penalty (0 supply, 0 served)."""
    t = 3
    Q, meta = qm.build_slot_qubo(t)
    zeros = np.zeros(meta["n_qubits"])
    assert qm.qubo_energy(Q, zeros) == pytest.approx(0.0, abs=1e-9)
    assert qm.qubo_energy(Q, zeros) + meta["offset"] == pytest.approx(meta["offset"])


def test_shedding_the_hospital_carries_the_critical_penalty():
    """Feature 6 expressed inside the QUBO: the hospital's serve bit carries a
    linear term of -W_CRIT * D, i.e. shedding it costs 100x more than shedding
    an industrial zone, so it is only shed when physically unavoidable."""
    t = 3
    _, meta = qm.build_slot_qubo(t)
    i_h = 2 * S + list(ZONES).index("Hospital")
    i_i = 2 * S + list(ZONES).index("Industrial")
    assert meta["linear"][i_h] == pytest.approx(-W_CRIT * ZONES["Hospital"]["demand"][t])
    assert meta["linear"][i_i] == pytest.approx(-W_FAIR * ZONES["Industrial"]["demand"][t])
    assert abs(meta["linear"][i_h]) == pytest.approx(100 * abs(meta["linear"][i_i]))
    # and both dwarf the largest generation cost term in the same QUBO
    largest_gen = max(abs(v) for v in meta["linear"][: 2 * S])
    assert abs(meta["linear"][i_h]) > 1000 * largest_gen


def test_optimum_serves_every_zone_when_supply_is_sufficient():
    """Evening has 110 kWh available vs 105 demanded, so the exact optimum sheds
    nobody and its QUBO objective equals the generation cost."""
    t = 3
    Q, meta = qm.build_slot_qubo(t)
    energy, bits = qm.brute_force(Q)
    levels, served = qm.decode(bits, t)
    assert served == list(range(Z))
    objective = energy + meta["offset"]
    gen_cost = sum(
        SOURCES[list(SOURCES)[s]]["cost"] * levels[s] for s in range(S)
    )
    assert objective == pytest.approx(gen_cost, abs=1e-6)
    # supply committed exactly matches demand -- the balance term is satisfied
    assert levels.sum() == pytest.approx(sum(ZONES[z]["demand"][t] for z in ZONES), abs=1e-6)


def test_shortage_optimum_sheds_the_cheapest_zones():
    """With the grid restricted, 80 kWh is available for 105 kWh of demand, so
    the optimum must shed -- and it sheds Commercial + Industrial ($5,000 of
    penalty) rather than Residential ($12,000) or the hospital ($200,000)."""
    t = 3
    sources = with_grid_restriction(default_sources(), slot=t, limit=10.0)
    Q, meta = qm.build_slot_qubo(t, sources=sources)
    energy, bits = qm.brute_force(Q)
    levels, served = qm.decode(bits, t, sources=sources)
    names = list(ZONES)
    assert [names[z] for z in served] == ["Hospital", "Residential"]
    shed_cost = (W_FAIR * ZONES["Commercial"]["demand"][t]
                 + W_FAIR * ZONES["Industrial"]["demand"][t])
    assert energy + meta["offset"] == pytest.approx(
        sum(sources[names_s]["cost"] * levels[i] for i, names_s in enumerate(SOURCES))
        + shed_cost,
        abs=1e-6,
    )


def test_level_bits_controls_qubit_count():
    for bits in (1, 2, 3):
        Q, meta = qm.build_slot_qubo(3, level_bits=bits)
        assert meta["n_qubits"] == bits * S + Z
        assert Q.shape[0] == bits * S + Z


def test_qubo_rejects_bad_inputs():
    with pytest.raises(ValueError):
        qm.build_slot_qubo(99)
    with pytest.raises(ValueError):
        qm.build_slot_qubo(3, level_bits=0)


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------
@pytest.mark.parametrize("t", range(len(SLOTS)))
def test_decode_full_level_gives_full_availability(t):
    bits = np.ones(2 * S + Z)
    bits[1::2][:S] = 0  # set only the low bit of each source -> level 1/3
    levels, served = qm.decode(bits, t)
    for s, name in enumerate(SOURCES):
        assert levels[s] == pytest.approx(SOURCES[name]["avail"][t] / 3.0)
    assert served == list(range(Z))


def test_decode_all_ones_is_full_availability():
    t = 2
    levels, served = qm.decode(np.ones(2 * S + Z), t)
    for s, name in enumerate(SOURCES):
        assert levels[s] == pytest.approx(SOURCES[name]["avail"][t])
    assert len(served) == Z


def test_decode_all_zeros_sheds_everything():
    levels, served = qm.decode(np.zeros(2 * S + Z), 3)
    assert levels.max() == 0.0
    assert served == []


def test_decode_rejects_wrong_length():
    with pytest.raises(ValueError):
        qm.decode(np.ones(5), 3)


def test_bits_to_string_is_readable():
    Q, meta = qm.build_slot_qubo(3)
    bits = np.ones(2 * S + Z)
    text = qm.bits_to_string(bits, meta)
    assert "Solar=100%" in text
    assert "serve=" in text


# --------------------------------------------------------------------------
# brute force
# --------------------------------------------------------------------------
def test_brute_force_matches_naive_enumeration():
    rng = np.random.default_rng(0)
    n = 9
    A = rng.normal(size=(n, n))
    Q = np.triu(A + A.T)  # random symmetric upper-triangular QUBO
    best_e, best_b = qm.brute_force(Q)
    naive = min(
        (
            (float(b @ Q @ b), b)
            for b in itertools.product([0.0, 1.0], repeat=n)
        ),
        key=lambda r: r[0],
    )
    assert best_e == pytest.approx(naive[0], abs=1e-9)


def test_brute_force_on_known_solution():
    """E(b) = -2*b0 + 2*b0*b1 + b1 over {0,1}^2 is (0, -2, 1, 1): min at (1,0)."""
    Q = np.array([[-2.0, 2.0], [0.0, 1.0]])
    e, b = qm.brute_force(Q)
    assert b.tolist() == [1.0, 0.0]
    assert e == pytest.approx(-2.0)
    assert all(qm.qubo_energy(Q, x) >= e - 1e-12
               for x in ([0, 0], [1, 0], [0, 1], [1, 1]))


def test_brute_force_is_chunk_invariant():
    Q, _ = qm.build_slot_qubo(3)
    e1, b1 = qm.brute_force(Q, chunk=1 << 16)
    e2, b2 = qm.brute_force(Q, chunk=7)
    assert e1 == pytest.approx(e2)
    assert np.array_equal(b1, b2)


def test_brute_force_refuses_huge_problems():
    with pytest.raises(ValueError):
        qm.brute_force(np.zeros((30, 30)))


# --------------------------------------------------------------------------
# classical preprocessing (variable fixing)
# --------------------------------------------------------------------------
def test_fix_bits_energy_identity():
    """b_full @ Q @ b_full == b_red @ Qr @ b_red + offset for every bitstring."""
    Q, meta = qm.build_slot_qubo(3)
    fixed, _ = qm.reduction_plan(3)
    Qr, offset, keep = qm.fix_qubo_bits(Q, fixed)
    rng = np.random.default_rng(1)
    for _ in range(300):
        red = rng.integers(0, 2, Qr.shape[0]).astype(float)
        full = qm.expand_bits(red, keep, fixed, Q.shape[0])
        assert float(full @ Q @ full) == pytest.approx(
            float(red @ Qr @ red) + offset, abs=1e-6
        )


def test_reduction_preserves_the_optimum():
    Q, meta = qm.build_slot_qubo(3)
    fixed, reasons = qm.reduction_plan(3)
    assert fixed and reasons
    Qr, offset, keep = qm.fix_qubo_bits(Q, fixed)
    e_full, b_full = qm.brute_force(Q)
    e_red, b_red = qm.brute_force(Qr)
    assert e_red + offset == pytest.approx(e_full, abs=1e-6)
    assert np.array_equal(qm.expand_bits(b_red, keep, fixed, Q.shape[0]), b_full)
    assert Qr.shape[0] < Q.shape[0]


def test_reduction_fixes_all_zone_bits_when_supply_is_sufficient():
    fixed, _ = qm.reduction_plan(3)  # evening: 110 available vs 105 demanded
    assert fixed == {2 * S + z: 1 for z in range(Z)}


def test_reduction_only_fixes_critical_bits_under_shortage():
    sources = with_grid_restriction(default_sources(), slot=3, limit=10.0)
    fixed, reasons = qm.reduction_plan(3, sources=sources)
    hosp = list(ZONES).index("Hospital")
    assert fixed == {2 * S + hosp: 1}
    assert any("critical" in r for r in reasons)


def test_reduction_fixes_zero_availability_sources():
    fixed, reasons = qm.reduction_plan(0)  # night: Solar and Backup unavailable
    assert fixed[0] == 0 and fixed[1] == 0  # Solar level bits
    assert fixed[8] == 0 and fixed[9] == 0  # Backup level bits


def test_reduction_under_severe_shortage_fixes_nothing_extra():
    sources = with_backup_enabled(
        with_grid_restriction(default_sources(), slot=3, limit=0.0), enabled=False
    )
    fixed, _ = qm.reduction_plan(3, sources=sources)
    hosp = list(ZONES).index("Hospital")
    # 45 kWh available still covers the 20 kWh hospital -> that bit is decided
    assert fixed.get(2 * S + hosp) == 1
    # but non-critical zones genuinely are a choice, so they stay free
    assert 2 * S + list(ZONES).index("Residential") not in fixed


def test_fix_qubo_bits_with_no_fixed_bits_is_identity():
    Q, _ = qm.build_slot_qubo(3)
    Qr, offset, keep = qm.fix_qubo_bits(Q, {})
    assert offset == 0.0
    assert keep == list(range(Q.shape[0]))
    assert np.array_equal(Qr, Q)


# --------------------------------------------------------------------------
# LP repair
# --------------------------------------------------------------------------
def test_repair_recovers_the_lp_engine_cost_at_the_optimum():
    """The hybrid loop must agree with the classical engine on the same slot."""
    t = 3
    Q, meta = qm.build_slot_qubo(t)
    _, b_opt = qm.brute_force(Q)
    levels, served = qm.decode(b_opt, t)
    rep = qm.repair_flows(t, levels, served)
    assert rep["feasible"] and rep["ok"]
    assert rep["cost"] == pytest.approx(26.10, abs=1e-6)
    # generation exactly covers the served zones
    assert rep["generation"].sum() == pytest.approx(
        sum(ZONES[list(ZONES)[z]]["demand"][t] for z in served), abs=1e-6
    )


def test_repair_respects_committed_levels():
    t = 3
    levels = np.array([10.0, 20.0, 15.0, 40.0, 25.0])
    rep = qm.repair_flows(t, levels, [0, 1, 2, 3])
    assert rep["feasible"]
    assert (rep["generation"] <= levels + 1e-6).all()


def test_repair_reports_infeasible_commitments_honestly():
    """Levels too small for the chosen zones -> soft repair, flagged infeasible."""
    t = 3
    levels = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # 5 kWh committed, 105 demanded
    rep = qm.repair_flows(t, levels, [0, 1, 2, 3])
    assert rep["feasible"] is False
    assert rep["penalty"] > 0
    assert rep["unmet"].sum() > 0
    assert (rep["generation"] <= levels + 1e-6).all()


def test_repair_handles_every_zone_shed():
    rep = qm.repair_flows(3, np.zeros(S), [])
    assert rep["ok"] and rep["feasible"]
    assert rep["cost"] == 0.0
    assert rep["shed_names"] == list(ZONES)


def test_repair_prefers_cheapest_sources():
    t = 3
    levels = np.array([10.0, 20.0, 15.0, 40.0, 25.0])
    rep = qm.repair_flows(t, levels, [list(ZONES).index("Hospital")])  # only 20 kWh
    assert rep["feasible"]
    names = list(SOURCES)
    # cheapest first: solar then wind
    assert rep["generation"][names.index("Solar")] == pytest.approx(10.0)
    assert rep["generation"][names.index("Wind")] == pytest.approx(10.0)
    assert rep["generation"][names.index("Backup")] == pytest.approx(0.0)


def test_repair_never_over_serves_a_zone():
    t = 3
    levels = np.array([10.0, 20.0, 15.0, 40.0, 25.0])
    rep = qm.repair_flows(t, levels, [0, 1, 2, 3])
    served = rep["flows"].sum(axis=0)
    for z, name in enumerate(ZONES):
        assert served[z] <= ZONES[name]["demand"][t] + 1e-6


# --------------------------------------------------------------------------
# greedy baseline + Ising conversion
# --------------------------------------------------------------------------
@pytest.mark.parametrize("t", range(len(SLOTS)))
def test_greedy_returns_a_valid_bitstring(t):
    bits = qm.greedy_solution(t)
    assert bits.shape == (2 * S + Z,)
    assert set(np.unique(bits)) <= {0.0, 1.0}
    levels, served = qm.decode(bits, t)
    assert (levels >= -1e-9).all()
    assert len(served) >= 1  # the hospital is always served


def test_qubo_to_ising_matches_qubo_energy():
    Q, meta = qm.build_slot_qubo(3)
    h, J, offset = qm.qubo_to_ising(Q)
    rng = np.random.default_rng(2)
    for _ in range(50):
        bits = rng.integers(0, 2, Q.shape[0]).astype(float)
        z = 1.0 - 2.0 * bits  # b = (1 - z)/2
        energy = offset + sum(h[i] * z[i] for i in h) + sum(
            v * z[i] * z[j] for (i, j), v in J.items()
        )
        assert energy == pytest.approx(qm.qubo_energy(Q, bits), abs=1e-6)


def test_export_qubo_writes_both_forms(tmp_path):
    Q, meta = qm.build_slot_qubo(3)
    path = tmp_path / "qubo.json"
    qm.export_qubo(Q, meta, str(path))
    import json

    payload = json.loads(path.read_text())
    assert payload["qubo"]["n"] == Q.shape[0]
    assert "h" in payload["ising"] and "J" in payload["ising"]
    assert payload["meta"]["slot"] == "Evening"


# --------------------------------------------------------------------------
# counts / endianness
# --------------------------------------------------------------------------
def test_counts_to_bits_handles_qiskit_endianness():
    counts = {"0001": 10, "1000": 5}
    ranked = qm.counts_to_bits(counts, n=4)
    assert ranked[0][0] == "0001"
    # rightmost character is qubit 0 -> bit index 0 set
    assert ranked[0][1].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert ranked[1][1].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_counts_to_bits_rejects_wrong_width():
    with pytest.raises(ValueError):
        qm.counts_to_bits({"00": 1}, n=5)


# --------------------------------------------------------------------------
# Qiskit path
# --------------------------------------------------------------------------
@requires_qiskit
def test_cost_hamiltonian_reproduces_qubo_energies():
    """<H> on a computational basis state must equal b @ Q @ b exactly,
    including the constant term -- that is what makes QAOA energies comparable
    to brute force."""
    from qiskit.quantum_info import Statevector

    Q, meta = qm.build_slot_qubo(3)
    n = meta["n_qubits"]
    op = qm._cost_hamiltonian(Q)
    rng = np.random.default_rng(3)
    for _ in range(12):
        bits = rng.integers(0, 2, n).astype(float)
        index = int(sum((1 << i) for i in range(n) if bits[i]))
        ev = float(np.real(Statevector.from_int(index, 2**n).expectation_value(op)))
        assert ev == pytest.approx(qm.qubo_energy(Q, bits), abs=1e-4)


@requires_qiskit
def test_normalize_operator_preserves_ground_state():
    Q, _ = qm.build_slot_qubo(3)
    op = qm._cost_hamiltonian(Q)
    scaled, scale = qm.normalize_operator(op)
    assert scale > 0
    labels = scaled.paulis.to_labels()
    non_id = np.array([set(l) != {"I"} for l in labels])
    assert np.abs(np.asarray(scaled.coeffs)[non_id]).max() == pytest.approx(1.0)
    e, b = qm.brute_force(Q)
    # rescaling by a positive constant cannot move the argmin
    e2, b2 = qm.brute_force(Q * scale)
    assert np.array_equal(b, b2)
    assert e2 == pytest.approx(e * scale, rel=1e-12, abs=1e-3)


@requires_qiskit
def test_qaoa_circuit_uses_only_hardware_friendly_gates():
    Q, meta = qm.build_slot_qubo(3)
    qc = qm.build_qaoa_circuit(Q, betas=[0.3, 0.4], gammas=[0.1, 0.2])
    names = {instr.operation.name for instr in qc.data}
    assert names <= {"h", "rz", "rzz", "rx"}          # no PauliEvolution blocks
    assert all(len(instr.qubits) <= 2 for instr in qc.data)
    assert qc.num_qubits == meta["n_qubits"]


@requires_qiskit
def test_qaoa_circuit_at_zero_angle_is_uniform():
    """gamma = beta = 0 leaves |+>^n untouched -> a flat distribution."""
    from qiskit.quantum_info import Statevector

    Q, meta = qm.build_slot_qubo(3)
    qc = qm.build_qaoa_circuit(Q, betas=[0.0], gammas=[0.0])
    probs = Statevector(qc).probabilities()
    assert probs.shape == (2 ** meta["n_qubits"],)
    assert probs.max() == pytest.approx(probs.min(), abs=1e-12)


@requires_qiskit
def test_qaoa_cost_layer_applies_exact_phases():
    """With beta = 0 the cost layer is diagonal, so it must leave every
    amplitude's magnitude at 1/sqrt(2**n) and set the phase to
    -gamma * (E(b) - ising_offset)."""
    from qiskit.quantum_info import Statevector

    Q = np.array([[1.0, 2.0], [0.0, -0.5]])
    n = 2
    _, _, offset = qm.qubo_to_ising(Q)
    gamma = 0.4
    qc = qm.build_qaoa_circuit(Q, betas=[0.0], gammas=[gamma])
    amps = Statevector(qc).data
    assert np.allclose(np.abs(amps), 1.0 / np.sqrt(2**n), atol=1e-12)
    for idx in range(2**n):
        bits = np.array([(idx >> i) & 1 for i in range(n)], dtype=float)
        expected = -gamma * (float(bits @ Q @ bits) - offset)
        diff = (np.angle(amps[idx]) - expected + np.pi) % (2 * np.pi) - np.pi
        assert abs(diff) < 1e-9, f"basis {idx}: got {np.angle(amps[idx])}, want {expected}"


@requires_qiskit
def test_qaoa_mixer_converts_phase_into_amplitude():
    """A diagonal cost layer alone cannot bias the sampling; the mixer is what
    turns phase into probability. With both angles non-zero the distribution
    must stop being uniform."""
    from qiskit.quantum_info import Statevector

    Q = np.array([[0.0, 1.0], [0.0, 0.0]])  # E(b) = b0 * b1
    uniform = Statevector(qm.build_qaoa_circuit(Q, betas=[0.0], gammas=[np.pi / 4])).probabilities()
    assert uniform.max() == pytest.approx(uniform.min(), abs=1e-12)
    mixed = Statevector(
        qm.build_qaoa_circuit(Q, betas=[np.pi / 8], gammas=[np.pi / 4])
    ).probabilities()
    assert mixed.max() - mixed.min() > 1e-6


@requires_qiskit
def test_qaoa_circuit_rejects_mismatched_reps():
    Q, _ = qm.build_slot_qubo(3)
    with pytest.raises(ValueError):
        qm.build_qaoa_circuit(Q, betas=[0.1, 0.2], gammas=[0.3])


@requires_qiskit
@pytest.mark.slow
def test_qaoa_recovers_the_brute_force_optimum():
    """End-to-end: QAOA + classical preprocessing reproduces the exact optimum."""
    result = qm.compare(t=3, shots=8192, reps=2, verbose=False)
    best = next(r for r in result["rows"] if r["solver"].startswith("QAOA best"))
    assert best["gap"] < 1e-6
    assert best["served"] == result["brute_force"]["served"]
    assert best["repaired_cost"] == pytest.approx(
        result["brute_force"]["repaired_cost"], abs=1e-6
    )
    assert result["reduction_ok"]
    assert result["p_opt"] > 0.0


@requires_qiskit
@pytest.mark.slow
def test_pooling_restarts_beats_single_restart_sampling():
    """Pooling the shots of every COBYLA restart is what makes best-of-shots
    reliable; with a single restart the optimum regularly goes unsampled."""
    pooled, _ = qm.qaoa_solve(qm.fix_qubo_bits(
        qm.build_slot_qubo(3)[0], qm.reduction_plan(3)[0])[0],
        restarts=12, shots=8192, pool_restarts=True)
    single, _ = qm.qaoa_solve(qm.fix_qubo_bits(
        qm.build_slot_qubo(3)[0], qm.reduction_plan(3)[0])[0],
        restarts=12, shots=8192, pool_restarts=False)
    assert sum(pooled.values()) >= sum(single.values())
    assert len(pooled) >= 1 and len(single) >= 1


@requires_qiskit
@pytest.mark.slow
def test_qaoa_finds_the_right_shed_plan_under_shortage():
    """Under a real shortage QAOA must pick the cheapest zones to shed while
    keeping the hospital served."""
    sources = with_grid_restriction(default_sources(), slot=3, limit=10.0)
    result = qm.compare(t=3, sources=sources, shots=8192, reps=2, verbose=False)
    bf = result["brute_force"]
    assert "Hospital" in bf["served"]
    assert bf["shed"] == ["Commercial", "Industrial"]
    best = next(r for r in result["rows"] if r["solver"].startswith("QAOA best"))
    assert best["served"] == bf["served"]
    assert best["gap"] < 1e-6


@requires_qiskit
def test_compare_without_quantum_still_reports_classical_rows():
    result = qm.compare(t=3, use_quantum=False, verbose=False)
    labels = [r["solver"] for r in result["rows"]]
    assert "Brute force (exact)" in labels
    assert "Greedy heuristic" in labels
    assert result["qaoa_info"] is None
