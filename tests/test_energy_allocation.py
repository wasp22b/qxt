"""Tests for the classical LP engine (Features 1-9, 11)."""

from __future__ import annotations

import numpy as np
import pytest

import energy_allocation as ea
from energy_allocation import (
    SCENARIOS,
    SLOTS,
    SOURCES,
    ZONES,
    default_sources,
    default_zones,
    solve,
    to_frames,
    validate,
    with_backup_enabled,
    with_grid_restriction,
)

T = len(SLOTS)
CRITICAL = [z for z in ZONES if ZONES[z].get("critical")]
NONCRITICAL = [z for z in ZONES if not ZONES[z].get("critical")]


@pytest.fixture
def sol():
    return solve()


# --------------------------------------------------------------------------
# basic shape / feasibility
# --------------------------------------------------------------------------
def test_plan_shapes(sol):
    assert sol["p"].shape == (len(SOURCES), len(ZONES), T)
    assert sol["served"].shape == (len(ZONES), T)
    assert sol["generation"].shape == (len(SOURCES), T)
    assert sol["curtailment_rate"].shape == (T,)


def test_default_plan_validates(sol):
    assert validate(sol, verbose=False)


def test_all_scenarios_validate():
    for name in SCENARIOS:
        assert validate(solve(scenario=name), verbose=False), name


def test_energy_conservation(sol):
    assert sol["total_generation"] == pytest.approx(sol["total_served"], abs=1e-6)
    assert sol["total_served"] == pytest.approx(sol["total_demand"], abs=1e-6)


def test_no_negative_flows(sol):
    assert (sol["p"] >= -1e-9).all()


# --------------------------------------------------------------------------
# Feature 4: capacity constraints
# --------------------------------------------------------------------------
def test_sources_respect_per_slot_availability(sol):
    for s, name in enumerate(sol["src"]):
        for t in range(T):
            assert sol["generation"][s, t] <= SOURCES[name]["avail"][t] + 1e-6


def test_thermal_daily_cap_is_enforced_and_binding(sol):
    idx = sol["src"].index("Thermal")
    used = sol["generation"][idx].sum()
    cap = SOURCES["Thermal"]["storage"]
    assert used <= cap + 1e-6
    assert used == pytest.approx(cap, abs=1e-6)  # cap actually binds on a normal day


def test_surplus_is_never_forced_into_load(sol):
    assert (sol["served"] <= sol["demand"] + 1e-9).all()


# --------------------------------------------------------------------------
# Feature 3: cost optimisation
# --------------------------------------------------------------------------
def test_reported_cost_matches_flows(sol):
    recomputed = sum(
        SOURCES[sol["src"][s]]["cost"] * sol["p"][s].sum() for s in range(len(sol["src"]))
    )
    assert sol["cost"] == pytest.approx(recomputed, abs=1e-6)


def test_engine_beats_a_naive_expensive_plan(sol):
    """A plan that can only run the expensive sources must cost more."""
    grid_only = default_sources()
    for name in grid_only:
        grid_only[name]["avail"] = [0.0] * T
    grid_only["Grid"]["avail"] = [1000.0] * T
    grid_only["Backup"]["avail"] = [1000.0] * T
    alt = solve(sources=grid_only)
    assert alt["cost"] > sol["cost"]


def test_renewables_are_consumed_before_grid(sol):
    """Wherever solar/wind is available and needed, it is taken in full."""
    for name in ("Solar", "Wind"):
        s = sol["src"].index(name)
        total_demand = sol["demand"].sum(axis=0)
        for t in range(T):
            avail = SOURCES[name]["avail"][t]
            if avail > 0 and avail < total_demand[t]:
                assert sol["generation"][s, t] == pytest.approx(avail, abs=1e-6)


# --------------------------------------------------------------------------
# Feature 6: critical protection
# --------------------------------------------------------------------------
def test_critical_zones_fully_served_in_every_scenario():
    for name in SCENARIOS:
        s = solve(scenario=name)
        z = s["zon"].index(CRITICAL[0])
        assert s["unmet"][z].max() <= 1e-6
        assert s["critical_ok"]


def test_critical_protection_survives_a_severe_outage():
    sources = with_backup_enabled(
        with_grid_restriction(default_sources(), slot=3, limit=0.0), enabled=False
    )
    s = solve(sources=sources)
    z = s["zon"].index(CRITICAL[0])
    assert s["unmet"][z].max() <= 1e-6
    assert validate(s, verbose=False)


def test_penalty_ordering_is_what_protects_critical_load():
    """Collapse W_CRIT to zero and the guarantee disappears -- proving the
    protection comes from the penalty ordering, not from an accident."""
    sources = with_backup_enabled(
        with_grid_restriction(default_sources(), slot=3, limit=0.0), enabled=False
    )
    protected = solve(sources=sources)
    unprotected = solve(sources=sources, w_crit=0.0, w_fair=200.0)
    z = protected["zon"].index(CRITICAL[0])
    assert protected["unmet"][z, 3] <= 1e-6          # W_CRIT = 10,000 -> hospital safe
    assert unprotected["unmet"][z, 3] > 1.0          # W_CRIT = 0 -> hospital curtailed


# --------------------------------------------------------------------------
# Feature 7: fairness / uniform curtailment
# --------------------------------------------------------------------------
def test_curtailment_is_uniform_across_non_critical_zones():
    sources = with_backup_enabled(with_grid_restriction(default_sources()), enabled=False)
    s = solve(sources=sources)
    for t in range(T):
        rates = []
        for name in NONCRITICAL:
            z = s["zon"].index(name)
            d = s["demand"][z, t]
            if d > 1e-9:
                rates.append(s["unmet"][z, t] / d)
        assert max(rates) - min(rates) <= 1e-6
        assert rates[0] == pytest.approx(s["curtailment_rate"][t], abs=1e-6)


def test_shortage_actually_triggers_curtailment():
    sources = with_backup_enabled(with_grid_restriction(default_sources()), enabled=False)
    s = solve(sources=sources)
    assert s["curtailment_rate"][3] > 0.1
    assert s["total_unmet"] > 0
    assert s["critical_ok"]


def test_curtailment_rate_stays_in_unit_interval(sol):
    assert ((sol["curtailment_rate"] >= -1e-9) & (sol["curtailment_rate"] <= 1 + 1e-9)).all()


# --------------------------------------------------------------------------
# Feature 8: surplus handling / backup discipline
# --------------------------------------------------------------------------
def test_surplus_reported_when_renewables_exceed_demand():
    zones = default_zones()
    for z in zones:  # tiny demand -> lots of renewable headroom
        zones[z]["demand"] = [1.0] * T
    s = solve(zones=zones)
    assert s["surplus"].sum() > 0
    assert any(
        "surplus" in msg
        for rec in s["recommendations"]
        for msg in rec["messages"]
    )


def test_backup_runs_only_when_cheaper_options_are_exhausted():
    s = solve()  # evening needs backup: non-backup supply is 85 < 105
    assert s["backup_used"]
    b = s["src"].index("Backup")
    for name in ("Solar", "Wind", "Grid"):
        i = s["src"].index(name)
        assert s["generation"][i, 3] == pytest.approx(SOURCES[name]["avail"][3], abs=1e-6)
    assert s["generation"][b, :3].max() <= 1e-9  # never used outside the evening


def test_backup_can_be_disabled():
    s = solve(sources=with_backup_enabled(default_sources(), enabled=False))
    assert not s["backup_used"]
    assert s["generation"][s["src"].index("Backup")].max() <= 1e-9


# --------------------------------------------------------------------------
# Feature 9: dynamic re-optimisation
# --------------------------------------------------------------------------
def test_solve_is_a_pure_function_of_its_inputs():
    a, b = solve(), solve()
    np.testing.assert_allclose(a["p"], b["p"], atol=1e-9)
    assert a["cost"] == pytest.approx(b["cost"])


def test_inputs_are_not_mutated():
    sources, zones = default_sources(), default_zones()
    before = np.array([sources[s]["avail"] for s in sources], dtype=float)
    solve(sources=sources, zones=zones)
    after = np.array([sources[s]["avail"] for s in sources], dtype=float)
    np.testing.assert_allclose(before, after)


def test_changing_availability_changes_the_plan():
    base = solve()
    tight = default_sources()
    tight["Grid"]["avail"] = [5.0] * T
    s = solve(sources=tight)
    assert s["cost"] != pytest.approx(base["cost"], abs=1e-6)
    assert validate(s, verbose=False)


def test_scenario_multiplier_scales_demand():
    for name, factor in SCENARIOS.items():
        s = solve(scenario=name)
        expected = np.array(
            [[factor * d for d in ZONES[z]["demand"]] for z in s["zon"]]
        )
        np.testing.assert_allclose(s["demand"], expected, atol=1e-9)


def test_custom_forecast_can_drive_demand():
    zones = default_zones()
    zones["Residential"]["demand"] = [10.0, 10.0, 10.0, 10.0]
    s = solve(zones=zones)
    assert s["demand"][s["zon"].index("Residential")].max() == pytest.approx(10.0)
    assert validate(s, verbose=False)


# --------------------------------------------------------------------------
# Feature 11: validation must actually be able to fail
# --------------------------------------------------------------------------
def test_validate_flags_capacity_violation():
    s = solve()
    i = s["src"].index("Grid")
    s["generation"][i, 0] += 500.0
    s["p"][i, :, 0] += 500.0 / s["p"].shape[1]
    assert not validate(s, verbose=False)


def test_validate_flags_critical_shortfall():
    s = solve()
    z = s["zon"].index(CRITICAL[0])
    s["unmet"][z, 0] = 5.0
    s["served"][z, 0] -= 5.0
    assert not validate(s, verbose=False)


def test_validate_flags_over_served_zone():
    s = solve()
    s["served"][0, 0] += 10.0
    s["unmet"][0, 0] -= 10.0
    assert not validate(s, verbose=False)


def test_validate_flags_broken_uniform_curtailment():
    sources = with_backup_enabled(with_grid_restriction(default_sources()), enabled=False)
    s = solve(sources=sources)
    z = s["zon"].index(NONCRITICAL[0])
    s["unmet"][z, 3] *= 0.5
    assert not validate(s, verbose=False)


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["Solar"].pop("avail"),
        lambda s: s["Solar"].update(avail=[1, 2, 3]),
        lambda s: s["Solar"].update(avail=[-1, 0, 0, 0]),
        lambda s: s.clear(),
    ],
)
def test_bad_source_models_raise(mutate):
    sources = default_sources()
    mutate(sources)
    with pytest.raises(ValueError):
        solve(sources=sources)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda z: z["Hospital"].pop("demand"),
        lambda z: z["Hospital"].update(demand=[1, 2]),
        lambda z: z["Hospital"].update(demand=[-1, 0, 0, 0]),
    ],
)
def test_bad_zone_models_raise(mutate):
    zones = default_zones()
    mutate(zones)
    with pytest.raises(ValueError):
        solve(zones=zones)


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        solve(scenario="apocalypse")


def test_grid_restriction_helper_leaves_original_alone():
    sources = default_sources()
    restricted = with_grid_restriction(sources, slot=3, limit=10.0)
    assert restricted["Grid"]["avail"][3] == 10.0
    assert sources["Grid"]["avail"][3] == 40


def test_backup_toggle_is_reversible():
    sources = default_sources()
    off = with_backup_enabled(sources, enabled=False)
    assert off["Backup"]["avail"] == [0.0] * T
    on = with_backup_enabled(off, enabled=True)
    assert on["Backup"]["avail"] == SOURCES["Backup"]["avail"]


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------
def test_report_returns_text_and_mentions_key_sections(sol, capsys):
    text = ea.report(sol)
    assert "=== Evening ===" in text
    assert "Total generation" in text
    assert capsys.readouterr().out  # it also prints


def test_to_frames_shapes(sol):
    frames = to_frames(sol)
    assert list(frames["generation_by_slot"].index) == list(SLOTS)
    assert list(frames["generation_by_slot"].columns) == sol["src"]
    assert list(frames["zone_supply"].columns) == sol["zon"]
    assert len(frames["curtailment"]) == T
