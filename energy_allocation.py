"""
Classical energy allocation engine (Features 1-9 and 11).

One linear program decides, for every time slot:
    * how much each source generates, and to which zone it is delivered
    * how much demand each zone is served
    * a single per-slot curtailment rate shared by all non-critical zones

Design notes
------------
The day-scale problem is linear and continuous, so a classical LP (HiGHS via
scipy) is the right engine: it is exact and solves in milliseconds.  The
combinatorial layer -- which sources run at which level and which zones get
shed during a shortage -- is handled by :mod:`quantum_module`.

Fairness (Feature 7)
    A naive formulation adds one "unmet demand" variable per zone.  Because
    every unit of unmet demand carries the same penalty, the LP is free to dump
    the entire shortage on a single zone.  Instead we introduce one curtailment
    *rate* r_t in [0, 1] per slot and force

        served[z, t] = (1 - r_t) * demand[z, t]      for every non-critical z

    so every non-critical zone loses the same percentage.  The model stays
    linear, and r_t doubles as a human-readable fairness indicator.

Run directly for a report::

    python energy_allocation.py --scenario peak --grid-restriction --validate
"""

from __future__ import annotations

import argparse
import copy
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

__all__ = [
    "SLOTS",
    "SOURCES",
    "ZONES",
    "SCENARIOS",
    "W_CRIT",
    "W_FAIR",
    "default_sources",
    "default_zones",
    "solve",
    "report",
    "validate",
    "to_frames",
    "with_grid_restriction",
    "with_backup_enabled",
    "scale_scenario",
]

# --------------------------------------------------------------------------
# Feature 2: time slots
# --------------------------------------------------------------------------
SLOTS: Tuple[str, ...] = ("Night", "Morning", "Afternoon", "Evening")

# --------------------------------------------------------------------------
# Feature 1: source management
#   cost        $/kWh
#   avail       kWh available per slot (a "<=" bound: surplus is never forced)
#   emissions   kg CO2 per kWh
#   renewable   True for must-take / weather-dependent sources
#   dispatchable True for sources the operator can ramp
#   emergency   True for sources that should only run when nothing else can
#   storage     optional daily energy cap (kWh summed over all slots)
# --------------------------------------------------------------------------
SOURCES: Dict[str, Dict[str, Any]] = {
    "Solar": {
        "cost": 0.05,
        "emissions": 0.00,
        "renewable": True,
        "dispatchable": False,
        "avail": [0, 40, 60, 10],
    },
    "Wind": {
        "cost": 0.08,
        "emissions": 0.00,
        "renewable": True,
        "dispatchable": False,
        "avail": [30, 30, 25, 20],
    },
    "Thermal": {
        "cost": 0.10,
        "emissions": 0.45,
        "renewable": False,
        "dispatchable": True,
        "avail": [15, 15, 15, 15],
        "storage": 30.0,  # daily energy budget -> hoarded for the evening peak
    },
    "Grid": {
        "cost": 0.20,
        "emissions": 0.35,
        "renewable": False,
        "dispatchable": True,
        "avail": [80, 80, 80, 40],  # evening import restriction
    },
    "Backup": {
        "cost": 0.60,
        "emissions": 0.80,
        "renewable": False,
        "dispatchable": True,
        "emergency": True,
        "avail": [0, 0, 0, 25],  # emergency diesel, evening only
    },
}

# --------------------------------------------------------------------------
# Feature 2: zone demand + criticality (Feature 6)
# --------------------------------------------------------------------------
ZONES: Dict[str, Dict[str, Any]] = {
    "Hospital": {"demand": [20, 25, 25, 20], "critical": True},
    "Residential": {"demand": [45, 30, 25, 60], "critical": False},
    "Commercial": {"demand": [10, 45, 50, 15], "critical": False},
    "Industrial": {"demand": [15, 20, 20, 10], "critical": False},
}

# --------------------------------------------------------------------------
# Feature 6: penalty ordering.  W_CRIT >> W_FAIR >> any generation cost, so the
# optimiser exhausts every kWh of supply (including expensive backup) before it
# sheds a critical zone, and sheds non-critical load before buying backup.
# --------------------------------------------------------------------------
W_CRIT: float = 10_000.0  # $/kWh unserved at a critical zone
W_FAIR: float = 200.0  # $/kWh unserved at a non-critical zone

# Feature 2: demand scenarios, applied as a multiplier on every zone.
SCENARIOS: Dict[str, float] = {"normal": 1.0, "peak": 1.25, "event": 1.10}

TOL = 1e-6


# --------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------
def default_sources() -> Dict[str, Dict[str, Any]]:
    """A deep copy of :data:`SOURCES`, safe to mutate."""
    return copy.deepcopy(SOURCES)


def default_zones() -> Dict[str, Dict[str, Any]]:
    """A deep copy of :data:`ZONES`, safe to mutate."""
    return copy.deepcopy(ZONES)


def _check_model(sources, zones, slots) -> None:
    """Fail loudly on malformed input rather than deep inside HiGHS."""
    if not sources:
        raise ValueError("sources must not be empty")
    if not zones:
        raise ValueError("zones must not be empty")
    T = len(slots)
    for name, src in sources.items():
        if "cost" not in src:
            raise ValueError(f"source {name!r} is missing 'cost'")
        avail = src.get("avail")
        if avail is None or len(avail) != T:
            raise ValueError(
                f"source {name!r} needs 'avail' with {T} entries (one per slot)"
            )
        if any(a < 0 for a in avail):
            raise ValueError(f"source {name!r} has negative availability")
    for name, zone in zones.items():
        dem = zone.get("demand")
        if dem is None or len(dem) != T:
            raise ValueError(
                f"zone {name!r} needs 'demand' with {T} entries (one per slot)"
            )
        if any(d < 0 for d in dem):
            raise ValueError(f"zone {name!r} has negative demand")


def with_grid_restriction(
    sources: Dict[str, Dict[str, Any]],
    slot: int = 3,
    limit: float = 10.0,
) -> Dict[str, Dict[str, Any]]:
    """Feature 9: a dynamic event -- the grid import cap collapses in a slot."""
    out = copy.deepcopy(sources)
    if "Grid" not in out:
        raise KeyError("no 'Grid' source in this model")
    out["Grid"]["avail"][slot] = min(out["Grid"]["avail"][slot], limit)
    return out


def with_backup_enabled(
    sources: Dict[str, Dict[str, Any]], enabled: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Feature 1: toggle the emergency fleet (operator policy switch)."""
    out = copy.deepcopy(sources)
    for name, src in out.items():
        if src.get("emergency"):
            if enabled:
                src["avail"] = list(src.get("standby_avail", src["avail"]))
            else:
                src.setdefault("standby_avail", list(src["avail"]))
                src["avail"] = [0.0] * len(src["avail"])
    return out


def scale_scenario(
    zones: Dict[str, Dict[str, Any]], factor: float
) -> Dict[str, Dict[str, Any]]:
    """Apply a raw demand multiplier (used when a forecast overrides ZONES)."""
    out = copy.deepcopy(zones)
    for zone in out.values():
        zone["demand"] = [d * factor for d in zone["demand"]]
    return out


# --------------------------------------------------------------------------
# Feature 5: the allocation engine
# --------------------------------------------------------------------------
def solve(
    sources: Dict[str, Dict[str, Any]] | None = None,
    zones: Dict[str, Dict[str, Any]] | None = None,
    slots: Sequence[str] | None = None,
    scenario: str = "normal",
    w_crit: float | None = None,
    w_fair: float | None = None,
) -> Dict[str, Any]:
    """Solve one LP over sources x zones x time slots.

    This is a pure function of its inputs: change the data and call it again
    (Feature 9 -- dynamic re-optimisation).

    Returns a dict with the full dispatch plan plus derived reporting fields.
    """
    sources = default_sources() if sources is None else sources
    zones = default_zones() if zones is None else zones
    slots = list(SLOTS) if slots is None else list(slots)
    if scenario not in SCENARIOS:
        raise ValueError(
            f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}"
        )
    _check_model(sources, zones, slots)

    w_crit = float(W_CRIT) if w_crit is None else float(w_crit)
    w_fair = float(W_FAIR) if w_fair is None else float(w_fair)

    factor = float(SCENARIOS[scenario])
    src = list(sources)
    zon = list(zones)
    S, Z, T = len(src), len(zon), len(slots)

    demand = np.array(
        [[factor * float(d) for d in zones[z]["demand"]] for z in zon], dtype=float
    )  # (Z, T)
    avail = np.array(
        [[float(a) for a in sources[s]["avail"]] for s in src], dtype=float
    )  # (S, T)
    cost = np.array([float(sources[s]["cost"]) for s in src], dtype=float)  # (S,)
    emissions = np.array(
        [float(sources[s].get("emissions", 0.0)) for s in src], dtype=float
    )

    crit = [z for z in range(Z) if zones[zon[z]].get("critical")]
    noncrit = [z for z in range(Z) if z not in crit]

    # ---- variable layout -------------------------------------------------
    #   p[s, z, t] : kWh from source s to zone z in slot t   (S*Z*T)
    #   r[t]       : uniform curtailment rate for non-critical zones, in [0,1]
    #   uc[k, t]   : kWh unserved at critical zone k in slot t
    NP, NR, NU = S * Z * T, T, len(crit) * T
    n = NP + NR + NU

    def ip(s: int, z: int, t: int) -> int:
        return (s * Z + z) * T + t

    def ir(t: int) -> int:
        return NP + t

    def iu(k: int, t: int) -> int:
        return NP + NR + k * T + t

    # ---- Feature 3: objective -------------------------------------------
    c = np.zeros(n)
    for s in range(S):
        for z in range(Z):
            c[[ip(s, z, t) for t in range(T)]] = cost[s]
    # Cost of raising the curtailment rate in slot t: it sheds
    # r_t * (total non-critical demand) kWh at W_FAIR $/kWh.
    for t in range(T):
        c[ir(t)] = w_fair * float(demand[noncrit, t].sum()) if noncrit else 0.0
    for k in range(len(crit)):
        c[[iu(k, t) for t in range(T)]] = w_crit

    # ---- constraints -----------------------------------------------------
    A_eq: List[np.ndarray] = []
    b_eq: List[float] = []
    A_ub: List[np.ndarray] = []
    b_ub: List[float] = []

    # Zone energy balance (Feature 2 / 6 / 7).
    for z in range(Z):
        for t in range(T):
            row = np.zeros(n)
            for s in range(S):
                row[ip(s, z, t)] = 1.0
            if z in crit:
                # sum_s p[s,z,t] + uc[k,t] = demand[z,t]
                row[iu(crit.index(z), t)] = 1.0
            elif demand[z, t] > 0:
                # sum_s p[s,z,t] + r_t * demand[z,t] = demand[z,t]
                row[ir(t)] = demand[z, t]
            A_eq.append(row)
            b_eq.append(demand[z, t])

    # Feature 4: per-slot source capacity (a "<=" bound -> surplus allowed).
    for s in range(S):
        for t in range(T):
            row = np.zeros(n)
            for z in range(Z):
                row[ip(s, z, t)] = 1.0
            A_ub.append(row)
            b_ub.append(avail[s, t])

    # Feature 4: optional daily energy cap (e.g. thermal fuel / storage).
    for s in range(S):
        cap = sources[src[s]].get("storage", sources[src[s]].get("energy_cap"))
        if cap is not None:
            row = np.zeros(n)
            for z in range(Z):
                for t in range(T):
                    row[ip(s, z, t)] = 1.0
            A_ub.append(row)
            b_ub.append(float(cap))

    bounds = [(0.0, None)] * NP + [(0.0, 1.0)] * NR + [(0.0, None)] * NU

    res = linprog(
        c,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=bounds,
        method="highs",
    )
    if res.status != 0:
        raise RuntimeError(f"LP failed (status {res.status}): {res.message}")

    x = res.x
    p = np.array(x[:NP], dtype=float).reshape(S, Z, T)
    p[p < 0] = 0.0  # clip solver noise
    rate = np.array(x[NP : NP + NR], dtype=float)
    uc = np.array(x[NP + NR :], dtype=float).reshape(len(crit), T) if NU else np.zeros((0, T))

    generation = p.sum(axis=1)  # (S, T)
    served = p.sum(axis=0)  # (Z, T)
    unmet = demand - served  # (Z, T)
    unmet_crit = np.zeros_like(demand)
    if len(crit):
        unmet_crit[crit, :] = uc

    generation_cost = float((cost[:, None] * generation).sum())
    source_cost = (cost[:, None] * generation).sum(axis=1)  # (S,)
    slot_cost = (cost[:, None] * generation).sum(axis=0)  # (T,)
    fair_penalty = float(
        sum(w_fair * rate[t] * demand[noncrit, t].sum() for t in range(T))
    ) if noncrit else 0.0
    crit_penalty = float(w_crit * uc.sum())
    surplus = np.maximum(avail - generation, 0.0)  # (S, T)

    sol: Dict[str, Any] = {
        # model actually used (report/validate must not reach for globals)
        "sources": copy.deepcopy(sources),
        "zones": copy.deepcopy(zones),
        "slots": list(slots),
        "scenario": scenario,
        "factor": factor,
        "w_crit": w_crit,
        "w_fair": w_fair,
        "src": src,
        "zon": zon,
        # plan
        "p": p,
        "generation": generation,
        "availability": avail,
        "surplus": surplus,
        "utilization": np.divide(
            generation, avail, out=np.zeros_like(generation), where=avail > 0
        ),
        "served": served,
        "demand": demand,
        "unmet": unmet,
        "unmet_critical": unmet_crit,
        "curtailment_rate": rate,
        # economics / environment
        "cost": generation_cost,
        "source_cost": source_cost,
        "slot_cost": slot_cost,
        "fair_penalty": fair_penalty,
        "crit_penalty": crit_penalty,
        "objective": float(res.fun),
        "emissions": float((emissions[:, None] * generation).sum()),
        # flags
        "critical_ok": bool((unmet_crit <= 1e-6).all()) if len(crit) else True,
        "backup_used": bool(
            any(
                generation[s].max() > 1e-6
                for s in range(S)
                if sources[src[s]].get("emergency")
            )
        ),
        "total_demand": float(demand.sum()),
        "total_served": float(served.sum()),
        "total_unmet": float(unmet.sum()),
        "total_generation": float(generation.sum()),
        "recommendations": _recommendations(
            src, slots, sources, generation, surplus, rate, crit, zon
        ),
        "lp": res,
    }
    return sol


def _recommendations(src, slots, sources, generation, surplus, rate, crit, zon):
    """Feature 8: turn surplus / shortage / backup usage into operator advice.

    Surplus is *reported*, never forced into a load: sources are bounded by
    ``<=`` availability, so unused renewable headroom becomes an explicit
    recommendation (shift flexible loads, charge storage, export, curtail).
    """
    recs = []
    S, T = generation.shape
    noncrit = [z for z in range(len(zon)) if z not in crit]
    for t in range(T):
        messages = []
        for s in range(S):
            spec = sources[src[s]]
            used, head = generation[s, t], surplus[s, t]

            if spec.get("emergency") and used > 1e-6:
                messages.append(
                    f"{src[s]}: RUNNING at {used:.1f} kWh -- emergency source "
                    "active, all cheaper options are exhausted"
                )
            if head <= 1e-6:
                continue
            if spec.get("renewable"):
                messages.append(
                    f"{src[s]}: {head:.1f} kWh surplus -> shift flexible loads, "
                    "charge storage, export, or curtail"
                )
            elif spec.get("emergency"):
                messages.append(
                    f"{src[s]}: {head:.1f} kWh emergency headroom held in reserve"
                )
            else:
                messages.append(
                    f"{src[s]}: {head:.1f} kWh dispatchable headroom -> keep as reserve"
                )
        if noncrit and rate[t] > 1e-6:
            messages.append(
                f"shortage: all non-critical zones curtailed uniformly by "
                f"{rate[t] * 100:.1f}% (fairness rate r_t)"
            )
        if not messages:
            messages.append("supply meets demand; no action required")
        recs.append(
            {
                "slot": slots[t],
                "index": t,
                "curtailment_rate": float(rate[t]),
                "surplus": {src[s]: float(surplus[s, t]) for s in range(S)},
                "generation": {src[s]: float(generation[s, t]) for s in range(S)},
                "messages": messages,
            }
        )
    return recs


# --------------------------------------------------------------------------
# Feature 12 (data) / demo output
# --------------------------------------------------------------------------
def report(sol: Dict[str, Any]) -> str:
    """Human-readable per-slot report.  Returns the text and prints it."""
    sources, zones = sol["sources"], sol["zones"]
    src, zon, slots = sol["src"], sol["zon"], sol["slots"]
    generation = sol["generation"]
    lines: List[str] = []

    lines.append(
        f"Scenario: {sol['scenario']} (x{sol['factor']:.2f})   "
        f"Slots: {len(slots)}   Sources: {len(src)}   Zones: {len(zon)}"
    )
    for t, name in enumerate(slots):
        lines.append("")
        lines.append(f"=== {name} ===  slot cost ${sol['slot_cost'][t]:.2f}")
        for s, sname in enumerate(src):
            used, av = generation[s, t], sol["availability"][s, t]
            tags = []
            if sources[sname].get("emergency") and used > 1e-6:
                tags.append("<-- BACKUP IN USE")
            head = av - used
            if head > 1e-6 and sources[sname].get("renewable"):
                tags.append("surplus -> flexible loads / storage / export / curtail")
            elif head > 1e-6 and sources[sname].get("dispatchable"):
                tags.append("reserve headroom")
            bar = ("  " + "  ".join(tags)) if tags else ""
            lines.append(f"  {sname:9s} used {used:6.1f} / {av:5.1f}{bar}")
        for z, zname in enumerate(zon):
            d, sd = sol["demand"][z, t], sol["served"][z, t]
            tag = "CRITICAL" if zones[zname].get("critical") else ""
            pct = (sd / d * 100) if d > 0 else 100.0
            lines.append(
                f"  {zname:12s} served {sd:6.1f} / {d:5.1f}  ({pct:5.1f}%)  "
                f"unmet {d - sd:5.1f}  {tag}"
            )
        r = sol["curtailment_rate"][t]
        if r > 1e-6:
            lines.append(f"  fairness rate r_t = {r * 100:.1f}% (uniform curtailment)")
    lines.append("")
    lines.append(
        f"Generation cost : ${sol['cost']:.2f}"
        f"   |   shortage penalties: fairness ${sol['fair_penalty']:.2f}"
        f", critical ${sol['crit_penalty']:.2f}"
    )
    lines.append(
        f"Total generation: {sol['total_generation']:.1f} kWh   "
        f"demand {sol['total_demand']:.1f} kWh   "
        f"served {sol['total_served']:.1f} kWh   "
        f"unmet {sol['total_unmet']:.1f} kWh"
    )
    lines.append(
        f"CO2             : {sol['emissions']:.2f} kg   "
        f"Critical demand met: {'YES' if sol['critical_ok'] else 'NO'}   "
        f"Backup used: {'YES' if sol['backup_used'] else 'NO'}"
    )
    text = "\n".join(lines)
    print(text)
    return text


def to_frames(sol: Dict[str, Any]):
    """Pandas views of the plan, for the Streamlit dashboard / notebooks."""
    import pandas as pd

    src, zon, slots = sol["src"], sol["zon"], sol["slots"]
    sources = sol["sources"]
    per_slot = pd.DataFrame(
        sol["generation"], index=src, columns=slots
    ).T  # rows = slots
    availability = pd.DataFrame(sol["availability"], index=src, columns=slots).T
    surplus = pd.DataFrame(sol["surplus"], index=src, columns=slots).T
    zone_supply = pd.DataFrame(sol["served"], index=zon, columns=slots).T
    zone_demand = pd.DataFrame(sol["demand"], index=zon, columns=slots).T
    zone_unmet = pd.DataFrame(sol["unmet"], index=zon, columns=slots).T
    source_table = pd.DataFrame(
        {
            "cost_$/kWh": [sources[s]["cost"] for s in src],
            "emissions_kg/kWh": [sources[s].get("emissions", 0.0) for s in src],
            "renewable": [bool(sources[s].get("renewable")) for s in src],
            "emergency": [bool(sources[s].get("emergency")) for s in src],
            "daily_cap": [sources[s].get("storage") for s in src],
            "total_available": [sol["availability"][i].sum() for i in range(len(src))],
            "total_used": [sol["generation"][i].sum() for i in range(len(src))],
            "total_cost_$": [sol["source_cost"][i] for i in range(len(src))],
        },
        index=src,
    )
    return {
        "generation_by_slot": per_slot,
        "availability_by_slot": availability,
        "surplus_by_slot": surplus,
        "zone_supply": zone_supply,
        "zone_demand": zone_demand,
        "zone_unmet": zone_unmet,
        "sources": source_table,
        "curtailment": pd.Series(sol["curtailment_rate"], index=slots, name="r_t"),
    }


# --------------------------------------------------------------------------
# Feature 11: independent validation
# --------------------------------------------------------------------------
def validate(sol: Dict[str, Any], tol: float = 1e-6, verbose: bool = True) -> bool:
    """Re-check every constraint from scratch (not from the LP's own view)."""
    sources, zones = sol["sources"], sol["zones"]
    src, zon, slots = sol["src"], sol["zon"], sol["slots"]
    p, generation = sol["p"], sol["generation"]
    S, Z, T = p.shape
    checks: List[Tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    add("LP reported optimal", sol["lp"].status == 0, str(sol["lp"].message))
    add("flows are finite", bool(np.isfinite(p).all()))
    add("flows non-negative", bool((p >= -tol).all()), f"min={p.min():.2e}")

    # Feature 4: capacity
    worst_cap = float("inf")
    for s in range(S):
        for t in range(T):
            slack = sources[src[s]]["avail"][t] - generation[s, t]
            worst_cap = min(worst_cap, slack)
            if slack < -tol:
                add(
                    f"{src[s]} within capacity @ {slots[t]}",
                    False,
                    f"over by {-slack:.3f} kWh",
                )
    add("all sources within per-slot capacity", worst_cap >= -tol, f"min slack={worst_cap:.3f}")

    # Feature 4: daily energy caps
    for s in range(S):
        cap = sources[src[s]].get("storage", sources[src[s]].get("energy_cap"))
        if cap is not None:
            used = float(generation[s].sum())
            add(
                f"{src[s]} daily energy cap ({cap:g} kWh)",
                used <= cap + tol,
                f"used {used:.3f}",
            )

    # Feature 2: zone balance  served + unmet == demand
    balance = np.abs(sol["served"] + sol["unmet"] - sol["demand"]).max()
    add("zone balance served + unmet = demand", balance <= 1e-4, f"max err={balance:.2e}")

    # Feature 8: never over-serve a zone (surplus is not dumped on load)
    over = (sol["served"] - sol["demand"]).max()
    add("no zone over-served", over <= tol, f"max excess={over:.2e}")

    # Global energy conservation
    gen_total = float(generation.sum())
    add(
        "generation equals served demand",
        abs(gen_total - float(sol["served"].sum())) <= 1e-4,
        f"gen={gen_total:.3f} served={sol['served'].sum():.3f}",
    )

    # Feature 6: critical protection
    crit_idx = [z for z in range(Z) if zones[zon[z]].get("critical")]
    if crit_idx:
        worst = float(sol["unmet"][crit_idx].max())
        add("critical zones fully served", worst <= tol, f"max unmet={worst:.2e}")
    else:
        add("critical zones fully served", True, "no critical zones in model")

    # Feature 7: uniform curtailment across non-critical zones
    noncrit_idx = [z for z in range(Z) if z not in crit_idx]
    fair = True
    fair_detail = ""
    for t in range(T):
        rates = []
        for z in noncrit_idx:
            d = sol["demand"][z, t]
            if d > tol:
                rates.append(sol["unmet"][z, t] / d)
        if rates and (max(rates) - min(rates)) > 1e-4:
            fair = False
            fair_detail = f"{slots[t]}: rates {[f'{x:.4f}' for x in rates]}"
            break
    add("non-critical curtailment is uniform", fair, fair_detail)

    add(
        "curtailment rate within [0, 1]",
        bool(((sol["curtailment_rate"] >= -tol) & (sol["curtailment_rate"] <= 1 + tol)).all()),
        f"rates={np.round(sol['curtailment_rate'], 4).tolist()}",
    )

    # Feature 3: reported cost is consistent with the flows
    recomputed = sum(
        float(sources[src[s]]["cost"]) * float(p[s].sum()) for s in range(S)
    )
    add(
        "reported cost matches flows",
        abs(recomputed - sol["cost"]) <= 1e-4,
        f"reported {sol['cost']:.4f} vs recomputed {recomputed:.4f}",
    )

    ok = all(passed for _, passed, _ in checks)
    if verbose:
        for name, passed, detail in checks:
            mark = "PASS" if passed else "FAIL"
            suffix = f"   [{detail}]" if detail and not passed else ""
            print(f"{mark}  {name}{suffix}")
        print(f"\nValidation: {'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'} "
              f"({sum(1 for _, p_, _ in checks if p_)}/{len(checks)})")
    return ok


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _build_sources(args) -> Dict[str, Dict[str, Any]]:
    sources = default_sources()
    if args.grid_restriction:
        sources = with_grid_restriction(sources, slot=args.slot, limit=args.grid_limit)
    if not args.backup:
        sources = with_backup_enabled(sources, enabled=False)
    return sources


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Classical LP engine for city-scale energy allocation."
    )
    ap.add_argument("--scenario", default="normal", choices=sorted(SCENARIOS))
    ap.add_argument(
        "--grid-restriction",
        action="store_true",
        help="simulate an evening grid import restriction",
    )
    ap.add_argument("--slot", type=int, default=3, help="slot index for the restriction")
    ap.add_argument("--grid-limit", type=float, default=10.0)
    ap.add_argument("--no-backup", dest="backup", action="store_false", help="disable emergency fleet")
    ap.add_argument("--validate", action="store_true", help="run independent validation")
    args = ap.parse_args(list(argv) if argv is not None else None)

    sources = _build_sources(args)
    sol = solve(sources=sources, scenario=args.scenario)
    report(sol)
    if args.validate:
        print()
        ok = validate(sol)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
