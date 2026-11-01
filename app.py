"""
Streamlit dashboard (Feature 12).

Run with::

    streamlit run app.py

Tabs: Dispatch (the LP plan), Zones (who gets served), Quantum (the QAOA vs
brute-force comparison) and Validation (the independent constraint checks).
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from energy_allocation import (
    SCENARIOS,
    SLOTS,
    W_CRIT,
    W_FAIR,
    default_sources,
    default_zones,
    report,
    solve,
    to_frames,
    validate,
    with_backup_enabled,
    with_grid_restriction,
)

st.set_page_config(
    page_title="Quantum-Assisted City Energy Allocator",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Quantum-Assisted City Energy Allocator")
st.caption(
    "Cost-optimal multi-source, multi-zone dispatch with critical-load protection, "
    "fair curtailment and a QUBO/QAOA combinatorial layer."
)

# --------------------------------------------------------------------------
# Sidebar: Features 1, 2, 9
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    scenario = st.selectbox(
        "Demand scenario",
        list(SCENARIOS),
        format_func=lambda k: f"{k} (x{SCENARIOS[k]:.2f})",
        help="Feature 2/9: scales every zone's demand, then re-solves.",
    )
    st.divider()
    st.subheader("Dynamic events (Feature 9)")
    outage = st.checkbox(
        "Evening grid import restriction",
        help="Drops the grid import cap in the selected slot.",
    )
    outage_slot = st.select_slider(
        "Restriction slot", options=list(range(len(SLOTS))), format_func=lambda i: SLOTS[i], value=3
    )
    grid_limit = st.slider("Grid cap in that slot (kWh)", 0.0, 80.0, 10.0, 5.0)
    backup_on = st.checkbox("Allow emergency backup generators", value=True)
    st.divider()
    st.subheader("Penalties (Feature 6)")
    w_crit = st.number_input("W_CRIT ($/kWh unserved, critical)", value=float(W_CRIT), step=1000.0)
    w_fair = st.number_input("W_FAIR ($/kWh unserved, non-critical)", value=float(W_FAIR), step=50.0)
    st.caption("Critical protection holds while W_CRIT >> W_FAIR >> generation cost.")
    st.divider()
    st.subheader("Source availability (Feature 1)")
    base = default_sources()
    edited = {}
    with st.expander("Edit per-slot availability"):
        for name, spec in base.items():
            vals = st.text_input(
                f"{name} avail (kWh per slot, $/kWh={spec['cost']})",
                value=", ".join(str(v) for v in spec["avail"]),
                key=f"avail_{name}",
            )
            edited[name] = vals
    if st.button("Re-optimise", type="primary", width="stretch"):
        st.session_state["force"] = True

sources = default_sources()
zones = default_zones()

# apply sidebar edits
for name, raw in edited.items():
    try:
        parsed = [float(x) for x in raw.split(",") if x.strip() != ""]
        if len(parsed) == len(SLOTS):
            sources[name]["avail"] = parsed
    except ValueError:
        st.sidebar.error(f"Could not parse availability for {name}")

if outage:
    sources = with_grid_restriction(sources, slot=int(outage_slot), limit=float(grid_limit))
if not backup_on:
    sources = with_backup_enabled(sources, enabled=False)

sol = solve(
    sources=sources,
    zones=zones,
    scenario=scenario,
    w_crit=float(w_crit),
    w_fair=float(w_fair),
)
frames = to_frames(sol)

# --------------------------------------------------------------------------
# Headline metrics
# --------------------------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Generation cost", f"${sol['cost']:,.2f}")
m2.metric(
    "Unmet demand",
    f"{sol['total_unmet']:,.1f} kWh",
    delta=f"{100 * sol['total_served'] / max(sol['total_demand'], 1e-9):.1f}% served",
    delta_color="normal",
)
m3.metric(
    "Critical load",
    "protected" if sol["critical_ok"] else "AT RISK",
    delta="0 kWh unserved" if sol["critical_ok"] else f"{sol['unmet_critical'].sum():.1f} kWh",
    delta_color="off" if sol["critical_ok"] else "inverse",
)
m4.metric("CO2", f"{sol['emissions']:,.1f} kg")
m5.metric("Backup used", "YES" if sol["backup_used"] else "no")

worst_rate = float(np.max(sol["curtailment_rate"]))
if worst_rate > 1e-6:
    st.warning(
        f"**Shortage:** non-critical zones are curtailed by a uniform "
        f"**{worst_rate * 100:.1f}%** (fairness rate r_t). Every non-critical zone "
        "loses the same percentage -- this is Feature 7 by construction."
    )
else:
    st.success("Supply covers all demand in every slot; no curtailment required.")

tabs = st.tabs(["Dispatch", "Zones", "Quantum module", "Validation", "Model"])

# --------------------------------------------------------------------------
# Tab 1: dispatch
# --------------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Generation mix by slot")
        fig = go.Figure()
        for s_name in sol["src"]:
            fig.add_trace(
                go.Bar(
                    name=s_name,
                    x=sol["slots"],
                    y=frames["generation_by_slot"][s_name].values,
                )
            )
        fig.update_layout(
            barmode="stack",
            yaxis_title="kWh",
            height=380,
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Supply vs demand")
        fig2 = go.Figure()
        fig2.add_bar(x=sol["slots"], y=frames["generation_by_slot"].sum(axis=1).values, name="generation")
        fig2.add_scatter(
            x=sol["slots"],
            y=frames["zone_demand"].sum(axis=1).values,
            name="demand",
            mode="lines+markers",
            line=dict(width=3, color="crimson"),
        )
        fig2.add_scatter(
            x=sol["slots"],
            y=frames["availability_by_slot"].sum(axis=1).values,
            name="availability",
            mode="lines",
            line=dict(width=2, dash="dot", color="grey"),
        )
        fig2.update_layout(height=320, yaxis_title="kWh", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, width="stretch")

    with right:
        st.subheader("Surplus & utilisation (Feature 8)")
        t_idx = st.selectbox(
            "Slot", range(len(SLOTS)), format_func=lambda i: SLOTS[i], key="disp_slot"
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "used": sol["generation"][:, t_idx],
                    "available": sol["availability"][:, t_idx],
                    "surplus": sol["surplus"][:, t_idx],
                    "utilisation": np.round(sol["utilization"][:, t_idx] * 100, 1),
                    "cost $": np.round(
                        sol["p"][:, :, t_idx].sum(axis=1)
                        * np.array([sol["sources"][s]["cost"] for s in sol["src"]]),
                        2,
                    ),
                },
                index=sol["src"],
            ),
            width="stretch",
        )
        st.markdown("**Operator recommendations**")
        for msg in sol["recommendations"][t_idx]["messages"]:
            st.markdown(f"- {msg}")

    st.subheader("Slot detail")
    with st.expander("Full text report"):
        buf = io.StringIO()
        with redirect_stdout(buf):
            report(sol)
        st.code(buf.getvalue(), language="text")

# --------------------------------------------------------------------------
# Tab 2: zones
# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Served vs demanded energy")
    fig3 = go.Figure()
    for z_name in sol["zon"]:
        fig3.add_bar(name=z_name, x=sol["slots"], y=frames["zone_supply"][z_name].values)
    fig3.update_layout(
        barmode="stack", yaxis_title="kWh served", height=380,
        legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig3, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Zone supply**")
        st.dataframe(frames["zone_supply"], width="stretch")
        st.markdown("**Unmet demand**")
        st.dataframe(frames["zone_unmet"], width="stretch")
    with c2:
        st.markdown("**Zone demand (scenario-adjusted)**")
        st.dataframe(frames["zone_demand"], width="stretch")
        st.markdown("**Uniform curtailment rate r_t (Feature 7)**")
        st.dataframe(
            pd.DataFrame({"r_t": frames["curtailment"].values,
                          "curtailed %": np.round(frames["curtailment"].values * 100, 2)},
                         index=sol["slots"]),
            width="stretch",
        )
    st.caption(
        "Critical zones are never curtailed: their shortfall is a separate "
        "penalised variable, so the optimiser exhausts every kWh of supply "
        "(including expensive backup) before a hospital loses power."
    )

# --------------------------------------------------------------------------
# Tab 3: quantum module
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Feature 10 — QUBO / QAOA on one time slot")
    st.caption(
        "Quantum optimisation handles the discrete layer (which sources run at "
        "which level, which zones get shed); a classical LP then repairs the "
        "exact flows. QAOA runs on a statevector simulator."
    )
    qc1, qc2, qc3 = st.columns([1, 1, 2])
    q_slot = qc1.selectbox("Slot", range(len(SLOTS)), format_func=lambda i: SLOTS[i], index=3)
    use_reduce = qc2.toggle("Classical preprocessing", value=True,
                            help="Fix bits that are provably determined before running QAOA.")
    reps = qc3.slider("QAOA depth p", 1, 4, 2)

    if st.button("Run quantum comparison", type="primary"):
        with st.spinner("Building QUBO and running QAOA..."):
            from quantum_module import compare

            res = compare(
                t=int(q_slot),
                sources=sources,
                zones=zones,
                factor=float(SCENARIOS[scenario]),
                reps=int(reps),
                reduce_qubo=bool(use_reduce),
                verbose=False,
            )
            st.session_state["quantum_result"] = res

    res = st.session_state.get("quantum_result")
    if res:
        meta = res["meta"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Full encoding", f"{meta['n_qubits']} qubits")
        k2.metric("QAOA ran on", f"{res['Q_reduced'].shape[0]} qubits")
        k3.metric("Search space", f"2^{meta['n_qubits']} = {meta['n_states']:,}")
        k4.metric("P(optimum)", f"{res['p_opt']:.4f}" if res["p_opt"] is not None else "-")

        if res["fixed_bits"]:
            ok = res["reduction_ok"]
            (st.success if ok else st.error)(
                f"Preprocessing fixed {len(res['fixed_bits'])} bits. "
                f"Reduction preserves the optimum: {'YES' if ok else 'NO'}"
            )
            for reason in res["reduction_reasons"]:
                st.markdown(f"- {reason}")

        rows = pd.DataFrame(
            [
                {
                    "solver": r["solver"],
                    "qubits": r.get("qubits"),
                    "QUBO energy": r.get("qubo_energy"),
                    "gap": r.get("gap"),
                    "repaired $": r.get("repaired_cost"),
                    "feasible": r.get("feasible"),
                    "P(opt)": r.get("p_opt"),
                    "runtime ms": r.get("runtime_ms"),
                }
                for r in res["rows"]
                if "error" not in r
            ]
        ).set_index("solver")
        st.dataframe(rows.round(4), width="stretch")

        best = next((r for r in res["rows"] if r["solver"].startswith("QAOA best")), None)
        bf = res["brute_force"]
        if best:
            if abs(best["gap"]) < 1e-6:
                st.success("QAOA reproduced the brute-force optimum exactly.")
            else:
                st.warning(f"QAOA best sample is {best['gap']:.2f} above the optimum.")
        st.markdown("**Decisions**")
        for r in res["rows"]:
            if "error" in r:
                st.error(r["error"])
                continue
            st.markdown(
                f"- **{r['solver']}** — serve {r['served']}, shed {r['shed']}  \n"
                f"  levels: "
                + ", ".join(f"{nm}={lv:.1f}" for nm, lv in zip(meta["sources"], r["levels"]))
            )
        st.caption(
            "Honest scope: a statevector simulation is classical computation. It "
            "validates the QUBO formulation; IBM Quantum / D-Wave hardware is the "
            "next step. The classical LP already solves this problem size "
            "optimally in milliseconds -- quantum's advantage appears when the "
            "discrete decisions explode (hundreds of generators, multi-day "
            "commitment)."
        )

# --------------------------------------------------------------------------
# Tab 4: validation
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Feature 11 — independent constraint validation")
    st.caption("Every constraint is re-checked from the raw flows, not from the solver's own view.")
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = validate(sol)
    st.code(buf.getvalue(), language="text")
    st.success("All constraints satisfied") if ok else st.error("Validation failures present")

# --------------------------------------------------------------------------
# Tab 5: model
# --------------------------------------------------------------------------
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sources (Feature 1)**")
        st.dataframe(frames["sources"], width="stretch")
    with c2:
        st.markdown("**Zones (Feature 2)**")
        st.dataframe(
            pd.DataFrame(
                {
                    "demand": [zones[z]["demand"] for z in sol["zon"]],
                    "critical": [bool(zones[z].get("critical")) for z in sol["zon"]],
                },
                index=sol["zon"],
            ),
            width="stretch",
        )
    st.markdown("**Flow matrix p[source, zone, slot]**")
    slot_pick = st.selectbox("Slot", range(len(SLOTS)), format_func=lambda i: SLOTS[i], key="flow_slot")
    st.dataframe(
        pd.DataFrame(sol["p"][:, :, slot_pick], index=sol["src"], columns=sol["zon"]).round(2),
        width="stretch",
    )
