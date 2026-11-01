# ⚡ Quantum-Assisted City Energy Allocator

Cost-optimal electricity allocation across **multiple generation sources**, **multiple
city zones** and **time slots**, with hard protection for critical infrastructure,
*fair* load-shedding when supply runs short, and a **QUBO/QAOA quantum module** for the
combinatorial layer underneath.

Built as a hybrid quantum-classical system: a classical LP solves the continuous
day-scale dispatch exactly, and QAOA attacks the *discrete* decisions (which sources run
at which level, which zones get shed) before a classical LP "repair" turns that decision
back into exact, feasible flows.

```
sources / zones / scenarios ──► solve()  [classical LP]  ──► validate() ──► dashboard
                                     ▲
              quantum module:  QUBO ──► preprocessing ──► QAOA ──► decode ──► LP repair
              (discrete decisions: source output levels + zone shedding)
```

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python demo.py                 # the full narrated demo (all 12 features)
python demo.py --no-quantum    # same, without QAOA (fast, no Qiskit needed)

python energy_allocation.py --scenario peak --validate
python energy_allocation.py --grid-restriction --no-backup --validate   # shortage

python quantum_module.py                       # QUBO + QAOA vs brute force
python quantum_module.py --grid-restriction    # the interesting shortage case
python quantum_module.py --no-reduce           # same QUBO without preprocessing

streamlit run app.py           # dashboard
pytest                         # 99 tests
```

---

## Feature → mechanism → code

| # | Feature | How it works | Where |
|---|---|---|---|
| 1 | Source management | `SOURCES` dict: cost, per-slot availability, emissions, daily energy cap, `renewable`/`dispatchable`/`emergency` flags | `energy_allocation.SOURCES` |
| 2 | Zone demand | `ZONES` dict: per-slot demand + `critical` flag; `SCENARIOS` demand multipliers | `energy_allocation.ZONES`, `.SCENARIOS` |
| 3 | Cost optimisation | LP objective = Σ cost·generation; QUBO linear terms | `solve()` |
| 4 | Capacity constraints | per-slot `≤` availability inequalities + optional daily energy cap; binary level encoding in the QUBO | `solve()`, `build_slot_qubo()` |
| 5 | Allocation engine | one LP over sources × zones × slots → flow matrix `p[s, z, t]` | `solve()` |
| 6 | Critical protection | unserved-demand penalty `W_CRIT = 10 000` ≫ `W_FAIR = 200` ≫ any generation cost | `solve()`, `build_slot_qubo()` |
| 7 | Shortage + fairness | a single per-slot curtailment **rate** `r_t` forces every non-critical zone to lose the *same percentage* | `solve()` |
| 8 | Surplus handling | availability is a `≤` bound, so surplus is reported and never forced into load; structured operator recommendations | `_recommendations()` |
| 9 | Dynamic re-optimisation | `solve()` is a pure function of its inputs — change the data, re-solve | `solve()`, `with_grid_restriction()` |
| 10 | Quantum module | QUBO → classical preprocessing → QAOA → decode → LP repair | `quantum_module.py` |
| 11 | Validation | `validate()` re-derives every constraint from the raw flows | `validate()` |
| 12 | Dashboard | Streamlit, 5 tabs, every control re-solves | `app.py` |

---

## The classical engine

### Variables

For `S` sources, `Z` zones and `T` slots the LP has:

- `p[s, z, t]` — kWh delivered from source `s` to zone `z` in slot `t`
- `r[t]` — the **uniform curtailment rate** for non-critical zones, in `[0, 1]`
- `uc[k, t]` — kWh unserved at critical zone `k` in slot `t`

Subject to zone balance, per-slot source capacity, optional daily energy caps, and
non-negativity. Solved with HiGHS via `scipy.optimize.linprog`.

### Why fairness is a *rate*, not a per-zone variable

This is the design decision worth defending in a demo. A naive model adds one
"unmet demand" variable per zone. Because every unserved kWh at a non-critical zone
carries the *same* penalty, the LP is indifferent to which zone absorbs the shortage —
so it dumps the entire shortfall on one arbitrarily chosen zone.

Instead we introduce one curtailment rate `r_t` per slot and impose

```
served[z, t] = (1 − r_t) · demand[z, t]      for every non-critical zone z
```

Every non-critical zone loses the **same percentage**, structurally. The model stays
linear, and `r_t` doubles as a single human-readable fairness indicator. Measured on the
shortage scenario below: Residential, Commercial and Industrial are each curtailed by
exactly 58.8%, while the hospital is untouched.

### Verified results

`python demo.py --no-quantum` produces these numbers:

| Scenario | Generation cost | Unmet | Critical | Evening curtailment |
|---|---|---|---|---|
| normal (×1.00) | **$62.90** | 0.0 kWh | protected | 0% |
| event (×1.10) | $72.50 | 5.5 kWh | protected | — |
| peak (×1.25) | $82.40 | 21.2 kWh | protected | 20.0% uniform |
| grid restriction + no backup | $44.90 | 50.0 kWh | **protected** | **58.8% uniform** |

On a normal day the engine also demonstrates:

- cheapest-first dispatch (Solar \$0.05 and Wind \$0.08 taken in full wherever available);
- the Thermal **daily energy cap of 30 kWh binds exactly** (`15, 0, 0, 15`) — it is hoarded
  for the evening peak rather than burned at night;
- Backup runs only in the evening, and only after every cheaper source is exhausted
  (Solar, Wind and Grid are all at their caps when Backup switches on);
- renewable surplus is *reported* with recommendations, never forced into a load.

`validate()` re-derives all 12 constraint families from the raw flow matrix — capacity,
daily caps, zone balance, no over-serving, energy conservation, critical protection,
uniform curtailment, rate bounds and cost consistency — and the tests confirm it actually
**fails** when a plan is tampered with.

---

## The quantum module

### Where quantum computing genuinely fits

The day-scale problem is linear and continuous, so a classical LP is the right engine: it
is exact and solves in milliseconds. Pretending otherwise would not survive a judge's
question. The quantum module therefore targets the **combinatorial layer underneath**:

- which sources run, and at what output level (unit-commitment style), and
- which zones get shed during a shortage (load-shedding blocks).

Those are discrete decisions. QAOA picks the combination; a classical LP then guarantees
feasibility by computing exact flows for it.

### Encoding

Per time slot:

- **2 bits per source** → output level in `{0, ⅓, ⅔, 1} × availability`
- **1 bit per zone** → `1` = serve fully, `0` = shed
- `n = 2S + Z` qubits. For the default model (5 sources, 4 zones) that is
  **14 qubits / 16 384 states** — small enough to brute-force exactly and use as ground
  truth for validating QAOA.

```
E(b) = generation_cost + λ·(supply − served_demand)² + Σ_z W_z·D_z·(1 − serve_z)
```

`W_z` is `W_CRIT` for critical zones, so shedding the hospital costs 200 000 against
2 000 for an industrial zone — a 100× separation, tested explicitly. The hospital is shed
only when physically unavoidable.

### Classical preprocessing (the single biggest win)

Real quantum-classical pipelines shrink and condition the QUBO before handing it to a
solver. Two rules are provably safe here:

1. a source with **zero availability** in the slot cannot run → its level bits are `0`;
2. if availability covers total demand, **no zone needs shedding** → every serve bit is
   `1`. Otherwise, if it covers *critical* demand, the critical serve bits are `1`
   (that is not really a choice — it is Feature 6, decided classically).

`fix_qubo_bits()` substitutes those bits algebraically, and the tests verify the exact
energy identity

```
b_full @ Q @ b_full  ==  b_reduced @ Q_reduced @ b_reduced + offset
```

on 300 random bitstrings, plus that the reduced argmin maps back to the full-space
optimum.

Fixing the zone bits also removes the huge `W_CRIT · D ≈ 2×10⁵` coefficients from the
Hamiltonian, which is what lets QAOA concentrate amplitude at all. On the evening slot:

| QAOA ran on | qubits | P(sampling the optimum) | runtime |
|---|---|---|---|
| full QUBO | 14 | **0.0001** | 20.8 s |
| after preprocessing | 10 | **0.0498** | 6.9 s |

A ~500× improvement in the probability of sampling the optimum, *and* 3× faster.

### Making QAOA reliable

Three things were needed beyond the textbook circuit, all measured:

1. **Hamiltonian normalisation.** The QUBO coefficients span ~7 orders of magnitude
   (a 2×10⁵ shed penalty next to a \$1 generation cost), so `γ ∈ [0, 2π]` is meaningless
   on the raw operator. Dividing by the largest coefficient leaves the ground state
   unchanged and puts the angles on a searchable scale.
2. **Correct mixer range.** Betas drive `RX(2β)` and are sampled from `[0, π/2]`, the
   physically distinct range; gammas from a full turn. Seeding beta over `[0, 2π]`
   measurably degrades the result.
3. **Pooled multi-start sampling.** 12 COBYLA restarts, sampling from *every* restart and
   merging the counts, rather than sampling only from the lowest-expectation one.

Best-of-shots accuracy (does QAOA reproduce the exact brute-force optimum?) over seeds
1, 7, 11, 23, 42, 99:

| configuration | evening slot | shortage slot |
|---|---|---|
| no pooling, 6 restarts | 5/6 | 4/6 |
| pooling, 6 restarts | 6/6 | 5/6 |
| **pooling, 12 restarts (default)** | **6/6** | **6/6** |

### Example: the shortage slot

With the grid restricted (80 kWh available against 105 kWh demanded), brute force, the
greedy baseline and QAOA all agree on the same plan:

```
serve : ['Hospital', 'Residential']
shed  : ['Commercial', 'Industrial']
levels: Solar=10.0  Wind=20.0  Thermal=15.0  Grid=10.0  Backup=25.0
QUBO objective = $5020.60  =  $20.60 generation + $5000 shedding penalty
```

That is the *correct* economic choice: shedding Commercial (\$3 000) + Industrial (\$2 000)
costs \$5 000, versus \$12 000 for Residential and \$200 000 for the hospital. Backup runs
at full output because it is cheaper than shedding anyone. The LP repair then confirms the
committed levels cover the served zones exactly.

### Discretisation gap (be upfront about this)

The QUBO's quantised source levels cannot match the continuous LP exactly. On the evening
slot, measured:

| encoding | qubits | brute-force dispatch cost | exact search time |
|---|---|---|---|
| 2 bits per source | 14 | \$26.10 | 4 ms |
| 3 bits per source | 19 | \$24.39 | 78 ms |
| continuous LP | — | **\$23.60** | < 1 ms |

More level bits converge toward the LP optimum at the cost of qubits
(`--level-bits 3`). This is a real, quantifiable property of binary encodings — not a bug,
and worth showing.

### Honest scope

- QAOA here runs on a **statevector simulator**, which is classical computation. What it
  validates is the *formulation*. Real hardware (IBM Quantum, D-Wave) is the next step —
  `--export model.json` writes both the QUBO and its Ising form `(h, J, offset)` for that.
- The classical LP already solves this problem size optimally in milliseconds. Quantum's
  value proposition appears when the discrete decisions explode: hundreds of generators,
  multi-day unit commitment, network switching. This module is the correct *pattern* for
  that regime, demonstrated at a size where it can be verified exactly.
- Zone shedding (quantum, discrete) and proportional curtailment (LP, continuous) are two
  legitimate shortage policies. Showing both is a feature, not a contradiction.

---

## Dashboard

`streamlit run app.py` — five tabs:

- **Dispatch** — stacked generation mix, supply vs demand vs availability, per-slot
  surplus / utilisation, operator recommendations, full text report
- **Zones** — served vs demanded energy, unmet demand, and the uniform curtailment rate
- **Quantum module** — runs the QUBO/QAOA comparison for a chosen slot and shows qubit
  counts, `P(optimum)`, the reduction check and each solver's decision
- **Validation** — live output of all constraint checks
- **Model** — source and zone parameters plus the raw flow matrix `p[s, z, t]`

Every sidebar control (scenario, grid restriction, backup toggle, penalty weights,
per-slot availability) re-solves the LP, so the dashboard *is* the live interface to
Feature 9.

---

## Project layout

```
energy_allocation.py    classical LP engine: solve, report, validate, to_frames  (Features 1–9, 11)
quantum_module.py       QUBO, preprocessing, QAOA, LP repair, comparison         (Feature 10)
app.py                  Streamlit dashboard                                      (Feature 12)
demo.py                 narrated end-to-end demo in presentation order
tests/                  99 tests (41 engine + 58 quantum)
requirements.txt
pytest.ini
```

Both modules are also CLIs:

```
python energy_allocation.py --scenario {normal,peak,event} [--grid-restriction]
                            [--slot N] [--grid-limit KWH] [--no-backup] [--validate]

python quantum_module.py    [--slot N] [--penalty LAMBDA] [--factor F] [--level-bits N]
                            [--reps P] [--restarts N] [--shots N] [--seed S]
                            [--grid-restriction] [--no-backup] [--no-reduce]
                            [--no-quantum] [--export PATH]
```

---

## Testing

```bash
pytest              # everything, ~30 s (includes the QAOA end-to-end tests)
pytest -m "not slow"  # skips the QAOA optimisation loops, ~2 s
```

Qiskit-dependent tests skip automatically when Qiskit is absent. Coverage includes the
energy identity of the QUBO→Ising conversion and of bit-fixing, brute force against naive
enumeration, chunk-invariance of brute force, QAOA circuit structure (only `h`/`rz`/`rzz`/
`rx`, uniform at zero angle, exact phase behaviour of the cost layer), repair-LP
feasibility and its infeasible and all-shed paths, and negative tests proving
`validate()` fails on capacity violations, critical shortfalls, over-serving and broken
uniform curtailment.

---

## Natural extensions

- **Multi-slot coupled QUBO** — the Thermal daily cap links slots, so a multi-slot
  encoding needs those bits shared across time (more qubits, genuinely harder, and where
  quantum would start to earn its keep)
- **Transmission limits** — per-link bounds on the flow matrix `p[s, z, t]`
- **ML demand forecasting** — feed predicted slot demands into `solve(zones=...)`;
  `scale_scenario()` already exists for this
- **Noise models** — run the same circuit under `qiskit_aer` noise to study hardware
  readiness, and submit to a real IBM Quantum backend using the exported Ising model
- **Finer encodings** — more level bits, or one-hot encoding, to close the
  discretisation gap

---

## License

MIT — see [LICENSE](LICENSE).
