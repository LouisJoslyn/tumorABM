# Tumor–Immune Escape ABM — Design Spec (v1)

Stochastic, on-lattice, 2D agent-based model of a solid tumor growing in tissue
and the CD8 T-cell response against it. Question #1: **immunoediting** — from the
same seed, how often does the immune system *eliminate*, *hold in equilibrium*,
or *lose to* the tumor, and what spatial features decide it? (Immunotherapy /
anti-PD-1 is a later extension on the same model.)

## 1. Scales

| Quantity | Value |
|---|---|
| Grid | 300 × 300 square lattice (configurable), 15 µm per site → 4.5 × 4.5 mm slice |
| Agent time step `dt` | 6 min (0.1 h) |
| Run length | 30–60 days (default 40 d) |
| Snapshot interval | every 2 h of model time |

The lattice code is written over flat indices + neighbour offsets so a 3D grid is
a config change in a later version.

## 2. Agents

### Tumor cells (one per site, stored as per-site arrays)
| Field | Meaning |
|---|---|
| `state` | 0 empty · 1 proliferating · 2 quiescent · 3 necrotic · 4 apoptotic (T-cell killed) · 5 vessel |
| `lineage` | clone ID; a new ID is created when a mutation occurs |
| `antigen` | antigenicity a ∈ [0,1] — scales T-cell recognition |
| `pdl1` | PD-L1 level p ∈ [0,1] — inhibits/exhausts T cells on contact |
| `timer` | time since death (for clearing debris) |

Traits are inherited on division and mutate with probability `mu` per division
(Gaussian step, clipped to [0,1]); a mutation creates a new lineage ID.

### CD8 T cells (list of agents + occupancy layer, max 1 per site)
Fields: `x, y, alive, exhaustion e, kills, age, divisions`. T cells live in a
separate layer, so they can infiltrate between tumor cells.

### Vessels
Fixed sites seeded randomly in stroma (density `vessel_density`). They supply
oxygen and are the entry points for recruited T cells. Vessels engulfed by tumor
regress with probability `vessel_regress` per step when surrounded by tumor
(creates hypoxic / necrotic cores).

## 3. Fields (diffuse on the same grid, quasi-steady)

Solved each step with a few warm-started red–black Gauss–Seidel sweeps of
`D∇²c − k(x)c + S(x) = 0` (non-dimensionalised; units of sites).

| Field | Source | Sink | Role |
|---|---|---|---|
| Oxygen `O` | vessels (Dirichlet O = 1) | tissue (low), tumor cells (high) | quiescence < `O_q`, necrosis < `O_n` |
| Chemokine `C` (CXCL9/10-like) | tumor cells, ×`chemo_secrete`; boosted near kills | linear decay | T-cell recruitment at vessels + chemotaxis |
| Suppression `S` (TGF-β-like) | tumor cells, ×`supp_secrete` | linear decay | slows T-cell motility, lowers kill prob |

## 4. Rules (per step, random update order)

**Tumor cell (proliferating)**
1. If `O < O_n` → necrotic. If `O < O_q` → quiescent (reversible if O recovers).
2. Divide with prob `dt / T_div` into a random empty Moore neighbour; if none → quiescent
   (contact inhibition). Daughter inherits traits (+ mutation).
3. Spontaneous apoptosis with prob `dt / T_death`.

**T cell**
1. Age; die with prob `dt / T_life`; die faster when exhausted.
2. Scan Moore neighbourhood for live tumor cells. For a target with antigen `a`, PD-L1 `p`,
   local suppression `S`:  `P_kill = k_kill · a · (1 − p_block·p) · (1 − s_block·S) · (1 − e)`.
   - On kill: target → apoptotic, `kills += 1`, chemokine burst, chance `p_prolif` to divide
     (clonal expansion) into an empty neighbour.
   - Every contact: exhaustion `e += e_contact + e_pdl1·p`; `e ≥ 1` → exhausted (no killing).
3. Otherwise move to a neighbour, biased up the chemokine gradient (`chi`), with move
   probability reduced by suppression `S`.

**Recruitment**: after a priming delay `t_prime`, each vessel recruits a T cell with prob
`r_recruit · C_local · dt`.

## 5. Scenarios

| Scenario | Idea | Key parameter differences |
|---|---|---|
| **Hot** (inflamed) | high antigenicity, strong chemokine signal, little suppression | `antigen0` high, `chemo_secrete` high, `supp_secrete` low |
| **Cold** (immune desert) | low antigenicity, weak recruitment, suppressive | `antigen0` low, `chemo_secrete` low, `supp_secrete` high |

Later: **Excluded** (T cells stuck at margin), and **anti-PD-1** (`p_block → 0` from day X).

## 5b. Adaptive resistance (v2)

* **IFN-γ field `G`**: secreted by active T cells (`ifn_secrete`), short range (`k_ifn`). Tumor-cell effective PD-L1 =
  min(1, intrinsic + `pdl1_ifn` · G / (G + `ifn_half`)).
* **Chronic exhaustion**: T cells engaged with a target gain `e_chronic` per hour. Anti-PD-1 does not block this.
* **Terminal exhaustion persists**: exhausted T cells live `T_life_ex` = 120 h and keep their sites, so a long-untreated
  tumor accumulates T cells the drug cannot rescue.

## 5c. Macrophages (v2)

Agent list + occupancy layer (like T cells). Polarization m ∈ [0,1] (0 = M1, 1 = M2).

| Rule | Detail |
|---|---|
| Recruitment | CCL2-like field `Q` secreted by hypoxic tumor cells (O < `O_hyp`); monocytes enter at vessels with prob ∝ `r_mac`·Q, arriving at m = 0.5. Tissue-resident macrophages are seeded at start (`macrophages.resident_density`, m0). |
| Migration | biased walk up the CCL2 gradient (`chi_mac`) and toward low oxygen (`chi_hyp`); slower than T cells (`mac_move`). |
| Polarization | dm/dt = `k_to_m2`·(S + hypoxia)·(1 − m) − `k_to_m1`·IFN-γ/(IFN-γ + 0.1)·m |
| M1 effects | phagocytose an adjacent tumor cell (`mac_kill`·(1 − m) per step); add CXCL9/10-like chemokine (`m1_chemo`·(1 − m)), which recruits T cells |
| M2 effects | add suppressor (`m2_supp`·m), which slows T cells and blunts killing and is not blocked by anti-PD-1; sprout a vessel in hypoxic tissue (`m2_angio`·m) |

## 5d. Anti-PD-1 extension

Config section `treatment: {day, drug_eff}` (CLI: `--treat <day>`). From the treatment day, `p_block` (PD-L1 inhibition
of killing) and `e_pdl1` (PD-L1-driven exhaustion) are multiplied by `1 − drug_eff` (0.9). Already-exhausted T cells
are not rescued, and chronic exhaustion and suppression are unaffected.

Results with the full v2 model (seeds 1–3 unless noted):

| Scenario | Config | Result |
|---|---|---|
| PD-L1-high inflamed, untreated | `pdl1_high.yaml` | escape 2/2 (~63k cells) |
| + anti-PD-1 day 14 | `pdl1_high_rx.yaml` | elimination 3/3 by about day 20 |
| + anti-PD-1 day 28 | `pdl1_high_late.yaml` | deep regression (~33k → 190–360 cells), residual disease 3/3 |
| Cold + anti-PD-1 day 14 | `cold_rx.yaml` | escape 2/2; T cells about 170 → 350, macrophages ~550 M2 vs ~2 M1 |
| Hot (no treatment) | `hot.yaml` | elimination 1/3; seeds 1–2 controlled then escape (seed 2 by antigen loss 0.90 → 0.68) |
| Immunoediting | `edited.yaml` | equilibrium 1/3; seed 3 relapses from low-antigen clones (0.80 → 0.42) |

Limitations: no T-cell exclusion by stroma, no explicit antigen-loss variants (only drift), and treatment efficacy is a
fixed fraction. The outcome labels come from a simple rule on the tumor trajectory (`model/outcome.py`).

## 6. Outputs

Per run (zarr, one folder per run + frozen copy of config):
* snapshots: `state`, `lineage`, `oxygen` (float16), T-cell positions/exhaustion, kill events
* time series: live tumor, necrotic, T cells (active / exhausted), infiltration
  (T cells inside tumor vs margin), cumulative kills, mean antigenicity, mean PD-L1
* outcome label: **elimination** (no tumor cells), **equilibrium** (size bounded), **escape** (growth)

## 7. Visualisation

Rendering is decoupled from the simulation (reads snapshots only).
* Dark background, cells as soft glowing blobs; proliferating rim amber, quiescent dim
  orange, necrotic core deep red-black texture, apoptotic magenta flash.
* T cells bright cyan with fading motion trails; exhausted T cells grey-blue.
* Optional oxygen underlay; kill events flash white.
* Alternate colouring by lineage (+ Muller plot) for evolution.
* MP4 (H.264) per run; dashboard plays videos alongside synced time-series charts.

## 8. Code layout (as built)

```
tumor_abm/
  config/     default.yaml (all parameters) · hot.yaml · cold.yaml · edited.yaml (overrides via `base:`)
  model/      params.py (YAML -> flat param array) · fields.py (O2 / chemokine / suppressor / IFN-γ / CCL2 SOR solver)
              agents.py (Numba kernels: tumor, T cells, macrophages) · sim.py (driver, snapshots, metrics) · outcome.py
  viz/        render.py (parallel frame renderer -> MP4 + poster; --stills for PNG checks)
  dashboard/  index.html · build.py (copies runs -> data/, writes data.js)
  runs/       <scenario>_s<seed>/  config.yaml · snap.zarr · kills.npy · timeseries.json · video*.mp4
  run.py · batch.sh (parallel configs × seeds) · summarize.py · render_all.sh
```

## 8b. Calibration notes (v1 tuning loop)
* Per-contact kill strength `k_kill` sits near a sharp threshold: 0.03 means every hot seed escapes, 0.04 means
  hot seeds are eliminated by about day 20 when priming is at day 7. Later priming (larger tumor at first contact) moves
  the threshold up, so tumor size at first T-cell contact matters.
* The final hot scenario (chemokine 2.0, antigen 0.8) is stochastic: seeds 1 and 3 are eliminated, seed 2 escapes.
* `edited` (intermediate inflammation, priming at day 9) gives control, then equilibrium, then relapse. In seed 1 the relapse is
  low-antigen clones (mean antigenicity 0.76 → 0.56), which is immunoediting arising from the model rules, not scripted.
* Speed: about 2.3 s per simulated day on the 300×300 grid. Rendering takes about 2 min per 40-day video with 8 workers.

## 9. Reproducibility
Every run is defined by its YAML config + integer seed; Numba kernels use the
seeded NumPy-compatible RNG (`np.random.seed` inside jitted code).
