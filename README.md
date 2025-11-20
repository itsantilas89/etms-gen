# ETMS-GEN – Crete 2030 Synthetic Operating Points

This repo contains **code only** for working with a CGMES model of the Crete 2030 network and minute-level power profiles.  
Raw data (CGMES XML, Excel snapshots, processed arrays) are **not** included in the repo.

## Directory layout

- `scripts/`
  - All Python scripts (pipeline stages).
- `Crete 2030/` (ignored)
  - CGMES XML files (`*EQ*.xml`, `*TP*.xml`, `*SSH*.xml`, `*SV*.xml`, `*DL*.xml`, `*GL*.xml`).
- `snapshots/` (ignored)
  - Excel files with one-minute P/Q profiles (two days).
- `processed/` (ignored)
  - Intermediate artefacts (JSON net, parquet P/Q, mapping, NPZ time series).

## Environment

Create and activate a venv, then install:

```bash
pip install -r requirements.txt
# or explicitly:
# pandapower[converter] numba cimpy rdflib lxml igraph pandas numpy scipy matplotlib
```

## Scripts overview

### `s1-import-cgmes.py`

* Input: `Crete 2030/*.xml` (CGMES EQ, TP, DL, SSH, SV, GL).
* Uses `pandapower.converter.cim.from_cim` to:

  * Parse CGMES v2.4.15 model.
  * Build a `pandapowerNet` with buses, lines, trafos, loads, gens, shunts, etc.
* Output: `crete2030_net.json` in repo root (ignored by git).

### `s2-inspect-net.py`

* Input: `crete2030_net.json`.
* Loads the net with `pp.from_json`.
* Prints basic stats:

  * Number of buses / lines / trafos / loads / gens / sgen / shunts.
  * First rows of `net.bus`, `net.line`, `net.load`.
* Purpose: sanity-check CGMES import.

### `s3-runopp.py`

* Input: `crete2030_net.json`.
* Loads `net`, runs an AC power flow (`pp.runpp`) or OPF if configured.
* Prints:

  * Convergence flag.
  * First few bus voltages (vm_pu, va_degree).
  * First few line loadings.
* Purpose: confirm that the imported network is electrically consistent.

### `s4-plot-grid.py`

* Input: `crete2030_net.json`.
* Uses `pandapower.plotting.simple_plot` to generate a quick layout:

  * If geodata available, uses it.
  * Otherwise, creates generic coordinates (requires `python-igraph`).
* Output: a plotted network (displayed via `matplotlib` window; optionally saved as PNG).
* Purpose: quick visual sanity check of network topology.

### `s5-read-snapshots.py`

* Input: `snapshots/PowerProfilesData-Jan10.xlsx`, `snapshots/PowerProfilesData-Jun10.xlsx`.

  * Each file has two sheets:

    * `active power (MW)`
    * `reactive power (MVAR)`
  * Each sheet has 3 header rows, then 1440 rows (one per minute).
* Logic:

  * Reads both days’ P and Q sheets with 3 header rows (MultiIndex columns).
  * Flattens headers to single-level column names:

    * `TIMESTAMP`
    * equipment IDs like `load_291231_1_MV`, `machine_291231_W3_MV`, `fixed shunt_291231_1_MV`, etc.
  * Drops rows without TIMESTAMP.
  * Concatenates the two days for P and Q separately.
  * Removes `Unnamed:*` columns.
  * Aligns P and Q on common timestamps.
* Output:

  * `processed/P_all.parquet` – aligned active power time series.
  * `processed/Q_all.parquet` – aligned reactive power time series.
* Purpose: turn messy Excel into clean, aligned P/Q time series.

### `s6-build-mapping.py`

* Input:

  * `crete2030_net.json`
  * `processed/P_all.parquet` / `processed/Q_all.parquet`
* Logic:

  * Builds a map from **Excel column name** → **(pandapower table, row index)**.
  * Extracts a bus ID from column names, e.g.:

    * `load_291231_1_MV` → bus id `291231`
    * `machine_291231_W3_MV` → bus id `291231`
    * `switched_shunt_61833_150kV` → bus id `61833`
  * Maps bus id to `net.bus.index` based on bus name prefix.
  * For each column:

    * `load_...`    → match `net.load` with that `bus`.
    * `machine_...` → prefer `net.sgen`, fallback to `net.gen`.
    * `fixed shunt_...` / `switched_shunt_...` → `net.shunt`.
* Output:

  * `processed/mapping.json`:

    * `"column_name": ["table", pp_index]`, e.g. `["load_291531_1_MV", ["load", 0]]`.
* Purpose: link time series columns to concrete pandapower elements.

### `s7-build-timeseries.py`

* Input:

  * `processed/P_all.parquet`
  * `processed/Q_all.parquet`
  * `processed/mapping.json`
* Logic:

  * Selects only the **mapped** columns (those with a valid pandapower element).
  * Extracts:

    * `t`: timestamps (`TIMESTAMP` column).
    * `P`: matrix of active power, shape `(T, N_mapped)`.
    * `Q`: matrix of reactive power, shape `(T, N_mapped)`.
  * Builds a metadata table with:

    * column name
    * pandapower table (`load` / `sgen` / `gen` / `shunt`)
    * pandapower element index.
* Output:

  * `processed/timeseries_mapped.npz` (compressed):

    * `t`, `P`, `Q`, `columns`.
  * `processed/timeseries_meta.csv`.
* Purpose: final clean time-series representation ready for:

  * model training,
  * synthetic operating point generation,
  * replay into `pandapower` for power-flow feasibility checks.

## Data privacy / confidentiality

* **Not** committed to the repo:

  * CGMES XML files (grid topology and parameters).
  * Excel snapshots (minute-level P/Q).
  * All derived time-series and mapping artefacts.
* **Commited** only:

  * Scripts (`scripts/*.py`).
  * `requirements.txt`, `README.md`, and non-sensitive documentation.





---

# Roadmap

## 1. Import the Crete CGMES network into Python

You already have a full CGMES export (EQ, TP, DL, SSH, SV, GL). You do not need PowerFactory to use it.

Use `pandapower` with its CGMES converter:

* `pip install "pandapower[converter]" numba`
* Import and convert:

```python
from pandapower.converter.cim import from_cim as cim2pp

cgmes_files = [
    r"C:\Users\itsantilas\Documents\Coding\PythonProjects\cretevalley\Crete 2030\20250429T1349Z__EQ_.xml",
    r"C:\Users\itsantilas\Documents\Coding\PythonProjects\cretevalley\Crete 2030\20250429T1349Z___TP_.xml",
    r"C:\Users\itsantilas\Documents\Coding\PythonProjects\cretevalley\Crete 2030\20250429T1349Z___DL_.xml",
    r"C:\Users\itsantilas\Documents\Coding\PythonProjects\cretevalley\Crete 2030\20250429T1349Z___SSH_.xml",
    r"C:\Users\itsantilas\Documents\Coding\PythonProjects\cretevalley\Crete 2030\20250429T1349Z___SV_.xml",
    r"C:\Users\itsantilas\Documents\Coding\PythonProjects\cretevalley\Crete 2030\20250429T1349Z___GL_.xml",
]

net = cim2pp.from_cim(file_list=cgmes_files, cgmes_version="2.4.15")
```

The `from_cim` converter is designed exactly for CGMES v2.4.15/3.0 and supports EQ, TP, SSH, SV, DL, GL profiles, converting them into a `pandapowerNet` with buses, lines, transformers, generators, loads, shunts, etc.([pandapower.readthedocs.io][1])

Key points:

* The resulting `net` has tables like `net.bus`, `net.line`, `net.trafo`, `net.gen`, `net.sgen`, `net.load`, etc.
* Every component carries an `origin_id` (the CGMES UUID) and usually the CIM `name` field as `net.<table>.name`.([pandapower.readthedocs.io][1])
* Topology is built from TopologicalNodes or ConnectivityNodes automatically.

This handles step (1).

## 2. Link the Excel snapshots to buses/elements in the model

Your Excel columns look like:

* Human-readable substation name: `AG. VARVARA`
* Equipment class: `LOAD`, `RES PRODUCTION`, `COMPENSATOR`
* Equipment identifier: `load_291231_1_MV`, `machine_291231_W3_MV`, `fixed shunt_291231_1_MV`

These identifiers will map to CIM objects in the EQ/SSH profiles, which in turn are mapped to pandapower elements:

* `load_...` → `cim:ConformLoad`/`EnergyConsumer` → `net.load`
* `machine_...` (RES production) → `SynchronousMachine`/`GeneratingUnit` or RES → `net.gen` or `net.sgen`
* `fixed shunt_...` → `ShuntCompensator` → `net.shunt`

PowerFactory typically exports its internal “element names” as `cim:IdentifiedObject.name` in EQ. Pandapower preserves those names in the `name` column.([pandapower.readthedocs.io][1])

Procedure:

1. After CGMES import, inspect your pandapower net:

   ```python
   net.load[["name", "bus", "p_mw", "q_mvar"]].head()
   net.sgen[["name", "bus", "p_mw", "q_mvar"]].head()
   net.shunt[["name", "bus", "q_mvar"]].head()
   ```

2. Build a mapping from Excel column IDs to pandapower index:

   * Parse the header row that contains `load_291231_1_MV` etc.
   * Look up `net.load[net.load.name == "load_291231_1_MV"].index[0]`.
   * Do the same for RES production (`net.sgen` or `net.gen`) and compensators (`net.shunt`).

3. Store these mappings in a configuration dict, e.g.:

   ```python
   mapping = {
       "load_291231_1_MV": ("load", load_idx),
       "machine_291231_W3_MV": ("sgen", sgen_idx),
       "fixed shunt_291231_1_MV": ("shunt", shunt_idx),
       ...
   }
   ```

4. For each timestamp row in the Excel MW sheet, create arrays:

   * `p_load[m, i]` for each load
   * `p_gen[m, j]` for each RES generator
   * From MVAr sheet, `q_load`, `q_gen`, `q_shunt`

What this gives you:

* A consistent multi-dimensional time series of active/reactive power for specific elements in `net`, over 1440×2 minutes.
* Exact alignment between CGMES model and your snapshots.

## 3. Learn temporal and cross-bus correlations from 2 days

With only 2 days of minute-resolution data (2×1440=2880 samples), deep neural models with many parameters are overkill and will overfit. You need a low-parameter generative structure.

Define your state vector for each minute `t`:

* Let `x_t ∈ ℝ^(2N)` be all P and Q values for N elements (loads, gens, shunts) at time t, after stacking.

Basic preprocessing:

1. Separate deterministic daily profile and residuals:

   * For each element `k`, compute average profile over the two days:

     ```text
     μ_k(τ) = average of x_{t,k} for t with same minute-of-day τ
     ```

     where τ = 0..1439.
   * Define residuals `r_t = x_t − μ(τ(t))`.

2. Standardize residuals per dimension:

   ```text
   r̂_t,k = (r_t,k − mean_k) / std_k
   ```

Now you have:

* A baseline “typical day” profile μ(τ) per element.
* Zero-mean standardized residuals capturing variability plus cross-bus correlation.

### 3a. Minimal but usable generative model (statistical, not neural)

Given the data constraints, a compact stochastic model is sane:

1. Temporal model on residuals:

   * Fit an AR(1) or low-order VAR (vector autoregression) on the first few principal components of `r̂_t`.
   * Steps:

     * Run PCA on residuals across all t:

       ```text
       r̂_t ≈ W z_t, where z_t ∈ ℝ^K, K << 2N
       ```
     * Keep K small (e.g. 5–10) so 2880 samples are enough to estimate VAR parameters.
     * Fit VAR(1):

       ```text
       z_t = A z_{t−1} + ε_t,   ε_t ~ N(0, Σ_ε)
       ```

2. Generation:

   * For a synthetic day:

     * For τ=0..1439:

       * Step the VAR for `z_t`.
       * Reconstruct residuals `r̂_t = W z_t`.
       * De-standardize: `r_t = r̂_t * std + mean`.
       * Add daily profile: `x_t = μ(τ) + r_t`.

3. Cross-bus correlations come via the PCA basis W, which is learned from the real data.

This gives you a simple, explainable generator that respects:

* Time-of-day structure (via μ(τ)).
* Temporal correlation (via VAR in z-space).
* Cross-bus correlation (via PCA loadings).

### 3b. Neural generator option

If you insist on a neural approach with so little data, you must heavily constrain capacity:

* First reduce dimensionality with PCA or an autoencoder to a small latent `z_t`.
* Then use a small temporal generator in latent space, e.g.:

  * A small GRU-based sequence model
  * Or a 1D temporal convolutional VAE

Example structure:

* Encoder: linear or shallow MLP that maps `x_t` to `z_t` (dimension 5–10).
* Temporal model: GRU that takes `z_t` as sequence and learns dynamics.
* Generator: GRU + decoder to produce new `x_t`.

You then train:

* Reconstruction loss between generated and real `x_t` (per time step).
* Maybe adversarial loss on marginal statistics if you go for GAN-like behavior.

But expect heavy regularization and small networks; otherwise you just memorize the two days.

## 4. Hybrid workflow: use pandapower AC load flow as feasibility filter/corrector

You now have a CGMES-derived pandapower network `net` and a generator that outputs candidate P/Q time series for each element.

You implement a loop over generated snapshots:

1. Apply P/Q injections:

   For a given time `t`:

   ```python
   # For each mapped element:
   for elem_id, (kind, idx) in mapping.items():
       P, Q = x_t_for_that_column  # synthetic values, in MW/MVAr

       if kind == "load":
           net.load.at[idx, "p_mw"] = max(P, 0.0)
           net.load.at[idx, "q_mvar"] = Q
       elif kind == "sgen":
           net.sgen.at[idx, "p_mw"] = P      # sign convention depends on your data
           net.sgen.at[idx, "q_mvar"] = Q
       elif kind == "shunt":
           net.shunt.at[idx, "q_mvar"] = Q   # shunts typically have only Q
   ```

   Careful with sign conventions: in CIM/pandapower, loads are usually positive P for consumption, generators positive for injection. Your Excel probably uses positive for both load and generation, so you invert generator sign when assigning.

2. Run AC power flow with pandapower:

   ```python
   import pandapower as pp

   pp.runpp(net, algorithm="nr", init="results")  # Newton-Raphson, warm-start
   feasible = net.converged
   ```

   `runpp` is the standard AC load flow in pandapower.([pandapower.readthedocs.io][2])

3. If converged → mark sample as feasible and record:

   * All bus voltages: `net.res_bus.vm_pu`, `va_degree`
   * Line loadings: `net.res_line.loading_percent`
   * Trafo loadings: `net.res_trafo.loading_percent`

4. If not converged → either discard or correct.

Two basic correction strategies without needing full OPF economics:

### 4a. Simple scaling correction

* If `runpp` fails for a given snapshot:

  * Try scaling all loads and generators towards the slack in a scalar fashion:

    ```python
    scale_lo, scale_hi = 0.2, 1.0
    for _ in range(max_iter):
        scale = 0.5 * (scale_lo + scale_hi)
        apply_scale(scale)   # multiply all |P| and |Q| by 'scale' around some base
        pp.runpp(net, algorithm="nr", init="results")
        if net.converged:
            scale_lo = scale
        else:
            scale_hi = scale
    ```

  * This binary search gives you the maximum loading factor for which the grid remains solvable and close to the candidate.

* Store the rescaled snapshot as “corrected” version of the original sample.

This is crude but robust, and it keeps cross-bus ratios while reducing global stress.

### 4b. Optimization-based correction (AC-OPF)

Pandapower also has AC-OPF (`pp.runopp`) that can enforce voltage and branch constraints.([pandapower.readthedocs.io][2])

Given you lack cost curves, you can:

* Treat loads as fixed.
* Treat generators as controllable with quadratic penalty on deviation from the generated P/Q setpoints.

In principle:

1. Set generator P bounds around the generated values, e.g. `P_gen ± ΔP`.
2. Set standard voltage limits, e.g. 0.9–1.1 p.u.
3. Set thermal limits from CGMES current limits or pandapower default ratings. CGMES OperationalLimitSet and CurrentLimit/VoltageLimit map into pandapower limits during conversion.([pandapower.readthedocs.io][1])
4. Run OPF to find the closest feasible operating point.

This is more work to configure but produces physically consistent corrections rather than scaling everything.

## 5. Evaluate synthetic data: statistics

Given real time series {x_t} and synthetic {x̃_t}, implement:

1. Marginal distributions:

   * For each element k, compare histograms or kernel density of real vs synthetic P and Q.
   * Use simple metrics:

     * Mean, std, skewness.
     * Wasserstein distance or KL divergence (approximate) for distributions.

2. Temporal structure:

   * Autocorrelation function ACF for each element k on the real and synthetic series.
   * Lag-1, lag-5, lag-60 (1 hour) correlations to check persistence and daily rhythm.

3. Ramps:

   * Compute Δx_t = x_t − x_{t−1} per element.
   * Compare ramp histograms (e.g. 5-minute, 15-minute aggregated ramps).
   * Focus on heavy tails: fraction of ramps exceeding thresholds.

4. Cross-bus correlations:

   * Compute correlation matrix Corr(x_t) across elements at each time or over all times.
   * Compare leading eigenvalues/eigenvectors of real vs synthetic correlation matrices.

All of this is standard numpy/pandas work once you have real and synthetic arrays.

## 6. Evaluate physical feasibility and violations

Using pandapower results for each AC-PF run:

1. Feasibility rate:

   * `feasible_rate = (# PF converged) / (# synthetic snapshots)`
   * This is a direct measure of how well your generator respects physical constraints implicitly.

2. Voltage violations:

   * Set bus voltage limits (unless already in the model), typically 0.9–1.1 p.u.
   * Count violations per snapshot:

     ```python
     vm = net.res_bus.vm_pu.values
     v_viol = ((vm < 0.9) | (vm > 1.1)).sum()
     ```
   * Aggregate statistics: average violations per snapshot, worst-case, etc.

3. Line and transformer overloads:

   * Use `loading_percent` from `net.res_line` and `net.res_trafo`.
   * Count elements with loading > 100% (or your chosen safety margin).

4. Spatial distribution:

   * For each bus/line, compute frequency of being in violation.
   * This will show you “weak spots” in the grid where your generator tends to push too hard.

These metrics cover steps (6) and (7) in your list.

## 7. Output validated synthetic dataset

Final step:

1. Keep only feasible or corrected snapshots:

   * For each synthetic scenario (e.g. a full 1440-minute day), keep:

     * The original generated P/Q.
     * The corrected P/Q after PF or OPF (if correction is used).
     * The network state (bus voltages, line loading, etc.).

2. Store in a structured format:

   * HDF5 or Parquet:

     * Dimensions: time × element × {P, Q, vm_pu, va_deg, line_loading, ...}.
   * Keep the mapping (element ID ↔ CIM origin_id ↔ pandapower index ↔ Excel column name) alongside.

This gives you a synthetic dataset that:

* Matches short-term statistical behavior of the two reference days.
* Comes with explicit feasibility labels and detailed physical states.
* Is directly usable as input for forecasting models or optimization experiments (e.g. OPF, security analysis).

## 8. End-to-end vs hybrid: feasibility verdict

Given:

* Only 2 days of training data.
* A detailed physical model available via CGMES → pandapower.

Conclusion:

* Pure end-to-end neural generator that “magically” learns feasibility is theoretically possible but practically weak with this data volume; it will either overfit or generate many infeasible points.
* Hybrid approach (statistical or small-neural generator + pandapower AC PF/OPF as a feasibility filter/corrector) is fully feasible and aligns with how CGMES and pandapower are designed to be used.([pandapower.readthedocs.io][1])

Your task is therefore possible end-to-end using only free tools:

* CGMES → pandapower via `from_cim`.
* Excel mapping → P/Q time series.
* Low-dimensional generative model (PCA + VAR or small neural net).
* Pandapower AC PF/OPF loop for feasibility and correction.
* Quantitative evaluation on both statistics and physical constraints.


