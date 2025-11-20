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
```

## Scripts overview

### s1-import-cgmes.py

**Input:**  
- A set of CGMES XML files (EQ, TP, DL, SSH, SV, GL) describing a power grid: equipment, topology, operating state, and geographic info.

**Goal:**  
- Parse the raw CGMES XML package and convert it into a single in-memory network object using `pandapower` (a Python library for power-grid modeling).  
- Turn the power grid into something that looks like a graph: buses (nodes), lines/transformers (edges), and components like loads and generators with attributes attached.  
- Persist this graph in a JSON file so later steps can load and reuse the exact same network without re-parsing CGMES.

**Output:**  
- `crete2030_net.json` containing the full pandapower network model (graph + attributes) in a JSON format.

---

### s2-inspect-net.py

**Input:**  
- `crete2030_net.json`.

**Goal:**  
- Load the stored network graph and print basic statistics: counts of nodes (buses) and elements (lines, transformers, loads, generators, shunts).  
- Show the first few rows of the main tables (buses, lines, loads) so a programmer can visually verify that the data looks reasonable (names, voltages, lengths, power values).  
- This is a quick sanity check that the CGMES import produced a usable network before doing any simulations or time-series work.

**Output:**  
- Console dump with object counts and small table slices for manual inspection.

---

### s3-runpp.py

**Input:**  
- `crete2030_net.json`.

**Goal:**  
- Run a single AC power-flow calculation using `pandapower` (think: solve for voltages and flows in the network given the current loads and generators).  
- Check whether the numerical solver converges, and inspect a subset of the results: bus voltages and line loading percentages.  
- This validates that the network is not only structurally correct but also numerically stable for a basic power-flow computation.

**Output:**  
- Console report showing solver convergence status, first few bus voltages, and line loading values.

---

### s4-plot-grid.py

**Input:**  
- `crete2030_net.json`.

**Goal:**  
- Load the network and create a simple 2D plot of its topology using `pandapower.plotting`.  
- Treat the grid as a graph: nodes and edges laid out using available coordinates or a simple layout routine.  
- Provide a quick visual check that the network is roughly what you expect (e.g. not empty, no obvious structural nonsense).

**Output:**  
- A Matplotlib figure window showing the network graph.

---

### s5-read-snapshots.py

**Input:**  
- Daily Excel files with multi-level headers, containing time series of active power (P in MW) and reactive power (Q in MVAr) for many pieces of equipment.  
- Two sheets per file: one for P, one for Q (e.g. "active power (MW)" and "reactive power (MVAR)").

**Goal:**  
- Load the raw Excel data where columns are labeled by a 3-level header (substation, category, equipment id) and timestamps appear in a special column.  
- Flatten the multi-level headers into simple string column names, clean out junk rows with missing timestamps, and make sure P and Q use the same timestamps and columns.  
- Concatenate multiple days into continuous P and Q dataframes and drop unneeded "Unnamed" columns, producing cleaned, aligned time series ready for further mapping.

**Output:**  
- `processed/P_all.parquet`: cleaned and aligned active power time series.  
- `processed/Q_all.parquet`: cleaned and aligned reactive power time series.

---

### s6-build-mapping.py

**Input:**  
- `crete2030_net.json`.  
- `processed/P_all.parquet`.  
- `processed/Q_all.parquet`.

**Goal:**  
- Take each time-series column name (e.g. `load_291231_1_MV`, `machine_61833_W3_MV`) and infer which network element it refers to (which bus, which load, which generator, which shunt).  
- Construct a mapping from Excel-style column names to `(pandapower_table, index)` pairs, by extracting numeric IDs from column names and matching them to bus IDs and then to elements connected to those buses.  
- Produce a machine-usable lookup that allows later steps to write each time-series value into the correct place in the network model.

**Output:**  
- `processed/mapping.json`: JSON dict from `excel_column_name` → `[table_name, element_index]`.  
- Console summary of how many columns were successfully mapped vs. left unmatched.

---

### s7-build-timeseries.py

**Input:**  
- `processed/P_all.parquet`, `processed/Q_all.parquet`.  
- `processed/mapping.json`.

**Goal:**  
- Filter the cleaned P/Q time series to only those columns that have a valid mapping to the network model.  
- Build dense NumPy arrays: timestamps `t`, active power matrix `P`, reactive power matrix `Q`, where each column corresponds to a mapped equipment item.  
- Generate a compact metadata table that describes each column: original name, target table (load/gen/sgen/shunt), and pandapower index.

**Output:**  
- `processed/timeseries_mapped.npz` containing arrays `t`, `P`, `Q`, and `columns`.  
- `processed/timeseries_meta.csv` listing metadata per column.

---

### s8-fit-statmodel.py

**Input:**  
- `processed/timeseries_mapped.npz`.

**Goal:**  
- Interpret the P/Q values as a high-dimensional state vector at each timestamp: concatenate P and Q into `X[t] = [P_t, Q_t]`.  
- Compute a typical daily pattern (mean behavior at each minute of the day), subtract it to get residuals, standardize them, and perform PCA to reduce the dimension to a smaller latent space.  
- Fit a VAR(1) model in that latent space (`Z_next = A * Z_prev + noise`), capturing how deviations from the daily pattern evolve over time, and store all parameters for later simulation.

**Output:**  
- `processed/stat_model.npz` containing:  
  - `mu`: typical daily profile for each minute and each dimension.  
  - `r_mean`, `r_std`: residual normalization parameters.  
  - `pca_components`, `pca_mean`: PCA model.  
  - `A`, `cov_eps`: VAR(1) transition matrix and noise covariance.

---

### s9-generate-synthetic_day.py

**Input:**  
- `processed/timeseries_mapped.npz`.  
- `processed/stat_model.npz`.

**Goal:**  
- Use the learned VAR(1) model in the PCA latent space to simulate an entirely new synthetic day of deviations (`Z_syn`), transform back through PCA to residuals, and re-add the typical daily profile `mu`.  
- Split the reconstructed state vectors into synthetic P and Q for each mapped equipment column, enforcing simple sign constraints (e.g. loads and generators must have non-negative active power).  
- Build a new timestamp range placed directly after the training period, so the synthetic day is temporally consistent with the original data.

**Output:**  
- `processed/synthetic_day_001.npz` containing synthetic `t`, `P`, `Q`, and `columns` in the same format as the training data.

---

### s10-check-feasibility.py

**Input:**  
- `crete2030_net.json`.  
- `processed/timeseries_meta.csv`.  
- `processed/synthetic_day_001.npz`.

**Goal:**  
- For each minute of the synthetic day, write the corresponding P/Q values into the pandapower network using the metadata mapping.  
- Run an AC power-flow simulation and check if it converges and respects basic engineering limits: bus voltages inside [v_min, v_max], and line/transformer loadings below a given percentage.  
- Collect time-series of feasibility metrics (convergence flags, min/max voltages, max line/trafo loading, and counts of violations) for downstream analysis.

**Output:**  
- Console summary of overall convergence and constraint ranges.  
- `processed/synthetic_day_001_pf_results.npz` storing all per-timestep feasibility metrics and limits.

---

### s11-summarize_pf_results.py

**Input:**  
- `processed/synthetic_day_001_pf_results.npz`.

**Goal:**  
- Load the feasibility metrics from the previous step and compute aggregate counts: how many time steps converge, how many satisfy voltage limits, line limits, transformer limits, and all constraints together.  
- Report extremal values (best/worst voltages, max line/trafo loading) over the subset of fully feasible time steps.  
- Provide a compact textual summary so a human can quickly judge whether the synthetic day is “good enough” from a grid-feasibility perspective.

**Output:**  
- Console summary of counts and extremal metrics across all time steps.

---

### s12-scale_correct_day.py

**Input:**  
- `crete2030_net.json`.  
- `processed/timeseries_meta.csv`.  
- `processed/synthetic_day_001.npz`.  
- `processed/synthetic_day_001_pf_results.npz` (for the chosen limits).

**Goal:**  
- For each synthetic time step, check whether the original P/Q snapshot is feasible; if not, find the largest global scaling factor α ∈ (0, 1] such that scaling all injections by α makes the power flow feasible (voltages and loadings within limits).  
- Implement this as a small binary search in α: try midpoints, run power flow, and keep the best feasible α; if none works, fall back to zero injections for that minute.  
- Produce a “corrected” synthetic day that stays as close as possible to the original synthetic profile while respecting grid constraints, and store both the corrected series and the α factors.

**Output:**  
- Console report of how many time steps admit a feasible α.  
- `processed/synthetic_day_001_corrected.npz` containing corrected `t`, `P`, `Q`, `columns`, per-step `alpha`, feasibility flags, and the constraint limits.
