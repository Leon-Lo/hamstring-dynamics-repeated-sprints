# Hamstring Length During Repeated Sprints

Code accompanying publication on hamstring muscle-fiber mechanics during repeated sprint exercise using musculoskeletal simulations.

## Overview

This repository provides the full analysis pipeline used to study how hamstring fiber length and musculotendon mechanics change across a repeated sprint protocol. The pipeline processes OpenSim simulation outputs (IK, ID, static optimization, muscle analysis) and produces the statistical analyses and figures reported in the publication.

## Repository Structure

```text
├── src/                        # Core simulation pipeline
│   ├── model_manager.py        # OpenSim model loading (data/models/)
│   ├── simulation_runner.py    # IK, ID, static optimisation, muscle analysis
│   └── simulation_analyzer.py # High-level pipeline orchestration
│
├── utils/                      # Shared utilities
│   ├── analysis_utils.py       # Signal processing and result aggregation
│   ├── data_utils.py           # DataFrame manipulation
│   ├── file_utils.py           # Path resolution for the data/ structure
│   ├── plot_utils.py           # Plotting helpers
│   ├── subject_utils.py        # Subject list and grouping
│   └── xml_utils.py            # OpenSim XML setup file editing
│
├── analysis/                   # Analysis and figure scripts (run in order)
│   ├── core/
│   │   └── data_processing.py  # Step 1 — run simulations, build master_results.csv
│   ├── figures/
│   │   ├── publication_figures.py  # Step 2 — final publication-quality figures
│   │   └── combined_figures.py     # Combined multi-panel figures
│   ├── tables/
│   │   └── combined_tables.py      # Combined results tables
│   ├── validation/
│   │   ├── ceinms_analysis.py      # CEINMS EMG-driven results
│   │   ├── emg_validation.py       # EMG signal validation
│   │   ├── gamma_correlation.py    # Gamma correlation checks
│   │   └── opensim_validation.py   # Simulation validation checks
│   ├── demographics.py             # Participant demographics and trial counts
│   ├── spatiotemporal.py           # Spatio-temporal variables (speed, contact time, …)
│   ├── fatigue_effects.py          # Core publication results: RM ANOVA, post-hoc tests
│   ├── muscle_mechanics.py         # Fiber length trajectories and SPM waveforms
│   ├── eccentric_loading.py        # Eccentric impulse, work, and ANCOVA
│   ├── kinematics.py               # Joint angles and moments SPM waveforms
│   ├── risk_stratification.py      # Responder / non-responder classification
│   ├── clinical_groups.py          # CON / FAIM / FAIS group comparisons
│   └── exploratory.py              # Exploratory data analysis
│
├── ceinms/                     # CEINMS EMG-driven simulation scripts (Windows only)
│
├── data/
│   ├── generic_setup_files/    # Generic OpenSim XML templates (IK, ID, SO, MA)
│   ├── default_models/         # Generic Rajagopal2015 base model
│   ├── models/                 # Subject-specific scaled models (not in repo, see below)
│   ├── simulations/            # Raw simulation outputs (not in repo, see below)
│   └── master_results.csv      # Aggregated results used by analysis scripts
│
└── tests/                      # Unit tests
```

## Requirements

### Software

| Dependency | Version | Install |
| --- | --- | --- |
| Python | ≥ 3.10 | via conda |
| OpenSim | ≥ 4.5 | via conda (see below) |
| spm1d | 0.4.53 | via pip |

### Environment Setup

```bash
conda env create -n opensim -f environment.yml
conda activate opensim
```

This installs OpenSim, NumPy, pandas, matplotlib, seaborn, scikit-learn, statsmodels, scipy, spm1d, pingouin, and all other dependencies.

## Data Structure

The pipeline expects subject data under `data/` in the following layout:

```text
data/
├── models/
│   └── {subject_id}/
│       └── {subject_id}_*_originalMass_opt_N10_hans.osim
└── simulations/
    └── {subject_id}/
        └── pre/
            └── {trial_name}/
                ├── markers.trc
                ├── emg.mot
                ├── externalloads.xml
                └── (IK / ID / SO / MA outputs written here)
```

Subject IDs are zero-padded integers (e.g. `012`, `034`). The pre-computed aggregated results are provided in `data/master_results.csv` so the analysis scripts can be run without re-running all simulations.

## Reproducing the Results

### Option A — Use pre-computed results (recommended)

The aggregated results file `data/master_results.csv` is included. Run the analysis scripts directly:

```bash
conda activate opensim
python analysis/demographics.py
python analysis/spatiotemporal.py
python analysis/fatigue_effects.py
python analysis/muscle_mechanics.py
python analysis/eccentric_loading.py
python analysis/kinematics.py
python analysis/risk_stratification.py
python analysis/clinical_groups.py
python analysis/figures/publication_figures.py   # generates figures to data/plot_results/
python analysis/tables/combined_tables.py        # assembles combined results tables
```

### Option B — Re-run from raw simulation data

If you have the raw OpenSim data (not distributed in this repository) set up under `data/simulations/` and `data/models/`:

```bash
conda activate opensim
python analysis/core/data_processing.py      # runs IK → ID → SO → MA for all subjects
```

This writes outputs back into each trial directory and rebuilds `data/master_results.csv`.

## CEINMS

The `ceinms/` directory contains scripts for EMG-driven simulations using [CEINMS](https://simtk.org/projects/ceinms). The results in `data/master_results.csv` are CEINMS-based, so CEINMS is required to fully re-run the pipeline from raw data (Option B).

CEINMS executables are Windows-only and are not distributed in this repository — they must be obtained separately from the [CEINMS project page](https://simtk.org/projects/ceinms) and placed in `ceinms/executables/`. The pre-computed `data/master_results.csv` is included so the analysis scripts (Option A) can be run without CEINMS.

## Data Availability

The aggregated results (`data/master_results.csv`) required to reproduce all analyses and figures are included in this repository. Raw simulation data (scaled OpenSim models, motion capture, EMG, and ground reaction force files) are not distributed here. Raw data are available from the corresponding author on reasonable request.

## Tests

```bash
conda activate opensim
pytest
```

## Citation

If you use this code or data, please cite:

> [Authors]. [Title]. *[Journal]*, [Year]. DOI: [DOI]

A `CITATION.cff` file is also provided for citation managers and GitHub's "Cite this repository" feature.

## License

MIT — see [LICENSE](LICENSE).
