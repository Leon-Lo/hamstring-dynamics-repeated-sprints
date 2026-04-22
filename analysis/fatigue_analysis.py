"""
Fatigue Analysis - Main Thesis Results.

========================================
Core analyses for the thesis: fatigue effects on muscle activation,
forces, work, and coordination patterns.
"""

# %% Setup - Add project root to path
# ruff: noqa: E402
import os
import sys
from pathlib import Path

# Add project root to Python path BEFORE other imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)
# Now import everything else

from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")  # Use non-interactive backend for performance in multi-threading
import seaborn as sns
from tqdm import tqdm

from utils import (
    Analysis_utils,
    Data_utils,
    File_utils,
    Plot_utils,
    Subject_utils,
)

# Configure matplotlib for threading
mpl.rcParams["figure.max_open_warning"] = 0  # Suppress warning since we handle it

plots = Plot_utils()
file_utils = File_utils()
subject_utils = Subject_utils()
data_utils = Data_utils()
analysis_utils = Analysis_utils()

# %% Configuration
SAVE_FIGURES: bool = True  # Set to True to save figures
OUTPUT_DIR = Path("data/plot_results/fatigue_analysis")

# Define subdirectories for organized plot storage
OVERVIEW_DIR = OUTPUT_DIR / "overview"
BIOMECHANICS_DIR = OUTPUT_DIR / "biomechanics"
MUSCLE_ACTIVATION_DIR = OUTPUT_DIR / "muscle_activation"
MUSCLE_GROUPS_DIR = OUTPUT_DIR / "muscle_groups"
MUSCLE_PARAMS_DIR = OUTPUT_DIR / "muscle_parameters"
FORCES_STRAIN_DIR = OUTPUT_DIR / "forces_and_strain"
VELOCITY_STRAIN_DIR = OUTPUT_DIR / "velocity_and_strain"
WORK_POWER_DIR = OUTPUT_DIR / "work_and_power"
REGRESSION_DIR = OUTPUT_DIR / "regression"
REGRESSION_STRAIN_DIR = REGRESSION_DIR / "strain"
REGRESSION_DELTA_DIR = REGRESSION_DIR / "delta_strain"
PER_SUBJECT_DIR = OUTPUT_DIR / "per_subject"

if SAVE_FIGURES:
    for dir_path in [
        OUTPUT_DIR,
        OVERVIEW_DIR,
        BIOMECHANICS_DIR,
        MUSCLE_ACTIVATION_DIR,
        MUSCLE_GROUPS_DIR,
        MUSCLE_PARAMS_DIR,
        FORCES_STRAIN_DIR,
        WORK_POWER_DIR,
        REGRESSION_STRAIN_DIR,
        REGRESSION_DELTA_DIR,
        PER_SUBJECT_DIR,
        PER_SUBJECT_DIR / "master_plots",
        PER_SUBJECT_DIR / "moments_torques",
        PER_SUBJECT_DIR / "emg",
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

EXAMPLE_SUBJECT = "024"  # Subject ID for single subject examples

# %% ================================================
# %%    ALL SUBJECTS ANALYSES
# %% ================================================

# %% ==================== MASTER PLOTS (OVERVIEW) ====================

# %% Master plot - ALL subjects before/after fatigue
try:
    result = plots.plot_master_plot_before_after_all()
    if result is not None:
        fig, axes = result
        if fig is not None and SAVE_FIGURES:
            plots.save_figure(fig, OVERVIEW_DIR, "master_plot_before_after_all.png")
        plt.show()
except KeyError as e:
    print(f"⚠️  Column naming mismatch: {e}")
    print("   Regenerate master_results.csv in data_processing.py")

# %% Master plot - ALL subjects before/after fatigue - individual lines (interactive)
import plotly.io as pio

fig_individual = plots.plot_master_plot_before_after_all_individual_interactive()
if fig_individual is not None and SAVE_FIGURES:
    pio.write_html(fig_individual, OVERVIEW_DIR / "master_plot_before_after_all_individual_interactive.html")
plt.show()

# %% Master plot with group splits
result = plots.plot_master_plot_with_after_all_group_split()
if result is not None:
    fig, axes = result
    if fig is not None and SAVE_FIGURES:
        plots.save_figure(fig, OVERVIEW_DIR, "master_plot_group_split.png")
    plt.show()

# %% Master plot with individual group plots
result = plots.plot_master_plot_with_after_all_group_split_individual()
if result is not None:
    fig, axes = result
    if fig is not None and SAVE_FIGURES:
        plots.save_figure(fig, OVERVIEW_DIR, "master_plot_individual_groups.png")
    plt.show()


# %% ==================== MOMENTS AND TORQUES ====================

# %% ID moments vs CEINMS torques - all subjects
result = plots.plot_moments_torques()
if result is not None:
    fig, axes = result
    if fig is not None and SAVE_FIGURES:
        plots.save_figure(fig, BIOMECHANICS_DIR, "moments_torques_all.png")
    plt.show()

# %% ==================== EMG ANALYSIS ====================

# %% EMG and adjusted EMG - all subjects
result = plots.plot_emg_adjusted_emg()
if result is not None:
    fig, axes = result
    if fig is not None and SAVE_FIGURES:
        plots.save_figure(fig, MUSCLE_ACTIVATION_DIR, "emg_adjusted_all.png")
    plt.show()

# %% ==================== MUSCLE GROUP ANALYSES ====================

# %% Muscle group activations - all groups
result = plots.plot_muscle_group_activations()
if result is not None:
    fig, axes = result
    if fig is not None and SAVE_FIGURES:
        plots.save_figure(fig, MUSCLE_GROUPS_DIR, "muscle_group_activations_all.png")
    plt.show()

# %% Specific muscle groups - hip flexors and extensors
result = plots.plot_muscle_group_activations(
    muscle_groups=["hip_flexors", "hip_extensors"],
)
if result is not None:
    fig, axes = result
    if fig is not None and SAVE_FIGURES:
        plots.save_figure(fig, MUSCLE_GROUPS_DIR, "hip_muscle_activations.png")
    plt.show()

# %% Specific muscle groups - knee
result = plots.plot_muscle_group_activations(
    muscle_groups=["knee_flexors", "knee_extensors"],
)
if result is not None:
    fig, axes = result
    if fig is not None and SAVE_FIGURES:
        plots.save_figure(fig, MUSCLE_GROUPS_DIR, "knee_muscle_activations.png")
    plt.show()

# %% Specific muscle groups - ankle
# NOTE: Only ankle_plantarflexors is defined (no dorsiflexors group)
result = plots.plot_muscle_group_activations(muscle_groups=["ankle_plantarflexors"])
if result is not None:
    fig, axes = result
    if fig is not None and SAVE_FIGURES:
        plots.save_figure(fig, MUSCLE_GROUPS_DIR, "ankle_muscle_activations.png")
    plt.show()

# %% Muscle group forces - all groups
# NOTE: plot_muscle_group_forces only accepts anatomical muscle groups
# Available groups: Gluteus Maximus, Gluteus Medius, Gluteus Minimus,
# Adductor Magnus, Biceps Femoris, Semimembranosus, Semitendinosus,
# Rectus Femoris, Vasti, Triceps Surae
result = plots.plot_muscle_group_forces()
if result is not None:
    fig, axes = result
    if fig is not None and SAVE_FIGURES:
        plots.save_figure(fig, MUSCLE_GROUPS_DIR, "muscle_group_forces_all.png")
    plt.show()

# %% ==================== MUSCLE PARAMETERS ====================

# %% Muscle parameters comparison
result = plots.plot_muscle_parameters(interactive=True)
if result is not None:
    if isinstance(result, tuple):
        fig, axes = result
        if fig is not None and SAVE_FIGURES:
            plots.save_figure(fig, MUSCLE_PARAMS_DIR, "muscle_parameters.png")
    # Plotly figure - save as HTML
    elif SAVE_FIGURES:
        import plotly.io as pio

        pio.write_html(
            result,
            MUSCLE_PARAMS_DIR / "muscle_parameters_interactive.html",
        )
    plt.show()

# %% Normalized fiber length pre/post fatigue
result = plots.plot_norm_fiber_length_pre_post()
if result is not None:
    fig, axes = result
    if fig is not None and SAVE_FIGURES:
        plots.save_figure(fig, MUSCLE_PARAMS_DIR, "norm_fiber_length_pre_post.png")
    plt.show()

# %% Maximum model force comparison
result = plots.plot_max_model_force_comp(interactive=True)
if result is not None:
    if isinstance(result, tuple):
        fig, axes = result
        if fig is not None and SAVE_FIGURES:
            plots.save_figure(fig, MUSCLE_PARAMS_DIR, "max_model_force_comparison.png")
    # Plotly figure - save as HTML
    elif SAVE_FIGURES:
        import plotly.io as pio

        pio.write_html(
            result,
            MUSCLE_PARAMS_DIR / "max_model_force_comparison_interactive.html",
        )
    plt.show()

# %% ==================== FORCE AND STRAIN ====================

# %% Peak normalized force
result = plots.plot_peak_norm_force(interactive=True)
if result is not None:
    if isinstance(result, tuple):
        fig, axes = result
        if fig is not None and SAVE_FIGURES:
            plots.save_figure(fig, FORCES_STRAIN_DIR, "peak_norm_force.png")
    # Plotly figure - save as HTML
    elif SAVE_FIGURES:
        import plotly.io as pio

        pio.write_html(
            result,
            FORCES_STRAIN_DIR / "peak_norm_force_interactive.html",
        )
    plt.show()

# %% Peak normalized fiber length
result = plots.plot_peak_norm_fiber_length(interactive=True)
if result is not None:
    if isinstance(result, tuple):
        fig, axes = result
        if fig is not None and SAVE_FIGURES:
            plots.save_figure(fig, FORCES_STRAIN_DIR, "peak_norm_fiber_length.png")
    # Plotly figure - save as HTML
    elif SAVE_FIGURES:
        import plotly.io as pio

        pio.write_html(result, FORCES_STRAIN_DIR / "peak_norm_fiber_length_interactive.html")
    plt.show()

    # %% Peak force change vs peak normalized fiber length change
result = plots.plot_peak_norm_force_change_vs_peak_norm_fiber_length_change(interactive=True)
if result is not None:
    if isinstance(result, tuple):
        fig, axes = result
        if fig is not None and SAVE_FIGURES:
            plots.save_figure(fig, FORCES_STRAIN_DIR, "peak_force_vs_norm_fiber_length.png")
    # Plotly figure - save as HTML
    elif SAVE_FIGURES:
        import plotly.io as pio

        pio.write_html(
            result,
            FORCES_STRAIN_DIR / "peak_force_vs_norm_fiber_length_interactive.html",
        )
    plt.show()

# %% Velocity vs strain scatter analysis
try:
    fig = plots.velocity_strain_scatter_plot()
    if fig is not None:
        if SAVE_FIGURES:
            plots.save_figure(fig, FORCES_STRAIN_DIR, "velocity_strain_scatter.png")
        plt.show()
except KeyError as e:
    print(f"⚠️  Column naming mismatch: {e}")
    print("   Regenerate master_results.csv in data_processing.py")

# %% ==================== VELOCITY AND STRAIN ====================
# %% Velocity vs strain scatter analysis with responders/non-responders highlighted
# Pre
try:
    if SAVE_FIGURES:
        import plotly.io as pio

    fig = plots.plot_peak_velocity_strain("pre", "bflh")
    if fig is not None and SAVE_FIGURES:
        pio.write_html(
            fig,
            FORCES_STRAIN_DIR / "velocity_strain_scatter_bflh_pre.html",
        )
    fig = plots.plot_peak_velocity_strain("pre", "semiten")
    if fig is not None and SAVE_FIGURES:
        pio.write_html(
            fig,
            FORCES_STRAIN_DIR / "velocity_strain_scatter_semiten_pre.html",
        )
    fig = plots.plot_peak_velocity_strain("pre", "semimem")
    if fig is not None and SAVE_FIGURES:
        pio.write_html(
            fig,
            FORCES_STRAIN_DIR / "velocity_strain_scatter_semimem_pre.html",
        )

    # Change
    fig = plots.plot_peak_velocity_strain("change", "bflh")
    if fig is not None and SAVE_FIGURES:
        pio.write_html(
            fig,
            FORCES_STRAIN_DIR / "velocity_strain_scatter_bflh_change.html",
        )
    fig = plots.plot_peak_velocity_strain("change", "semiten")
    if fig is not None and SAVE_FIGURES:
        pio.write_html(
            fig,
            FORCES_STRAIN_DIR / "velocity_strain_scatter_semiten_change.html",
        )
    fig = plots.plot_peak_velocity_strain("change", "semimem")
    if fig is not None and SAVE_FIGURES:
        pio.write_html(
            fig,
            FORCES_STRAIN_DIR / "velocity_strain_scatter_semimem_change.html",
        )


except KeyError as e:
    print(f"⚠️  Column naming mismatch: {e}")
    print("   Regenerate master_results.csv in data_processing.py")


# %% ==================== WORK AND POWER ====================

# %% Muscle work analysis
result = plots.plot_muscle_work(return_plotly=True)
if result is not None:
    if isinstance(result, tuple):
        fig, axes = result
        if fig is not None and SAVE_FIGURES:
            plots.save_figure(fig, WORK_POWER_DIR, "muscle_work.png")
    # Plotly figure - save as HTML
    elif SAVE_FIGURES:
        import plotly.io as pio

        pio.write_html(result, WORK_POWER_DIR / "muscle_work_interactive.html")
    plt.show()

# %% Power overview
result = plots.plot_power_overview()
if result is not None:
    fig, axes = result
    if fig is not None and SAVE_FIGURES:
        plots.save_figure(fig, WORK_POWER_DIR, "power_overview.png")
    plt.show()

# %% =================== MULTIPLE LINEAR REGRESSION ====================

# Create markdown file for full regression results
REGRESSION_RESULTS_FILE = Path("data/regression_results.md")
if REGRESSION_RESULTS_FILE.exists():
    REGRESSION_RESULTS_FILE.unlink()  # Remove old file if exists

from pathlib import Path

# Write header
with REGRESSION_RESULTS_FILE.open("w", encoding="utf-8") as f:
    f.write("# Multiple Linear Regression Results\n\n")
    f.write("Factors influencing muscle strain during running.\n\n")
    f.write("---\n\n")

# %%
# Correlation matrix of output variables for pre
reg_df = analysis_utils.get_regression_df()
excluded_cols = ["subject_id"] + [col for col in reg_df.columns if "change" in col or "post" in col]
filtered_df = reg_df[[col for col in reg_df.columns if col not in excluded_cols]]
corr_matrix = filtered_df.corr()
fig = plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=1,
)
if SAVE_FIGURES:
    plots.save_figure(fig, REGRESSION_DIR, "correlation_matrix.png")
plt.show()

# %%
# Correlation matrix of output variables for changes (delta)
excluded_cols = ["subject_id"] + [col for col in reg_df.columns if "pre" in col or "post" in col]

delta_df = reg_df[[col for col in reg_df.columns if col not in excluded_cols]]
delta_corr_matrix = delta_df.corr()
fig = plt.figure(figsize=(10, 8))
sns.heatmap(
    delta_corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=1,
)
if SAVE_FIGURES:
    plots.save_figure(fig, REGRESSION_DIR, "delta_correlation_matrix.png")
plt.show()

# %% Factors influencing strain (baseline values)
print("\n" + "=" * 60)
print("REGRESSION ANALYSIS: Factors Influencing Baseline Strain")
print("=" * 60)

# Write section header to markdown
with REGRESSION_RESULTS_FILE.open("a", encoding="utf-8") as f:
    f.write("# Baseline Strain Analysis\n\n")

# BFLH
reg_bflh_strain = analysis_utils.multiple_linear_regression(
    type="strain",
    muscle="bflh",
    check_assumptions=True,
    create_plots=True,
    output_file=REGRESSION_RESULTS_FILE,
    interactive=True,
)
if SAVE_FIGURES and "figures" in reg_bflh_strain.get("assumptions", {}):
    import plotly.io as pio

    for fig, plot_type in reg_bflh_strain["assumptions"]["figures"]:
        if hasattr(fig, "write_html"):  # Plotly figure
            pio.write_html(
                fig,
                REGRESSION_STRAIN_DIR / f"bflh_{plot_type}_interactive.html",
            )
        else:  # Matplotlib figure
            plots.save_figure(fig, REGRESSION_STRAIN_DIR, f"bflh_{plot_type}.png")
            plt.close(fig)
# Semitendinosus
reg_semiten_strain = analysis_utils.multiple_linear_regression(
    type="strain",
    muscle="semiten",
    check_assumptions=True,
    create_plots=True,
    output_file=REGRESSION_RESULTS_FILE,
    interactive=True,
)
if SAVE_FIGURES and "figures" in reg_semiten_strain.get("assumptions", {}):
    import plotly.io as pio

    for fig, plot_type in reg_semiten_strain["assumptions"]["figures"]:
        if hasattr(fig, "write_html"):  # Plotly figure
            pio.write_html(
                fig,
                REGRESSION_STRAIN_DIR / f"semiten_{plot_type}_interactive.html",
            )
        else:  # Matplotlib figure
            plots.save_figure(fig, REGRESSION_STRAIN_DIR, f"semiten_{plot_type}.png")
            plt.close(fig)
# Semimembranosus
reg_semimem_strain = analysis_utils.multiple_linear_regression(
    type="strain",
    muscle="semimem",
    check_assumptions=True,
    create_plots=True,
    output_file=REGRESSION_RESULTS_FILE,
    interactive=True,
)
if SAVE_FIGURES and "figures" in reg_semimem_strain.get("assumptions", {}):
    import plotly.io as pio

    for fig, plot_type in reg_semimem_strain["assumptions"]["figures"]:
        if hasattr(fig, "write_html"):  # Plotly figure
            pio.write_html(
                fig,
                REGRESSION_STRAIN_DIR / f"semimem_{plot_type}_interactive.html",
            )
        else:  # Matplotlib figure
            plots.save_figure(fig, REGRESSION_STRAIN_DIR, f"semimem_{plot_type}.png")
            plt.close(fig)

# %% Factors influencing delta strain (fatigue-induced changes)
print("\n" + "=" * 60)
print("REGRESSION ANALYSIS: Factors Influencing Strain Changes")
print("=" * 60)

# Write section header to markdown
with REGRESSION_RESULTS_FILE.open("a", encoding="utf-8") as f:
    f.write("# Strain Changes Analysis (Delta)\n\n")

# BFLH
reg_bflh_delta = analysis_utils.multiple_linear_regression(
    type="delta_strain",
    muscle="bflh",
    check_assumptions=True,
    create_plots=True,
    output_file=REGRESSION_RESULTS_FILE,
    interactive=True,
)
if SAVE_FIGURES and "figures" in reg_bflh_delta.get("assumptions", {}):
    import plotly.io as pio

    for fig, plot_type in reg_bflh_delta["assumptions"]["figures"]:
        if hasattr(fig, "write_html"):  # Plotly figure
            pio.write_html(
                fig,
                REGRESSION_DELTA_DIR / f"bflh_{plot_type}_interactive.html",
            )
        else:  # Matplotlib figure
            plots.save_figure(fig, REGRESSION_DELTA_DIR, f"bflh_{plot_type}.png")
            plt.close(fig)
# Semitendinosus
reg_semiten_delta = analysis_utils.multiple_linear_regression(
    type="delta_strain",
    muscle="semiten",
    check_assumptions=True,
    create_plots=True,
    output_file=REGRESSION_RESULTS_FILE,
    interactive=True,
)
if SAVE_FIGURES and "figures" in reg_semiten_delta.get("assumptions", {}):
    import plotly.io as pio

    for fig, plot_type in reg_semiten_delta["assumptions"]["figures"]:
        if hasattr(fig, "write_html"):  # Plotly figure
            pio.write_html(
                fig,
                REGRESSION_DELTA_DIR / f"semiten_{plot_type}_interactive.html",
            )
        else:  # Matplotlib figure
            plots.save_figure(fig, REGRESSION_DELTA_DIR, f"semiten_{plot_type}.png")
            plt.close(fig)
# Semimembranosus
reg_semimem_delta = analysis_utils.multiple_linear_regression(
    type="delta_strain",
    muscle="semimem",
    check_assumptions=True,
    create_plots=True,
    output_file=REGRESSION_RESULTS_FILE,
    interactive=True,
)
if SAVE_FIGURES and "figures" in reg_semimem_delta.get("assumptions", {}):
    import plotly.io as pio

    for fig, plot_type in reg_semimem_delta["assumptions"]["figures"]:
        if hasattr(fig, "write_html"):  # Plotly figure
            pio.write_html(
                fig,
                REGRESSION_DELTA_DIR / f"semimem_{plot_type}_interactive.html",
            )
        else:  # Matplotlib figure
            plots.save_figure(fig, REGRESSION_DELTA_DIR, f"semimem_{plot_type}.png")
            plt.close(fig)

print("\n" + "=" * 60)

# %% ==================== REGRESSION SCATTER PLOTS ====================

# %% Scatter plots for baseline strain regression
print("\n" + "=" * 60)
print("Creating scatter plots for baseline strain regression...")
print("=" * 60)

scatter_figs_strain = plots.create_regression_scatter_plots(
    type="strain",
    interactive=True,
)

if SAVE_FIGURES and scatter_figs_strain:
    import plotly.io as pio

    REGRESSION_SCATTER_STRAIN_DIR = REGRESSION_STRAIN_DIR / "scatter_plots"
    REGRESSION_SCATTER_STRAIN_DIR.mkdir(parents=True, exist_ok=True)

    for fig, muscle_code, var_name in scatter_figs_strain:
        if hasattr(fig, "write_html"):  # Plotly figure
            filename = f"{muscle_code}_scatter_{var_name}_interactive.html"
            pio.write_html(fig, REGRESSION_SCATTER_STRAIN_DIR / filename)
            print(f"  Saved: {filename}")
        else:  # Matplotlib figure
            filename = f"{muscle_code}_scatter_{var_name}.png"
            plots.save_figure(fig, REGRESSION_SCATTER_STRAIN_DIR, filename)
            plt.close(fig)
            print(f"  Saved: {filename}")

# %% Scatter plots for delta strain regression
print("\n" + "=" * 60)
print("Creating scatter plots for delta strain regression...")
print("=" * 60)

scatter_figs_delta = plots.create_regression_scatter_plots(
    type="delta_strain",
    interactive=True,
)

if SAVE_FIGURES and scatter_figs_delta:
    import plotly.io as pio

    REGRESSION_SCATTER_DELTA_DIR = REGRESSION_DELTA_DIR / "scatter_plots"
    REGRESSION_SCATTER_DELTA_DIR.mkdir(parents=True, exist_ok=True)

    for fig, muscle_code, var_name in scatter_figs_delta:
        if hasattr(fig, "write_html"):  # Plotly figure
            filename = f"{muscle_code}_scatter_{var_name}_interactive.html"
            pio.write_html(fig, REGRESSION_SCATTER_DELTA_DIR / filename)
            print(f"  Saved: {filename}")
        else:  # Matplotlib figure
            filename = f"{muscle_code}_scatter_{var_name}.png"
            plots.save_figure(fig, REGRESSION_SCATTER_DELTA_DIR, filename)
            plt.close(fig)
            print(f"  Saved: {filename}")

print("\n" + "=" * 60)

# %% Output variable correlation plots
print("\n" + "=" * 60)
print("Creating correlation plots between muscle strain outputs...")
print("=" * 60)

output_correlation_figs = plots.create_output_correlation_plots(interactive=True)

if SAVE_FIGURES and output_correlation_figs:
    import plotly.io as pio

    REGRESSION_CORRELATION_DIR = REGRESSION_DIR / "output_correlations"
    REGRESSION_CORRELATION_DIR.mkdir(parents=True, exist_ok=True)

    for fig, plot_description in output_correlation_figs:
        if hasattr(fig, "write_html"):  # Plotly figure
            filename = f"{plot_description}_interactive.html"
            pio.write_html(fig, REGRESSION_CORRELATION_DIR / filename)
            print(f"  Saved: {filename}")
        else:  # Matplotlib figure
            filename = f"{plot_description}.png"
            plots.save_figure(fig, REGRESSION_CORRELATION_DIR, filename)
            plt.close(fig)
            print(f"  Saved: {filename}")

print("\n" + "=" * 60)


# %%
# PCA
print("\n" + "=" * 60)

# %%

# # PCA bflh
# reg_df = analysis_utils.get_regression_df()
# X = reg_df[[col for col in reg_df.columns if col != "change"]]
# pca = analysis_utils.pca(X, )

# %% ================================================
# %%    SINGLE SUBJECT ANALYSES
# %% ================================================

# %% ==================== EXAMPLE SUBJECT PLOTS ====================

# %% Master plot for single subject
try:
    result = plots.plot_master_plot_before_after(EXAMPLE_SUBJECT)
    if result is not None:
        fig, axes = result
        if fig is not None and SAVE_FIGURES:
            plots.save_figure(
                fig,
                PER_SUBJECT_DIR / "master_plots",
                f"master_plot_{EXAMPLE_SUBJECT}.png",
            )
        plt.show()
except KeyError as e:
    print(f"⚠️  Column naming mismatch: {e}")
    print("   Regenerate master_results.csv in data_processing.py")
    plt.show()

# %% Single subject moments/torques
result = plots.plot_moments_torques_single(EXAMPLE_SUBJECT)
if result is not None:
    fig, axes = result
    if fig is not None and SAVE_FIGURES:
        plots.save_figure(
            fig,
            PER_SUBJECT_DIR / "moments_torques",
            f"moments_torques_{EXAMPLE_SUBJECT}.png",
        )
    plt.show()

# %% Single subject EMG
result = plots.plot_emg_adjusted_emg_single(EXAMPLE_SUBJECT)
if result is not None:
    fig, axes = result
    if fig is not None and SAVE_FIGURES:
        plots.save_figure(
            fig,
            PER_SUBJECT_DIR / "emg",
            f"emg_adjusted_{EXAMPLE_SUBJECT}.png",
        )
    plt.show()

# %% ==================== BATCH PROCESSING - ALL SUBJECTS ====================


# %% Generate all subject-specific plots
def process_subject_fatigue(subject_id):
    """Process fatigue plots for a single subject."""
    plots = Plot_utils()
    errors = []

    try:
        # Master plot
        result = plots.plot_master_plot_before_after(subject_id)
        if result is not None:
            fig, _axes = result
            if fig is not None and SAVE_FIGURES:
                plots.save_figure(
                    fig,
                    PER_SUBJECT_DIR / "master_plots",
                    f"master_plot_{subject_id}.png",
                )
            plt.close()

        # Moments/torques
        result = plots.plot_moments_torques_single(subject_id)
        if result is not None:
            fig, _axes = result
            if fig is not None and SAVE_FIGURES:
                plots.save_figure(
                    fig,
                    PER_SUBJECT_DIR / "moments_torques",
                    f"moments_torques_{subject_id}.png",
                )
            plt.close()

        # EMG
        result = plots.plot_emg_adjusted_emg_single(subject_id)
        if result is not None:
            fig, _axes = result
            if fig is not None and SAVE_FIGURES:
                plots.save_figure(fig, PER_SUBJECT_DIR / "emg", f"emg_{subject_id}.png")
            plt.close()
    except RuntimeError as e:
        errors.append(f"Error processing {subject_id}: {e}")

    return subject_id, errors


all_subjects = subject_utils.get_all_subjects()

# Use ThreadPoolExecutor for parallel processing (stable in VSCode)
max_workers = min(4, len(all_subjects))  # Limit to 4 workers for stability

print(
    f"Processing fatigue analysis for {len(all_subjects)} subjects using {max_workers} threads...",
)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_subject_fatigue, subj): subj for subj in all_subjects}

    for future in tqdm(
        as_completed(futures),
        total=len(all_subjects),
        desc="Processing subjects",
    ):
        subject_id, errors = future.result()
        if errors:
            for error in errors:
                print(f"  {error}")

# %% ================================================
# %%    FINALIZATION
# %% ================================================

print("\n" + "=" * 60)
print("Fatigue analysis complete!")
print(f"Figures saved to: {OUTPUT_DIR.absolute()}")
print("=" * 60)

# %% Generate HTML index
if SAVE_FIGURES:
    print("\nGenerating HTML index...")
    html_path = plots.generate_html_index(
        OUTPUT_DIR,
        title="Fatigue Analysis - Plot Overview",
        columns=3,
        analysis_type="fatigue_analysis",
    )
    print(f"📄 HTML index created: {html_path.absolute()}")

    # Generate central hub
    hub_path = plots.generate_hub_index()
    print(f"🎯 Central hub updated: {hub_path.absolute()}")
    print("Open hub in browser to access all analyses")

# %%
