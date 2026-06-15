"""
Publication Figures.

===================
Final, publication-ready figures for thesis and papers.
High-quality outputs with careful styling and formatting.
"""

# %% Setup - Add project root to path
import os
import sys
from pathlib import Path

# Add project root to Python path and set as working directory
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

import matplotlib.pyplot as plt  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import (  # noqa: E402
    Analysis_utils,
    Data_utils,
    File_utils,
    Plot_utils,
    Subject_utils,
)

plots = Plot_utils()
file_utils = File_utils()
subject_utils = Subject_utils()
data_utils = Data_utils()
analysis_utils = Analysis_utils()

# %% Publication configuration
OUTPUT_DIR = Path("data/plot_results/publication")

# Define subdirectories for organized plot storage
OVERVIEW_DIR = OUTPUT_DIR / "overview"
MUSCLE_ANALYSIS_DIR = OUTPUT_DIR / "muscle_analysis"
BIOMECHANICS_DIR = OUTPUT_DIR / "biomechanics"
REGRESSION_DIR = OUTPUT_DIR / "regression"
PER_SUBJECT_DIR = OUTPUT_DIR / "per_subject"

for dir_path in [
    OUTPUT_DIR,
    OVERVIEW_DIR,
    MUSCLE_ANALYSIS_DIR,
    BIOMECHANICS_DIR,
    REGRESSION_DIR,
    PER_SUBJECT_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# High-quality figure settings
DPI = 600
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_DOUBLE = (16, 6)
FORMAT = "png"  # or "pdf", "svg" for vector graphics

REP_SUBJECT = "019"  # Representative subject for single-subject examples
_rep_subject_trials = plots.master_df[plots.master_df["subject_id"] == int(REP_SUBJECT)]["trial"].unique()
from utils.plot_utils import _get_last_post_trial as _plt_get_last  # noqa: E402
REP_TRIAL = _plt_get_last(_rep_subject_trials) or "RunL1"

# %% ================================================
# %%    ALL SUBJECTS FIGURES
# %% ================================================

# %% ==================== MAIN RESULTS FIGURES ====================

# %% Figure 1: Master plot - Overview of fatigue effects
print("Generating Figure 1: Master plot...")
try:
    result = plots.plot_master_plot_before_after_all()
    if result is not None:
        fig, axes = result
        if fig is not None:
            plots.save_figure(
                fig,
                OVERVIEW_DIR,
                f"figure_1_master_plot.{FORMAT}",
                dpi=DPI,
                format=FORMAT,
            )
            plt.show()
            print("✓ Figure 1 saved")
        else:
            print("✗ Figure 1 - fig is None")
    else:
        print("✗ Figure 1 - method returned None")
except KeyError as e:
    print(f"✗ Figure 1 failed - Column naming mismatch: {e}")
    print("   Regenerate master_results.csv in data_processing.py")

# %% Figure 1b: Master plot with official SPM panels
print("Generating Figure 1b: Master plot with SPM panels...")
try:
    result = plots.plot_master_plot_before_after_all_with_spm(
        alpha=0.05,
        two_tailed=False,
    )
    if result is not None:
        fig, axes = result
        if fig is not None:
            plots.save_figure(
                fig,
                OVERVIEW_DIR,
                f"figure_1b_master_plot_spm.{FORMAT}",
                dpi=DPI,
                format=FORMAT,
            )
            plt.show()
            print("✓ Figure 1b saved")
        else:
            print("✗ Figure 1b - fig is None")
    else:
        print("✗ Figure 1b - method returned None")
except (KeyError, RuntimeError, ValueError) as e:
    print(f"✗ Figure 1b failed: {e}")

# %% Figure 2: Master plot without peak normalized fiber lengths
print("Generating Figure 2: Master plot (angles, moments, normalized forces)...")
try:
    result = plots.master_plot_with_peak_norm_fiber_lengths(show_fiber_rows=False)
    if result is not None:
        fig, axes = result
        if fig is not None:
            plots.save_figure(
                fig,
                OVERVIEW_DIR,
                f"figure_2_master_plot.{FORMAT}",
                dpi=DPI,
                format=FORMAT,
            )
            plt.show()
            print("✓ Figure 2 saved")
        else:
            print("✗ Figure 2 - fig is None")
    else:
        print("✗ Figure 2 - method returned None")
except RuntimeError as e:
    print(f"✗ Figure 2 failed: {e}")

# %% Figure 2b: Peak normalized fiber lengths only
print("Generating Figure 2b: Peak normalized fiber lengths...")
try:
    result = plots.plot_peak_fiber_lengths_only()
    if result is not None:
        fig, axes = result
        if fig is not None:
            plots.save_figure(
                fig,
                OVERVIEW_DIR,
                f"figure_2b_peak_norm_fiber_lengths.{FORMAT}",
                dpi=DPI,
                format=FORMAT,
            )
            plt.show()
            print("✓ Figure 2b saved")
        else:
            print("✗ Figure 2b - fig is None")
    else:
        print("✗ Figure 2b - method returned None")
except RuntimeError as e:
    print(f"✗ Figure 2b failed: {e}")

# %% Figure 3: Muscle group activations
print("Generating Figure 3: Muscle activations...")
result = plots.plot_muscle_group_activations()
if result is not None:
    fig, axes = result
    if fig is not None:
        plots.save_figure(
            fig,
            MUSCLE_ANALYSIS_DIR,
            f"figure_3_muscle_activations.{FORMAT}",
            dpi=DPI,
            format=FORMAT,
        )
        plt.show()
        print("✓ Figure 3 saved")
    else:
        print("✗ Figure 3 - fig is None")
else:
    print("✗ Figure 3 - method returned None")

# %% Figure 4: Muscle group forces
print("Generating Figure 4: Muscle forces...")
result = plots.plot_muscle_group_forces()
if result is not None:
    fig, axes = result
    if fig is not None:
        plots.save_figure(
            fig,
            MUSCLE_ANALYSIS_DIR,
            f"figure_4_muscle_forces.{FORMAT}",
            dpi=DPI,
            format=FORMAT,
        )
        plt.show()
        print("✓ Figure 4 saved")
    else:
        print("✗ Figure 4 - fig is None")
else:
    print("✗ Figure 4 - method returned None")

# %% Figure 5: ID moments vs CEINMS torques
print("Generating Figure 5: Moments validation...")
result = plots.plot_moments_torques()
if result is not None:
    fig, axes = result
    if fig is not None:
        plots.save_figure(
            fig,
            BIOMECHANICS_DIR,
            f"figure_5_moments_torques.{FORMAT}",
            dpi=DPI,
            format=FORMAT,
        )
        plt.show()
        print("✓ Figure 5 saved")
    else:
        print("✗ Figure 5 - fig is None")
else:
    print("✗ Figure 5 - method returned None")

# %% Figure 6: Muscle work
print("Generating Figure 6: Muscle work...")
result = plots.plot_muscle_work()
if result is not None:
    fig, axes = result
    if fig is not None:
        plots.save_figure(
            fig,
            MUSCLE_ANALYSIS_DIR,
            f"figure_6_muscle_work.{FORMAT}",
            dpi=DPI,
            format=FORMAT,
        )
        plt.show()
        print("✓ Figure 6 saved")
    else:
        print("✗ Figure 6 - fig is None")
else:
    print("✗ Figure 6 - method returned None")

# %% Figure 7: Power overview
print("Generating Figure 7: Power analysis...")
result = plots.plot_power_overview()
if result is not None:
    fig, axes = result
    if fig is not None:
        plots.save_figure(
            fig,
            MUSCLE_ANALYSIS_DIR,
            f"figure_7_power_overview.{FORMAT}",
            dpi=DPI,
            format=FORMAT,
        )
        plt.show()
        print("✓ Figure 7 saved")
    else:
        print("✗ Figure 7 - fig is None")
else:
    print("✗ Figure 7 - method returned None")

# %% ==================== SUPPLEMENTARY FIGURES ====================

# %% Supplementary Figure 1: Group split analysis
print("Generating Supplementary Figure 1...")
result = plots.plot_master_plot_with_after_all_group_split()
if result is not None:
    fig, axes = result
    if fig is not None:
        plots.save_figure(
            fig,
            OVERVIEW_DIR,
            f"supp_figure_1_group_split.{FORMAT}",
            dpi=DPI,
            format=FORMAT,
        )
        plt.show()
        print("✓ Supp Figure 1 saved")
    else:
        print("✗ Supp Figure 1 - fig is None")
else:
    print("✗ Supp Figure 1 - method returned None")

# %% Supplementary Figure 2: Muscle parameters
print("Generating Supplementary Figure 2...")
result = plots.plot_muscle_parameters()
if result is not None:
    fig, axes = result
    if fig is not None:
        plots.save_figure(
            fig,
            MUSCLE_ANALYSIS_DIR,
            f"supp_figure_2_muscle_parameters.{FORMAT}",
            dpi=DPI,
            format=FORMAT,
        )
        plt.show()
        print("✓ Supp Figure 2 saved")
    else:
        print("✗ Supp Figure 2 - fig is None")
else:
    print("✗ Supp Figure 2 - method returned None")

# %% Supplementary Figure 3: Fiber length
print("Generating Supplementary Figure 3...")
result = plots.plot_norm_fiber_length_pre_post()
if result is not None:
    fig, axes = result
    if fig is not None:
        plots.save_figure(
            fig,
            MUSCLE_ANALYSIS_DIR,
            f"supp_figure_3_fiber_length.{FORMAT}",
            dpi=DPI,
            format=FORMAT,
        )
        plt.show()
        print("✓ Supp Figure 3 saved")
    else:
        print("✗ Supp Figure 3 - fig is None")
else:
    print("✗ Supp Figure 3 - method returned None")

# %% Supplementary Figure 4: Peak force vs normalized fiber length
print("Generating Supplementary Figure 4...")
result = plots.plot_peak_norm_force_change_vs_peak_norm_fiber_length_change()
if result is not None:
    fig, axes = result
    if fig is not None:
        plots.save_figure(
            fig,
            MUSCLE_ANALYSIS_DIR,
            f"supp_figure_4_force_vs_norm_fiber_length.{FORMAT}",
            dpi=DPI,
            format=FORMAT,
        )
        plt.show()
        print("✓ Supp Figure 4 saved")
    else:
        print("✗ Supp Figure 4 - fig is None")
else:
    print("✗ Supp Figure 4 - method returned None")

# %% Supplementary Figure 5: EMG validation
print("Generating Supplementary Figure 5...")
result = plots.plot_emg_adjusted_emg()
if result is not None:
    fig, axes = result
    if fig is not None:
        plots.save_figure(
            fig,
            BIOMECHANICS_DIR,
            f"supp_figure_5_emg_validation.{FORMAT}",
            dpi=DPI,
            format=FORMAT,
        )
        plt.show()
        print("✓ Supp Figure 5 saved")
    else:
        print("✗ Supp Figure 5 - fig is None")
else:
    print("✗ Supp Figure 5 - method returned None")

# %% Supplementary Figure 6: EMG vs CEINMS Activations
print("Generating Supplementary Figure 6...")
result = plots.plot_emg_ceinms_activations()
if result is not None:
    fig, axes = result
    if fig is not None:
        plots.save_figure(
            fig,
            BIOMECHANICS_DIR,
            f"supp_figure_6_emg_vs_ceinms_activations.{FORMAT}",
            dpi=DPI,
            format=FORMAT,
        )
        plt.show()
        print("✓ Supp Figure 6 saved")
    else:
        print("✗ Supp Figure 6 - fig is None")
else:
    print("✗ Supp Figure 6 - method returned None")

# %% ================================================
# %%    REPRESENTATIVE SUBJECT FIGURES
# %% ================================================

# %% Representative subject (high quality example)
print(f"\nGenerating representative subject figures ({REP_SUBJECT})...")

# %% Single subject master plot
result = plots.plot_master_plot_before_after(REP_SUBJECT)
if result is not None:
    fig, axes = result
    if fig is not None:
        plots.save_figure(
            fig,
            PER_SUBJECT_DIR,
            f"representative_master_plot_{REP_SUBJECT}.{FORMAT}",
            dpi=DPI,
            format=FORMAT,
        )
        plt.close()
    else:
        print("✗ Representative master plot - fig is None")
else:
    print("✗ Representative master plot - method returned None")

# %% Single subject CEINMS validation
result = plots.plot_experimental_vs_ceinms(REP_SUBJECT, REP_TRIAL)
if result is not None:
    emg_fig, emg_axes, moments_fig, moments_axes = result
    plots.save_figure(
        emg_fig,
        PER_SUBJECT_DIR,
        f"representative_emg_validation_{REP_SUBJECT}.{FORMAT}",
        dpi=DPI,
        format=FORMAT,
    )
    plots.save_figure(
        moments_fig,
        PER_SUBJECT_DIR,
        f"representative_moments_validation_{REP_SUBJECT}.{FORMAT}",
        dpi=DPI,
        format=FORMAT,
    )
    plt.close("all")
else:
    print("✗ Representative CEINMS validation - method returned None")

print("✓ Representative figures saved")

# %% ================================================
# %%    BATCH FIGURE GENERATION
# %% ================================================

# %% ==================== COMPLETE FIGURE SET ====================


# %% Generate all publication figures at once
def _generate_single_figure(name, plot_func, target_dir, has_interactive):
    """Generate one figure (static + optional interactive) and return a status tuple."""
    try:
        result = plot_func()
        if result is None:
            return name, False, "Method returned None"
        fig, _ = result
        if fig is None:
            return name, False, "fig is None"

        plots.save_figure(fig, target_dir, f"{name}.{FORMAT}", dpi=DPI, format=FORMAT)
        plt.close(fig)

        if has_interactive:
            if "muscle_work" in name:
                fig_plotly = plots.plot_muscle_work(return_plotly=True)
            elif "power_overview" in name:
                fig_plotly = plots.plot_power_overview(return_plotly=True)
            else:
                fig_plotly = None

            if fig_plotly is not None:
                plots.save_plotly_figure(
                    fig_plotly, target_dir, f"{name}_interactive", subfolder="interactive"
                )
                return name, True, "with interactive ✓"
            return name, True, "plotly not available"

        return name, True, None
    except RuntimeError as e:
        return name, False, str(e)


def generate_all_publication_figures():
    figures = {
        "figure_1_master_plot": (plots.plot_master_plot_before_after_all, OVERVIEW_DIR, False),
        "figure_1b_master_plot_spm": (plots.plot_master_plot_before_after_all_with_spm, OVERVIEW_DIR, False),
        "figure_2_master_plot": (
            lambda: plots.master_plot_with_peak_norm_fiber_lengths(show_fiber_rows=False),
            OVERVIEW_DIR,
            False,
        ),
        "figure_2b_peak_norm_fiber_lengths": (plots.plot_peak_fiber_lengths_only, OVERVIEW_DIR, False),
        "figure_3_muscle_activations": (plots.plot_muscle_group_activations, MUSCLE_ANALYSIS_DIR, False),
        "figure_4_muscle_forces": (plots.plot_muscle_group_forces, MUSCLE_ANALYSIS_DIR, False),
        "figure_5_moments_torques": (plots.plot_moments_torques, BIOMECHANICS_DIR, False),
        "figure_6_muscle_work": (plots.plot_muscle_work, MUSCLE_ANALYSIS_DIR, True),
        "figure_7_power_overview": (plots.plot_power_overview, MUSCLE_ANALYSIS_DIR, True),
        "supp_1_group_split": (plots.plot_master_plot_with_after_all_group_split, OVERVIEW_DIR, False),
        "supp_2_muscle_parameters": (plots.plot_muscle_parameters, MUSCLE_ANALYSIS_DIR, False),
        "supp_3_fiber_length": (plots.plot_norm_fiber_length_pre_post, MUSCLE_ANALYSIS_DIR, False),
        "supp_4_force_vs_norm_fiber_length": (
            plots.plot_peak_norm_force_change_vs_peak_norm_fiber_length_change,
            MUSCLE_ANALYSIS_DIR,
            False,
        ),
        "supp_5_emg_validation": (plots.plot_emg_adjusted_emg, BIOMECHANICS_DIR, False),
    }

    print("\n" + "=" * 60)
    print("GENERATING ALL PUBLICATION FIGURES")
    print("=" * 60 + "\n")

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_generate_single_figure, name, func, target_dir, interactive): name
            for name, (func, target_dir, interactive) in figures.items()
        }
        for future in as_completed(futures):
            name, success, message = future.result()
            status = "✓" if success else "✗"
            suffix = f" ({message})" if message else ""
            print(f"  {status} {name}{suffix}")

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 60)


# Uncomment to generate all at once:
generate_all_publication_figures()

print("\n" + "=" * 60)
print("Publication figures ready!")
print(f"Output directory: {OUTPUT_DIR.absolute()}")
print(f"Format: {FORMAT}, DPI: {DPI}")
print("=" * 60)

# %% ================================================
# %%    FINALIZATION
# %% ================================================

# %% Generate HTML index
print("\nGenerating HTML index...")
html_path = plots.generate_html_index(
    OUTPUT_DIR,
    title="Publication Figures - High Quality Outputs",
    columns=3,
    analysis_type="publication_figures",
)
print(f"📄 HTML index created: {html_path.absolute()}")

# Generate central hub
hub_path = plots.generate_hub_index()
print(f"🎯 Central hub updated: {hub_path.absolute()}")
print("   Open hub in browser to access all figures")
# %%
