"""
CEINMS Analysis and Validation.

Analyze CEINMS model calibration, optimization results, and validate
against experimental data (EMG and inverse dynamics).
"""

# %% Setup - Add project root to path
import os
import sys
import time
from pathlib import Path

# Add project root to Python path and set as working directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")  # Use non-interactive backend for performance in batch processing
import matplotlib.pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402

from utils import (  # noqa: E402
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
SAVE_FIGURES = True
OUTPUT_DIR = Path("data/plot_results/ceinms_analysis")

# Define subdirectories for organized plot storage
CALIBRATION_DIR = OUTPUT_DIR / "calibration"
OPTIMIZATION_DIR = OUTPUT_DIR / "optimization"
VALIDATION_DIR = OUTPUT_DIR / "validation"
PER_SUBJECT_DIR = OUTPUT_DIR / "per_subject"

if SAVE_FIGURES:
    for dir_path in [
        OUTPUT_DIR,
        CALIBRATION_DIR,
        OPTIMIZATION_DIR,
        VALIDATION_DIR,
        PER_SUBJECT_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

TEST_SUBJECT = "024"
# TEST_TRIAL = "Run_baselineA1"
# TEST_TRIAL = "Run_baselineB1"
TEST_TRIAL = "RunL1"

all_subjects = subject_utils.get_all_subjects()

max_workers = min(4, len(all_subjects))  # Limit to 4 workers for stability
print(f"Configured to use up to {max_workers} threads for batch processing.")

# %% ================================================
# %%    SINGLE SUBJECT EXAMPLES
# %% ================================================

# %% ==================== MODEL CALIBRATION ====================

# %% CEINMS model parameters (calibrated)
fig, axes = plots.plot_ceinms_model_parameters(TEST_SUBJECT)
if SAVE_FIGURES:
    plots.save_figure(fig, CALIBRATION_DIR, f"ceinms_parameters_{TEST_SUBJECT}.png")
plt.show()

# %% Compare uncalibrated vs calibrated models
fig, axes = plots.plot_compare_ceinms_models(TEST_SUBJECT)
if SAVE_FIGURES:
    plots.save_figure(
        fig,
        CALIBRATION_DIR,
        f"ceinms_model_comparison_{TEST_SUBJECT}.png",
    )
plt.show()

# %% CEINMS calibration results (all muscles grouped)
plots_list = plots.plot_ceinms_calibration_results(TEST_SUBJECT)
if SAVE_FIGURES:
    for i, (fig, _axes, title) in enumerate(plots_list):
        plots.save_figure(
            fig,
            CALIBRATION_DIR,
            f"ceinms_calibration_{TEST_SUBJECT}_{i}_{title}.png",
        )
plt.show()

# %% Calibration torques vs ID moments
fig, axes = plots.plot_moments_calibration_results(TEST_SUBJECT, TEST_TRIAL)
if SAVE_FIGURES:
    plots.save_figure(
        fig,
        CALIBRATION_DIR,
        f"ceinms_moments_calibration_{TEST_SUBJECT}_{TEST_TRIAL}.png",
    )
plt.show()

# %% ==================== OPTIMIZATION RESULTS ====================

# %% CEINMS optimization outputs (activations, forces, etc.)
plots_list = plots.plot_optimisation_results(TEST_SUBJECT, TEST_TRIAL)
if SAVE_FIGURES:
    for i, (fig, _axes, title) in enumerate(plots_list):
        plots.save_figure(
            fig,
            OPTIMIZATION_DIR,
            f"ceinms_optimization_{TEST_SUBJECT}_{TEST_TRIAL}_{i}_{title}.png",
        )
plt.show()

# %% CEINMS muscle forces with activation overlay
fig, axes = plots.plot_ceinms_muscle_forces(TEST_SUBJECT, TEST_TRIAL)
if SAVE_FIGURES:
    plots.save_figure(
        fig,
        OPTIMIZATION_DIR,
        f"ceinms_muscle_forces_{TEST_SUBJECT}_{TEST_TRIAL}.png",
    )
plt.show()

# %% ==================== EXPERIMENTAL VALIDATION ====================

# %% EMG vs CEINMS activations + ID moments vs CEINMS torques
emg_fig, emg_axes, moments_fig, moments_axes = plots.plot_experimental_vs_ceinms(
    TEST_SUBJECT,
    TEST_TRIAL,
)

if SAVE_FIGURES:
    plots.save_figure(
        emg_fig,
        VALIDATION_DIR,
        f"emg_vs_ceinms_{TEST_SUBJECT}_{TEST_TRIAL}.png",
    )
    plots.save_figure(
        moments_fig,
        VALIDATION_DIR,
        f"moments_vs_ceinms_{TEST_SUBJECT}_{TEST_TRIAL}.png",
    )

plt.show()

# %% ================================================
# %%    BATCH PROCESSING - ALL SUBJECTS
# %% ================================================


# %% Process all subjects - CEINMS model parameters
def process_subject_models(subject_id):
    """Process CEINMS model plots for a single subject."""
    # Use non-interactive backend for performance

    plots = Plot_utils()
    errors = []

    try:
        fig, _axes = plots.plot_ceinms_model_parameters(subject_id)
        if SAVE_FIGURES:
            plots.save_figure(
                fig,
                PER_SUBJECT_DIR,
                f"ceinms_parameters_{subject_id}.png",
            )
        plt.close(fig)

        fig, _axes = plots.plot_compare_ceinms_models(subject_id)
        if SAVE_FIGURES:
            plots.save_figure(
                fig,
                PER_SUBJECT_DIR,
                f"ceinms_comparison_{subject_id}.png",
            )
        plt.close(fig)

        plots_list = plots.plot_ceinms_calibration_results(subject_id)
        if SAVE_FIGURES:
            for i, (fig, _axes, title) in enumerate(plots_list):
                plots.save_figure(
                    fig,
                    PER_SUBJECT_DIR,
                    f"ceinms_calibration_{subject_id}_{i}_{title}.png",
                )
                plt.close(fig)
            for fig, _, _ in plots_list:
                plt.close(fig)
            plt.close("all")
    except RuntimeError as e:
        errors.append(f'Error processing {subject_id}: {e}"')
    finally:
        # Ensure all figures are closed even if error occurs
        plt.close("all")

    return subject_id, errors


USE_THREADING = False  # Set to True/False to toggle

all_subjects = subject_utils.get_all_subjects()

start_time = time.time()

if USE_THREADING:
    print(f"Processing CEINMS models for {len(all_subjects)} subjects using {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_subject_models, subj): subj for subj in all_subjects}
        for future in tqdm(as_completed(futures), total=len(all_subjects), desc="Processing subjects"):
            subject_id, errors = future.result()
            if errors:
                for error in errors:
                    print(f"  {error}")
else:
    print(f"Processing CEINMS models for {len(all_subjects)} subjects sequentially...")
    for subject_id in tqdm(all_subjects, desc="Processing subjects"):
        subject_id_results, errors = process_subject_models(subject_id)
        if errors:
            for error in errors:
                print(f"  {error}")

elapsed_time = time.time() - start_time
print("\nCEINMS analysis complete!")
print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")


# %% Process all subjects - validation plots (first trial only)
def process_subject_validation(subject_id):
    """Process CEINMS validation plots for a single subject."""
    plots = Plot_utils()
    subject_utils = Subject_utils()
    errors = []

    try:
        trials = subject_utils.get_all_relevant_trials(subject_id)
        trial = trials[0]  # First trial

        # Optimization results
        plots_list = plots.plot_optimisation_results(subject_id, trial)
        if SAVE_FIGURES:
            for i, (fig, _axes, title) in enumerate(plots_list):
                plots.save_figure(
                    fig,
                    PER_SUBJECT_DIR,
                    f"optimization_{subject_id}_{trial}_{i}_{title}.png",
                )
                plt.close(fig)
        else:
            for fig, _, _ in plots_list:
                plt.close(fig)

        # Experimental validation
        emg_fig, _emg_axes, moments_fig, _moments_axes = plots.plot_experimental_vs_ceinms(subject_id, trial)
        if SAVE_FIGURES:
            plots.save_figure(
                emg_fig,
                PER_SUBJECT_DIR,
                f"emg_validation_{subject_id}_{trial}.png",
            )
            plots.save_figure(
                moments_fig,
                PER_SUBJECT_DIR,
                f"moments_validation_{subject_id}_{trial}.png",
            )
        plt.close(emg_fig)
        plt.close(moments_fig)
    except RuntimeError as e:
        errors.append(f'Error processing {subject_id}: {e}"')
    finally:
        # Ensure all figures are closed even if error occurs
        plt.close("all")

    return subject_id, errors


USE_THREADING = False  # Set to True/False to toggle

all_subjects = subject_utils.get_all_subjects()

start_time = time.time()

if USE_THREADING:
    print(f"Processing CEINMS models for {len(all_subjects)} subjects using {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_subject_validation, subj): subj for subj in all_subjects}
        for future in tqdm(as_completed(futures), total=len(all_subjects), desc="Processing subjects"):
            subject_id, errors = future.result()
            if errors:
                for error in errors:
                    print(f"  {error}")
else:
    print(f"Processing CEINMS models for {len(all_subjects)} subjects sequentially...")
    for subject_id in tqdm(all_subjects, desc="Processing subjects"):
        subject_id_results, errors = process_subject_validation(subject_id)
        if errors:
            for error in errors:
                print(f"  {error}")

elapsed_time = time.time() - start_time
print("\nCEINMS analysis complete!")
print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

# %% ================================================
# %%    FINALIZATION
# %% ================================================

# %% Generate HTML index
if SAVE_FIGURES:
    print("\nGenerating HTML index...")
    html_path = plots.generate_html_index(
        OUTPUT_DIR,
        title="CEINMS Analysis - Model Calibration & Validation",
        columns=3,
        analysis_type="ceinms_analysis",
    )
    print(f"📄 HTML index created: {html_path.absolute()}")

    # Generate central hub
    hub_path = plots.generate_hub_index()
    print(f"🎯 Central hub updated: {hub_path.absolute()}")
    print("   Open hub in browser to access all analyses")

# %%
