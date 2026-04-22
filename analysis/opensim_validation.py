"""
OpenSim Validation Analysis.

============================
Validation plots for inverse kinematics (IK) and inverse dynamics (ID).
Check marker tracking quality, joint angles, and joint moments.
"""

# %% Setup - Add project root to path
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

# Add project root to Python path and set as working directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src import Simulation_analyzer  # noqa: E402
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
tools = Simulation_analyzer()

# %% Configuration
SAVE_FIGURES = True  # Set to False to skip saving
OUTPUT_DIR = Path("data/plot_results/opensim_validation")

# Define subdirectories for organized plot storage
KINEMATICS_DIR = OUTPUT_DIR / "kinematics"
KINETICS_DIR = OUTPUT_DIR / "kinetics"
MUSCLE_FORCES_DIR = OUTPUT_DIR / "muscle_forces"
PER_SUBJECT_DIR = OUTPUT_DIR / "per_subject"

if SAVE_FIGURES:
    for dir_path in [
        OUTPUT_DIR,
        KINEMATICS_DIR,
        KINETICS_DIR,
        MUSCLE_FORCES_DIR,
        PER_SUBJECT_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

TRIAL = "RunL1"  # Standard trial for comparison

# Threading configuration
cpu_count = os.cpu_count() or 4

max_workers = min(
    cpu_count - 2, len(subject_utils.get_all_subjects())
)  # Leave some cores free for system responsiveness
print(f"Configured to use up to {max_workers} threads for batch processing.")

# %% ================================================
# %%    ALL SUBJECTS ANALYSES
# %% ================================================

# %% Compare kinematics across all subjects for a trial
result = plots.plot_all_ik(TRIAL)
if result is not None:
    fig, *_ = result  # Unpack first element, ignore the rest
    if not isinstance(fig, Figure):
        msg = f"Expected 'fig' to be a matplotlib.figure.Figure, got {type(fig)}"
        raise TypeError(msg)
    if SAVE_FIGURES:
        plots.save_figure(fig, KINEMATICS_DIR, f"all_subjects_ik_{TRIAL}.png")
    plt.show()
else:
    print(f"⚠️  No data available for plot_all_ik({TRIAL})")

# %% Compare kinetics across all subjects for a trial
result = plots.plot_all_id(TRIAL)
if result is not None:
    fig, *_ = result  # Unpack first element, ignore the rest
    if not isinstance(fig, Figure):
        msg = f"Expected 'fig' to be a matplotlib.figure.Figure, got {type(fig)}"
        raise TypeError(msg)
    if SAVE_FIGURES:
        plots.save_figure(fig, KINETICS_DIR, f"all_subjects_id_{TRIAL}.png")
        # Generate interactive version
        fig_plotly = plots.plot_all_id(TRIAL, return_plotly=True)
        if fig_plotly is not None:
            plots.save_plotly_figure(
                fig_plotly,
                KINETICS_DIR,
                f"all_subjects_id_{TRIAL}_interactive",
                subfolder="interactive",
            )
    plt.show()
else:
    print(f"⚠️  No data available for plot_all_id({TRIAL})")

# %% Normalized muscle forces across all subjects
# NOTE: Requires master_results.csv with correct column names
try:
    result = plots.plot_all_norm_forces(TRIAL)
    if result is not None:
        fig, *_ = result  # Unpack first element, ignore the rest
        if not isinstance(fig, Figure):
            msg = f"Expected 'fig' to be a matplotlib.figure.Figure, got {type(fig)}"
            raise TypeError(msg)
        if SAVE_FIGURES:
            plots.save_figure(
                fig,
                MUSCLE_FORCES_DIR,
                f"all_subjects_norm_forces_{TRIAL}.png",
            )
            # Generate interactive version
            fig_plotly = plots.plot_all_norm_forces(TRIAL, return_plotly=True)
            if fig_plotly is not None:
                plots.save_plotly_figure(
                    fig_plotly,
                    MUSCLE_FORCES_DIR,
                    f"all_subjects_norm_forces_{TRIAL}_interactive",
                    subfolder="interactive",
                )
        plt.show()
    else:
        print(f"⚠️  No data available for plot_all_norm_forces({TRIAL})")
except KeyError as e:
    print(f"⚠️  Column naming mismatch: {e}")
    print("   Regenerate master_results.csv in data_processing.py")

# %% Strain rates across all subjects
# NOTE: Requires master_results.csv with correct column names
try:
    result = plots.plot_all_strain_rates(TRIAL)
    if result is not None:
        fig, *_ = result  # Unpack first element, ignore the rest
        if not isinstance(fig, Figure):
            msg = f"Expected 'fig' to be a matplotlib.figure.Figure, got {type(fig)}"
            raise TypeError(msg)
        if SAVE_FIGURES:
            plots.save_figure(
                fig,
                MUSCLE_FORCES_DIR,
                f"all_subjects_strain_rates_{TRIAL}.png",
            )
        plt.show()
    else:
        print(f"⚠️  No data available for plot_all_strain_rates({TRIAL})")
except KeyError as e:
    print(f"⚠️  Column naming mismatch: {e}")
    print("   Regenerate master_results.csv in data_processing.py")

# %% ================================================
# %%    SINGLE SUBJECT ANALYSES
# %% ================================================


# %% Individual subject validation (iterate through all subjects)
def process_subject_validation(subject_id):
    """Process validation plots for a single subject."""
    # Use non-interactive backend for performance
    mpl.use("Agg")

    plots = Plot_utils()
    subject_utils = Subject_utils()
    errors = []

    try:
        trials = subject_utils.get_all_relevant_trials(subject_id)

        # Plot first and last trial for comparison
        for trial in [trials[0], trials[-1]]:
            # Kinematics
            result = plots.plot_single_ik(subject_id, trial)
            if result is not None:
                fig, *_ = result  # Unpack first element, ignore the rest
                if not isinstance(fig, Figure):
                    msg = f"Expected 'fig' to be a matplotlib.figure.Figure, got {type(fig)}"
                    errors.append(str(msg))
                    continue
                if SAVE_FIGURES:
                    plots.save_figure(
                        fig,
                        PER_SUBJECT_DIR,
                        f"ik_{subject_id}_{trial}.png",
                    )
                plt.close(fig)
            else:
                continue

            # Kinetics
            result = plots.plot_single_id(subject_id, trial)
            if result is not None:
                fig, *_ = result  # Unpack first element, ignore the rest
                if not isinstance(fig, Figure):
                    msg = f"Expected 'fig' to be a matplotlib.figure.Figure, got {type(fig)}"
                    errors.append(str(msg))
                    continue
                if SAVE_FIGURES:
                    plots.save_figure(
                        fig,
                        PER_SUBJECT_DIR,
                        f"id_{subject_id}_{trial}.png",
                    )
                plt.close(fig)
            else:
                continue
    except RuntimeError as e:
        errors.append(f"Error processing {subject_id}: {e}")
    finally:
        # Ensure all figures are closed even if error occurs
        plt.close("all")

    return subject_id, errors


all_subjects = subject_utils.get_all_subjects()

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(process_subject_validation, subj): subj for subj in all_subjects
    }

    for future in tqdm(
        as_completed(futures),
        total=len(all_subjects),
        desc="Processing subjects",
    ):
        subject_id, errors = future.result()
        if errors:
            for error in errors:
                print(f"  {error}")

print("\nValidation analysis complete!")

# %% ================================================
# %%    ADDITIONAL VALIDATION
# %% ================================================

# %% Calculate marker errors for specific subject/trial
TEST_SUBJECT = "019"
TEST_TRIAL = TRIAL

marker_errors = analysis_utils.calculate_marker_errors(TEST_SUBJECT, TEST_TRIAL)
print(f"Marker errors for {TEST_SUBJECT} - {TEST_TRIAL}:")
print(marker_errors)

# %% ================================================
# %%    FINALIZATION
# %% ================================================

# %% Generate HTML index
if SAVE_FIGURES:
    print("\nGenerating HTML index...")
    html_path = plots.generate_html_index(
        OUTPUT_DIR,
        title="OpenSim Validation - Kinematics & Kinetics",
        columns=3,
        analysis_type="opensim_validation",
    )
    print(f"📄 HTML index created: {html_path.absolute()}")

    # Generate central hub
    hub_path = plots.generate_hub_index()
    print(f"🎯 Central hub updated: {hub_path.absolute()}")
    print("   Open hub in browser to access all analyses")

# %%
