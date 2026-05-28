"""
Exploratory Analysis - Quick tests and debugging.

=================================================
Use this file for quick data exploration, testing individual subjects/trials,
and debugging analysis pipelines.
"""

# %% Setup - Add project root to path
import os
import sys
from pathlib import Path

# Add project root to Python path and set as working directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

import matplotlib.pyplot as plt  # noqa: E402

from src import Simulation_analyzer  # noqa: E402
from utils import (  # noqa: E402
    Analysis_utils,
    Data_utils,
    File_utils,
    Plot_utils,
    Subject_utils,
)

# Initialize utilities
plots = Plot_utils()
file_utils = File_utils()
subject_utils = Subject_utils()
data_utils = Data_utils()
analysis_utils = Analysis_utils()
tools = Simulation_analyzer()

# %% Configure subject and trial for testing
TEST_SUBJECT = "019"
_test_subject_trials = plots.master_df[plots.master_df["subject_id"] == int(TEST_SUBJECT)]["trial"].unique()
TEST_TRIAL = analysis_utils._get_last_post_trial(_test_subject_trials) or "RunL1"

# %% Single trial kinematics (IK)
fig, axes = plots.plot_single_ik(TEST_SUBJECT, TEST_TRIAL)
plt.show()

# %% Single trial kinetics (ID)
fig, axes = plots.plot_single_id(TEST_SUBJECT, TEST_TRIAL)
plt.show()

# %% Single trial master plot
# NOTE: Column naming mismatch between master_results.csv and plot functions
# Your CSV has abbreviated names (e.g., 'bflh_norm_force')
# But plot functions expect full names (e.g., 'biceps_femoris_long_head_norm_force')
try:
    fig, axes = plots.plot_single_master_plot(TEST_SUBJECT, TEST_TRIAL)
    plt.show()
except KeyError as e:
    print(f"⚠️  Column naming mismatch: {e}")
    print("   Your master_results.csv uses abbreviated muscle names")
    print("   But the plot function expects full names")
    print("   To fix: Regenerate master_results.csv")
    print("   1. Open data_processing.py")
    print("   2. Run the 'Create master results CSV' cell")

# %% EMG and adjusted EMG for single subject
fig, axes = plots.plot_emg_adjusted_emg_single(TEST_SUBJECT)
plt.show()

# %% ID moments vs CEINMS torques (single subject)
result = plots.plot_moments_torques_single(TEST_SUBJECT)
if result is not None:
    fig, axes = result
    plt.show()

# %% Get all subjects for batch operations
all_subjects = subject_utils.get_all_subjects()
print(f"Found {len(all_subjects)} subjects: {all_subjects}")

# %% Get trials for a specific subject
trials = subject_utils.get_all_relevant_trials(TEST_SUBJECT)
print(f"Trials for {TEST_SUBJECT}: {trials}")

# %% Quick data check - load and inspect IK results
ik_file = file_utils.get_ik_file_path(TEST_SUBJECT, TEST_TRIAL)
ik_data = data_utils.create_df(ik_file)
print(f"IK data shape: {ik_data.shape}")
print(f"IK columns: {list(ik_data.columns)}")

# %% Quick data check - load and inspect ID results
id_file = file_utils.get_id_file_path(TEST_SUBJECT, TEST_TRIAL)
id_data = data_utils.create_df(id_file)
print(f"ID data shape: {id_data.shape}")
print(f"ID columns: {list(id_data.columns)}")

# %%
