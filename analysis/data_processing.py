"""
Data Processing and Results Generation.

=======================================
Create master results CSVs and process raw data for analysis.
Run these before generating plots.
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

from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src import OpensimModelManager, Simulation_analyzer  # noqa: E402
from utils import Analysis_utils, Subject_utils  # noqa: E402

analysis_utils = Analysis_utils()
subject_utils = Subject_utils()
tools = Simulation_analyzer()

# PARALLELIZATION SETTINGS
# Increase max_workers (1-3) for faster processing if your system can handle it
# 1 = safe/sequential, 2-3 = faster but more resource intensive
MAX_WORKERS = 1  # Default to 1 worker (safe, no crashes)

# %% Get all subject IDs
subject_ids = OpensimModelManager.get_ids()
print(f"Found {len(subject_ids)} subjects: {subject_ids}")

# %% ==================== GENERATE MASTER RESULTS ====================

# %% Create master results CSV (time normalized)
print("Creating master results CSV (time normalized)...")
master_results = analysis_utils.create_master_results_df()
save_path = Path("data/master_results.csv")
master_results.to_csv(save_path, index=False)
print(f"✓ Saved to: {save_path.absolute()}")
print(f"  Shape: {master_results.shape}")
print(f"  Columns: {list(master_results.columns)}")

# %% Create master results CSV (no time normalization)
print("\nCreating master results CSV (no time normalization)...")
master_results_no_norm = analysis_utils.create_master_results_df_no_time_norm()
save_path_no_norm = Path("data/master_results_no_time_norm.csv")
master_results_no_norm.to_csv(save_path_no_norm, index=False)
print(f"✓ Saved to: {save_path_no_norm.absolute()}")
print(f"  Shape: {master_results_no_norm.shape}")

# %% ==================== CREATE SPLIT RESULTS ====================

# %% Create split results files (various metrics)
print("\nCreating split results files...")
analysis_utils.create_split_results()
analysis_utils.create_torques_results_wide_df()
print("✓ Split results created in data/split_results/")

# %% Create muscle parameters CSV
print("\nExtracting muscle parameters...")
muscle_params = analysis_utils.get_muscle_parameters()
muscle_params_path = Path("data/muscle_parameters.csv")
muscle_params.to_csv(muscle_params_path, index=False)
print(f"✓ Saved to: {muscle_params_path.absolute()}")
print(f"  Shape: {muscle_params.shape}")

# %% Create peak normalized fiber length CSV
print("\nCalculating peak normalized fiber lengths...")
peak_norm_fiber_length = analysis_utils.get_peak_norm_fiber_length_df(
    muscles=["biceps_femoris_long_head", "semitendinosus", "semimembranosus"],
)
peak_norm_fiber_length_path = Path("data/peak_norm_fiber_length.csv")
peak_norm_fiber_length.to_csv(peak_norm_fiber_length_path, index=False)
print(f"✓ Saved to: {peak_norm_fiber_length_path.absolute()}")
print(f"  Shape: {peak_norm_fiber_length.shape}")

# %% ==================== VERIFY DATA ====================

# %% Check what data files exist
print("\n" + "=" * 60)
print("EXISTING DATA FILES")
print("=" * 60)

data_dir = Path("data")
csv_files = list(data_dir.glob("*.csv"))
try:
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(f"\n{csv_file.name}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {len(df.columns)}")
except FileNotFoundError:
    print("No CSV files found in data/ directory. Please run the processing steps above.")
print("\n" + "=" * 60)
print("Data processing complete!")
print("=" * 60)

# %% ==================== RUN OPENSIM ANALYSES ====================

# %% Run IK for all subjects (SLOW - only if needed)
# WARNING: This will take a long time!
run_ik = True  # Set to True to run

if run_ik:

    def run_ik_for_subject(subject_id):
        # Run IK for all trials of a subject.""
        subject_utils = Subject_utils()
        tools = Simulation_analyzer()
        errors = []

        trials = subject_utils.get_all_relevant_trials(subject_id)
        try:
            for trial in trials:
                tools.run_specific_tools(subject_id, trial, "ik")
        except RuntimeError:
            errors.append(f"{subject_id}: IK failed for one or more trials")

        return subject_id, errors

    print("Running Inverse Kinematics for all subjects...")
    workers = min(MAX_WORKERS, len(subject_ids))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_ik_for_subject, subj): subj for subj in subject_ids}

        for future in tqdm(
            as_completed(futures),
            total=len(subject_ids),
            desc="Running IK",
        ):
            subject_id, errors = future.result()
            if errors:
                for error in errors:
                    print(f"  Error: {error}")

    print("✓ IK complete")

# %% Run ID for all subjects (SLOW - only if needed)
run_id = True  # Set to True to run

if run_id:

    def run_id_for_subject(subject_id):
        # Run ID for all trials of a subject.
        subject_utils = Subject_utils()
        tools = Simulation_analyzer()
        errors = []

        trials = subject_utils.get_all_relevant_trials(subject_id)
        try:
            for trial in trials:
                tools.run_specific_tools(subject_id, trial, "id")
        except RuntimeError:
            errors.append(f"{subject_id}: ID failed for one or more trials")

        return subject_id, errors

    print("Running Inverse Dynamics for all subjects...")
    workers = min(MAX_WORKERS, len(subject_ids))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_id_for_subject, subj): subj for subj in subject_ids}

        for future in tqdm(
            as_completed(futures),
            total=len(subject_ids),
            desc="Running ID",
        ):
            subject_id, errors = future.result()
            if errors:
                for error in errors:
                    print(f"  Error: {error}")

    print("✓ ID complete")

# %% Run Muscle Analysis for all subjects (VERY SLOW - only if needed)
run_ma = True  # Set to True to run

if run_ma:

    def run_ma_for_subject(subject_id):
        # Run Muscle Analysis for all trials of a subject

        subject_utils = Subject_utils()
        tools = Simulation_analyzer()
        errors = []

        trials = subject_utils.get_all_relevant_trials(subject_id)
        try:
            for trial in trials:
                tools.run_specific_tools(subject_id, trial, "ma")
        except RuntimeError:
            errors.append(f"{subject_id}: MA failed for one or more trials")

        return subject_id, errors

    print("Running Muscle Analysis for all subjects...")
    workers = min(MAX_WORKERS, len(subject_ids))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_ma_for_subject, subj): subj for subj in subject_ids}

        for future in tqdm(
            as_completed(futures),
            total=len(subject_ids),
            desc="Running MA",
        ):
            subject_id, errors = future.result()
            if errors:
                for error in errors:
                    print(f"  Error: {error}")

    print("✓ Muscle Analysis complete")

# %% ==================== SINGLE SUBJECT TESTING ====================

# %% Test specific tool on single subject
TEST_SUBJECT = "022"
TEST_TRIAL = "Run_baselineA1"

# %% Run IK for test subject
ik_result = tools.run_specific_tools(TEST_SUBJECT, TEST_TRIAL, "ik")
print(f"IK for {TEST_SUBJECT} - {TEST_TRIAL}: {ik_result}")

# %% Run ID for test subjectabs_
id_result = tools.run_specific_tools(TEST_SUBJECT, TEST_TRIAL, "id")
print(f"ID for {TEST_SUBJECT} - {TEST_TRIAL}: {id_result}")

# %% Run MA for test subject
ma_result = tools.run_specific_tools(TEST_SUBJECT, TEST_TRIAL, "ma")
print(f"MA for {TEST_SUBJECT} - {TEST_TRIAL}: {ma_result}")

# %% Calculate marker errors for test subject
marker_errors = analysis_utils.calculate_marker_errors(TEST_SUBJECT, TEST_TRIAL)
print(f"\nMarker errors for {TEST_SUBJECT} - {TEST_TRIAL}:")
print(marker_errors)

# %% Run special Tools with free parameters for test subject
model = Path("data/models")
