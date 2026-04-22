from pathlib import Path

import pandas as pd

from utils.file_utils import File_utils


class Subject_utils:
    """
    A utility class for managing subject-related information and operations.

    This class provides methods for retrieving subject IDs, measured legs,
    trial information, and other subject-specific metadata.
    """

    def __init__(self):
        """Initialize the Subject_utils instance."""

    @staticmethod
    def get_all_subjects(
        base_path: Path | None = None,
        use_leading_zeros: bool = True,
    ) -> list:
        """
        Get a list of all subject IDs from the simulations directory.

        Returns:
            Sorted list of subject ID strings

        Note:
            Excludes the '000' directory (template/test directory)

        """
        # Define the base path where subject directories are stored
        if base_path is None:
            base_path = Path("data/simulations")
        if not base_path.exists():
            raise FileNotFoundError(f"Base path {base_path} does not exist.")
        subject_list = []

        # Iterate through all directories in the simulations folder
        for subject_dir in base_path.iterdir():
            if subject_dir.is_dir():
                # Skip the template directory
                if subject_dir.name == "000":
                    continue  # Skip the '000' directory
                # Add valid subject directory names to the list
                if not use_leading_zeros:
                    if subject_dir.name.startswith("0"):
                        subject_dir_name = subject_dir.name.lstrip("0")
                        subject_list.append(subject_dir_name)
                else:
                    subject_list.append(subject_dir.name)

        # Return sorted list for consistent ordering
        return sorted(subject_list)

    def get_measured_leg(
        self,
        subject_id: str,
        subject_demographics_csv: Path | None = None,
    ) -> str:
        """
        Get the measured leg (left or right) for a specific subject.

        Args:
            subject_id: The subject identifier
            subject_demographics_csv: Optional path to demographics CSV file

        Returns:
            String indicating the measured leg (typically 'L' or 'R')

        Raises:
            FileNotFoundError: If the demographics CSV file doesn't exist
            ValueError: If subject ID is invalid or not found in the CSV
            KeyError: If required columns are missing from the CSV

        """
        # Use default demographics file if not specified
        if subject_demographics_csv is None:
            subject_demographics_csv = Path("data/subject_demographics.csv")

        # Read the demographics CSV with error handling
        try:
            df_subject_demographics = pd.read_csv(subject_demographics_csv)
        except FileNotFoundError:  # Be specific about the exception type
            raise FileNotFoundError(
                f"Could not read subject demographics CSV at {subject_demographics_csv}. File not found.",
            )
        except pd.errors.EmptyDataError:  # Handle empty CSV files
            raise ValueError(
                f"Subject demographics CSV at {subject_demographics_csv} is empty.",
            )
        except Exception as e:  # Catch other potential pandas errors
            raise OSError(
                f"An error occurred while reading subject demographics CSV at {subject_demographics_csv}: {e}",
            )

        # Validate and normalize the subject ID
        try:
            processed_id = str(int(subject_id))  # Ensure it's a string of an integer
        except ValueError:
            raise ValueError(
                f"Invalid subject ID: '{subject_id}'. Subject ID must be convertible to an integer (e.g., '001', 1).",
            )

        # Validate that required columns exist
        required_columns = ["Subject", "Measured Leg"]
        for col in required_columns:
            if col not in df_subject_demographics.columns:
                raise KeyError(
                    f"Required column '{col}' not found in subject demographics CSV at {subject_demographics_csv}.",
                )

        # Check if the subject ID exists in the CSV
        if (
            processed_id not in df_subject_demographics["Subject"].astype(str).values
        ):  # .astype(str) for robust comparison
            raise ValueError(
                f"Subject ID '{processed_id}' not found in 'Subject' column of CSV at {subject_demographics_csv}.",
            )

        # Retrieve the measured leg value for this subject
        measured_legs = df_subject_demographics.loc[
            df_subject_demographics["Subject"].astype(str) == processed_id,
            "Measured Leg",
        ].values

        # Return the measured leg if found
        if len(measured_legs) > 0:
            return measured_legs[0]
        raise ValueError(
            f"No measured leg found for subject ID '{processed_id}' in CSV at {subject_demographics_csv}.",
        )

    def get_body_mass(self, subject_id: str) -> float:
        demographics_csv = Path("data/subject_demographics.csv")
        df_subject_demographics = pd.read_csv(demographics_csv)
        body_masses = df_subject_demographics.loc[
            df_subject_demographics["Subject"].astype(str) == str(int(subject_id)),
            "Weight",
        ].values
        if len(body_masses) > 0:
            return float(body_masses[0])
        raise ValueError(
            f"No body mass found for subject ID '{subject_id}' in CSV at {demographics_csv}.",
        )

    def get_body_height(self, subject_id: str) -> float:
        demographics_csv = Path("data/subject_demographics.csv")
        df_subject_demographics = pd.read_csv(demographics_csv)
        body_heights = df_subject_demographics.loc[
            df_subject_demographics["Subject"].astype(str) == str(int(subject_id)),
            "Height",
        ].values
        if len(body_heights) > 0:
            return float(body_heights[0])
        raise ValueError(
            f"No body height found for subject ID '{subject_id}' in CSV at {demographics_csv}.",
        )

    def get_last_trial(self, subject_id: str) -> str | None:
        """
        Get the last (final) trial for a subject, excluding baseline trials.

        Args:
            subject_id: The subject identifier

        Returns:
            Trial name string, or None if no matching trial found

        """
        # Get the session directory for this subject
        session_path = File_utils().get_session_dir(subject_id)
        trials = []

        # Iterate through folders in sorted order
        for folder in sorted(session_path.iterdir()):
            if folder.is_dir():
                # Look for trial folders that start with "Run", end with "1", and aren't baseline
                if (
                    folder.name.startswith("Run")
                    and folder.name.endswith("1")
                    and "baseline" not in folder.name.lower()
                ):
                    trials.append(folder.name)

        # Return the last trial in the sorted list, or None if empty
        return trials[-1] if trials else None

    def get_L1_trial(self, subject_id: str) -> str | None:
        """
        Get the L1 trial (first fatigue level) for a subject.

        Args:
            subject_id: The subject identifier

        Returns:
            Trial name string starting with "RunL1", or None if not found

        """
        # Get the session directory for this subject
        session_path = File_utils().get_session_dir(subject_id)
        # Search for folders starting with "RunL1"
        for folder in session_path.iterdir():
            if folder.is_dir():
                if folder.name.startswith("RunL1"):
                    return folder.name

    def get_all_baseline_trials(self, subject_id: str) -> list:
        """
        Get all baseline trials for a subject.

        Args:
            subject_id: The subject identifier

        Returns:
            List of baseline trial names

        """
        # Get the session directory for this subject
        session_path = File_utils().get_session_dir(subject_id)
        baseline_trials = []
        # Search for folders starting with "Run_baseline"
        for folder in session_path.iterdir():
            if folder.is_dir():
                if folder.name.startswith("Run_baseline"):
                    baseline_trials.append(folder.name)
        return baseline_trials

    def get_all_baseline_and_last_trial(self, subject_id: str) -> list:
        """
        Get a list of all baseline trials and the last trial for a given subject ID.

        Args:
            subject_id: The subject identifier

        Returns:
            List containing all baseline trials plus the last trial

        Raises:
            ValueError: If no last trial is found

        """
        # Get all baseline trials
        baseline_trials = self.get_all_baseline_trials(subject_id)
        # Get the final trial
        last_trial = self.get_last_trial(subject_id)

        # Validate that a last trial exists
        if last_trial is None:
            raise ValueError(
                f"Could not find last trial for subject ID '{subject_id}'.",
            )

        # Combine baseline trials with the last trial
        return baseline_trials + [last_trial]

    def get_all_relevant_trials(self, subject_id: str) -> list:
        """
        Get all relevant trials for analysis (baseline, L1, and last).

        Args:
            subject_id: The subject identifier

        Returns:
            List of trial names including baseline, L1 (if different from last), and last trial

        """
        # Get baseline trials
        baseline_trials = self.get_all_baseline_trials(subject_id)
        # Get the L1 (first fatigue level) trial
        L1_trial = self.get_L1_trial(subject_id)
        # Get the final trial
        last_trial = self.get_last_trial(subject_id)

        # If L1 and last trial are the same, don't duplicate
        if last_trial == L1_trial:
            return baseline_trials + [last_trial]
        # Include all three: baseline, L1, and last
        return baseline_trials + [L1_trial, last_trial]
