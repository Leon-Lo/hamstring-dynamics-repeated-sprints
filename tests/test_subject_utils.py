from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from utils.subject_utils import Subject_utils


class Test_subject_utils:
    @pytest.fixture
    # Create temporary subject demographics DataFrame
    def temp_subject_demographics_csv(self, tmp_path: Path):
        # Create a temporary CSV file for testing
        file_path = tmp_path / "subject_demographics.csv"
        data = {
            "Subject": ["1", "2", "3"],
            "Measured Leg": ["left", "right", "left"],
        }
        try:
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            yield file_path
        finally:
            if file_path.exists():
                file_path.unlink()

    @pytest.fixture
    def temp_subject_processing_csv(self, tmp_path: Path):
        # Create a temporary CSV file for testing
        file_path = tmp_path / "subject_processing.csv"
        data = {
            "Labels meaning": [
                "Run_baselineA1",
                "Run_baselineB1",
                "RunA1",
                "RunB1",
            ],
            "s001": [
                "NA",
                "",
                "",
                "NH",
            ],
            "s002": [
                "",
                "",
                "",
                "",
            ],
            "s003": [
                "",
                "",
                "NA",
                "",
            ],
        }
        try:
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            yield file_path
        finally:
            if file_path.exists():
                file_path.unlink()

    def test_get_measured_leg(self, temp_subject_demographics_csv):
        # Test getting measured leg for a valid subject ID
        subject_utils = Subject_utils()
        measured_leg_001 = subject_utils.get_measured_leg(
            "001",
            temp_subject_demographics_csv,
        )
        assert measured_leg_001 == "left"
        measured_leg_002 = subject_utils.get_measured_leg(
            "002",
            temp_subject_demographics_csv,
        )
        assert measured_leg_002 == "right"
        measured_leg_003 = subject_utils.get_measured_leg(
            "003",
            temp_subject_demographics_csv,
        )
        assert measured_leg_003 == "left"

        with pytest.raises(ValueError):
            subject_utils.get_measured_leg("one", temp_subject_demographics_csv)

        with pytest.raises(ValueError):
            subject_utils.get_measured_leg("004", temp_subject_demographics_csv)

    @patch("utils.subject_utils.File_utils")
    def test_get_last_trial(self, mock_file_utils, temp_subject_processing_csv):
        # Mock the get_trial_dir method to not raise FileNotFoundError
        mock_file_utils.return_value.get_trial_dir.return_value = Path("/mock/path")

        # Test getting the last trial for a valid subject ID
        subject_utils = Subject_utils()
        last_trial_001 = subject_utils.get_last_trial(
            "001",
            temp_subject_processing_csv,
        )
        last_trial_002 = subject_utils.get_last_trial(
            "002",
            temp_subject_processing_csv,
        )
        last_trial_003 = subject_utils.get_last_trial(
            "003",
            temp_subject_processing_csv,
        )
        assert last_trial_001 == "RunA1"
        assert last_trial_002 == "RunB1"
        assert last_trial_003 == "RunB1"
