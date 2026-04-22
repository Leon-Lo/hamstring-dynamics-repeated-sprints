from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.data_utils import Data_utils


class Test_data_utils:
    @pytest.fixture
    def temp_trc_file(self, tmp_path: Path):
        file_path = tmp_path / "test.trc"
        # Create a temporary TRC file for testing
        with open(file_path, "w") as file:
            with open(file_path, "w") as file:
                file.write(
                    "PathFileType\t4\t(X/Y/Z)\tC:/path/to/sample_file.trc\n"
                    "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits	OrigDataRate\tDataStartFrame\n"
                    "100\t100\t10\t5\tmm\t100\t1\n"
                    "endheader\n"
                    "Frame#\tTime\tMarker1\t\t\tMarker2\t\t\tMarker3\t\t\tMarker4\t\t\tMarker5\t\t\n"
                    "\t\tX1\tY1\tZ1\tX2\tY2\tZ2\tX3\tY3\tZ3\tX4\tY4\tZ4\tX5\tY5\tZ5\n"
                    "1\t0.010000\t100.000000\t200.000000\t300.000000\t150.000000\t250.000000\t350.000000\t200.000000\t300.000000\t400.000000\t250.000000\t350.000000\t450.000000\t300.000000\t400.000000\t500.000000\n"
                    "2\t0.020000\t101.000000\t201.000000\t301.000000\t151.000000\t251.000000\t351.000000\t201.000000\t301.000000\t401.000000\t251.000000\t351.000000\t451.000000\t301.000000\t401.000000\t501.000000\n"
                    "3\t0.030000\t102.000000\t202.000000\t302.000000\t152.000000\t252.000000\t352.000000\t202.000000\t302.000000\t402.000000\t252.000000\t352.000000\t452.000000\t302.000000\t402.000000\t502.000000\n"
                    "4\t0.040000\t103.000000\t203.000000\t303.000000\t153.000000\t253.000000\t353.000000\t203.000000\t303.000000\t403.000000\t253.000000\t353.000000\t453.000000\t303.000000\t403.000000\t503.000000\n"
                    "5\t0.050000\t104.000000\t204.000000\t304.000000\t154.000000\t254.000000\t354.000000\t204.000000\t304.000000\t404.000000\t254.000000\t354.000000\t454.000000\t304.000000\t404.000000\t504.000000\n"
                    "6\t0.060000\t105.000000\t205.000000\t305.000000\t155.000000\t255.000000\t355.000000\t205.000000\t305.000000\t405.000000\t255.000000\t355.000000\t455.000000\t305.000000\t405.000000\t505.000000\n"
                    "7\t0.070000\t106.000000\t206.000000\t306.000000\t156.000000\t256.000000\t356.000000\t206.000000\t306.000000\t406.000000\t256.000000\t356.000000\t456.000000\t306.000000\t406.000000\t506.000000\n"
                    "8\t0.080000\t107.000000\t207.000000\t307.000000\t157.000000\t257.000000\t357.000000\t207.000000\t307.000000\t407.000000\t257.000000\t357.000000\t457.000000\t307.000000\t407.000000\t507.000000\n"
                    "9\t0.090000\t108.000000\t208.000000\t308.000000\t158.000000\t258.000000\t358.000000\t208.000000\t308.000000\t408.000000\t258.000000\t358.000000\t458.000000\t308.000000\t408.000000\t508.000000\n"
                    "10\t0.100000\t109.000000\t209.000000\t309.000000\t159.000000\t259.000000\t359.000000\t209.000000\t309.000000\t409.000000\t259.000000\t359.000000\t459.000000\t309.000000\t409.000000\t509.000000\n",
                )
        yield file_path
        # Cleanup the temporary file
        if file_path.exists():
            file_path.unlink()

    @pytest.fixture
    def temp_csv_file(self, tmp_path: Path):
        # Create a temporary CSV file for testing
        file_path = tmp_path / "test.csv"
        # header tuples
        df = pd.DataFrame(
            {
                "time": np.arange(0, 10, 0.1),
                "test_value1": np.random.rand(100),
                "test_value2": np.random.rand(100),
            },
        )
        original_columns = df.columns
        new_header_tuple = [("endheader", col) for col in original_columns]
        multi_index_header = pd.MultiIndex.from_tuples(new_header_tuple)
        df.columns = multi_index_header
        try:
            df.to_csv(file_path, index=False, sep="\t", header=True)
            yield file_path
        finally:
            # Cleanup the temporary file
            if file_path.exists():
                file_path.unlink()

    def test_create_df(self, temp_csv_file: Path):
        df = Data_utils().create_df(temp_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert "time" in df.columns
        assert "test_value1" in df.columns
        assert "test_value2" in df.columns
        assert df.shape[0] == 100
        assert df.shape[1] == 3

    def test_get_skiprows_count(self, temp_trc_file: Path):
        data_utils = Data_utils()
        skiprows_count = data_utils.get_skiprows_count(temp_trc_file)
        assert isinstance(skiprows_count, int)
        assert skiprows_count == 4

    def test_get_skiprows_content(self, temp_trc_file: Path):
        data_utils = Data_utils()
        skiprows_content = data_utils.get_skiprows_content(temp_trc_file)
        assert isinstance(skiprows_content, list)
        assert len(skiprows_content) == 4
        assert (
            skiprows_content[0]
            == "PathFileType\t4\t(X/Y/Z)\tC:/path/to/sample_file.trc"
        )
        assert (
            skiprows_content[1]
            == "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits	OrigDataRate\tDataStartFrame"
        )
        assert skiprows_content[2] == "100\t100\t10\t5\tmm\t100\t1"
        assert skiprows_content[3] == "endheader"

    def test_get_endheader(self, temp_trc_file: Path):
        endheader = Data_utils()
        get_endheader = endheader.get_skiprows_count(temp_trc_file)
        assert isinstance(get_endheader, int)
        assert get_endheader == 4

    def test_trc_header_mod(self, temp_trc_file: Path):
        trc = Data_utils()
        modified_df = trc.trc_header_mod(temp_trc_file)
        assert isinstance(modified_df, pd.DataFrame)
        assert (
            modified_df.iloc[1, 0]
            == "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
        )
        assert modified_df.iloc[2, 0] == "100\t100\t10\t5\tmm\t100\t1\t10"
