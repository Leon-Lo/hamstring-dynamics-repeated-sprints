from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.analysis_utils import Analysis_utils
from utils.data_utils import Data_utils


class Test_analysis_utils:
    @pytest.fixture
    def temp_csv_file(self, tmp_path: Path):
        # Create a temporary CSV file for testing
        file_path = tmp_path / "test.csv"
        # header tuples
        columns = [
            "time",
            "addbrev_r",
            "addlong_r",
            "addmagDist_r",
            "addmagIsch_r",
            "addmagMid_r",
            "addmagProx_r",
            "bflh_r",
            "bfsh_r",
            "edl_r",
            "ehl_r",
            "fdl_r",
            "fhl_r",
            "gaslat_r",
            "gasmed_r",
            "glmax1_r",
            "glmax2_r",
            "glmax3_r",
            "glmed1_r",
            "glmed2_r",
            "glmed3_r",
            "glmin1_r",
            "glmin2_r",
            "glmin3_r",
            "grac_r",
            "iliacus_r",
            "perbrev_r",
            "perlong_r",
            "piri_r",
            "psoas_r",
            "recfem_r",
            "sart_r",
            "semimem_r",
            "semiten_r",
            "soleus_r",
            "tfl_r",
            "tibant_r",
            "tibpost_r",
            "vasint_r",
            "vaslat_r",
            "vasmed_r",
            "addbrev_l",
            "addlong_l",
            "addmagDist_l",
            "addmagIsch_l",
            "addmagMid_l",
            "addmagProx_l",
            "bflh_l",
            "bfsh_l",
            "edl_l",
            "ehl_l",
            "fdl_l",
            "fhl_l",
            "gaslat_l",
            "gasmed_l",
            "glmax1_l",
            "glmax2_l",
            "glmax3_l",
            "glmed1_l",
            "glmed2_l",
            "glmed3_l",
            "glmin1_l",
            "glmin2_l",
            "glmin3_l",
            "grac_l",
            "iliacus_l",
            "perbrev_l",
            "perlong_l",
            "piri_l",
            "psoas_l",
            "recfem_l",
            "sart_l",
            "semimem_l",
            "semiten_l",
            "soleus_l",
            "tfl_l",
            "tibant_l",
            "tibpost_l",
            "vasint_l",
            "vaslat_l",
            "vasmed_l",
        ]
        num_rows = 100
        time = np.linspace(0, 10, num_rows)
        df = pd.DataFrame({"time": time})
        for col in columns[1:]:
            df[col] = np.random.uniform(-10, 100, num_rows)

        try:
            df.to_csv(file_path, index=False, sep="\t", header=True)
            yield file_path
        finally:
            # Cleanup the temporary file
            if file_path.exists():
                file_path.unlink()

    @pytest.fixture
    def mock_opensim_model_manager(self, monkeypatch):
        class MockOpensimModelManager:
            def __init__(self, id):
                self.id = id

            def get_muscle_groups(self):
                return {
                    "right_leg": [
                        "addbrev_r",
                        "addlong_r",
                        "addmagDist_r",
                        "addmagIsch_r",
                        "addmagMid_r",
                        "addmagProx_r",
                        "bflh_r",
                        "bfsh_r",
                        "edl_r",
                        "ehl_r",
                        "fdl_r",
                        "fhl_r",
                        "gaslat_r",
                        "gasmed_r",
                        "glmax1_r",
                        "glmax2_r",
                        "glmax3_r",
                        "glmed1_r",
                        "glmed2_r",
                        "glmed3_r",
                        "glmin1_r",
                        "glmin2_r",
                        "glmin3_r",
                        "grac_r",
                        "iliacus_r",
                        "perbrev_r",
                        "perlong_r",
                        "piri_r",
                        "psoas_r",
                        "recfem_r",
                        "sart_r",
                        "semimem_r",
                        "semiten_r",
                        "soleus_r",
                        "tfl_r",
                        "tibant_r",
                        "tibpost_r",
                        "vasint_r",
                        "vaslat_r",
                        "vasmed_r",
                    ],
                    "left_leg": [
                        "addbrev_l",
                        "addlong_l",
                        "addmagDist_l",
                        "addmagIsch_l",
                        "addmagMid_l",
                        "addmagProx_l",
                        "bflh_l",
                        "bfsh_l",
                        "edl_l",
                        "ehl_l",
                        "fdl_l",
                        "fhl_l",
                        "gaslat_l",
                        "gasmed_l",
                        "glmax1_l",
                        "glmax2_l",
                        "glmax3_l",
                        "glmed1_l",
                        "glmed2_l",
                        "glmed3_l",
                        "glmin1_l",
                        "glmin2_l",
                        "glmin3_l",
                        "grac_l",
                        "iliacus_l",
                        "perbrev_l",
                        "perlong_l",
                        "piri_l",
                        "psoas_l",
                        "recfem_l",
                        "sart_l",
                        "semimem_l",
                        "semiten_l",
                        "soleus_l",
                        "tfl_l",
                        "tibant_l",
                        "tibpost_l",
                        "vasint_l",
                        "vaslat_l",
                        "vasmed_l",
                    ],
                    "hip_add_r": [
                        "addbrev_r",
                        "addlong_r",
                        "addmagDist_r",
                        "addmagIsch_r",
                        "addmagMid_r",
                        "addmagProx_r",
                        "bflh_r",
                        "grac_r",
                        "semimem_r",
                        "semiten_r",
                    ],
                    "hip_abd_r": [
                        "glmax1_r",
                        "glmed1_r",
                        "glmed2_r",
                        "glmed3_r",
                        "glmin1_r",
                        "glmin2_r",
                        "glmin3_r",
                        "piri_r",
                        "sart_r",
                        "tfl_r",
                    ],
                    "hip_flex_r": [
                        "addbrev_r",
                        "addlong_r",
                        "glmin1_r",
                        "grac_r",
                        "iliacus_r",
                        "psoas_r",
                        "recfem_r",
                        "sart_r",
                        "tfl_r",
                    ],
                    "hip_ext_r": [
                        "addlong_r",
                        "addmagDist_r",
                        "addmagIsch_r",
                        "addmagMid_r",
                        "addmagProx_r",
                        "bflh_r",
                        "glmax1_r",
                        "glmax2_r",
                        "glmax3_r",
                        "glmed1_r",
                        "glmed2_r",
                        "glmed3_r",
                        "glmin3_r",
                        "semimem_r",
                        "semiten_r",
                    ],
                    "hip_inrot_r": [
                        "glmed1_r",
                        "glmed2_r",
                        "glmed3_r",
                        "glmin1_r",
                        "iliacus_r",
                        "psoas_r",
                        "tfl_r",
                    ],
                    "hip_exrot_r": ["glmin3_r", "piri_r"],
                    "knee_flex_r": [
                        "bflh_r",
                        "bfsh_r",
                        "gaslat_r",
                        "gasmed_r",
                        "grac_r",
                        "sart_r",
                        "semimem_r",
                        "semiten_r",
                    ],
                    "knee_ext_r": ["recfem_r", "vasint_r", "vaslat_r", "vasmed_r"],
                    "ankle_df_r": ["edl_r", "ehl_r", "tibant_r"],
                    "ankle_pf_r": [
                        "fdl_r",
                        "fhl_r",
                        "gaslat_r",
                        "gasmed_r",
                        "perbrev_r",
                        "perlong_r",
                        "soleus_r",
                        "tibpost_r",
                    ],
                    "everter_r": ["edl_r", "perbrev_r", "perlong_r"],
                    "inverter_r": ["ehl_r", "fdl_r", "fhl_r", "tibant_r", "tibpost_r"],
                    "hip_add_l": [
                        "addbrev_l",
                        "addlong_l",
                        "addmagDist_l",
                        "addmagIsch_l",
                        "addmagMid_l",
                        "addmagProx_l",
                        "bflh_l",
                        "grac_l",
                        "semimem_l",
                        "semiten_l",
                    ],
                    "hip_abd_l": [
                        "glmax1_l",
                        "glmed1_l",
                        "glmed2_l",
                        "glmed3_l",
                        "glmin1_l",
                        "glmin2_l",
                        "glmin3_l",
                        "piri_l",
                        "sart_l",
                        "tfl_l",
                    ],
                    "hip_flex_l": [
                        "addbrev_l",
                        "addlong_l",
                        "glmin1_l",
                        "grac_l",
                        "iliacus_l",
                        "psoas_l",
                        "recfem_l",
                        "sart_l",
                        "tfl_l",
                    ],
                    "hip_ext_l": [
                        "addlong_l",
                        "addmagDist_l",
                        "addmagIsch_l",
                        "addmagMid_l",
                        "addmagProx_l",
                        "bflh_l",
                        "glmax1_l",
                        "glmax2_l",
                        "glmax3_l",
                        "glmed1_l",
                        "glmed2_l",
                        "glmed3_l",
                        "glmin3_l",
                        "semimem_l",
                        "semiten_l",
                    ],
                    "hip_exrot_l": ["glmin3_l", "piri_l"],
                    "hip_inrot_l": [
                        "glmed1_l",
                        "glmed2_l",
                        "glmed3_l",
                        "glmin1_l",
                        "iliacus_l",
                        "psoas_l",
                        "tfl_l",
                    ],
                    "knee_flex_l": [
                        "bflh_l",
                        "bfsh_l",
                        "gaslat_l",
                        "gasmed_l",
                        "grac_l",
                        "sart_l",
                        "semimem_l",
                        "semiten_l",
                    ],
                    "knee_ext_l": ["recfem_l", "vasint_l", "vaslat_l", "vasmed_l"],
                    "ankle_df_l": ["edl_l", "ehl_l", "tibant_l"],
                    "ankle_pf_l": [
                        "fdl_l",
                        "fhl_l",
                        "gaslat_l",
                        "gasmed_l",
                        "perbrev_l",
                        "perlong_l",
                        "soleus_l",
                        "tibpost_l",
                    ],
                    "everter_l": ["edl_l", "perbrev_l", "perlong_l"],
                    "inverter_l": ["ehl_l", "fdl_l", "fhl_l", "tibant_l", "tibpost_l"],
                }

        monkeypatch.setattr(
            "utils.analysis_utils.OpensimModelManager",
            MockOpensimModelManager,
        )
        return MockOpensimModelManager

    def test_time_normalise_df(self, temp_csv_file: Path):
        df = Data_utils().create_df(temp_csv_file)
        normalised_df = Analysis_utils().time_normalise_df(df)
        assert isinstance(normalised_df, pd.DataFrame)
        assert normalised_df.shape == (101, df.shape[1])

    def test_get_muscle_groupmean(self, temp_csv_file, mock_opensim_model_manager):
        test_id = "test"
        df = Data_utils().create_df(temp_csv_file)
        mean_df = Analysis_utils().get_muscle_group_mean(df, test_id)

        mock_manager = mock_opensim_model_manager(test_id)

        assert isinstance(mean_df, pd.DataFrame)
        assert mean_df.shape[0] == 100
        assert mean_df.shape[1] == len(mock_manager.get_muscle_groups()) + 1
        assert "time" in mean_df.columns
        assert all(col.endswith("_mean") for col in mean_df.columns if col != "time")

    def test_get_muscle_group_sum(self, temp_csv_file, mock_opensim_model_manager):
        test_id = "test"
        df = Data_utils().create_df(temp_csv_file)
        sum_df = Analysis_utils().get_muscle_group_sum(df, test_id)

        mock_manager = mock_opensim_model_manager(test_id)

        assert isinstance(sum_df, pd.DataFrame)
        assert sum_df.shape[0] == 100
        assert sum_df.shape[1] == len(mock_manager.get_muscle_groups()) + 1
        assert "time" in sum_df.columns
        assert all(col.endswith("_sum") for col in sum_df.columns if col != "time")

    def test_get_pos_and_neg_fiber_velocity_index(self, tmp_path, mocker):
        test_id = "test"
        test_trial = "test_trial"
        test_muscle = "psoas_r"
        fixture_dir = Path("tests/fixtures")
        ma_dir = fixture_dir / "simulations" / "000" / "pre" / "Run_baselineA0" / "ma"

        test_mtu_file = ma_dir / "_MuscleAnalysis_Length.sto"
        test_fl_path = ma_dir / "_MuscleAnalysis_Length.sto"
        test_fv_path = ma_dir / "_MuscleAnalysis_FiberVelocity.sto"
        test_ff_file = ma_dir / "_MuscleAnalysis_FiberForce.sto"

        mocker.patch(
            "utils.file_utils.File_utils.get_mtu_length_file_path",
            return_value=test_mtu_file,
        )
        mocker.patch(
            "utils.file_utils.File_utils.get_fiber_length_file_path",
            return_value=test_fl_path,
        )
        mocker.patch(
            "utils.file_utils.File_utils.get_fiber_velocity_file_path",
            return_value=test_fv_path,
        )
        mocker.patch(
            "utils.file_utils.File_utils.get_fiber_force_file_path",
            return_value=test_ff_file,
        )

        results = Analysis_utils().get_pos_and_neg_fiber_velocity_index(
            test_id,
            test_trial,
            test_muscle,
        )

        assert isinstance(results, dict)
        assert "positive_index" in results
        assert "negative_index" in results

    def test_get_muscle_work_single_muscle(self, tmp_path, mocker):
        test_id = "test"
        test_trial = "test_trial"
        test_muscle = "psoas_r"
        fixture_dir = Path("tests/fixtures")
        ma_dir = fixture_dir / "simulations" / "000" / "pre" / "Run_baselineA0" / "ma"

        test_mtu_file = ma_dir / "_MuscleAnalysis_Length.sto"
        test_fl_path = ma_dir / "_MuscleAnalysis_FiberLength.sto"
        test_fv_path = ma_dir / "_MuscleAnalysis_FiberVelocity.sto"
        test_ff_file = ma_dir / "_MuscleAnalysis_FiberForce.sto"

        mocker.patch(
            "utils.file_utils.File_utils.get_mtu_length_file_path",
            return_value=test_mtu_file,
        )
        mocker.patch(
            "utils.file_utils.File_utils.get_fiber_length_file_path",
            return_value=test_fl_path,
        )
        mocker.patch(
            "utils.file_utils.File_utils.get_fiber_velocity_file_path",
            return_value=test_fv_path,
        )
        mocker.patch(
            "utils.file_utils.File_utils.get_fiber_force_file_path",
            return_value=test_ff_file,
        )

        result = Analysis_utils().get_muscle_work_single_muscle(
            test_id,
            test_trial,
            test_muscle,
        )
        print(result)
        assert isinstance(result, dict)
        assert "positive_work" in result
        assert "negative_work" in result
        assert len(result) == 2
        assert isinstance(result["positive_work"], float)
        assert isinstance(result["negative_work"], float)
