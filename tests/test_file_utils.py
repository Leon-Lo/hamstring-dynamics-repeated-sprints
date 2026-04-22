from pathlib import Path

import pytest

from utils.file_utils import File_utils

Project_root = Path(__file__).parent.parent


class Test_file_utils:
    @pytest.fixture
    def fixture_base_dir(self):
        """Base directory for test fixtures"""
        return Path(__file__).parent / "fixtures"

    @pytest.fixture
    def file_utils_with_fixtures(self, fixture_base_dir: Path):
        """File_utils instance configured with fixture directories"""
        return File_utils(
            sim_base_dir=fixture_base_dir / "simulations",
            model_base_dir=fixture_base_dir / "models",
        )

    def test_get_sim_run_dir(self, file_utils_with_fixtures):
        sim_run_dir = file_utils_with_fixtures.get_sim_run_dir("000")
        expected_dir = Path(__file__).parent / "fixtures/simulations/000"

        assert sim_run_dir == expected_dir
        assert sim_run_dir.exists()

    def test_get_trial_dirs(self, file_utils_with_fixtures):
        trial_dirs = file_utils_with_fixtures.get_trial_dirs("000")
        expected_dirs = [
            Path(__file__).parent / "fixtures/simulations/000/pre/Run_baselineA0",
        ]

        assert trial_dirs == expected_dirs
        assert isinstance(trial_dirs, list)
        assert "pre" in str(trial_dirs[0])

    def test_get_trial_dir(self, file_utils_with_fixtures):
        trial_dir = file_utils_with_fixtures.get_trial_dir("000", "Run_baselineA0")
        expected_dir = (
            Path(__file__).parent / "fixtures/simulations/000/pre/Run_baselineA0"
        )

        assert trial_dir == expected_dir
        assert trial_dir.exists()
        assert isinstance(trial_dir, Path)

    def test_get_output_dir(self, file_utils_with_fixtures):
        output_dir = file_utils_with_fixtures.get_output_dir("000", "Run_baselineA0")
        expected_dir = (
            Path(__file__).parent / "fixtures/simulations/000/pre/Run_baselineA0"
        )

        assert output_dir == expected_dir
        assert output_dir.exists()
        assert isinstance(output_dir, Path)

    def test_get_model_path(self, mocker):
        # Create a file_utils instance
        file_utils = File_utils(model_base_dir=Path("/fake/models"))

        # Mock the is_dir method to return True
        mocker.patch.object(Path, "is_dir", return_value=True)

        mocker.patch.object(Path, "is_file", return_value=True)

        # Mock the glob method to return a list of fake files
        mocker.patch.object(
            Path,
            "glob",
            return_value=[Path("/fake/models/010_model.osim")],
        )

        # Test the function
        model_path = file_utils.get_model_path("010")
        expected_path = Path("/fake/models/010_model.osim")

        # Assert expected path (without needing real files)
        assert model_path == expected_path

    def test_get_ik_file_path(self, file_utils_with_fixtures):
        ik_file_path = file_utils_with_fixtures.get_ik_file_path(
            "000", "Run_baselineA0"
        )
        expected_path = (
            Path(__file__).parent / "fixtures/simulations/000/pre/Run_baselineA0/ik.mot"
        )

        assert ik_file_path == expected_path
        assert ik_file_path.exists()
        assert isinstance(ik_file_path, Path)

    def test_get_id_file_path(self, file_utils_with_fixtures):
        id_file_path = file_utils_with_fixtures.get_id_file_path(
            "000", "Run_baselineA0"
        )
        expected_path = (
            Path(__file__).parent / "fixtures/simulations/000/pre/Run_baselineA0/id.sto"
        )

        assert id_file_path == expected_path
        assert id_file_path.exists()
        assert isinstance(id_file_path, Path)

    def test_get_so_forces_file_path(self, file_utils_with_fixtures):
        so_forces_file_path = file_utils_with_fixtures.get_so_forces_file_path(
            "000",
            "Run_baselineA0",
        )
        expected_path = (
            Path(__file__).parent
            / "fixtures/simulations/000/pre/Run_baselineA0/_StaticOptimization_force.sto"
        )

        assert so_forces_file_path == expected_path
        assert so_forces_file_path.exists()
        assert isinstance(so_forces_file_path, Path)

    def test_get_mtu_length_file_path(self, mocker):
        # Create a file_utils instance
        file_utils = File_utils(sim_base_dir=Path("/fake/simulations"))

        # Mock the is_dir method to return True
        mocker.patch.object(Path, "is_dir", return_value=True)

        # Mock the glob method to return a list of fake files
        mocker.patch.object(
            Path,
            "glob",
            return_value=[Path("/fake/simulations/010/pre/trial1/MTULength.sto")],
        )

        # Test the function
        mtu_length_file_path = file_utils.get_mtu_length_file_path("010", "trial1")
        expected_path = Path("/fake/simulations/010/pre/trial1/MTULength.sto")

        # Assert expected path (without needing real files)
        assert mtu_length_file_path == expected_path

    def test_get_fiber_length_file_path(self, mocker):
        # Create a file_utils instance
        file_utils = File_utils(sim_base_dir=Path("/fake/simulations"))

        # Mock the is_dir method to return True
        mocker.patch.object(Path, "is_dir", return_value=True)

        # Mock the glob method to return a list of fake files
        mocker.patch.object(
            Path,
            "glob",
            return_value=[
                Path("/fake/simulations/010/pre/trial1/NormalizedFiberLength.sto"),
            ],
        )

        # Test the function
        fiber_length_file_path = file_utils.get_fiber_length_file_path("010", "trial1")
        expected_path = Path(
            "/fake/simulations/010/pre/trial1/NormalizedFiberLength.sto",
        )

        # Assert expected path (without needing real files)
        assert fiber_length_file_path == expected_path

    def get_norm_fiber_length_file_path(self, mocker):
        # Create a file_utils instance
        file_utils = File_utils(sim_base_dir=Path("/fake/simulations"))

        # Mock the is_dir method to return True
        mocker.patch.object(Path, "is_dir", return_value=True)

        # Mock the glob method to return a list of fake files
        mocker.patch.object(
            Path,
            "glob",
            return_value=[
                Path("/fake/simulations/010/pre/trial1/NormalizedFiberLength.sto"),
            ],
        )

        # Test the function
        norm_fiber_length_file_path = file_utils.get_norm_fiber_length_file_path(
            "010",
            "trial1",
        )
        expected_path = Path(
            "/fake/simulations/010/pre/trial1/NormalizedFiberLength.sto",
        )

        # Assert expected path (without needing real files)
        assert norm_fiber_length_file_path == expected_path

    def test_get_fiber_velocity_file_path(self, mocker):
        # Create a file_utils instance
        file_utils = File_utils(sim_base_dir=Path("/fake/simulations"))

        # Mock the is_dir method to return True
        mocker.patch.object(Path, "is_dir", return_value=True)

        # Mock the glob method to return a list of fake files
        mocker.patch.object(
            Path,
            "glob",
            return_value=[
                Path("/fake/simulations/010/pre/trial1/NormalizedFiberVelocity.sto"),
            ],
        )

        # Test the function
        fiber_velocity_file_path = file_utils.get_fiber_velocity_file_path(
            "010",
            "trial1",
        )
        expected_path = Path(
            "/fake/simulations/010/pre/trial1/NormalizedFiberVelocity.sto",
        )

        # Assert expected path (without needing real files)
        assert fiber_velocity_file_path == expected_path

    def test_get_fiber_force_file_path(self, mocker):
        # Create a file_utils instance
        file_utils = File_utils(sim_base_dir=Path("/fake/simulations"))

        # Mock the is_dir method to return True
        mocker.patch.object(Path, "is_dir", return_value=True)

        # Mock the glob method to return a list of fake files
        mocker.patch.object(
            Path,
            "glob",
            return_value=[Path("/fake/simulations/010/pre/trial1/FiberForce.sto")],
        )

        # Test the function
        fiber_force_file_path = file_utils.get_fiber_force_file_path("010", "trial1")
        expected_path = Path("/fake/simulations/010/pre/trial1/FiberForce.sto")

        # Assert expected path (without needing real files)
        assert fiber_force_file_path == expected_path

    def test_get_muscle_activation_file_path(self, mocker):
        # Create a file_utils instance
        file_utils = File_utils(sim_base_dir=Path("/fake/simulations"))

        # Mock the is_dir method to return True
        mocker.patch.object(Path, "is_dir", return_value=True)

        # Mock the glob method to return a list of fake files
        mocker.patch.object(
            Path,
            "glob",
            return_value=[
                Path("/fake/simulations/010/pre/trial1/MuscleActivation.sto"),
            ],
        )

        # Test the function
        muscle_activation_file_path = file_utils.get_muscle_activation_file_path(
            "010",
            "trial1",
        )
        expected_path = Path("/fake/simulations/010/pre/trial1/MuscleActivation.sto")

        # Assert expected path (without needing real files)
        assert muscle_activation_file_path == expected_path

    def test_get_ceinms_setup_file_path(self, file_utils_with_fixtures):
        ceinms_setup_file_path = file_utils_with_fixtures.get_ceinms_setup_file_path(
            "000",
            "Run_baselineA0",
        )
        expected_path = (
            Path(__file__).parent
            / "fixtures/simulations/000/pre/Run_baselineA0/setup_ceinms_optimise.xml"
        )

        assert ceinms_setup_file_path == expected_path
        assert ceinms_setup_file_path.exists()
