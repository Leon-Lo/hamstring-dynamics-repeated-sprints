from pathlib import Path

import pytest

from src.simulation_runner import Run_tools


class Test_runner:
    @pytest.fixture
    def mock_model_manager(self, mocker):
        mock_manager = mocker.patch("src.simulation_runner.OpensimModelManager")
        mock_instance = mocker.Mock()
        mock_manager.return_value = mock_instance
        mock_instance.model_path = Path("data/models/009_Rajagopal2015_FAI.osim")
        return mock_manager

    def test_run_ik(self, mocker, mock_model_manager):
        # Mock File_utils and its methods
        mock_file_utils = mocker.patch("src.simulation_runner.File_utils")
        mock_file_utils_instance = mocker.Mock()
        mock_file_utils.return_value = mock_file_utils_instance

        # Mock paths and directories
        mock_sim_dir = Path("mocked/sim/dir")
        mock_results_dir = mock_sim_dir
        mock_marker_file = Path("mocked/marker/file.trc")

        # Configure mocks
        mock_file_utils_instance.get_trial_dir.return_value = mock_sim_dir
        mock_file_utils_instance.get_marker_file_path.return_value = mock_marker_file

        # Mock opensim.MarkerData
        mock_marker_data = mocker.patch("src.simulation_runner.osim.MarkerData")
        mock_marker_data_instance = mocker.Mock()
        mock_marker_data.return_value = mock_marker_data_instance
        mock_marker_data_instance.getStartFrameTime.return_value = 0.0
        mock_marker_data_instance.getLastFrameTime.return_value = 1.0

        # Mock Path operations
        mocker.patch.object(Path, "exists", return_value=False)
        mock_mkdir = mocker.patch.object(Path, "mkdir")

        # Mock InverseKinematicsTool
        mock_ik = mocker.patch("src.simulation_runner.osim.InverseKinematicsTool")
        mock_ik_instance = mocker.Mock()
        mock_ik.return_value = mock_ik_instance

        # Mock XML operations
        mock_xml_utils = mocker.patch("src.simulation_runner.XML_utils")
        mock_xml_utils_instance = mocker.Mock()
        mock_xml_utils.return_value = mock_xml_utils_instance

        mock_task_set = mocker.Mock()
        mock_xml_parse_result = mocker.Mock()
        mock_xml_parse_result.find.return_value = mock_task_set
        mock_xml_utils_instance.xml_parse.return_value = mock_xml_parse_result

        # Create runner instance with our mocks
        runner = Run_tools("009", "trial1")
        runner.sim_dir = mock_sim_dir
        runner.results_dir = mock_results_dir

        # Run the method being tested
        runner.run_ik()

        # Verify directory creation
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify IK tool was configured correctly
        mock_ik_instance.setDebugLevel.assert_called_once_with(0)
        mock_ik_instance.set_model_file.assert_called_once_with(
            str(mock_model_manager.return_value.model_path),
        )
        mock_ik_instance.set_marker_file.assert_called_once_with(
            str(mock_marker_file.resolve()),
        )
        mock_ik_instance.set_output_motion_file.assert_called_once()
        mock_ik_instance.setStartTime.assert_called_once_with(0.0)
        mock_ik_instance.setEndTime.assert_called_once_with(1.0)

        # Verify XML manipulation for task set
        generic_task_set_path = Path("data/generic_setup_files/setup_ik_taskset.xml")
        mock_xml_utils_instance.xml_parse.assert_called_once_with(generic_task_set_path)
        mock_xml_utils_instance.xml_add_element.assert_called_once_with(
            mocker.ANY,
            "InverseKinematicsTool",
            mock_task_set,
        )

        # Verify IK tool was run
        mock_ik_instance.run.assert_called_once()

    def test_run_id(self, mocker, mock_model_manager):
        # Mock File_utils and its methods
        mock_file_utils = mocker.patch("src.simulation_runner.File_utils")
        mock_file_utils_instance = mocker.Mock()
        mock_file_utils.return_value = mock_file_utils_instance

        # Mock paths and directories
        mock_sim_dir = Path("mocked/sim/dir")
        mock_results_dir = mock_sim_dir
        mock_coordinate_file = Path("mocked/results/dir/ik.mot")
        mock_external_loads_file = Path("mocked/external_loads.xml")

        # Configure mocks
        mock_file_utils_instance.get_trial_dir.return_value = mock_sim_dir
        mock_file_utils_instance.get_sim_file_path.side_effect = (
            lambda file_type, dir_path: {
                "coordinate_file": mock_coordinate_file,
                "external_loads_file": mock_external_loads_file,
            }.get(file_type)
        )

        # Mock Storage
        mock_storage = mocker.patch("src.simulation_runner.osim.Storage")
        mock_storage_instance = mocker.Mock()
        mock_storage.return_value = mock_storage_instance
        mock_storage_instance.getFirstTime.return_value = 0.0
        mock_storage_instance.getLastTime.return_value = 1.0

        # Mock Path operations
        mocker.patch.object(Path, "exists", return_value=False)
        mock_mkdir = mocker.patch.object(Path, "mkdir")
        mocker.patch.object(Path, "resolve", return_value=Path("mocked/resolved/path"))

        # Mock InverseDynamicsTool
        mock_id = mocker.patch("src.simulation_runner.osim.InverseDynamicsTool")
        mock_id_instance = mocker.Mock()
        mock_id.return_value = mock_id_instance

        # Mock XML operations
        mock_xml_utils = mocker.patch("src.simulation_runner.XML_utils")
        mock_xml_utils_instance = mocker.Mock()
        mock_xml_utils.return_value = mock_xml_utils_instance

        # Create runner instance with our mocks
        runner = Run_tools("009", "trial1")
        runner.sim_dir = mock_sim_dir
        runner.results_dir = mock_results_dir

        # Run the method being tested
        runner.run_id(lowpass_cutoff_frequency=8)

        # Verify directory creation
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify ID tool was configured correctly
        mock_id_instance.setDebugLevel.assert_called_once_with(0)
        mock_id_instance.setModelFileName.assert_called_once_with(
            str(mock_model_manager.return_value.model_path),
        )
        mock_id_instance.setCoordinatesFileName.assert_called_once()
        mock_id_instance.setExternalLoadsFileName.assert_called_once()
        mock_id_instance.setOutputGenForceFileName.assert_called_once()
        mock_id_instance.setLowpassCutoffFrequency.assert_called_once_with(8)
        mock_id_instance.setStartTime.assert_called_once_with(0.0)
        mock_id_instance.setEndTime.assert_called_once_with(1.0)

        # Verify XML modification for muscle exclusion
        mock_xml_utils_instance.xml_add_text.assert_called_once_with(
            mocker.ANY,
            "forces_to_exclude",
            " Muscles",
        )

        # Verify ID tool was run
        mock_id_instance.run.assert_called_once()

    def test_run_so(self, mocker, mock_model_manager):
        # Mock File_utils and its methods
        mock_file_utils = mocker.patch("src.simulation_runner.File_utils")
        mock_file_utils_instance = mocker.Mock()
        mock_file_utils.return_value = mock_file_utils_instance

        # Mock paths and directories
        mock_sim_dir = Path("mocked/sim/dir")
        mock_results_dir = mock_sim_dir
        mock_coordinate_file = Path("mocked/results/dir/ik.mot")
        mock_external_loads_file = Path("mocked/external_loads.xml")
        mock_actuator_file = Path("mocked/actuators.xml")

        # Configure mocks
        mock_file_utils_instance.get_trial_dir.return_value = mock_sim_dir
        mock_file_utils_instance.get_sim_file_path.side_effect = (
            lambda file_type, dir_path: {
                "coordinate_file": mock_coordinate_file,
                "external_loads_file": mock_external_loads_file,
                "actuator_so": mock_actuator_file,
            }.get(file_type)
        )

        # Mock Storage
        mock_storage = mocker.patch("src.simulation_runner.osim.Storage")
        mock_storage_instance = mocker.Mock()
        mock_storage.return_value = mock_storage_instance
        mock_storage_instance.getFirstTime.return_value = 0.0
        mock_storage_instance.getLastTime.return_value = 1.0

        # Mock Path operations
        mocker.patch.object(Path, "exists", return_value=False)
        mock_mkdir = mocker.patch.object(Path, "mkdir")
        mocker.patch.object(Path, "resolve", return_value=Path("mocked/resolved/path"))

        # Mock AnalyzeTool
        mock_analyze = mocker.patch("src.simulation_runner.osim.AnalyzeTool")
        mock_analyze_instance = mocker.Mock()
        mock_analyze.return_value = mock_analyze_instance

        # Mock StaticOptimization
        mock_static_opt = mocker.patch("src.simulation_runner.osim.StaticOptimization")
        mock_static_opt_instance = mocker.Mock()
        mock_static_opt.return_value = mock_static_opt_instance

        # Mock XML operations
        mock_xml_utils = mocker.patch("src.simulation_runner.XML_utils")
        mock_xml_utils_instance = mocker.Mock()
        mock_xml_utils.return_value = mock_xml_utils_instance

        mock_static_opt_element = mocker.Mock()
        mock_xml_parse_result = mocker.Mock()
        mock_xml_parse_result.find.return_value = mock_static_opt_element
        mock_xml_utils_instance.xml_parse.return_value = mock_xml_parse_result

        # Create runner instance with our mocks
        runner = Run_tools("009", "trial1")
        runner.sim_dir = mock_sim_dir
        runner.results_dir = mock_results_dir

        # Run the method being tested
        runner.run_so(lowpass_cutoff_frequency=8, precision=8)

        # Verify directory creation
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify AnalyzeTool was configured correctly
        mock_analyze_instance.setModelFilename.assert_called_once()
        mock_analyze_instance.setReplaceForceSet.assert_called_once_with(False)
        mock_analyze_instance.setResultsDir.assert_called_once()
        mock_analyze_instance.setOutputPrecision.assert_called_once_with(8)
        mock_analyze_instance.setStartTime.assert_called_once_with(0.0)
        mock_analyze_instance.setFinalTime.assert_called_once_with(1.0)
        mock_analyze_instance.setLowpassCutoffFrequency.assert_called_once_with(8)

        # Verify StaticOptimization was configured correctly
        mock_static_opt_instance.setName.assert_called_once_with("StaticOptimization")
        mock_static_opt_instance.setOn.assert_called_once_with(True)
        mock_static_opt_instance.setStartTime.assert_called_once_with(0.0)
        mock_static_opt_instance.setEndTime.assert_called_once_with(1.0)

        # Verify XML manipulations
        assert mock_xml_utils_instance.xml_add_element.call_count >= 1
        mock_xml_utils_instance.xml_add_text.assert_called_once()

        # Verify AnalyzeTool was run
        mock_analyze_instance.run.assert_called_once()

    def test_run_ma(self, mocker, mock_model_manager):
        # Mock File_utils and its methods
        mock_file_utils = mocker.patch("src.simulation_runner.File_utils")
        mock_file_utils_instance = mocker.Mock()
        mock_file_utils.return_value = mock_file_utils_instance

        # Mock paths and directories
        mock_sim_dir = Path("mocked/sim/dir")
        mock_results_dir = mock_sim_dir / "ma"
        mock_coordinate_file = Path("mocked/results/dir/ik.mot")
        mock_external_loads_file = Path("mocked/external_loads.xml")

        # Configure mocks
        mock_file_utils_instance.get_trial_dir.return_value = mock_sim_dir
        mock_file_utils_instance.get_sim_file_path.side_effect = (
            lambda file_type, dir_path: {
                "coordinate_file": mock_coordinate_file,
                "external_loads_file": mock_external_loads_file,
            }.get(file_type)
        )

        # Mock Storage
        mock_storage = mocker.patch("src.simulation_runner.osim.Storage")
        mock_storage_instance = mocker.Mock()
        mock_storage.return_value = mock_storage_instance
        mock_storage_instance.getFirstTime.return_value = 0.0
        mock_storage_instance.getLastTime.return_value = 1.0

        # Mock Path operations
        mocker.patch.object(Path, "exists", return_value=False)
        mock_mkdir = mocker.patch.object(Path, "mkdir")
        mocker.patch.object(Path, "resolve", return_value=Path("mocked/resolved/path"))

        # Mock AnalyzeTool
        mock_analyze = mocker.patch("src.simulation_runner.osim.AnalyzeTool")
        mock_analyze_instance = mocker.Mock()
        mock_analyze.return_value = mock_analyze_instance

        # Mock MuscleAnalysis
        mock_muscle_analysis = mocker.patch("src.simulation_runner.osim.MuscleAnalysis")
        mock_muscle_analysis_instance = mocker.Mock()
        mock_muscle_analysis.return_value = mock_muscle_analysis_instance

        # Mock XML operations
        mock_xml_utils = mocker.patch("src.simulation_runner.XML_utils")
        mock_xml_utils_instance = mocker.Mock()
        mock_xml_utils.return_value = mock_xml_utils_instance

        mock_muscle_analysis_element = mocker.Mock()
        mock_xml_parse_result = mocker.Mock()
        mock_xml_parse_result.find.return_value = mock_muscle_analysis_element
        mock_xml_utils_instance.xml_parse.return_value = mock_xml_parse_result

        # Create runner instance with our mocks
        runner = Run_tools("009", "trial1")
        runner.sim_dir = mock_sim_dir
        runner.results_dir = mock_results_dir

        # Run the method being tested
        runner.run_ma(lowpass_cutoff_frequency=8, precision=8)

        # Verify directory creation
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify AnalyzeTool was configured correctly
        mock_analyze_instance.setModelFilename.assert_called_once()
        mock_analyze_instance.setReplaceForceSet.assert_called_once_with(False)
        mock_analyze_instance.setResultsDir.assert_called_once()
        mock_analyze_instance.setOutputPrecision.assert_called_once_with(8)
        mock_analyze_instance.setStartTime.assert_called_once_with(0.0)
        mock_analyze_instance.setFinalTime.assert_called_once_with(1.0)
        mock_analyze_instance.setLowpassCutoffFrequency.assert_called_once_with(8)

        # Verify MuscleAnalysis was configured correctly
        mock_muscle_analysis_instance.setName.assert_called_once_with("MuscleAnalysis")
        mock_muscle_analysis_instance.setOn.assert_called_once_with(True)
        mock_muscle_analysis_instance.setStartTime.assert_called_once_with(0.0)
        mock_muscle_analysis_instance.setEndTime.assert_called_once_with(1.0)
        mock_muscle_analysis_instance.setComputeMoments.assert_called_once_with(True)

        # Verify XML manipulation
        mock_xml_utils_instance.xml_add_element.assert_called_once_with(
            mocker.ANY,
            "AnalysisSet/objects",
            mock_muscle_analysis_element,
        )

        # Verify AnalyzeTool was run
        mock_analyze_instance.run.assert_called_once()
