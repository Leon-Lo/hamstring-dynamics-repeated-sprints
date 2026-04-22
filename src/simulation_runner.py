# Import necessary modules for OpenSim simulation tools
import datetime
import os
import shutil
import timeit
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import opensim as osim

from src.model_manager import OpensimModelManager
from utils.file_utils import File_utils
from utils.xml_utils import XML_utils

# import platform


class Run_tools:
    """
    A class containing tools for running various OpenSim simulation analyses.

    This class provides methods to run:
    - Inverse Kinematics (IK)
    - Inverse Dynamics (ID)
    - Static Optimization (SO)
    - Muscle Analysis (MA)
    """

    def __init__(self, subject_id: str, trial_name: str):
        """
        Initialize the Run_tools instance.

        Args:
            subject_id: The subject identifier
            trial_name: The name of the trial to process

        """
        # Load the OpenSim model for this subject
        self.model_manager = OpensimModelManager(subject_id)
        # Get the path to the model file
        self.model = self.model_manager.model_path
        # Get the initialized model state
        self.state = self.model_manager.state
        # Get the simulation directory for this trial
        self.sim_dir = File_utils().get_trial_dir(subject_id, trial_name)
        # Store subject ID and trial name
        self.id = subject_id
        self.trial_name = trial_name
        # Set results directory to the simulation directory
        self.results_dir = self.sim_dir
        # Path to generic setup files
        self.generic_setup_path = Path("data/generic_setup_files")

    def run_ik(self):
        """
        Run Inverse Kinematics (IK) analysis.

        IK computes joint angles from experimental marker data by solving
        the optimization problem that minimizes marker position errors.
        """
        # Start timing the IK run
        start = timeit.default_timer()

        # Initialize file utilities
        file_access = File_utils()
        # Get the marker file path (TRC file with experimental marker data)
        try:
            marker_file = file_access.get_marker_file_path(self.id, self.trial_name)
        except Exception as e:
            raise ValueError(f"Error getting marker file: {e}")

        # Load marker data using OpenSim
        marker_file_osim = osim.MarkerData(str(marker_file))
        # Get the model path
        model_path = self.model_manager.model_path
        if model_path is None:
            raise ValueError("Model path is None. Please check the model manager.")

        # Ensure the results directory exists
        res_dir = self.results_dir
        if not res_dir.exists():
            res_dir.mkdir(parents=True, exist_ok=True)

        # Define output file path for IK results
        output_file = res_dir / "ik.mot"

        ik_tool = osim.InverseKinematicsTool()
        ik_tool.setDebugLevel(0)
        ik_tool.set_model_file(str(os.path.relpath(model_path, os.getcwd())))
        ik_tool.set_marker_file(str(marker_file.relative_to(self.sim_dir)))
        ik_tool.set_output_motion_file(str(output_file.relative_to(self.sim_dir)))
        ik_tool.set_results_directory(str(res_dir.relative_to(self.sim_dir)))
        ik_tool.set_constraint_weight(np.inf)
        ik_tool.set_accuracy(1.0000000000000004e-06)
        ik_tool.set_coordinate_file("Unassigned")
        ik_tool.setStartTime(marker_file_osim.getStartFrameTime())
        ik_tool.setEndTime(marker_file_osim.getLastFrameTime())
        ik_tool.set_report_errors(True)
        ik_tool.set_report_marker_locations(True)

        setup_file = self.sim_dir / "setup_ik.xml"
        ik_tool.printToXML(str(setup_file))

        generic_task_set_path = self.generic_setup_path / "setup_ik_taskset.xml"
        # Add the IKTaskSet element to the setup file
        generic_task_set_file = XML_utils().xml_parse(generic_task_set_path)
        generic_task_set = generic_task_set_file.find(".//IKTaskSet")

        # Add the generic IKTaskSet element to the setup file
        if generic_task_set is not None:
            XML_utils().xml_add_element(
                setup_file,
                "InverseKinematicsTool",
                generic_task_set,
            )
        else:
            raise ValueError(f"IKTaskSet not found in {generic_task_set_path}")

        try:
            ik_tool = osim.InverseKinematicsTool(str(setup_file))
            ik_tool.run()
        except Exception as e:
            raise RuntimeError(f"Error running IK tool: {e}")

        stop = timeit.default_timer()
        run_time = stop - start
        if run_time < 60:
            print(f"IK Tool run time: {run_time:.2f} seconds")
        else:
            run_time = datetime.timedelta(seconds=run_time)
            print(f"IK Tool run time: {run_time}")

    def run_ik_from_file(self, ik_setup_file):
        start = timeit.default_timer()

        try:
            ik_tool = osim.InverseKinematicsTool(str(ik_setup_file))
            ik_tool.run()
        except Exception as e:
            print(f"Error running IK tool: {e}")
            raise

        stop = timeit.default_timer()
        run_time = stop - start
        if run_time < 60:
            print(f"IK Tool run time: {run_time:.2f} seconds")
        else:
            run_time = datetime.timedelta(seconds=run_time)
            print(f"IK Tool run time: {run_time}")

    def run_id(self, lowpass_cutoff_frequency=None):
        start = timeit.default_timer()

        file_access = File_utils()

        try:
            coordinate_file = file_access.get_ik_file_path(self.id, self.trial_name)
        except Exception as e:
            raise ValueError(f"Error getting coordinate file: {e}")

        model_path = self.model_manager.model_path
        if model_path is None:
            raise ValueError("Model path is None. Please check the model manager.")
        try:
            external_loads_file = file_access.get_external_loads_xml_file_path(
                self.id,
                self.trial_name,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"External loads file not found for {self.id} {self.trial_name}. Please provide an external loads file.",
            )
        res_dir = self.results_dir
        if not res_dir.exists():
            res_dir.mkdir(parents=True, exist_ok=True)
        output_file = res_dir / "id.sto"

        if lowpass_cutoff_frequency is None:
            lowpass_cutoff_frequency = 6
        else:
            lowpass_cutoff_frequency = lowpass_cutoff_frequency

        try:
            coordinate_file_osim = osim.Storage(str(coordinate_file))
        except Exception as e:
            raise ValueError(f"Error reading coordinate file: {e}")

        try:
            id_tool = osim.InverseDynamicsTool()
            id_tool.setDebugLevel(0)
            id_tool.setModelFileName(str(os.path.relpath(model_path, os.getcwd())))
            id_tool.setCoordinatesFileName(
                str(coordinate_file.relative_to(self.sim_dir)),
            )
            id_tool.setExternalLoadsFileName(
                str(external_loads_file.relative_to(self.sim_dir)),
            )
            id_tool.setOutputGenForceFileName(
                str(output_file.relative_to(self.sim_dir)),
            )
            id_tool.setLowpassCutoffFrequency(lowpass_cutoff_frequency)
            id_tool.setStartTime(coordinate_file_osim.getFirstTime())
            id_tool.setEndTime(coordinate_file_osim.getLastTime())
            id_tool.set_results_directory(str(res_dir.relative_to(self.sim_dir)))
            setup_file = self.sim_dir / "setup_id.xml"
            id_tool.printToXML(str(setup_file))
        except Exception as e:
            raise RuntimeError(f"Error setting up ID tool: {e}")

        # Add muscle exclusion to ID tool via XML
        XML_utils().xml_add_text(setup_file, "forces_to_exclude", " Muscles")

        try:
            id_tool = osim.InverseDynamicsTool(str(setup_file))
            id_tool.run()
        except Exception as e:
            raise RuntimeError(f"Error running ID tool: {e}")

        stop = timeit.default_timer()
        run_time = stop - start
        if run_time < 60:
            print(f"ID Tool run time: {run_time:.2f} seconds")
        else:
            run_time = datetime.timedelta(seconds=run_time)
            print(f"ID Tool run time: {run_time}")

    def run_so(self, lowpass_cutoff_frequency=None, precision=None):
        start = timeit.default_timer()

        file_access = File_utils()
        try:
            coordinate_file = file_access.get_ik_file_path(self.id, self.trial_name)
        except Exception as e:
            raise ValueError(f"Error getting coordinate file: {e}")

        model_path = self.model_manager.model_path
        if model_path is None:
            raise ValueError("Model path is None. Please check the model manager.")
        external_loads_file = file_access.get_external_loads_xml_file_path(
            self.id,
            self.trial_name,
        )

        try:
            actuator_so_file = file_access.get_actuators_so_file_path(
                self.id,
                self.trial_name,
            )
        except FileNotFoundError:
            generic_actuator_file = self.generic_setup_path / "actuators_so.xml"
            actuator_so_file = self.sim_dir / "actuators_so.xml"
            shutil.copy(generic_actuator_file, self.sim_dir / "actuators_so.xml")

        if lowpass_cutoff_frequency is None:
            lowpass_cutoff_frequency = 6
        else:
            lowpass_cutoff_frequency = lowpass_cutoff_frequency
        if precision is None:
            precision = 6
        else:
            precision = precision

        try:
            coordinates_file_osim = osim.Storage(str(coordinate_file))
        except Exception as e:
            raise ValueError(f"Error reading coordinate file: {e}")

        res_dir = self.results_dir
        if not res_dir.exists():
            res_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create the AnalyzeTool
            analyze_tool = osim.AnalyzeTool()
            analyze_tool.setModelFilename(str(os.path.relpath(model_path, os.getcwd())))
            analyze_tool.setReplaceForceSet(False)
            analyze_tool.setResultsDir(str(res_dir.relative_to(self.sim_dir)))
            analyze_tool.setOutputPrecision(precision)
            analyze_tool.setStartTime(coordinates_file_osim.getFirstTime())
            analyze_tool.setFinalTime(coordinates_file_osim.getLastTime())
            analyze_tool.setSolveForEquilibrium(False)
            analyze_tool.setMaximumNumberOfSteps(20000)
            analyze_tool.setMaxDT(1)
            analyze_tool.setMinDT(1e-08)
            analyze_tool.setErrorTolerance(1.0000000000000001e-05)
            analyze_tool.setLowpassCutoffFrequency(lowpass_cutoff_frequency)

            # Initialize the StaticOptimization
            static_opt = osim.StaticOptimization()
            static_opt.setName("StaticOptimization")
            static_opt.setOn(True)
            static_opt.setStartTime(coordinates_file_osim.getFirstTime())
            static_opt.setEndTime(coordinates_file_osim.getLastTime())
            static_opt.setStepInterval(1)
            static_opt.setInDegrees(True)
            static_opt.setUseModelForceSet(True)
            static_opt.setActivationExponent(2)
            static_opt.setUseMusclePhysiology(True)
            static_opt.setConvergenceCriterion(0.0001)
            static_opt.setMaxIterations(100)
            analysis_set_file = self.sim_dir / "so_analysis_set.xml"
            static_opt.printToXML(str(analysis_set_file))

            # analyze
            analyze_tool.setExternalLoadsFileName(
                str(external_loads_file.relative_to(self.sim_dir)),
            )
            analyze_tool.setCoordinatesFileName(
                str(coordinate_file.relative_to(self.sim_dir)),
            )
            setup_file = self.sim_dir / "setup_so.xml"
            analyze_tool.printToXML(str(setup_file))

        except Exception as e:
            raise RuntimeError(f"Error setting up SO tool: {e}")

        analysis_set_xml = XML_utils().xml_parse(analysis_set_file)
        static_opt = analysis_set_xml.find(".//StaticOptimization")

        # Add the StaticOptimization element to the setup file
        if static_opt is not None:
            XML_utils().xml_add_element(setup_file, "AnalysisSet/objects", static_opt)
        else:
            raise ValueError(f"Static not found in {analysis_set_file}")

        # Add actuators to the setup file
        actuator_path = actuator_so_file.resolve()
        XML_utils().xml_add_text(setup_file, "force_set_files", str(actuator_path))

        try:
            analyze_tool = osim.AnalyzeTool(str(setup_file))
            analyze_tool.run()
        except Exception as e:
            raise RuntimeError(f"Error running SO tool: {e}")

        stop = timeit.default_timer()
        run_time = stop - start
        if run_time < 60:
            print(f"SO Tool run time: {run_time:.2f} seconds")
        else:
            run_time = datetime.timedelta(seconds=run_time)
            print(f"SO Tool run time: {run_time}")

    def run_ma(self, lowpass_cutoff_frequency=None, precision=None):
        start = timeit.default_timer()

        file_access = File_utils()
        model_path = self.model_manager.model_path
        if model_path is None:
            raise ValueError("Model path is None. Please check the model manager.")

        try:
            coordinate_file = file_access.get_ik_file_path(self.id, self.trial_name)
        except Exception as e:
            raise ValueError(f"Error getting coordinate file: {e}")

        coordinates_file_osim = osim.Storage(str(coordinate_file))

        external_loads_file = file_access.get_external_loads_xml_file_path(
            self.id,
            self.trial_name,
        )
        if not external_loads_file.exists():
            raise FileNotFoundError(
                f"External loads file not found: {external_loads_file}",
            )

        res_dir = self.results_dir / "ma"
        if not res_dir.exists():
            res_dir.mkdir(parents=True, exist_ok=True)

        if lowpass_cutoff_frequency is None:
            lowpass_cutoff_frequency = 6
        else:
            lowpass_cutoff_frequency = lowpass_cutoff_frequency
        if precision is None:
            precision = 6
        else:
            precision = precision

        analyze_tool = osim.AnalyzeTool()
        analyze_tool.setModelFilename(str(os.path.relpath(model_path, self.sim_dir)))
        analyze_tool.setReplaceForceSet(False)
        analyze_tool.setResultsDir(str(res_dir.relative_to(self.sim_dir)))
        analyze_tool.setOutputPrecision(precision)
        analyze_tool.setStartTime(coordinates_file_osim.getFirstTime())
        analyze_tool.setFinalTime(coordinates_file_osim.getLastTime())
        analyze_tool.setSolveForEquilibrium(False)
        analyze_tool.setMaximumNumberOfSteps(20000)
        analyze_tool.setMaxDT(1)
        analyze_tool.setMinDT(1e-08)
        analyze_tool.setErrorTolerance(1.0000000000000001e-05)

        # Initialize the muscle analysis
        muscle_analysis = osim.MuscleAnalysis()
        muscle_analysis.setName("MuscleAnalysis")
        muscle_analysis.setOn(True)
        muscle_analysis.setStartTime(coordinates_file_osim.getFirstTime())
        muscle_analysis.setEndTime(coordinates_file_osim.getLastTime())
        muscle_analysis.setStepInterval(1)
        muscle_analysis.setInDegrees(True)
        muscle_analysis.setComputeMoments(True)
        analysis_set_file = self.sim_dir / "ma_analysis_set.xml"
        muscle_analysis.printToXML(str(analysis_set_file))

        # Set the external loads file
        analyze_tool.setExternalLoadsFileName(
            str(external_loads_file.relative_to(self.sim_dir)),
        )
        analyze_tool.setCoordinatesFileName(
            str(coordinate_file.relative_to(self.sim_dir)),
        )
        analyze_tool.setLowpassCutoffFrequency(lowpass_cutoff_frequency)
        setup_file = self.sim_dir / "setup_ma.xml"
        analyze_tool.printToXML(str(setup_file))

        # Add the MuscleAnalysis element to the setup file
        analysis_set_xml = XML_utils().xml_parse(analysis_set_file)
        muscle_analysis = analysis_set_xml.find(".//MuscleAnalysis")
        if muscle_analysis is not None:
            XML_utils().xml_add_element(
                setup_file,
                "AnalysisSet/objects",
                muscle_analysis,
            )
        else:
            raise ValueError(f"MuscleAnalysis not found in {analysis_set_file}")

        analyze_tool = osim.AnalyzeTool(str(setup_file))
        analyze_tool.getModel().initSystem()
        analyze_tool.run()

        stop = timeit.default_timer()
        run_time = stop - start
        if run_time < 60:
            print(f"MA Tool run time: {run_time:.2f} seconds")
        else:
            run_time = datetime.timedelta(seconds=run_time)
            print(f"MA Tool run time: {run_time}")

    # def run_ceinms(self):
    #     # Check os
    #     if platform.system() != "Windows":
    #         raise NotImplementedError("CEINMS is only supported on Windows. Use a virtual machine or Docker container with Windows to run CEINMS.")
    #     start = timeit.default_timer()

    #     # Input data
    #     file_acess = File_utils()
    #     try:
    #         setup_files_path = Path("data/generic_setup_files")
    #         setup_input_data_file = setup_files_path / "setup_inputData.xml"
    #         input_data = self.sim_dir / "inputData.xml"
    #         # Copy the setup input data file to the simulation directory
    #         shutil.copy(setup_input_data_file, input_data)
    #     except FileNotFoundError as e:
    #         raise FileNotFoundError(f"Error copying input data file: {e}")

    #     # Load time for the input data file and write it to the XML
    #     try:
    #         coordinate_file = osim.Storage(str(self.results_dir / "ik.mot"))
    #         first_time = coordinate_file.getFirstTime()
    #         last_time = coordinate_file.getLastTime()
    #         formatted_str = f"{first_time:.3f} {last_time:.3f}"
    #         XML_utils().xml_add_text(input_data, "startStopTime", formatted_str)
    #     except Exception as e:
    #         raise ValueError(f"Error reading coordinate file: {e}")

    #     try:
    #         ceinms_exe = Path("executables/CEINMS.exe")
    #         ceinms_optimize_exe = Path("executables/CEINMSoptimise.exe")
    #         ceinms_calibrate_exe = Path("executables/CEINMS-nn-calibrate.exe")
    #         setup_ceinms = self.sim_dir / "setup_ceinms.xml"
    #         setup_ceinms_optimize = self.sim_dir / "setup_ceinms_optimize.xml"
    #         setup_ceinms_calibrate = self.sim_dir / "setup_ceinms_calibrate.xml"
    #     except FileNotFoundError as e:
    #         raise FileNotFoundError(f"Error finding CEINMS executables or setup files: {e}")

    #     # Run CEINMS calibration
    #     try:
    #         command = [ceinms_calibrate_exe, "-S", calibration_setup_file]

    @staticmethod
    def run_ik_standalone(model_path, marker_file, output_dir, task_set_file=None):
        """
        Run Inverse Kinematics (IK) standalone — no class or project utilities required.

        Args:
            model_path:   Path to the .osim model file.
            marker_file:  Path to the .trc experimental marker data file.
            output_dir:   Path to the directory where results (setup XML + .mot) will be written.
            task_set_file: (optional) Path to an XML file containing an IKTaskSet element.
                            Defaults to data/generic_setup_files/setup_ik_taskset.xml.
        """
        model_path = Path(model_path)
        marker_file = Path(marker_file)
        output_dir = Path(output_dir)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not marker_file.exists():
            raise FileNotFoundError(f"Marker file not found: {marker_file}")

        output_dir.mkdir(parents=True, exist_ok=True)

        start = timeit.default_timer()

        marker_data = osim.MarkerData(str(marker_file))

        ik_tool = osim.InverseKinematicsTool()
        ik_tool.setDebugLevel(0)
        ik_tool.set_model_file(str(model_path.resolve()))
        ik_tool.set_marker_file(str(marker_file.resolve()))
        ik_tool.set_output_motion_file("ik.mot")
        ik_tool.set_results_directory(".")
        ik_tool.set_constraint_weight(np.inf)
        ik_tool.set_accuracy(1.0000000000000004e-06)
        ik_tool.set_coordinate_file("Unassigned")
        ik_tool.setStartTime(marker_data.getStartFrameTime())
        ik_tool.setEndTime(marker_data.getLastFrameTime())
        ik_tool.set_report_errors(True)
        ik_tool.set_report_marker_locations(True)

        setup_file = output_dir / "setup_ik.xml"
        ik_tool.printToXML(str(setup_file))

        if task_set_file is None:
            task_set_file = Path("data/generic_setup_files/setup_ik_taskset.xml")
        else:
            task_set_file = Path(task_set_file)

        if not task_set_file.exists():
            raise FileNotFoundError(f"Task set file not found: {task_set_file}")
        task_tree = ET.parse(task_set_file)
        task_set_element = task_tree.find(".//IKTaskSet")
        if task_set_element is None:
            raise ValueError(f"IKTaskSet element not found in {task_set_file}")
        setup_tree = ET.parse(setup_file)
        parent = setup_tree.find(".//InverseKinematicsTool")
        if parent is None:
            raise ValueError("InverseKinematicsTool element not found in setup XML")
        parent.append(task_set_element)
        setup_tree.write(str(setup_file))

        try:
            ik_tool = osim.InverseKinematicsTool(str(setup_file))
            ik_tool.run()
        except Exception as e:
            raise RuntimeError(f"Error running IK tool: {e}")

        stop = timeit.default_timer()
        run_time = stop - start
        if run_time < 60:
            print(f"IK Tool run time: {run_time:.2f} seconds")
        else:
            print(f"IK Tool run time: {datetime.timedelta(seconds=run_time)}")
