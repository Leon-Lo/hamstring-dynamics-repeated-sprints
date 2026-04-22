from pathlib import Path

import pandas as pd

from src.simulation_runner import Run_tools
from utils.file_utils import File_utils


class Simulation_analyzer:
    """
    A class for running and analyzing OpenSim simulations.

    This class provides high-level methods to run complete simulation pipelines
    or specific analysis tools for biomechanical data processing.
    """

    def __init__(self):
        """Initialize the Simulation_analyzer instance."""

    def run_all_tools(self, subject_id: str, trial_dir: str):
        """
        Run all OpenSim analysis tools in sequence.

        Args:
            subject_id: The subject identifier
            trial_dir: The trial directory name

        Runs in order:
        1. Inverse Kinematics (IK)
        2. Inverse Dynamics (ID)
        3. Static Optimization (SO)
        4. Muscle Analysis (MA)

        """
        # Initialize the Run_tools instance for this subject and trial
        run_tools = Run_tools(subject_id, trial_dir)
        # Run each analysis tool in sequence
        run_tools.run_ik()
        run_tools.run_id()
        run_tools.run_so()
        run_tools.run_ma()

    def run_specific_tools(
        self,
        subject_id: str,
        trial_dir: str,
        tools: str | list[str],
    ):
        """
        Run specific OpenSim analysis tools based on the provided list.

        Args:
            subject_id: The subject identifier
            trial_dir: The trial directory name
            tools: A string or list of tool names to run (valid: 'ik', 'id', 'so', 'ma')

        Special handling for IK:
        - Checks subject_processing.csv for special treatment flags
        - If special treatment is needed, runs IK from existing setup file
        - Otherwise runs standard IK workflow

        """
        # Convert single tool string to list for uniform processing
        if isinstance(tools, str):
            tools = [tools]

        # Get the trial directory path
        trial_path = File_utils().get_trial_dir(subject_id, trial_dir)

        # Define valid tool names
        valid_tools = ["ik", "id", "so", "ma"]
        # Initialize Run_tools with subject and trial information
        run_tools = Run_tools(subject_id, trial_path.name)

        # Process tools that need special handling first
        tools_to_process = []
        for tool in tools:
            # Validate that the tool name is valid
            if tool not in valid_tools:
                raise ValueError(
                    f"Invalid tool specified: {tool}. Valid tools are {valid_tools}",
                )
            # IK requires special handling based on subject processing flags
            if tool == "ik":
                # Load subject processing CSV to check for special IK handling
                subject_processing_csv = Path("data/subject_processing.csv")
                df = pd.read_csv(subject_processing_csv)

                # Get the path to the IK setup file
                ik_setup_path = trial_path / "setup_ik.xml"
                # Format subject ID for CSV column lookup
                s_s_id = f"s{subject_id}"
                # Check if subject requires special IK treatment
                if s_s_id in df.columns:
                    # Get the special treatment flag (row 8)
                    csv_value = df[s_s_id].iloc[8]
                    if csv_value == "y":
                        # Special treatment required - use existing setup file
                        try:
                            run_tools.run_ik_from_file(ik_setup_path.absolute())
                        except Exception as e:
                            print(f"Error running IK from file for {trial_dir}: {e}")
                    else:
                        # No special treatment - run standard IK
                        try:
                            run_tools.run_ik()
                        except Exception as e:
                            print(f"Error running IK tool for {trial_dir}: {e}")
            else:
                # Other tools don't need special handling
                tools_to_process.append(tool)

        # Process remaining tools (ID, SO, MA)
        # Map tool names to their corresponding methods
        tool_methods = {
            "id": run_tools.run_id,
            "so": run_tools.run_so,
            "ma": run_tools.run_ma,
        }

        # Execute each tool method
        for tool in tools_to_process:
            try:
                # Call the corresponding tool method
                tool_methods[tool]()
            except Exception as e:
                # Log error and continue with remaining tools
                print(f"Error running {tool} tool for {trial_dir}: {e}")
                continue

    # def sprint_analysis(self, subject_id: str, trial_dir: str):
    #     """
    #     Perform a complete analysis of a sprint simulation.
    #     This includes running all necessary tools and generating plots.
    #     """
    #     # Import locally to avoid circular imports
    #     from utils.analysis_utils import Analysis_utils

    #     self.run_all_tools(subject_id, trial_dir)

    #     analysis_utils = Analysis_utils()
    #     muscle_work = analysis_utils.get_muscle_work(subject_id, trial_dir)

    #     # Generate plots for the analysis

    #     return muscle_work
