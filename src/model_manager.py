from pathlib import Path

# from utils.xml_utils import XML_utils
import opensim as osim


class OpensimModelManager:
    """Manager class for loading and interacting with OpenSim models."""

    def __init__(self, id: str, models_dir: Path | None = None):
        """
        Initialize the OpenSim model manager.

        Args:
            id: Subject identifier used to find the corresponding model file
            models_dir: Optional directory path where models are stored (defaults to "data/models")

        """
        # Store the subject ID
        self.id = id
        # Set the models directory, using default if not provided
        self.models_dir = models_dir if models_dir is not None else Path("data/models")
        # Locate the model file for this subject
        self.model_path = self.get_model_path()
        # Validate that the model file exists
        if self.model_path is None:
            raise FileNotFoundError(f"Model file for ID {self.id} not found.")
        # Load the OpenSim model
        self.model = self.load_model()

    def get_model_path(self) -> Path | None:
        """
        Find the model file path for the current subject ID.

        Looks inside the subject-specific subdirectory (models_dir / id) for a
        file matching the pattern ``{id}_*_originalMass_opt_N10_hans.osim``.

        Returns:
            Path to the .osim model file, or None if not found

        """
        subject_dir = self.models_dir / str(self.id)
        if not subject_dir.is_dir():
            return None
        last_match = None
        # Iterate through files in the subject subdirectory
        for files in subject_dir.iterdir():
            # Match: is a file, starts with subject ID, ends with the target suffix
            if (
                files.is_file()
                and files.name.startswith(str(self.id))
                and files.name.endswith("_originalMass_opt_N10_hans.osim")
            ):
                last_match = files
        return last_match

    def load_model(self):
        """
        Load the OpenSim model from the model file path.

        Returns:
            The loaded OpenSim model object

        """
        # Load the model from the file path
        self.model = osim.Model(str(self.model_path))
        # Initialize the system state for the model
        self.state = self.model.initSystem()
        return self.model

    def get_model(self) -> osim.Model:
        """Return the loaded OpenSim model."""
        return self.model

    def get_state(self) -> osim.State:
        """Return the OpenSim state."""
        return self.state

    def get_muscle_groups(self):
        """
        Extract muscle groups from the OpenSim model XML file.

        Returns:
            Dictionary mapping group names to lists of muscle names

        """
        from utils.xml_utils import XML_utils

        muscle_groups = {}
        # Get the path to the model file
        model_path = self.get_model_path()

        # Validate that the model file exists
        if model_path is None:
            raise FileNotFoundError(f"Model file for ID {self.id} not found.")
        # Parse the model XML file
        model_xml = XML_utils.xml_parse(model_path)
        # Find the ForceSet element in the XML
        force_set = model_xml.find(".//ForceSet")
        if force_set is None:
            raise ValueError("ForceSet not found in the model XML.")
        # Find all ObjectGroup elements (muscle groups)
        groups = force_set.findall(".//ObjectGroup")
        # Iterate through each muscle group
        for group in groups:
            # Get the group name
            group_name = group.attrib.get("name")
            # Find the members element containing muscle names
            members = group.find("members")
            if members is not None and members.text:
                # Split the space-separated muscle names and store in dictionary
                muscle_names = members.text.strip().split()
                muscle_groups[group_name] = muscle_names

        return muscle_groups

    @staticmethod
    def get_ids() -> list:
        """
        Get a list of all available subject IDs.

        Returns:
            List of subject IDs

        """
        from utils.subject_utils import Subject_utils

        # Create a Subject_utils instance to access subject data
        subject_utils = Subject_utils()
        # Return all available subject IDs
        return subject_utils.get_all_subjects()
