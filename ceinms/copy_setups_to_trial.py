import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import opensim as osim

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src import OpensimModelManager
from utils import File_utils, Subject_utils

all_ids = OpensimModelManager.get_ids()

input_data_setup_file = Path(r"ceinms/setupFiles/inputData.xml")

for id in all_ids:
    trials = Subject_utils().get_all_baseline_and_last_trial(id)
    session_path = File_utils().get_session_dir(id)
    for trial in trials:
        trial_path = File_utils().get_trial_dir(id, trial)
        # Copy and adjust inputData.xml
        try:
            ik_path = File_utils().get_ik_file_path(id, trial)
            coordinates = osim.Storage(str(ik_path))
            time = [coordinates.getFirstTime(), coordinates.getLastTime()]
        except:
            print("no Ik file found for ", id, " in ", trial)
        tree = ET.parse(input_data_setup_file)
        root = tree.getroot()
        elem = root.find("startStopTime")
        elem.text = f"{time[0]} {time[1]}"
        dst_inputData_file = trial_path / "inputData.xml"
        tree.write(dst_inputData_file)

    # Copy
