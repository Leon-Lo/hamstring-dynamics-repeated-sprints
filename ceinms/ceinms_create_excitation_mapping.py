import os
import shutil
import sys
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path

import opensim as osim
import paths

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils.subject_utils import Subject_utils


def save_pretty_xml(element, file_path):
    """Save an XML element to a file with pretty formatting."""
    tree = ET.ElementTree(element)
    with open(file_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
    # Pretty print the XML

    dom = xml.dom.minidom.parse(file_path)
    pretty_xml_as_string = dom.toprettyxml(indent="  ")
    with open(file_path, "w") as f:
        f.write(pretty_xml_as_string)


def copy_correct_excitation_generator_xml(save_path):

    id = paths.SUBJECT
    used_leg = Subject_utils().get_measured_leg(id)
    if used_leg is None:
        raise ValueError(f"Measured leg for subject {id} not found.")
    used_leg = used_leg.lower()
    setup_file_name = f"excitationGenerator_{used_leg}.xml"
    setup_file_path = Path(paths.SETUP_DIR) / setup_file_name

    dst_path = Path(save_path)

    shutil.copy(setup_file_path, dst_path)


if __name__ == "__main__":
    import time

    current_dir = os.path.dirname(os.path.abspath(__file__))
    CEINMS_NN_DIR = os.path.dirname(current_dir)

    print("\033[93mWARNING: Please check if the path is correct. If not change \033[0m")
    print(f"CEINMS NN directory: {CEINMS_NN_DIR}")
    osim_version = osim.__version__
    print(f"OpenSim version: {osim_version}")
    time.sleep(1)

    # Example usage
    subject_id = paths.SUBJECT
    emg_path = paths.EMG_MOT
    save_path = paths.CEINMS_EXCITATION_MAPPING
    copy_correct_excitation_generator_xml(save_path)
