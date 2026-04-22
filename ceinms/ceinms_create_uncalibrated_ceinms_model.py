import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import ceinms_utils as utils
import opensim as osim

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.file_utils import File_utils
from utils.subject_utils import Subject_utils


def create_ceinms_model(osimModelPath=None, outputCEINMSModelPath=None, leg=None):
    """
    Create a CEINMS subject XML file with muscle parameters extracted from the OpenSim model.
    """
    if not osimModelPath:
        osimModelPath = input("Enter path to OpenSim model (.osim): ").strip('"')

    if not outputCEINMSModelPath:
        outputCEINMSModelPath = input(
            "Enter path to output CEINMS model (.xml): ",
        ).strip('"')

    print("Creating CEINMS model from OpenSim model")
    # Load the OpenSim model
    model = osim.Model(osimModelPath)
    model.initSystem()

    # Create the root element
    root = ET.Element("subject")

    # Add mtuDefault section with default curves and parameters
    mtu_default = ET.SubElement(root, "mtuDefault")

    # Add default parameters
    ET.SubElement(mtu_default, "emDelay").text = "0.015"
    ET.SubElement(mtu_default, "percentageChange").text = "0.15"
    ET.SubElement(mtu_default, "damping").text = "0.1"

    # Add default curves (using the curves from your example)
    curves_data = {
        "activeForceLength": {
            "xPoints": "-5 0 0.401 0.402 0.4035 0.52725 0.62875 0.71875 0.86125 1.045 1.2175 1.4387 1.6187 1.62 1.621 2.2 5",
            "yPoints": "0 0 0 0 0 0.22667 0.63667 0.85667 0.95 0.99333 0.77 0.24667 0 0 0 0 0",
        },
        "passiveForceLength": {
            "xPoints": "-5 0.998 0.999 1 1.1 1.2 1.3 1.4 1.5 1.6 1.601 1.602 5",
            "yPoints": "0 0 0 0 0.035 0.12 0.26 0.55 1.17 2 2 2 2",
        },
        "forceVelocity": {
            "xPoints": "-10 -1 -0.6 -0.3 -0.1 0 0.1 0.3 0.6 0.8 10",
            "yPoints": "0 0 0.08 0.2 0.55 1 1.4 1.6 1.7 1.75 1.75",
        },
        "tendonForceStrain": {
            "xPoints": " ".join([str(i / 1000) for i in range(101)]),
            "yPoints": "0 0.0012652 0.0073169 0.016319 0.026613 0.037604 0.049078 0.060973 0.073315 0.086183 0.099678 0.11386 0.12864 0.14386 0.15928 0.17477 0.19041 0.20658 0.22365 0.24179 0.26094 0.28089 0.30148 0.32254 0.34399 0.36576 0.38783 0.41019 0.43287 0.45591 0.4794 0.50344 0.52818 0.55376 0.58022 0.60747 0.63525 0.66327 0.69133 0.71939 0.74745 0.77551 0.80357 0.83163 0.85969 0.88776 0.91582 0.94388 0.97194 1 1.0281 1.0561 1.0842 1.1122 1.1403 1.1684 1.1964 1.2245 1.2526 1.2806 1.3087 1.3367 1.3648 1.3929 1.4209 1.449 1.477 1.5051 1.5332 1.5612 1.5893 1.6173 1.6454 1.6735 1.7015 1.7296 1.7577 1.7857 1.8138 1.8418 1.8699 1.898 1.926 1.9541 1.9821 2.0102 2.0383 2.0663 2.0944 2.1224 2.1505 2.1786 2.2066 2.2347 2.2628 2.2908 2.3189 2.3469 2.375 2.4031 2.4311",
        },
    }

    for curve_name, points in curves_data.items():
        curve = ET.SubElement(mtu_default, "curve")
        ET.SubElement(curve, "name").text = curve_name
        ET.SubElement(curve, "xPoints").text = points["xPoints"]
        ET.SubElement(curve, "yPoints").text = points["yPoints"]

    # Add mtuSet section
    mtu_set = ET.SubElement(root, "mtuSet")

    # Extract muscle parameters from OpenSim model
    muscle_set = model.getMuscles()
    for i in range(muscle_set.getSize()):
        muscle = muscle_set.get(i)

        # Create mtu element for each muscle
        mtu = ET.SubElement(mtu_set, "mtu")

        # Add muscle parameters
        ET.SubElement(mtu, "name").text = muscle.getName()
        ET.SubElement(mtu, "c1").text = "-0.5"
        ET.SubElement(mtu, "c2").text = "-0.5"
        ET.SubElement(mtu, "shapeFactor").text = "0.1"
        ET.SubElement(mtu, "optimalFibreLength").text = str(
            muscle.getOptimalFiberLength(),
        )
        ET.SubElement(mtu, "pennationAngle").text = str(
            muscle.getPennationAngleAtOptimalFiberLength(),
        )
        ET.SubElement(mtu, "tendonSlackLength").text = str(
            muscle.getTendonSlackLength(),
        )
        ET.SubElement(mtu, "maxIsometricForce").text = str(
            muscle.getMaxIsometricForce(),
        )
        ET.SubElement(mtu, "strengthCoefficient").text = "1"

    # Add dofSet section
    dof_set = ET.SubElement(root, "dofSet")

    # Define DOFs and their associated muscles
    dof_muscles = {}
    coordinates = model.getCoordinateSet()
    print("Adding muscles to DOFs...")
    for i in range(coordinates.getSize()):
        coord = coordinates.get(i)
        coord_name = coord.getName()

        allowed_dofs = (
            [
                f"hip_flexion_{leg}",
                f"hip_adduction_{leg}",
                f"hip_rotation_{leg}",
                f"knee_angle_{leg}",
                f"ankle_angle_{leg}",
            ]
            if leg
            else [
                "hip_flexion_r",
                "hip_flexion_l",
                "hip_adduction_r",
                "hip_adduction_l",
                "hip_rotation_r",
                "hip_rotation_l",
                "knee_angle_r",
                "knee_angle_l",
                "ankle_angle_r",
                "ankle_angle_l",
            ]
        )
        if coord_name not in allowed_dofs:
            continue

        # Get muscles that cross this coordinate
        muscles_for_coord = []
        for j in range(muscle_set.getSize()):
            muscle = muscle_set.get(j)
            state = model.initSystem()
            model.realizePosition(state)

            try:
                moment_arm = muscle.computeMomentArm(state, coord)
                if abs(moment_arm) > 1e-6:  # Small threshold for numerical precision
                    muscles_for_coord.append(muscle.getName())
            except Exception as e:
                print(
                    f"Warning: Could not compute moment arm for muscle {muscle.getName()} and coordinate {coord_name}: {e}",
                )

        if muscles_for_coord:
            dof_muscles[coord_name] = muscles_for_coord

    # Create DOF elements
    for dof_name, muscle_names in dof_muscles.items():
        dof = ET.SubElement(dof_set, "dof")
        ET.SubElement(dof, "name").text = dof_name
        ET.SubElement(dof, "mtuNameSet").text = " ".join(muscle_names)

    # Add calibrationInfo section
    calibration_info = ET.SubElement(root, "calibrationInfo")
    uncalibrated = ET.SubElement(calibration_info, "uncalibrated")
    ET.SubElement(uncalibrated, "subjectID").text = os.path.basename(
        osimModelPath,
    ).replace(".osim", "")
    ET.SubElement(uncalibrated, "additionalInfo").text = ""

    # Add opensimModelFile reference
    ET.SubElement(root, "opensimModelFile").text = os.path.relpath(
        osimModelPath,
        os.path.dirname(outputCEINMSModelPath),
    )

    # Create the XML tree and save
    tree = ET.ElementTree(root)
    utils.save_pretty_xml(tree, outputCEINMSModelPath)

    print(f"CEINMS subject file created: {outputCEINMSModelPath}")


if __name__ == "__main__":
    subject_ids = Subject_utils().get_all_subjects()
    for subject_id in subject_ids:
        osim_model_path = str(
            Path("data") / "models" / subject_id / f"{subject_id}_Rajagopal2015_FAI_originalMass_opt_N10_hans.osim",
        )
        if not Path(osim_model_path).exists():
            print(f"Skipping {subject_id}: model path does not exist: {osim_model_path}")
            continue
        output_ceinms_model_path = str(File_utils().get_session_dir(subject_id) / "subjectUncalibrated.xml")
        leg = Subject_utils().get_measured_leg(subject_id).lower()

        create_ceinms_model(osim_model_path, output_ceinms_model_path, leg=leg)
