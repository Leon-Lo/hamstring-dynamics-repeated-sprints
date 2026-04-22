import datetime
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import File_utils, Subject_utils


def main():
    print("Running CEINMS calibration...")
    # Prepare CEINMS calibration executable and setup file paths

    # subject_ids = Subject_utils().get_all_subjects()
    subject_ids = ["024"]
    for subject_id in subject_ids:
        session_dir = File_utils().get_session_dir(subject_id).absolute()
        ceinms_calibration_exe = str(
            Path("ceinms/executables/ceinms-nn-calibrate.exe").absolute(),
        )
        calibration_setup = str(session_dir / "calibrationSetup_ceinms-nn_hybrid.xml")
        if not Path(ceinms_calibration_exe).exists():
            raise FileNotFoundError(
                f"CEINMS calibration executable not found at {ceinms_calibration_exe}",
            )
        if not Path(calibration_setup).exists():
            raise FileNotFoundError(
                f"CEINMS calibration setup file not found at {calibration_setup}",
            )

        # Parse the outputDirectory from the calibration setup XML
        tree = ET.parse(calibration_setup)
        root = tree.getroot()
        output_dir = None
        for elem in root.iter():
            if elem.tag == "outputDirectory":
                output_dir = elem.text
                break

        if output_dir:
            # Use pathlib to resolve relative paths
            output_path = Path(calibration_setup).parent / output_dir
            output_path = output_path.resolve()

            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Warning! Could not find output tag in {calibration_setup}")

        # Run the CEINMS calibration executable
        log_file_path = session_dir / "ceinms_cal.log"

        if not log_file_path.parent.exists():
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

        exe_dir = Path(ceinms_calibration_exe).parent

        try:
            print("Running CEINMS calibration...")
            print(f"Executable: {ceinms_calibration_exe}")
            print(f"Setup file: {calibration_setup}")
            print(f"Exe directory: {exe_dir}")

            # PowerShell script that adds exe dir to PATH
            ps_script = f"""
$ErrorActionPreference = "Continue"
$env:PATH = "{exe_dir};$env:PATH"
& "{ceinms_calibration_exe}" -S "{calibration_setup}" 2>&1
exit $LASTEXITCODE
"""

            cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_script]

            with open(log_file_path, "w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                # Stream output
                if process.stdout:
                    for line in process.stdout:
                        print(line, end="")
                        log_file.write(line)
                        log_file.flush()

                process.wait()
        except FileNotFoundError:
            raise RuntimeError(
                f"Could not execute {ceinms_calibration_exe}. Check if file exists and is executable.",
            )
        except Exception as e:
            raise RuntimeError(f"Error running CEINMS calibration: {e}")

        print(
            f"Finished calibrating model for {subject_id} at {datetime.datetime.now()}",
        )


if __name__ == "__main__":
    main()
