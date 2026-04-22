import datetime
import subprocess
import sys
import timeit
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.file_utils import File_utils


def run_ceinms_exe(subject: str | None = None, trial: str | None = None):
    # find ceinms executable
    project_path = Path(__file__).parent.parent

    ceinms_exe_search = list(project_path.rglob("CEINMS.exe"))

    ceinms_exe = ceinms_exe_search[0].resolve()  # Resolve once for executable
    if not ceinms_exe.is_file():
        raise FileNotFoundError("CEINMSoptimise.exe not found in the project directory")

    if subject is None or trial is None:
        raise ValueError("Both subject and trial must be provided")

    file_utils = File_utils()
    trial_path = file_utils.get_trial_dir(
        subject,
        trial,
    )  # Remove resolve - get_trial_dir likely returns absolute
    if not trial_path.exists():
        raise FileNotFoundError(f"Trial path does not exist: {trial_path}")

    session_path = trial_path.parent  # Already absolute
    if not session_path.exists():
        raise FileNotFoundError(f"Session path does not exist: {session_path}")

    ceinms_setup_file = trial_path / "setup_ceinms_optimise.xml"
    if not ceinms_setup_file.is_file():
        raise FileNotFoundError(
            f"CEINMS setup file not found: {ceinms_setup_file}",
        )

    cmd = f"{ceinms_exe!s} -S {ceinms_setup_file!s}"

    log_file_path = (
        File_utils().get_trial_dir(subject, trial) / "non_optimised" / "ceinms.log"
    )

    if not log_file_path.parent.exists():
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_with_redirect = f"{cmd} 2>&1 | Tee-Object -FilePath '{log_file_path}'; exit"

    cmd = f"{ceinms_exe!s} -S {ceinms_setup_file.resolve()!s}"

    log_file_path = (
        File_utils().get_trial_dir(subject, trial) / "Optimised" / "ceinms.log"
    )
    if not log_file_path.parent.exists():
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Method 1: Direct execution (most compatible)
    exe_dir = ceinms_exe.parent

    start_time = timeit.default_timer()

    try:
        print("Running CEINMS optimization...")
        print(f"Executable: {ceinms_exe}")
        print(f"Setup file: {ceinms_setup_file}")
        print(f"Trial directory: {trial_path}")
        print(f"Exe directory: {exe_dir}")

        # PowerShell script that adds exe dir to PATH, then runs from trial dir
        ps_script = f"""
$ErrorActionPreference = "Continue"
$env:PATH = "{exe_dir};$env:PATH"
Set-Location "{trial_path}"
& "{ceinms_exe}" -S "{ceinms_setup_file}"
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
                    #                    print(line, end="")
                    log_file.write(line)
                    log_file.flush()

            process.wait()
        stop_time = timeit.default_timer()
        run_time = stop_time - start_time

        print(f"\nOptimization complete. Log saved to: {log_file_path}")
        if run_time < 60:
            print(f"\nCeinms run time: {run_time:.2f}s")
        else:
            run_time = datetime.timedelta(seconds=run_time)
            print(f"\nCeinms run time: {run_time}")

    except FileNotFoundError:
        raise RuntimeError(
            f"Could not execute {ceinms_exe}. Check if file exists and is executable.",
        )
    except Exception as e:
        raise RuntimeError(f"Error running CEINMS: {e}")


if __name__ == "__main__":
    run_ceinms_exe(subject="029", trial="Run_baselineA1")
