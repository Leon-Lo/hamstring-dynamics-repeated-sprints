import os
import shutil

originDir = r"E:\DataFolder\Running_FAI\c3d_files"
destinationDir = (
    r"C:\Git\1_current_projects\Fatigue-prediction-MSC-Thesis\data\simulations"
)

fileDict = {
    "Run_baselineA1": ["Run_baselineA1", "Run_baseline1"],
    "Run_baselineB1": ["Run_baselineB1"],
    "RunL1": ["RunL1"],
    "RunM1": ["RunM1"],
}

# walk through origin dir and print all .c3d files
for root, dirs, files in os.walk(originDir):
    for filename in files:
        if filename.endswith(".c3d"):
            for main_name, possible_names in fileDict.items():
                if filename.strip(".c3d") in possible_names:
                    # change the name to key value
                    old_path = os.path.join(root, filename)
                    new_path = os.path.join(root, main_name + ".c3d")
                    if old_path != new_path:
                        os.rename(old_path, new_path)
                        print(f"Renamed {old_path} to {new_path}")

                    subject = new_path.split(os.sep)[-3]
                    session = new_path.split(os.sep)[-2]
                    dest_path = os.path.join(
                        destinationDir, subject, session, main_name
                    )

                    if os.path.exists(dest_path):
                        if os.path.exists(os.path.join(dest_path, main_name + ".c3d")):
                            continue

                        shutil.copy2(new_path, dest_path)
                        print(f"Copied {new_path} to {dest_path}")
