import shutil
import os

files_to_sync = {
    "server/app.py": [
        "clinical_trial_env/server/app.py",
        "clinical_trial_env/clinical_trial_env/server/app.py"
    ],
    "inference.py": [
        "clinical_trial_env/inference.py",
        "clinical_trial_env/clinical_trial_env/inference.py"
    ]
}

for src, destinations in files_to_sync.items():
    if not os.path.exists(src):
        print(f"Source missing: {src}")
        continue
    for dest in destinations:
        if os.path.exists(dest):
            shutil.copy2(src, dest)
            print(f"Synced {src} -> {dest}")
        else:
            print(f"Destination missing (skipping): {dest}")

