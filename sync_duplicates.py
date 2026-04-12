import shutil
import os

files_to_sync = {
    "server/app.py": [
        "clinical_trial_env/server/app.py",
        "clinical_trial_env/clinical_trial_env/server/app.py"
    ],
    "server/graders.py": [
        "clinical_trial_env/server/graders.py",
        "clinical_trial_env/clinical_trial_env/server/graders.py"
    ],
    "server/clinical_trial_environment.py": [
        "clinical_trial_env/server/clinical_trial_environment.py",
        "clinical_trial_env/clinical_trial_env/server/clinical_trial_environment.py"
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
        dest_dir = os.path.dirname(dest)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        shutil.copy2(src, dest)
        print(f"Synced {src} -> {dest}")
