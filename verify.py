import os
import ast
import subprocess

try:
    for f in ['inference.py','clinical_trial_env/server/app.py','clinical_trial_env/server/graders.py']:
        ast.parse(open(f).read())
    print('All syntax OK')
except Exception as e:
    print('Syntax ERROR:', e)

try:
    import yaml
    yaml.safe_load(open('openenv.yaml'))
    yaml.safe_load(open('clinical_trial_env/openenv.yaml'))
    print('YAML OK')
except Exception as e:
    print('YAML ERROR:', e)

try:
    src = open('inference.py').read()
    assert 'HF_TOKEN = os.getenv("HF_TOKEN")' in src or "HF_TOKEN = os.getenv('HF_TOKEN')" in src, 'HF_TOKEN must have NO default'
    assert 'API_BASE_URL = os.getenv' in src, 'API_BASE_URL missing'
    assert 'MODEL_NAME = os.getenv' in src, 'MODEL_NAME missing'
    assert '[START]' in src, '[START] log missing'
    assert '[STEP]' in src, '[STEP] log missing'
    assert '[END]' in src, '[END] log missing'
    assert 'from openai import OpenAI' in src, 'OpenAI client missing'
    print('inference.py checklist: ALL PASS')
except Exception as e:
    print('inference.py ERROR:', e)

print("Starting server background test...")
import time
proc = subprocess.Popen(["uvicorn", "clinical_trial_env.server.app:app", "--port", "7860"])
time.sleep(4)

try:
    import requests
    r1 = requests.get("http://localhost:7860/health")
    print("Health:", r1.status_code)
    
    r2 = requests.post("http://localhost:7860/baseline")
    d = r2.json()
    tasks = [k for k in d if k != 'average']
    scores = [d[k]['score'] for k in tasks]
    print('Scores:', scores)
    print('All in [0,1]:', all(0 <= s <= 1 for s in scores))
    
    r3 = requests.post("http://localhost:7860/grader", json={"task_id":"task_1"})
    print("Grader task_1:", r3.json())
    
    r4 = requests.get("http://localhost:7860/tasks")
    t = r4.json()
    print('action_schema present:', 'action_schema' in t[0])

finally:
    proc.terminate()
    proc.wait()
    
# test inference log
print("Testing inference.py local execution...")
proc2 = subprocess.Popen(["uvicorn", "clinical_trial_env.server.app:app", "--port", "7860"])
time.sleep(4)
try:
    os.environ["ENV_URL"] = "http://localhost:7860"
    os.environ["HF_TOKEN"] = "test"
    inf_proc = subprocess.run(["python", "inference.py"], capture_output=True, text=True)
    out = inf_proc.stdout
    lines = out.split("\n")
    print("inference output lines related to START/STEP/END:")
    for line in lines:
        if line.startswith("[START") or line.startswith("[STEP") or line.startswith("[END") or line.startswith("[SUM"):
            print(line[:80] + "...")
finally:
    proc2.terminate()
    proc2.wait()
