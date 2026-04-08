import ast
import json
import os
import subprocess
import time

TASK_IDS = ["task_1", "task_2", "task_3"]


def _assert_strict_score(label, value):
    if isinstance(value, bool):
        raise AssertionError(f"{label} must be numeric, got bool")
    if not isinstance(value, (int, float)):
        raise AssertionError(f"{label} must be numeric, got {type(value).__name__}")
    if not (0 < float(value) < 1):
        raise AssertionError(f"{label}={value} is outside strict range (0,1)")


def _check_syntax():
    for file_path in ["inference.py", "clinical_trial_env/server/app.py", "clinical_trial_env/server/graders.py"]:
        with open(file_path, "r", encoding="utf-8") as handle:
            ast.parse(handle.read())
    print("All syntax OK")


def _check_yaml():
    import yaml

    with open("openenv.yaml", "r", encoding="utf-8") as handle:
        yaml.safe_load(handle)
    with open("clinical_trial_env/openenv.yaml", "r", encoding="utf-8") as handle:
        yaml.safe_load(handle)
    print("YAML OK")


def _check_inference_source():
    with open("inference.py", "r", encoding="utf-8") as handle:
        src = handle.read()
    assert 'HF_TOKEN = os.getenv("HF_TOKEN")' in src or "HF_TOKEN = os.getenv('HF_TOKEN')" in src, "HF_TOKEN must have NO default"
    assert "API_BASE_URL = os.getenv" in src, "API_BASE_URL missing"
    assert "MODEL_NAME = os.getenv" in src, "MODEL_NAME missing"
    assert "[START]" in src, "[START] log missing"
    assert "[STEP]" in src, "[STEP] log missing"
    assert "[END]" in src, "[END] log missing"
    assert "from openai import OpenAI" in src, "OpenAI client missing"
    print("inference.py checklist: ALL PASS")


def _run_http_checks():
    import requests

    print("Starting server background test...")
    proc = subprocess.Popen(["uvicorn", "clinical_trial_env.server.app:app", "--port", "7860"])
    time.sleep(4)
    try:
        health = requests.get("http://localhost:7860/health", timeout=30)
        print("Health:", health.status_code)

        baseline = requests.post("http://localhost:7860/baseline", timeout=60).json()
        scores = []
        for task_id in TASK_IDS:
            score = baseline[task_id]["score"]
            _assert_strict_score(f"baseline.{task_id}.score", score)
            scores.append(score)
        print("Baseline scores:", scores)
        print("All in (0,1):", True)

        for task_id in TASK_IDS:
            grader = requests.post("http://localhost:7860/grader", json={"task_id": task_id}, timeout=60).json()
            _assert_strict_score(f"grader.{task_id}.score", grader.get("score"))
            print(f"Grader {task_id}:", grader)

        tasks_response = requests.get("http://localhost:7860/tasks", timeout=30).json()
        assert tasks_response and "action_schema" in tasks_response[0], "action_schema missing in /tasks"
        print("action_schema present:", True)
    finally:
        proc.terminate()
        proc.wait()


def _run_inference_checks():
    print("Testing inference.py local execution...")
    proc = subprocess.Popen(["uvicorn", "clinical_trial_env.server.app:app", "--port", "7860"])
    time.sleep(4)
    try:
        env = dict(os.environ)
        env["ENV_URL"] = "http://localhost:7860"
        env["HF_TOKEN"] = "test"
        env["API_BASE_URL"] = "http://127.0.0.1:9/v1/"
        env["MODEL_NAME"] = "dummy-model"
        inf_proc = subprocess.run(["python3", "inference.py"], capture_output=True, text=True, env=env)
        if inf_proc.returncode != 0:
            raise AssertionError(f"inference.py exited with {inf_proc.returncode}\nSTDERR:\n{inf_proc.stderr}")

        end_scores = {}
        print("inference output lines related to START/STEP/END:")
        for line in inf_proc.stdout.splitlines():
            if line.startswith("[START") or line.startswith("[STEP") or line.startswith("[END") or line.startswith("[SUMMARY"):
                print((line[:160] + "...") if len(line) > 160 else line)
            if line.startswith("[END] "):
                payload = json.loads(line[len("[END] "):])
                task_id = payload.get("task_id")
                if task_id in TASK_IDS:
                    end_scores[task_id] = payload.get("score")

        missing = [task_id for task_id in TASK_IDS if task_id not in end_scores]
        if missing:
            raise AssertionError(f"Missing [END] entries for tasks: {missing}")

        for task_id in TASK_IDS:
            _assert_strict_score(f"inference.END.{task_id}.score", end_scores[task_id])
        print("Inference END score strict check: PASS")
    finally:
        proc.terminate()
        proc.wait()


def main():
    _check_syntax()
    _check_yaml()
    _check_inference_source()
    _run_http_checks()
    _run_inference_checks()
    print("ALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
