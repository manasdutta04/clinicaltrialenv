# ClinicalTrialEnv Testing Guide

This guide reflects the current submission-ready interface for ClinicalTrialEnv, including the hackathon validator checks, the LLM agent runner, the dashboard, and the live Space deployment.

## Local Verification

From the repository root:

```bash
python3 -c "import ast; ast.parse(open('inference.py').read()); print('syntax OK')"
python3 -c "import ast; ast.parse(open('server/app.py').read()); print('syntax OK')"
python3 -c "import yaml; yaml.safe_load(open('openenv.yaml')); print('yaml OK')"
```

Start the local server:

```bash
python3 -m uvicorn server.app:app --port 7860
```

In another terminal, run the validator-style checks:

```bash
curl -s http://127.0.0.1:7860/health
curl -s -X POST http://127.0.0.1:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task":"task_1"}'
curl -s http://127.0.0.1:7860/tasks
curl -s -X POST http://127.0.0.1:7860/baseline
curl -s -X POST http://127.0.0.1:7860/grader \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task_1"}'
```

Expected behavior:
- `/health` returns `{"status":"ok","env":"ClinicalTrialEnv","version":"1.0.0"}`
- `/reset` returns `session_id`, `task`, and an `observation`
- `/tasks` includes `action_schema`
- `/baseline` returns per-task scores normalized to `[0.0, 1.0]`
- `/grader` returns the final normalized score for the most recent completed run

## LLM Agent Check

`inference.py` is the file judges use. It must live at the repository root and emit exact `[START]`, `[STEP]`, and `[END]` lines.

Run it locally against the local server:

```bash
HF_TOKEN=test \
API_BASE_URL=http://127.0.0.1:11434/v1/ \
MODEL_NAME=llama3 \
ENV_URL=http://127.0.0.1:7860 \
python3 inference.py
```

What to look for:
- A single `[START]` line at the beginning of each task
- One `[STEP]` line per step
- A single `[END]` line after each task
- `task_1`, `task_2`, and `task_3` run sequentially
- If the local model endpoint is unavailable, the heuristic fallback still completes the run

Example shape:

```text
[START] {"task_id": "task_1", "model": "llama3", "env_url": "http://127.0.0.1:7860"}
[STEP] {"step": 1, "action": {...}, "observation": {...}, "reward": 0.0197, "done": false}
[END] {"task_id": "task_1", "total_steps": 3, "total_reward": 0.7102, "score": 0.6248, "outcome": "success"}
```

## Dashboard Checks

Custom dashboard:
- Local: `http://localhost:7860/`
- Live: `https://manasdutta04-clinicaltrialenv.hf.space/`

OpenEnv built-in UI:
- Local: `http://localhost:7860/web`
- Live: `https://manasdutta04-clinicaltrialenv.hf.space/web`

Manual UI checks:
- The root URL should load directly with HTTP `200`, not redirect
- Task switching should reset the episode
- The allocation warning should only appear after user interaction
- Auto-run should show a visible `Cancel` button while running
- The p-value chart should include a real dashed threshold dataset at `p = 0.05`
- The score overlay should show both the numeric score and a short human-readable interpretation

## Notebook Check

`Environment_Demo.ipynb` is intended to be judge-friendly and Colab-friendly.

Quick validation:

```bash
python3 -m json.tool clinical_trial_env/Environment_Demo.ipynb >/dev/null && echo "notebook JSON OK"
```

Notebook contents:
- installs dependencies
- connects to the live WebSocket endpoint
- runs `task_1` with the heuristic agent
- prints interim p-values and response rates in a table
- plots p-value trajectories with a `0.05` threshold
- fetches and prints the grader breakdown

## Live Space Checks

Once the Space rebuilds, run:

```bash
curl -s https://manasdutta04-clinicaltrialenv.hf.space/health
curl -s -X POST https://manasdutta04-clinicaltrialenv.hf.space/baseline
curl -s -X POST https://manasdutta04-clinicaltrialenv.hf.space/grader \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task_1"}'
```

Expected live behavior after deployment:
- `/health` matches the local custom response
- `/baseline` includes normalized scores and grading breakdowns
- `/grader` returns a real score instead of the old `"Use WebSocket /ws..."` stub

## Submission Checklist

Before final submission, confirm:
- `inference.py` exists at the repository root
- `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, and `LOCAL_IMAGE_NAME` are read with `os.getenv()`
- only `API_BASE_URL` and `MODEL_NAME` have defaults
- all LLM calls go through `from openai import OpenAI`
- stdout uses exact `[START]`, `[STEP]`, and `[END]` prefixes
- `GET /` returns `200`
- `GET /tasks` includes `action_schema`
- `POST /baseline` and `POST /grader` return scores in `[0, 1]`
- `openenv.yaml` includes `inclusion_criteria_strictness`, the full observation schema, and `author: "Manas Dutta"`
