---
title: ClinicalTrialEnv
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# ClinicalTrialEnv

ClinicalTrialEnv is a live OpenEnv reinforcement-learning environment for adaptive Phase II clinical trial design. Agents act as trial statisticians: they choose cohort size, allocation across dose arms, early stopping, arm dropping, and patient inclusion strictness to reach significance quickly while protecting patient safety.

Built by Manas Dutta.

Live deployment:
- [Hugging Face Space](https://huggingface.co/spaces/manasdutta04/clinicaltrialenv)
- [GitHub repository](https://github.com/manasdutta04/clinicaltrialenv)

## Tasks

1. `task_1`: effective drug with a clear signal and a 200-patient budget
2. `task_2`: efficacy versus safety tradeoff with a risky high-dose arm
3. `task_3`: rare-disease setting with only 150 patients and weak effects

## Observation Space

The environment exposes trial progress, response rates, adverse-event rates, patient counts, p-values, Bayesian posterior probabilities, estimated power, active-arm flags, and stop metadata. The full schema is available from `GET /tasks` and `GET /schema`.

## Action Space

Agents submit:
- `n_next_cohort`
- `allocation_control`
- `allocation_low`
- `allocation_mid`
- `allocation_high`
- `stop_for_success`
- `stop_for_futility`
- `drop_arm`
- `inclusion_criteria_strictness`

## Setup

```bash
pip install openenv-core
pip install -e .
uvicorn server.app:app --port 7860 --reload
```

Open:
- `http://localhost:7860/` for the custom dashboard
- `http://localhost:7860/web` for the OpenEnv built-in UI

Run the LLM agent against the live environment:

```bash
API_BASE_URL=<url> MODEL_NAME=<model> HF_TOKEN=<token> python inference.py
```

## LLM Agent (`inference.py`)

`inference.py` is the hackathon submission agent used by judges against the deployed environment.

- It uses the OpenAI client for all model calls through an OpenAI-compatible endpoint.
- It emits exact `[START]`, `[STEP]`, and `[END]` structured logs for evaluation.
- By default it targets `meta-llama/Llama-3.1-8B-Instruct` through the Hugging Face Inference API.
- It can be redirected to any OpenAI-compatible API by setting `API_BASE_URL` and `MODEL_NAME`.
- If model calls fail, it falls back to a built-in heuristic so the run still completes.

## API Reference

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/` | GET | Serve the custom dashboard directly with HTTP 200 |
| `/web` | GET | OpenEnv built-in web UI |
| `/health` | GET | Healthcheck for deployment validators |
| `/tasks` | GET | List tasks plus `action_schema` |
| `/reset` | POST | Start an HTTP episode for validator compatibility |
| `/step` | POST | Step the latest active HTTP episode |
| `/grader` | POST | Return the final normalized score for the latest completed episode |
| `/baseline` | POST | Run the heuristic agent across all three tasks |
| `/ws` | WebSocket | Main OpenEnv interaction channel |

## Baseline Results

| Task | Typical Score | Outcome | Notes |
| :--- | :--- | :--- | :--- |
| `task_1` | ~0.8 | `success` | Clear efficacy signal with efficient stopping |
| `task_2` | ~0.55-0.65 | `success` | Balances efficacy with AE control |
| `task_3` | ~0.2-0.4 | `futility` | Futility is often the correct behavior because the task is designed to reward efficient rare-disease decision making, not forced over-enrollment |

## Notebook Demo

`Environment_Demo.ipynb` is a runnable Colab notebook that connects to the live Space, runs a heuristic policy over WebSocket, plots p-value trajectories, and fetches the final grader breakdown.

## Deploy to Hugging Face Spaces

```bash
openenv push --repo-id manasdutta04/clinicaltrialenv
```
