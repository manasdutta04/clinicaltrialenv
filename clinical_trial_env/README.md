---
title: ClinicalTrialEnv
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# ClinicalTrialEnv — Adaptive Clinical Trial Design RL Environment

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-gray)](https://github.com)

## Real-world motivation
Poorly designed clinical trials waste an estimated **$1.3 billion** per failed drug program. The FDA has recently issued guidance encouraging adaptive trial designs to make trials faster, safer, and more likely to succeed. This environment simulates adaptive trials, allowing RL agents to act as trial statisticians—designing phase II trials dynamically to minimize the number of patients exposed while maximizing statistical power.

## The pharmacological simulation
Patient responses are simulated using the Emax pharmacological model (Hill equation), which is the FDA/ICH standard model for dose-response relationships:
`response_rate(dose) = baseline + (Emax * dose^hill) / (ED50^hill + dose^hill)`
Adverse events are modeled using a quadratic dose-dependent relationship. The true underlying parameters are strictly hidden from the agent.

## Environment description
At each step (interim analysis), the RL agent observes the trial's progress, response rates, and adverse event (AE) rates. Based on these observations, it must determine the size of the next patient cohort, the allocation ratio across dose arms, whether to drop futile or unsafe arms, or whether to stop the trial early for success or futility.

## Observation space

| Feature | Type | Description |
| :--- | :--- | :--- |
| `interim_number` | `int` | Current number of interim analyses performed |
| `total_patients_enrolled` | `int` | Total enrolled patients across all arms |
| `budget_remaining` | `int` | Remaining patient spots before budget exhausted |
| `control_response_rate` | `float` | Observed response rate for control arm |
| `[arm]_response_rate` | `float` | Observed response rates for low, mid, high arms |
| `p_value_[arm]` | `float` | Two-sided Fisher's exact test p-value vs control |
| `prob_[arm]_beats_control` | `float` | Bayesian Beta-Binomial posterior probability |
| `any_arm_significant` | `bool` | True if any arm's p-value < 0.05 |
| `futility_flag` | `bool` | True if predictive probability of success < 10% |

## Action space

| Feature | Type | Description |
| :--- | :--- | :--- |
| `n_next_cohort` | `int` | Number of patients to enroll in next step (5-100) |
| `allocation_control` | `float` | Relative allocation for control [0, 1] |
| `allocation_[arm]` | `float` | Relative allocation for treatment arms [0, 1] |
| `stop_for_success` | `bool` | Signal early stop if p<0.05 |
| `stop_for_futility` | `bool` | Signal early stop for futility |
| `drop_arm` | `categorical` | Arm to permanently drop (`null`, `low`, `mid`, `high`) |

## Tasks

1. **task_1 (Easy): Phase II dose-finding — effective drug.** Find the optimal dose and reach p<0.05 using as few patients as possible from a 200 patient budget.
2. **task_2 (Medium): Phase II — efficacy vs safety tradeoff.** A drug with good efficacy at high doses but dangerous adverse events. Balance efficacy and safety.
3. **task_3 (Hard): Rare disease trial — weak signal, tiny budget.** Only 150 patients available for a modest effect size. Allocate efficiently to squeeze statistical power.

## Reward function
The reward function consists of a dense shaping reward at every step and a final terminal reward (grader score). The shaping reward provides:
1. `+ 0.02 * best_posterior` for appropriately prioritizing the strongest arm
2. `- 0.03 * (AE / threshold)` penalty for approaching the safety stopping boundary
3. `+ 0.05` bonus upon crossing statistical significance (p<0.05)

The terminal reward is the explicit grader score (0.0 to 1.0) returned via the `/grader` endpoint.

## Running the Baseline Solver Agent

To demonstrate environmental flexibility and solvability, you can run the included autonomous "Heuristic" baseline solver.

```bash
python3 baseline_agent.py
```
This boots up an autonomous agent that parses Bayesian posterior outcomes and dynamically tightens the Trial's **inclusion criteria** to force statistical significance while staying under budget. An identical training layout is provided in `Environment_Demo.ipynb`.

## Statistical methods
The environment uses `scipy` standard methodologies rather than approximations.
* **Fisher's Exact Test:** Computes strict p-values for small-sample responder counts.
* **Beta-Binomial Bayesian Inference:** Computes conservative posteriors using a continuous conjugate conjugate update.
* **Predictive Probability of Success:** Full Monte Carlo estimation of future trial success dynamically informs the futility flags.
* **Inclusion strictness:** 
    1. **Biomarker strictness vs Budget:** RL agent must tune inclusion criteria. High strictness equals massive pharmacological effect sizes but heavily penalizes patient budget tracking.
    2. **Patient Simulation Model**: Emax Model for Pharmacodynamics alongside Bayesian Inference testing.
    3. **Budget and Ethics Constraint**: Agent is bound by max_patients budget and forced to monitor adversarial AE thresholds.

## Visual dashboard
The built-in web dashboard provides a robust visualization of the trial trajectory:
![Dashboard Placeholder](dashboard.jpg)
The dashboard plots the p-values per interim analysis and allows users to manually progress through the environment steps.

## Setup locally

```bash
pip install openenv-core
pip install -e .
uvicorn clinical_trial_env.server.app:app --port 7860 --reload
# Open http://localhost:7860
```

## Docker

```bash
docker build -f clinical_trial_env/server/Dockerfile -t clinicaltrialenv .
docker run -p 7860:7860 -e ENABLE_WEB_INTERFACE=true clinicaltrialenv
```

## Deploy to HuggingFace Spaces

```bash
openenv push --repo-id <your-hf-username>/clinical-trial-env
```

## API reference

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/health` | GET | Check API server status |
| `/schema` | GET | Returns action and observation JSON schemas |
| `/reset` | POST | Initializes trial: `{"task_id": "task_1"}` |
| `/step` | POST | Send actions, returns WSObservationResponse |

## Example session

**WebSocket interactions:**
```json
// Reset the environment
{"type": "reset", "data": {"task_id": "task_1"}}

// Take a step
{"type": "step", "data": {"n_next_cohort": 25, "allocation_control": 0.25, "allocation_low": 0.25, "allocation_mid": 0.25, "allocation_high": 0.25, "stop_for_success": false, "stop_for_futility": false, "drop_arm": null}}
```

**HTTP interactions:**
```bash
curl http://localhost:7860/tasks
curl -X POST http://localhost:7860/baseline
```

## Baseline results table

| Task | Score | Outcome | Average Interims | Time |
| :--- | :--- | :--- | :--- | :--- |
| task_1 (easy) | ~ 0.81 | success | ~ 5 | <1.0s |
| task_2 (medium) | ~ 0.58 | success | ~ 6 | <1.0s |
| task_3 (hard) | ~ 0.35 | futility | ~ 6 | <1.0s |
