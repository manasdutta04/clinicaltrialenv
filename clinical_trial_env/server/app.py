from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openenv.core.env_server import create_web_interface_app
import os, asyncio, random, numpy as np

from .clinical_trial_environment import ClinicalTrialEnvironment
from ..models import TrialAction, TrialObservation

# Create base app using OpenEnv's built-in web interface
env = ClinicalTrialEnvironment()
app = create_web_interface_app(env, TrialAction, TrialObservation)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Serve the visual dashboard at root ────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
async def serve_dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# ── Health check (hackathon validator pings this first) ───────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "env": "ClinicalTrialEnv", "version": "1.0.0"}

# ── /tasks — required by hackathon spec ──────────────────────────────────────
@app.get("/tasks")
async def list_tasks():
    from .tasks import TASKS
    from ..models import TrialAction
    import dataclasses
    action_schema = {
        f.name: str(f.type) for f in dataclasses.fields(TrialAction)
    }
    return [
        {
            "task_id": t["task_id"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "max_patients": t["max_patients"],
            "doses": t["doses"],
            "action_schema": action_schema
        }
        for t in TASKS.values()
    ]

# ── /grader — required by hackathon spec ─────────────────────────────────────
@app.post("/grader")
async def grader():
    if env.episode_active:
        return JSONResponse(
            status_code=400,
            content={"error": "Episode still active. Complete it first."}
        )
    if env.task is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No episode run yet. Call reset first."}
        )
    result = env.grade()
    return {
        "score": result.score,
        "task_id": result.task_id,
        "trial_outcome": result.trial_outcome,
        "breakdown": result.breakdown
    }

# ── /baseline — required by hackathon spec ────────────────────────────────────
@app.post("/baseline")
async def baseline():
    """
    Run a full heuristic-agent episode for all 3 tasks.
    Heuristic: response-adaptive allocation + stop for success when p<0.05.
    Must complete in under 90 seconds.
    """
    np.random.seed(42)
    random.seed(42)
    scores = {}

    for task_id in ["task_1", "task_2", "task_3"]:
        await env.reset(task_id)
        done = False
        step_count = 0

        while not done and step_count < 30:
            obs_data = env._build_observation()

            # Heuristic: allocate proportional to posterior probabilities
            probs = {
                "low":  obs_data.prob_low_beats_control  if obs_data.low_active  else 0.0,
                "mid":  obs_data.prob_mid_beats_control  if obs_data.mid_active  else 0.0,
                "high": obs_data.prob_high_beats_control if obs_data.high_active else 0.0,
            }
            total_prob = sum(probs.values()) + 0.3  # +0.3 for control
            alloc_ctrl = 0.3 / total_prob
            alloc_low  = probs["low"]  / total_prob
            alloc_mid  = probs["mid"]  / total_prob
            alloc_high = probs["high"] / total_prob

            # Drop arm if AE > 80% of threshold
            ae_thresh = env.task["ae_stopping_threshold"]
            drop = None
            for arm, ae_r in [("low", obs_data.low_ae_rate),
                               ("mid", obs_data.mid_ae_rate),
                               ("high", obs_data.high_ae_rate)]:
                if ae_r > ae_thresh * 0.80 and arm not in env.dropped_arms:
                    drop = arm
                    break

            action = TrialAction(
                n_next_cohort=25,
                allocation_control=alloc_ctrl,
                allocation_low=alloc_low,
                allocation_mid=alloc_mid,
                allocation_high=alloc_high,
                stop_for_success=(
                    obs_data.any_arm_significant and
                    env.interim_number >= env.task["min_interims_before_stop"]
                ),
                stop_for_futility=(
                    obs_data.futility_flag and
                    env.interim_number >= env.task["min_interims_before_stop"]
                ),
                drop_arm=drop
            )

            result = await env.step(action)
            done = result.done
            step_count += 1

        grade_result = env.grade()
        scores[task_id] = round(grade_result.score, 4)

    scores["average"] = round(sum(scores.values()) / 3, 4)
    return scores
