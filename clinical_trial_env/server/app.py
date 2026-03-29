"""
FastAPI app for ClinicalTrialEnv.
Uses create_app from openenv-core (matching the scaffold template pattern).
"""
from openenv.core.env_server.http_server import create_app
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os, random
import numpy as np

from models import TrialAction, TrialObservation
from .clinical_trial_environment import ClinicalTrialEnvironment

# ── Core OpenEnv app ───────────────────────────────────────────────────────────
app = create_app(
    ClinicalTrialEnvironment,
    TrialAction,
    TrialObservation,
    env_name="ClinicalTrialEnv",
    max_concurrent_envs=10,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Root Redirect ──────────────────────────────────────────────────────────────
from fastapi.responses import RedirectResponse

@app.get("/")
async def root_redirect():
    # Hugging Face initially loads the root URL. 
    # Redirect it to OpenEnv's built-in web interface.
    return RedirectResponse(url="/web")

# ── Serve frontend dashboard ───────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "env": "ClinicalTrialEnv", "version": "1.0.0"}

# ── /tasks ─────────────────────────────────────────────────────────────────────
@app.get("/tasks")
async def list_tasks():
    from .tasks import TASKS
    return [
        {
            "task_id": t["task_id"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "max_patients": t["max_patients"],
            "doses": t["doses"],
        }
        for t in TASKS.values()
    ]

# ── /grader ────────────────────────────────────────────────────────────────────
@app.post("/grader")
async def grader():
    env = ClinicalTrialEnvironment()
    return JSONResponse(
        status_code=200,
        content={"message": "Use WebSocket /ws to run an episode, then grade from the observation's reward field."}
    )

# ── /baseline ──────────────────────────────────────────────────────────────────
@app.post("/baseline")
async def baseline():
    """Run a full heuristic episode for all 3 tasks."""
    np.random.seed(42)
    random.seed(42)
    scores = {}

    for task_id in ["task_1", "task_2", "task_3"]:
        env = ClinicalTrialEnvironment()
        env.reset(task_id)
        done = False
        step_count = 0
        cumulative_reward = 0.0
        dropped = set()

        while not done and step_count < 30:
            obs = env._build_observation()
            probs = {
                "low":  obs.prob_low_beats_control  if obs.low_active  else 0.0,
                "mid":  obs.prob_mid_beats_control  if obs.mid_active  else 0.0,
                "high": obs.prob_high_beats_control if obs.high_active else 0.0,
            }
            total_prob = sum(probs.values()) + 0.3
            drop = None
            ae_thresh = env.task["ae_stopping_threshold"]
            for arm, ae_r in [("low", obs.low_ae_rate), ("mid", obs.mid_ae_rate), ("high", obs.high_ae_rate)]:
                if ae_r > ae_thresh * 0.80 and arm not in dropped:
                    drop = arm
                    dropped.add(arm)
                    break

            action = TrialAction(
                n_next_cohort=25,
                allocation_control=0.3 / total_prob,
                allocation_low=probs["low"] / total_prob,
                allocation_mid=probs["mid"] / total_prob,
                allocation_high=probs["high"] / total_prob,
                stop_for_success=(obs.any_arm_significant and env.interim_number >= env.task["min_interims_before_stop"]),
                stop_for_futility=(obs.futility_flag and env.interim_number >= env.task["min_interims_before_stop"]),
                drop_arm=drop,
            )

            result_obs = env.step(action)
            done = result_obs.done
            cumulative_reward += result_obs.reward
            step_count += 1

        scores[task_id] = {
            "score": round(cumulative_reward, 4),
            "stop_reason": env.stop_reason,
            "patients_used": env.total_enrolled,
            "interims": env.interim_number,
        }

    scores["average"] = round(sum(v["score"] for v in scores.values()) / 3, 4)
    return scores
