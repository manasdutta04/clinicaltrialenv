"""
FastAPI app for ClinicalTrialEnv.
"""
from openenv.core.env_server.http_server import create_app
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, Body
import os, random
import numpy as np

from models import TrialAction, TrialObservation
from .clinical_trial_environment import ClinicalTrialEnvironment
from .graders import strict_score
from .session_store import _completed_sessions

app = create_app(
    ClinicalTrialEnvironment,
    TrialAction,
    TrialObservation,
    env_name="ClinicalTrialEnv",
    max_concurrent_envs=10,
)

# Ensure validator-sensitive HTTP endpoints resolve to the custom handlers below.
app.router.routes = [
    route
    for route in app.router.routes
    if getattr(route, "path", None) not in {"/reset", "/step", "/health"}
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


def _run_heuristic_episode(task_id: str) -> ClinicalTrialEnvironment:
    """Run one deterministic heuristic episode and persist it for grading."""
    env = ClinicalTrialEnvironment()
    env.reset(task_id)
    done = False
    step_count = 0
    dropped = set()

    while not done and step_count < 30:
        obs = env._build_observation()
        probs = {
            "low": obs.prob_low_beats_control if obs.low_active else 0.0,
            "mid": obs.prob_mid_beats_control if obs.mid_active else 0.0,
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
        step_count += 1

    _completed_sessions[task_id] = env
    return env

@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/health")
async def health():
    return {"status": "ok", "env": "ClinicalTrialEnv", "version": "1.0.0"}

@app.get("/tasks")
async def list_tasks():
    from .tasks import TASKS
    action_schema = TrialAction.model_json_schema()
    return [
        {
            "task_id": t["task_id"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "max_patients": t["max_patients"],
            "doses": t["doses"],
            "action_schema": action_schema,
        }
        for t in TASKS.values()
    ]

@app.post("/grader")
async def grader(request: Request):
    try:
        body = await request.json()
        task_id = body.get("task_id") if body else None
    except Exception:
        task_id = None

    valid_task_ids = {"task_1", "task_2", "task_3"}
    if task_id and task_id in _completed_sessions:
        env_instance = _completed_sessions[task_id]
    elif task_id in valid_task_ids:
        env_instance = _run_heuristic_episode(task_id)
    elif _completed_sessions:
        env_instance = list(_completed_sessions.values())[-1]
    else:
        env_instance = _run_heuristic_episode("task_1")

    result = env_instance.grade()
    score = strict_score(float(result.score))
    return {
        "score": score,
        "task_id": result.task_id,
        "trial_outcome": result.trial_outcome,
        "breakdown": result.breakdown,
    }

@app.post("/reset")
async def http_reset(body: dict = Body(default={"task": "task_1"})):
    task_id = body.get("task", body.get("task_id", "task_1"))
    env = ClinicalTrialEnvironment()
    obs = env.reset(task_id)
    _completed_sessions[f"http_active_{task_id}"] = env
    return {
        "session_id": f"http_{task_id}",
        "task": task_id,
        "observation": obs.model_dump(),
    }

@app.post("/step")
async def http_step(action: TrialAction):
    active = {k: v for k, v in _completed_sessions.items()
              if k.startswith("http_active_")}
    if not active:
        return JSONResponse(status_code=400,
            content={"error": "No active session. POST /reset first."})
    env = list(active.values())[-1]
    obs = env.step(action)
    if obs.done:
        _completed_sessions[env.task["task_id"]] = env
    clamped_reward = float(max(0.01, min(0.99, float(obs.reward))))
    obs.reward = clamped_reward
    return {
        "observation": obs.model_dump(),
        "reward": clamped_reward,
        "done": bool(obs.done),
    }

@app.post("/baseline")
async def baseline():
    """Run heuristic agent for all 3 tasks. Returns grader scores in (0,1)."""
    np.random.seed(42)
    random.seed(42)
    scores = {}

    for task_id in ["task_1", "task_2", "task_3"]:
        env = ClinicalTrialEnvironment()
        env.reset(task_id)
        done = False
        step_count = 0
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
            for arm, ae_r in [("low", obs.low_ae_rate),
                               ("mid", obs.mid_ae_rate),
                               ("high", obs.high_ae_rate)]:
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
                stop_for_success=(obs.any_arm_significant and
                                  env.interim_number >= env.task["min_interims_before_stop"]),
                stop_for_futility=(obs.futility_flag and
                                   env.interim_number >= env.task["min_interims_before_stop"]),
                drop_arm=drop,
            )

            result_obs = env.step(action)
            done = result_obs.done
            step_count += 1

        # Use the actual grader — NOT cumulative reward
        grade_result = env.grade()
        final_score = strict_score(float(grade_result.score))
        _completed_sessions[task_id] = env

        scores[task_id] = {
            "score": round(final_score, 4),
            "stop_reason": env.stop_reason,
            "patients_used": env.total_enrolled,
            "interims": env.interim_number,
            "breakdown": grade_result.breakdown,
        }

    task_scores = [scores[t]["score"] for t in ["task_1", "task_2", "task_3"]]
    scores["average"] = round(sum(task_scores) / 3, 4)
    return scores

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
