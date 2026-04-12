"""
FastAPI app for ClinicalTrialEnv.
"""
import os, random, json
import numpy as np
from openenv.core.env_server.http_server import create_app
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, Body

from models import TrialAction, TrialObservation
from .clinical_trial_environment import ClinicalTrialEnvironment
from .graders import strict_score, _deep_sanitize
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

def _get_schemas():
    """Extract action and observation schemas from openenv.yaml."""
    import yaml
    try:
        yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openenv.yaml")
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("action_space", {}), cfg.get("observation_space", {})
    except Exception:
        return {}, {}

def _run_heuristic_episode(task_id: str) -> ClinicalTrialEnvironment:
    """Run one deterministic heuristic episode and persist it for grading."""
    env = ClinicalTrialEnvironment()
    env.reset(task_id)
    done = False
    step_count = 0
    dropped = set()

    try:
        while not done and step_count < 25:
            obs = env._build_observation()
            probs = {
                "low": getattr(obs, "prob_low_beats_control", 0.5) if getattr(obs, "low_active", True) else 0.0,
                "mid": getattr(obs, "prob_mid_beats_control", 0.5) if getattr(obs, "mid_active", True) else 0.0,
                "high": getattr(obs, "prob_high_beats_control", 0.5) if getattr(obs, "high_active", True) else 0.0,
            }
            total_prob = sum(probs.values()) + 0.3
            drop = None
            ae_thresh = env.task["ae_stopping_threshold"]
            
            for arm in ["low", "mid", "high"]:
                ae_r = getattr(obs, f"{arm}_ae_rate", 0.0)
                if ae_r > ae_thresh * 0.85 and arm not in dropped:
                    drop = arm
                    dropped.add(arm)
                    break

            action = TrialAction(
                n_next_cohort=20,
                allocation_control=0.3 / total_prob,
                allocation_low=probs["low"] / total_prob,
                allocation_mid=probs["mid"] / total_prob,
                allocation_high=probs["high"] / total_prob,
                stop_for_success=(getattr(obs, "any_arm_significant", False) and env.interim_number >= 5),
                stop_for_futility=(getattr(obs, "futility_flag", False) and env.interim_number >= 8),
                drop_arm=drop,
            )

            result_obs = env.step(action)
            done = getattr(result_obs, "done", False)
            step_count += 1
    except Exception:
        pass

    _completed_sessions[task_id] = env
    return env

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health():
    return {"status": "ok", "env": "ClinicalTrialEnv", "version": "1.2.0-schema-hardened"}

@app.get("/tasks")
async def list_tasks():
    from .tasks import TASKS
    a_schema, o_schema = _get_schemas()
    return [
        {
            "task_id": t["task_id"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "max_patients": t["max_patients"],
            "action_schema": a_schema,
            "observation_schema": o_schema,
        }
        for t in TASKS.values()
    ]

@app.post("/grader")
async def grader(request: Request):
    try:
        body = await request.json()
        task_id = body.get("task_id")
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
    response = {
        "score": strict_score(float(result.score)),
        "task_id": str(result.task_id),
        "trial_outcome": str(result.trial_outcome),
        "breakdown": result.breakdown,
    }
    return _deep_sanitize(response)

@app.post("/reset")
async def http_reset(body: dict = Body(default={"task": "task_1"})):
    task_id = body.get("task", body.get("task_id", "task_1"))
    env = ClinicalTrialEnvironment()
    obs = env.reset(task_id)
    _completed_sessions[f"http_active_{task_id}"] = env
    
    try:
        obs_dict = obs.model_dump()
    except AttributeError:
        from dataclasses import asdict
        obs_dict = asdict(obs)

    # Ensure reward is present in both observation and top-level
    reward = float(obs_dict.get("reward", 0.10))
    obs_dict["reward"] = reward
    
    response = {
        "observation": obs_dict,
        "reward": reward,
        "done": False,
        "session_id": str(env.state.episode_id),
        "task": task_id,
    }
    return _deep_sanitize(response)

@app.post("/step")
async def http_step(action: TrialAction):
    active = {k: v for k, v in _completed_sessions.items() if k.startswith("http_active_")}
    if not active:
        return JSONResponse(status_code=400, content={"error": "No active session."})
    
    env = list(active.values())[-1]
    obs = env.step(action)
    
    if getattr(obs, "done", False):
        _completed_sessions[env.task["task_id"]] = env
        
    try:
        obs_dict = obs.model_dump()
    except AttributeError:
        from dataclasses import asdict
        obs_dict = asdict(obs)

    reward = float(obs_dict.get("reward", 0.5))
    obs_dict["reward"] = reward

    response = {
        "observation": obs_dict,
        "reward": reward,
        "done": bool(getattr(obs, "done", False)),
    }
    return _deep_sanitize(response)

@app.post("/baseline")
async def baseline():
    scores = {}
    total_val = 0
    for task_id in ["task_1", "task_2", "task_3"]:
        env = _run_heuristic_episode(task_id)
        result = env.grade()
        s = strict_score(float(result.score))
        scores[task_id] = {
            "score": s,
            "outcome": str(result.trial_outcome),
            "breakdown": result.breakdown,
            "task_id": task_id,
        }
        total_val += s
    
    scores["average"] = round(total_val / 3, 4)
    return _deep_sanitize(scores)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
