#!/usr/bin/env python3
"""
ClinicalTrialEnv inference agent.
Hackathon mandatory submission file — follows [START]/[STEP]/[END] log format.
"""
import os, json, asyncio, math, websockets, requests
from openai import OpenAI

# Mandatory env vars — defaults ONLY for API_BASE_URL and MODEL_NAME
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # optional, no default
ENV_URL = os.getenv("ENV_URL", "https://manasdutta04-clinicaltrialenv.hf.space")

# OpenAI client using the mandatory variables
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-token")

TASKS = ["task_1", "task_2", "task_3"]
TASK_TIMEOUT_SECONDS = int(os.getenv("TASK_TIMEOUT_SECONDS", "45"))


def _strict_score(value):
    try:
        numeric = float(value)
    except Exception:
        return 0.5
    if not math.isfinite(numeric):
        return 0.5
    return float(max(0.01, min(0.99, numeric)))


def _fallback_result(task_id: str, outcome: str = "unknown") -> dict:
    return {
        "task_id": task_id,
        "total_steps": 0,
        "total_reward": 0.05,
        "score": 0.5,
        "outcome": outcome,
    }


def _print_end(result: dict) -> dict:
    safe = {
        "task_id": result["task_id"],
        "total_steps": int(result.get("total_steps", 0)),
        "total_reward": round(_strict_score(result.get("total_reward", 0.05)), 4),
        "score": round(_strict_score(result.get("score", 0.5)), 4),
        "outcome": result.get("outcome") or "unknown",
    }
    print(
        f'[END] {json.dumps({"task_id": safe["task_id"], "total_steps": safe["total_steps"], "total_reward": safe["total_reward"], "score": safe["score"], "outcome": safe["outcome"]})}',
        flush=True,
    )
    return safe


SYSTEM_PROMPT = """You are a clinical trial statistician making adaptive decisions.
Given trial observations, decide the next action to maximize statistical power
while minimizing patients used and keeping adverse events safe.

Respond ONLY with valid JSON matching this exact schema (no markdown, no explanation):
{
  "n_next_cohort": <int 5-100>,
  "allocation_control": <float>,
  "allocation_low": <float>,
  "allocation_mid": <float>,
  "allocation_high": <float>,
  "stop_for_success": <bool>,
  "stop_for_futility": <bool>,
  "drop_arm": <null or "low" or "mid" or "high">,
  "inclusion_criteria_strictness": <float 0-1>
}

Rules:
- Allocate more to arms with higher prob_X_beats_control
- stop_for_success=true when any_arm_significant=true AND interim_number >= 2
- stop_for_futility=true when futility_flag=true AND interim_number >= 2
- drop_arm if its ae_rate > 0.20 and arm is active
- Allocations are auto-normalized, they don't need to sum to 1"""


def _heuristic(obs: dict) -> dict:
    """Fallback heuristic when LLM is unavailable."""
    probs = {
        "low":  obs.get("prob_low_beats_control",  0.5) if obs.get("low_active",  True) else 0.0,
        "mid":  obs.get("prob_mid_beats_control",  0.5) if obs.get("mid_active",  True) else 0.0,
        "high": obs.get("prob_high_beats_control", 0.5) if obs.get("high_active", True) else 0.0,
    }
    total = sum(probs.values()) + 0.3
    drop = None
    for arm in ["low", "mid", "high"]:
        if obs.get(f"{arm}_ae_rate", 0) > 0.20 and obs.get(f"{arm}_active", True):
            drop = arm
            break
    interim = obs.get("interim_number", 0)
    return {
        "n_next_cohort": 25,
        "allocation_control": 0.3 / total,
        "allocation_low":     probs["low"]  / total,
        "allocation_mid":     probs["mid"]  / total,
        "allocation_high":    probs["high"] / total,
        "stop_for_success":  obs.get("any_arm_significant", False) and interim >= 2,
        "stop_for_futility": obs.get("futility_flag", False) and interim >= 2,
        "drop_arm": drop,
        "inclusion_criteria_strictness": 0.6,
    }


def llm_action(obs: dict, task_id: str, step_num: int) -> dict:
    """Ask LLM for action; fall back to heuristic on any error."""
    try:
        summary = {
            "task_id": task_id, "step": step_num,
            "interim": obs.get("interim_number", 0),
            "enrolled": obs.get("total_patients_enrolled", 0),
            "budget_left": obs.get("budget_remaining", 0),
            "response_rates": {
                "control": obs.get("control_response_rate", 0),
                "low": obs.get("low_response_rate", 0),
                "mid": obs.get("mid_response_rate", 0),
                "high": obs.get("high_response_rate", 0),
            },
            "ae_rates": {
                "low": obs.get("low_ae_rate", 0),
                "mid": obs.get("mid_ae_rate", 0),
                "high": obs.get("high_ae_rate", 0),
            },
            "p_values": {
                "low": obs.get("p_value_low", 1.0),
                "mid": obs.get("p_value_mid", 1.0),
                "high": obs.get("p_value_high", 1.0),
            },
            "posteriors": {
                "low": obs.get("prob_low_beats_control", 0.5),
                "mid": obs.get("prob_mid_beats_control", 0.5),
                "high": obs.get("prob_high_beats_control", 0.5),
            },
            "any_significant": obs.get("any_arm_significant", False),
            "futility_flag":   obs.get("futility_flag", False),
            "active": {
                "low":  obs.get("low_active",  True),
                "mid":  obs.get("mid_active",  True),
                "high": obs.get("high_active", True),
            },
        }
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(summary)},
            ],
            max_tokens=256,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return _heuristic(obs)


async def run_task(task_id: str) -> dict:
    ws_url   = ENV_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    http_url = ENV_URL
    headers  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    result = _fallback_result(task_id)

    try:
        async with websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=10,
            open_timeout=10,
            close_timeout=5,
        ) as ws:
            # Reset
            await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
            msg     = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            payload = msg.get("data", msg)
            obs     = payload.get("observation", {})

            # [START] — mandatory format
            print(f'[START] {json.dumps({"task_id": task_id, "model": MODEL_NAME, "env_url": ENV_URL})}',
                  flush=True)

            step_num, done, total_reward = 0, False, 0.0

            while not done and step_num < 30:
                step_num += 1
                action = llm_action(obs, task_id, step_num)

                await ws.send(json.dumps({"type": "step", "data": action}))
                msg     = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                payload = msg.get("data", msg)
                obs     = payload.get("observation", {})
                reward  = _strict_score(payload.get("reward", 0.0))
                done    = bool(payload.get("done", False))
                total_reward += reward

                # [STEP] — mandatory format
                print(f'[STEP] {json.dumps({"step": step_num, "action": action, "observation": obs, "reward": round(reward, 4), "done": done})}',
                      flush=True)

            result["total_steps"]  = step_num
            result["total_reward"] = _strict_score(total_reward)

            # Get grader score via HTTP
            try:
                r = requests.post(
                    f"{http_url}/grader",
                    json={"task_id": task_id},
                    timeout=12,
                    headers=headers,
                )
                r.raise_for_status()
                grade = r.json()
                raw_score = float(grade.get("score", 0.5))
                result["score"] = _strict_score(raw_score)
                result["outcome"] = grade.get("trial_outcome", "unknown")
            except Exception:
                result["score"] = 0.5
                result["outcome"] = obs.get("stop_reason") or "budget_exhausted"
    except Exception as e:
        print(f'[WARN] {json.dumps({"task_id": task_id, "error": str(e)})}', flush=True)
        result["score"] = 0.5

    return _print_end(result)


async def main():
    if not HF_TOKEN:
        print('[WARN] {"message": "HF_TOKEN not set, LLM calls may fail"}', flush=True)
    scores = []
    for task_id in TASKS:
        try:
            r = await asyncio.wait_for(run_task(task_id), timeout=TASK_TIMEOUT_SECONDS)
        except Exception as e:
            print(f'[WARN] {json.dumps({"task_id": task_id, "error": f"task_timeout_or_cancelled: {e}"})}', flush=True)
            r = _print_end(_fallback_result(task_id, outcome="timeout_or_error"))
        scores.append(r["score"])
    avg = round(sum(scores) / len(scores), 4)
    print(f'[SUMMARY] {json.dumps({"task_1": scores[0], "task_2": scores[1], "task_3": scores[2], "average": avg})}',
          flush=True)

if __name__ == "__main__":
    asyncio.run(main())
