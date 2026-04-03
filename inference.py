#!/usr/bin/env python3
"""
ClinicalTrialEnv - LLM-powered inference agent.
Follows the mandatory Scaler x Meta PyTorch OpenEnv Hackathon format exactly.

Usage:
    API_BASE_URL=<url> MODEL_NAME=<model> HF_TOKEN=<token> python inference.py

Environment variables:
    API_BASE_URL     - OpenAI-compatible API base URL (required, has default)
    MODEL_NAME       - Model identifier (required, has default)
    HF_TOKEN         - Hugging Face token for the Space (required, no default)
    LOCAL_IMAGE_NAME - Optional Docker image name if running locally
"""
import asyncio
import json
import os

import requests
import websockets
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = os.getenv(
    "ENV_URL",
    "https://manasdutta04-clinicaltrialenv.hf.space",
)

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-token",
)

TASKS = ["task_1", "task_2", "task_3"]

SYSTEM_PROMPT = """You are an expert clinical trial statistician making adaptive
trial design decisions. You will receive observations from an adaptive clinical
trial and must decide how to proceed.

Your goal: Reach statistical significance (p < 0.05) while using as few patients
as possible, maintaining patient safety (keep AE rates below thresholds).

You must respond with ONLY a valid JSON object matching this schema:
{
  "n_next_cohort": <int 5-100>,
  "allocation_control": <float 0-1>,
  "allocation_low": <float 0-1>,
  "allocation_mid": <float 0-1>,
  "allocation_high": <float 0-1>,
  "stop_for_success": <bool>,
  "stop_for_futility": <bool>,
  "drop_arm": <null or "low" or "mid" or "high">,
  "inclusion_criteria_strictness": <float 0-1>,
  "reasoning": "<brief one-line reasoning>"
}

Allocation weights do NOT need to sum to 1 - they will be normalized automatically.

Strategy tips:
- Allocate more patients to arms with higher P(arm > control) (Bayesian posterior)
- Stop for success when any_arm_significant=true AND interim_number >= 2
- Stop for futility when futility_flag=true AND interim_number >= 2
- Drop an arm if its AE rate > 0.80 * threshold
- Use inclusion_criteria_strictness=0.7 for cleaner signal, 0.3 for speed"""


def _heuristic_fallback(obs: dict) -> dict:
    """Fallback heuristic when the LLM call fails."""
    probs = {
        "low": obs.get("prob_low_beats_control", 0.5) if obs.get("low_active", True) else 0.0,
        "mid": obs.get("prob_mid_beats_control", 0.5) if obs.get("mid_active", True) else 0.0,
        "high": obs.get("prob_high_beats_control", 0.5) if obs.get("high_active", True) else 0.0,
    }
    total = sum(probs.values()) + 0.3
    drop = None
    for arm, ae in (
        ("low", obs.get("low_ae_rate", 0.0)),
        ("mid", obs.get("mid_ae_rate", 0.0)),
        ("high", obs.get("high_ae_rate", 0.0)),
    ):
        if ae > 0.20 and obs.get(f"{arm}_active", True):
            drop = arm
            break

    interim = obs.get("interim_number", 0)
    return {
        "n_next_cohort": 25,
        "allocation_control": 0.3 / total,
        "allocation_low": probs["low"] / total,
        "allocation_mid": probs["mid"] / total,
        "allocation_high": probs["high"] / total,
        "stop_for_success": obs.get("any_arm_significant", False) and interim >= 2,
        "stop_for_futility": obs.get("futility_flag", False) and interim >= 2,
        "drop_arm": drop,
        "inclusion_criteria_strictness": 0.6,
    }


def llm_decide_action(obs: dict, task_id: str, step_num: int) -> dict:
    """Use the configured OpenAI-compatible client to choose the next action."""
    obs_summary = {
        "interim": obs.get("interim_number", 0),
        "enrolled": obs.get("total_patients_enrolled", 0),
        "budget_left": obs.get("budget_remaining", 0),
        "resp_rates": {
            "control": obs.get("control_response_rate", 0.0),
            "low": obs.get("low_response_rate", 0.0),
            "mid": obs.get("mid_response_rate", 0.0),
            "high": obs.get("high_response_rate", 0.0),
        },
        "ae_rates": {
            "low": obs.get("low_ae_rate", 0.0),
            "mid": obs.get("mid_ae_rate", 0.0),
            "high": obs.get("high_ae_rate", 0.0),
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
        "futility_flag": obs.get("futility_flag", False),
        "active_arms": {
            "low": obs.get("low_active", True),
            "mid": obs.get("mid_active", True),
            "high": obs.get("high_active", True),
        },
    }

    user_msg = f"""Task: {task_id} | Step: {step_num}
Current trial state:
{json.dumps(obs_summary, indent=2)}

Decide the next action. Remember: respond with ONLY valid JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=300,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action_with_reasoning = json.loads(raw.strip())
        action_with_reasoning.pop("reasoning", None)
        return action_with_reasoning
    except Exception:
        return _heuristic_fallback(obs)


async def run_task(task_id: str) -> dict:
    """Run one task from reset through grading."""
    ws_url = ENV_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    http_url = ENV_URL

    results = {
        "task_id": task_id,
        "total_steps": 0,
        "total_reward": 0.0,
        "score": 0.0,
        "outcome": "unknown",
    }

    async with websockets.connect(ws_url, ping_interval=30, ping_timeout=10) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
        msg = json.loads(await ws.recv())
        payload = msg.get("data", msg)
        obs = payload.get("observation", {})

        print(
            f'[START] {json.dumps({"task_id": task_id, "model": MODEL_NAME, "env_url": ENV_URL})}',
            flush=True,
        )

        step_num = 0
        done = False
        total_reward = 0.0

        while not done and step_num < 30:
            step_num += 1
            action = llm_decide_action(obs, task_id, step_num)

            await ws.send(json.dumps({"type": "step", "data": action}))
            msg = json.loads(await ws.recv())
            payload = msg.get("data", msg)
            new_obs = payload.get("observation", {})
            reward = float(payload.get("reward", 0.0))
            done = bool(payload.get("done", False))
            total_reward += reward

            print(
                f'[STEP] {json.dumps({"step": step_num, "action": action, "observation": new_obs, "reward": round(reward, 4), "done": done})}',
                flush=True,
            )

            obs = new_obs

        results["total_steps"] = step_num
        results["total_reward"] = round(total_reward, 4)

        try:
            response = requests.post(
                f"{http_url}/grader",
                json={"task_id": task_id},
                timeout=30,
                headers={"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {},
            )
            response.raise_for_status()
            grade = response.json()
            results["score"] = round(float(grade.get("score", 0.0)), 4)
            results["outcome"] = grade.get("trial_outcome", "unknown")
        except Exception:
            results["score"] = round(max(0.0, min(1.0, total_reward)), 4)
            results["outcome"] = obs.get("stop_reason", "unknown") or "budget_exhausted"

    print(
        f'[END] {json.dumps({"task_id": task_id, "total_steps": results["total_steps"], "total_reward": results["total_reward"], "score": results["score"], "outcome": results["outcome"]})}',
        flush=True,
    )
    return results


async def main() -> None:
    """Run all tasks sequentially."""
    if not HF_TOKEN:
        print("[WARN] HF_TOKEN not set - some requests may fail", flush=True)

    all_scores = []
    for task_id in TASKS:
        result = await run_task(task_id)
        all_scores.append(result["score"])

    average = round(sum(all_scores) / len(all_scores), 4)
    print(
        f'[SUMMARY] {json.dumps({"task_1": all_scores[0], "task_2": all_scores[1], "task_3": all_scores[2], "average": average})}',
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
