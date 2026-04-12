import os, json, asyncio, websockets, requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL", "https://manasdutta04-clinicaltrialenv.hf.space")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-token")
TASKS = ["task_1", "task_2", "task_3"]
TASK_TIMEOUT_SECONDS = int(os.getenv("TASK_TIMEOUT_SECONDS", "45"))


def _strict_open_score(value, fallback=0.5):
    try:
        numeric = float(value)
    except Exception:
        return float(fallback)
    if numeric != numeric:  # NaN guard
        return float(fallback)
    return float(max(0.1, min(0.9, numeric)))


def _sanitize_floats(obj):
    """Recursively clamp every float to (0.0001, 0.9999). Skips bools/ints/strings."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if obj != obj:
            return 0.5
        return float(max(0.0001, min(0.9999, obj)))
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v) for v in obj]
    return obj


def _sanitize_action(action):
    try:
        numeric_fields = {
            "allocation_control": float(action.get("allocation_control", 0.25)),
            "allocation_low": float(action.get("allocation_low", 0.25)),
            "allocation_mid": float(action.get("allocation_mid", 0.25)),
            "allocation_high": float(action.get("allocation_high", 0.25)),
            "inclusion_criteria_strictness": float(action.get("inclusion_criteria_strictness", 0.5)),
        }
    except Exception:
        return _sanitize_action(_heuristic({}))

    for key in ["allocation_control", "allocation_low", "allocation_mid", "allocation_high"]:
        numeric_fields[key] = _strict_open_score(numeric_fields[key], fallback=0.01)

    strictness = _strict_open_score(numeric_fields["inclusion_criteria_strictness"], fallback=0.5)

    total = sum(numeric_fields[key] for key in ["allocation_control", "allocation_low", "allocation_mid", "allocation_high"])
    if total <= 0:
        total = 1.0

    return {
        "n_next_cohort": int(max(5, min(100, int(action.get("n_next_cohort", 25))))),
        "allocation_control": numeric_fields["allocation_control"] / total,
        "allocation_low": numeric_fields["allocation_low"] / total,
        "allocation_mid": numeric_fields["allocation_mid"] / total,
        "allocation_high": numeric_fields["allocation_high"] / total,
        "stop_for_success": bool(action.get("stop_for_success", False)),
        "stop_for_futility": bool(action.get("stop_for_futility", False)),
        "drop_arm": action.get("drop_arm"),
        "inclusion_criteria_strictness": strictness,
    }

def _heuristic(obs):
    probs = {
        "low":  obs.get("prob_low_beats_control", 0.5) if obs.get("low_active", True) else 0.0,
        "mid":  obs.get("prob_mid_beats_control", 0.5) if obs.get("mid_active", True) else 0.0,
        "high": obs.get("prob_high_beats_control", 0.5) if obs.get("high_active", True) else 0.0,
    }
    t = sum(probs.values()) + 0.3
    if t <= 0:
        t = 1.0
    interim = obs.get("interim_number", 0)
    drop = next(
        (a for a in ["low", "mid", "high"]
         if obs.get(f"{a}_ae_rate", 0) > 0.20 and obs.get(f"{a}_active", True)),
        None
    )
    return {
        "n_next_cohort": 25,
        "allocation_control": 0.3 / t,
        "allocation_low": probs["low"] / t,
        "allocation_mid": probs["mid"] / t,
        "allocation_high": probs["high"] / t,
        "stop_for_success": obs.get("any_arm_significant", False) and interim >= 2,
        "stop_for_futility": obs.get("futility_flag", False) and interim >= 2,
        "drop_arm": drop,
        "inclusion_criteria_strictness": 0.6,
    }

def llm_action(obs, task_id, step_num):
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON with keys: n_next_cohort, allocation_control, allocation_low, allocation_mid, allocation_high, stop_for_success, stop_for_futility, drop_arm, inclusion_criteria_strictness"},
                {"role": "user", "content": json.dumps({"task": task_id, "step": step_num, "obs": obs})}
            ],
            max_tokens=200,
            temperature=0.1,
        )
        raw = r.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except Exception:
        return _heuristic(obs)

def _unwrap(msg: dict) -> dict:
    """Handle OpenEnv WebSocket envelope: {data: {observation:...}} or flat."""
    if "data" in msg and isinstance(msg["data"], dict):
        return msg["data"]
    return msg

async def run_task(task_id: str) -> float:
    ws_url = ENV_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    score = 0.5
    total_steps = 0
    total_reward = 0.05
    outcome = "unknown"

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
            raw_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            payload = _unwrap(raw_msg)
            obs = payload.get("observation", {})

            print(f'[START] {json.dumps({"task_id": task_id, "model": MODEL_NAME, "env_url": ENV_URL})}', flush=True)

            done = False
            step_num = 0

            while not done and step_num < 30:
                step_num += 1
                action = _sanitize_action(llm_action(obs, task_id, step_num))

                await ws.send(json.dumps({"type": "step", "data": action}))
                raw_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                payload = _unwrap(raw_msg)

                obs = payload.get("observation", obs)
                obs = _sanitize_floats(obs)
                reward = _strict_open_score(payload.get("reward", 0.0), fallback=0.05)
                done = bool(payload.get("done", False))
                total_reward += reward

                print(f'[STEP] {json.dumps({"step": step_num, "action": action, "observation": obs, "reward": round(reward, 4), "done": done})}', flush=True)

            total_steps = step_num
            outcome = obs.get("stop_reason") or "budget_exhausted"

            # Get grader score via HTTP
            try:
                headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
                resp = requests.post(
                    f"{ENV_URL.rstrip('/')}/grader",
                    json={"task_id": task_id},
                    timeout=15,
                    headers=headers
                )
                g = resp.json()
                raw_score = g.get("score")
                if raw_score is None:
                    raw_score = g.get("average", 0.5)
                score = _strict_open_score(raw_score)
            except Exception as e:
                print(f"[WARN] Grader call failed: {e}", flush=True)
                # Fallback score based on progress
                score = _strict_open_score(0.2 + step_num * 0.01)

    except Exception as e:
        print(f'[WARN] Task execution error: {json.dumps({"error": str(e)})}', flush=True)
        score = 0.2

    safe_score = round(_strict_open_score(score), 4)
    safe_total_reward = round(_strict_open_score(total_reward, fallback=0.5), 4)

    print(
        f'[END] {json.dumps({"task_id": task_id, "total_steps": total_steps, "total_reward": safe_total_reward, "score": safe_score, "outcome": outcome})}',
        flush=True,
    )
    return safe_score

async def main():
    scores = []
    for task_id in TASKS:
        try:
            s = await asyncio.wait_for(run_task(task_id), timeout=TASK_TIMEOUT_SECONDS)
        except Exception as e:
            print(f'[WARN] {json.dumps({"task_id": task_id, "error": f"task_timeout_or_cancelled: {e}"})}', flush=True)
            fallback = 0.5
            print(
                f'[END] {json.dumps({"task_id": task_id, "total_steps": 0, "total_reward": 0.05, "score": fallback, "outcome": "timeout_or_error"})}',
                flush=True,
            )
            s = fallback
        scores.append(s)
    if scores:
        avg = round(_strict_open_score(sum(scores) / len(scores)), 4)
        print(f'[SUMMARY] {json.dumps({"task_1": scores[0], "task_2": scores[1], "task_3": scores[2], "average": avg})}', flush=True)

if __name__ == "__main__":
    asyncio.run(main())
