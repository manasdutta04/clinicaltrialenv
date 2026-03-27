#!/usr/bin/env python3
"""
ClinicalTrialEnv baseline agent.
Uses the OpenEnv client (WebSocket) to interact with the running server.

Usage:
    # Server must be running first:
    uvicorn server.app:app --port 7860

    # Then run baseline:
    python baseline/run_baseline.py --host ws://localhost:7860
    python baseline/run_baseline.py --host wss://your-space.hf.space
"""
import argparse, asyncio, random, time
import numpy as np

# Try to use the installed client, fall back to websockets directly
try:
    from clinical_trial_env import ClinicalTrialEnv, TrialAction
    USE_CLIENT = True
except ImportError:
    import websockets, json
    USE_CLIENT = False


def heuristic_action(obs: dict, interim: int, min_interims: int,
                     dropped: set, ae_thresh: float = 0.25) -> dict:
    """Response-adaptive heuristic agent."""
    probs = {
        "low":  obs.get("prob_low_beats_control", 0.5)  if "low"  not in dropped else 0.0,
        "mid":  obs.get("prob_mid_beats_control", 0.5)  if "mid"  not in dropped else 0.0,
        "high": obs.get("prob_high_beats_control", 0.5) if "high" not in dropped else 0.0,
    }
    total = sum(probs.values()) + 0.30
    drop = None
    for arm, ae_r in [("low",  obs.get("low_ae_rate",  0)),
                      ("mid",  obs.get("mid_ae_rate",  0)),
                      ("high", obs.get("high_ae_rate", 0))]:
        if ae_r > ae_thresh * 0.80 and arm not in dropped:
            drop = arm
            break
    return {
        "n_next_cohort": 25,
        "allocation_control": 0.30 / total,
        "allocation_low":  probs["low"]  / total,
        "allocation_mid":  probs["mid"]  / total,
        "allocation_high": probs["high"] / total,
        "stop_for_success": (
            obs.get("any_arm_significant", False) and interim >= min_interims
        ),
        "stop_for_futility": (
            obs.get("futility_flag", False) and interim >= min_interims
        ),
        "drop_arm": drop
    }


async def run_task_websocket(host: str, task_id: str) -> tuple[float, int, float]:
    """Run one task episode via raw WebSocket."""
    import websockets, json
    uri = f"{host}/ws"
    async with websockets.connect(uri) as ws:
        # Reset
        await ws.send(json.dumps({"action": "reset", "task_id": task_id}))
        msg = json.loads(await ws.recv())
        obs = msg.get("observation", {})

        dropped = set()
        interim = 0
        min_i = 2  # conservative default
        done = False
        reward_sum = 0.0

        while not done and interim < 30:
            action = heuristic_action(obs, interim, min_i, dropped)
            await ws.send(json.dumps({"action": "step", "data": action}))
            msg = json.loads(await ws.recv())
            obs = msg.get("observation", {})
            reward_sum += msg.get("reward", 0.0)
            done = msg.get("done", False)
            interim += 1
            if action.get("drop_arm"):
                dropped.add(action["drop_arm"])

    return reward_sum, interim, done


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="ws://localhost:7860")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"\nClinicalTrialEnv — Baseline Agent (heuristic)")
    print(f"Host: {args.host.replace('ws','http')}")
    print(f"Seed: {args.seed}")
    print("─" * 50)

    import httpx
    http_host = args.host.replace("wss://", "https://").replace("ws://", "http://")

    results = {}
    for task_id in ["task_1", "task_2", "task_3"]:
        label = {"task_1": "easy", "task_2": "medium", "task_3": "hard"}[task_id]
        print(f"Running {task_id} ({label})...", end=" ", flush=True)
        t0 = time.time()

        reward_sum, interims, done = await run_task_websocket(args.host, task_id)

        # Get grader score via HTTP
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{http_host}/grader", timeout=30)
            r.raise_for_status()
            grade = r.json()

        elapsed = time.time() - t0
        score = grade.get("score", 0.0)
        outcome = grade.get("trial_outcome", "unknown")
        results[task_id] = score
        print(f"score={score:.4f}  outcome={outcome}  interims={interims}  t={elapsed:.1f}s")

    avg = sum(results.values()) / 3
    print("─" * 50)
    print(f"Task 1 (easy):    {results['task_1']:.4f}")
    print(f"Task 2 (medium):  {results['task_2']:.4f}")
    print(f"Task 3 (hard):    {results['task_3']:.4f}")
    print(f"Average:          {avg:.4f}")
    print("─" * 50)


if __name__ == "__main__":
    asyncio.run(main())
