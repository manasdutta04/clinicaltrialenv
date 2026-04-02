# ClinicalTrialEnv Testing Guide

Welcome! I built this project on top of the strict `openenv-core` framework. It has multiple layers of interaction, and I've designed it so you can test it thoroughly.

Here is my complete guide on testing every single part of my finished environment.

## 1. The Interactive API Playground (Swagger)
I've made sure the environment automatically generates interactive documentation where you can test my REST APIs without writing code.

1. Go to my API Docs: **https://manasdutta04-clinicaltrialenv.hf.space/docs**
2. Click on the green `GET /health` block.
3. Click **"Try it out"**, then **"Execute"**. You should see a `200 OK` response with `{"status": "ok", "env": "ClinicalTrialEnv"}`.
4. Click on the green `GET /schema` block and **Execute**. You will see the JSON outline of my `TrialAction` and `TrialObservation` models, proving they are properly structured for the OpenEnv scalable platform!

---

## 2. The Official Human Agent Debugger
This is the official tool built by Meta/OpenEnv to test RL environments manually. I've natively integrated my environment with it over WebSockets.

1. Go to my main URL: **https://huggingface.co/spaces/manasdutta04/clinicaltrialenv**
2. Wait for the green **Running** dot.
3. On the left side under **"Take Action"**, fill in the required inputs for a single cohort step:
   - **N Next Cohort**: 20
   - **Allocations**: 0.25 (for all four dose arms)
4. Click **Submit Action**.
5. Look at the right side under **"State Observer"** and **"Action History"**. You will see my environment print out the `"Current Observation"` in raw JSON (like `p-values`, `response rates`, etc.). This proves that my pharmacological math model successfully processed your action.

---

## 3. My Custom Visual Dashboard
I also went a step further and built a custom visual dashboard that listens to the backend WebSocket and mathematically plots the trial progress for you.

1. Go to my custom proxy: **https://manasdutta04-clinicaltrialenv.hf.space/dashboard**
2. (Important) If the UI ever feels stuck, refresh the page using **Cmd+Shift+R** to clear the browser's persistent cache.
3. Choose a difficulty level (Task 1, 2, or 3) at the top.
4. Adjust the patient allocation limits on the sliders or leave them at `25%` uniform.
5. Click **"Run Next Interim"**. Watch my charts update in real-time. 
6. Click **"Auto-run Heuristic"** to watch my bot rapid-fire execute and finish the trial itself using a preset heuristic baseline.

---

## 4. Programmatic Testing (Python RL Agent script)

This is how an actual AI agent (or the auto-grader) will test my application using Python. You can try this out yourself by copying the script below and running it in your terminal, VS Code, or Google Colab.

```python
import requests
import time

BASE_URL = "https://manasdutta04-clinicaltrialenv.hf.space"

print("1. Resetting Environment...")
res = requests.post(f"{BASE_URL}/reset", json={"task_id": "task_1"})
print("Reset response:", res.status_code)

print("\n2. Sending 1st Patient Cohort (Step 1)...")
action = {
    "action": {
        "n_next_cohort": 30,
        "allocation_control": 0.25,
        "allocation_low": 0.25,
        "allocation_mid": 0.25,
        "allocation_high": 0.25,
        "stop_for_success": False,
        "stop_for_futility": False,
        "drop_arm": None
    }
}
res2 = requests.post(f"{BASE_URL}/step", json=action)
observation = res2.json()

print(f"Step 1 Complete! Reward: {observation['reward']} | Done: {observation['done']}")
print("P-values achieved:", {
    "low": observation['observation']['p_value_low'],
    "mid": observation['observation']['p_value_mid'],
    "high": observation['observation']['p_value_high'],
})
```
