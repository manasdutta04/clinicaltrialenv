import requests
import time
import sys

# Change to https://manasdutta04-clinicaltrialenv.hf.space when running externally
BASE_URL = "http://localhost:7860" 

class HeuristicAgent:
    def __init__(self, task_id="task_1"):
        self.task_id = task_id
        
    def run(self):
        print(f"\033[94m[Agent] Starting Autonomous Trial for {self.task_id}...\033[0m")
        try:
            res = requests.post(f"{BASE_URL}/reset", json={"task_id": self.task_id})
            res.raise_for_status()
            obs = res.json()["observation"]
        except Exception as e:
            print(f"Failed to connect to API ({BASE_URL}). Are you sure the server is running?")
            sys.exit(1)
        
        while True:
            # 1. Safety Monitoring
            drop_arm = None
            if obs.get("low_ae_rate", 0) > 0.25 and obs.get("low_active", True): drop_arm = "low"
            elif obs.get("mid_ae_rate", 0) > 0.25 and obs.get("mid_active", True): drop_arm = "mid"
            elif obs.get("high_ae_rate", 0) > 0.25 and obs.get("high_active", True): drop_arm = "high"
                
            # 2. Bayesian Thomson Sampling-style Allocations
            probs = {
                "low": obs.get("prob_low_beats_control", 0) if obs.get("low_active", True) else 0.0,
                "mid": obs.get("prob_mid_beats_control", 0) if obs.get("mid_active", True) else 0.0,
                "high": obs.get("prob_high_beats_control", 0) if obs.get("high_active", True) else 0.0,
            }
            tp = probs["low"] + probs["mid"] + probs["high"] + 0.3
            
            # 3. Dynamic Inclusion Criteria (The Wow Factor)
            # If we isolate a good drug candidate (>60% posterior), we tighten criteria to boost statistical power.
            # If we are exploring, we keep criteria loose to save budget.
            best_prob = max(probs.values()) if probs else 0.5
            strictness = 0.8 if best_prob > 0.6 else 0.3 
            
            action = {
                "action": {
                    "n_next_cohort": 20,
                    "allocation_control": 0.3 / tp,
                    "allocation_low": probs["low"] / tp,
                    "allocation_mid": probs["mid"] / tp,
                    "allocation_high": probs["high"] / tp,
                    "stop_for_success": obs.get("any_arm_significant", False) and obs.get("interim_number", 0) >= 2,
                    "stop_for_futility": obs.get("futility_flag", False) and obs.get("interim_number", 0) >= 2,
                    "drop_arm": drop_arm,
                    "inclusion_criteria_strictness": strictness
                }
            }
            
            res = requests.post(f"{BASE_URL}/step", json=action)
            step_data = res.json()
            obs = step_data["observation"]
            
            print(f"\033[92m[Interim {obs['interim_number']}]\033[0m Budget: {obs['budget_remaining']} | Strictness: {strictness} | P-Values: Low({obs['p_value_low']}) Mid({obs['p_value_mid']}) High({obs['p_value_high']})")
            
            if step_data["done"]:
                print(f"\n\033[95m[Trial Finished] Reason: {obs['stop_reason']}\033[0m")
                print(f"\033[93m★ Final Grader Reward: {step_data['reward']}\033[0m")
                break
                
            time.sleep(0.1)

if __name__ == "__main__":
    agent = HeuristicAgent("task_2")
    agent.run()
