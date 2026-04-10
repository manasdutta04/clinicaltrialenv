import numpy as np
from dataclasses import dataclass
from typing import Optional

_SCORE_MIN = 0.02
_SCORE_MAX = 0.98

def strict_score(score: float) -> float:
    """Clamp to strictly open (0,1) — safe against float rounding on stdout."""
    return float(np.clip(score, _SCORE_MIN, _SCORE_MAX))


@dataclass
class GraderResult:
    score: float  # strictly in (0.02, 0.98)
    task_id: str
    trial_outcome: str  # "success" | "futility" | "budget_exhausted" | "safety_stop"
    breakdown: dict


def efficacy_grader(session_state: dict) -> GraderResult:
    """Task 1: Did the trial reach significance? How efficiently?"""
    stop = session_state["stop_reason"]
    enrolled = session_state["total_enrolled"]
    max_p = session_state["task"]["max_patients"]
    best_p = session_state.get("best_pvalue", 1.0)

    if stop == "success":
        efficiency = 1.0 - (enrolled / max_p)
        score = 0.60 + 0.40 * efficiency
    elif stop == "safety_stop":
        score = 0.15
    elif stop == "futility":
        score = 0.05
    else:  # budget_exhausted
        score = 0.30 * max(0.0, 1.0 - best_p / 0.05)

    return GraderResult(
        score=strict_score(score),
        task_id="task_1",
        trial_outcome=stop or "budget_exhausted",
        breakdown={
            "reached_significance": stop == "success",
            "patients_used": enrolled,
            "max_patients": max_p,
            "efficiency": round(1.0 - enrolled / max_p, 3),
            "best_pvalue": round(best_p, 4),
            "interims_run": session_state.get("interim_number", 0)
        }
    )


def tradeoff_grader(session_state: dict) -> GraderResult:
    """Task 2: Efficacy AND safety. Penalize patients given to unsafe arm."""
    stop = session_state["stop_reason"]
    enrolled = session_state["total_enrolled"]
    max_p = session_state["task"]["max_patients"]
    best_p = session_state.get("best_pvalue", 1.0)
    unsafe_patients = session_state.get("unsafe_arm_patients", 0)

    if stop == "success":
        eff = 1.0 - (enrolled / max_p)
        efficacy_score = 0.60 + 0.20 * eff
    else:
        efficacy_score = 0.25 * max(0.0, 1.0 - best_p / 0.10)

    safety_penalty = min(0.30, unsafe_patients * 0.01)
    safety_score = max(0.0, 0.40 - safety_penalty)

    score = efficacy_score + safety_score
    return GraderResult(
        score=strict_score(score),
        task_id="task_2",
        trial_outcome=stop or "budget_exhausted",
        breakdown={
            "efficacy_score": round(efficacy_score, 3),
            "safety_score": round(safety_score, 3),
            "unsafe_arm_patients": unsafe_patients,
            "reached_significance": stop == "success"
        }
    )


def efficiency_grader(session_state: dict) -> GraderResult:
    """Task 3: Rare disease — squeeze significance from 150 patients."""
    stop = session_state["stop_reason"]
    enrolled = session_state["total_enrolled"]
    max_p = session_state["task"]["max_patients"]
    best_p = session_state.get("best_pvalue", 1.0)
    best_posterior = session_state.get("best_posterior", 0.5)

    if stop == "success":
        patient_eff = 1.0 - (enrolled / max_p)
        score = 0.50 + 0.30 * patient_eff + 0.20 * best_posterior
    elif stop == "futility":
        score = 0.20
    else:
        score = 0.15 + 0.20 * max(0.0, 1.0 - best_p / 0.10)

    return GraderResult(
        score=strict_score(score),
        task_id="task_3",
        trial_outcome=stop or "budget_exhausted",
        breakdown={
            "reached_significance": stop == "success",
            "patients_used": enrolled,
            "budget": max_p,
            "best_pvalue": round(best_p, 4),
            "best_posterior_prob": round(best_posterior, 3)
        }
    )