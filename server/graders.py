import numpy as np
from dataclasses import dataclass
from typing import Optional

STRICT_SCORE_MIN = 0.0001  # Minimum: rounds to 0.0001, never 0.0
STRICT_SCORE_MAX = 0.9999  # Maximum: rounds to 0.9999, never 1.0


def strict_score(value: float) -> float:
    """Keep task scores safely inside the open interval (0, 1)."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not np.isfinite(numeric):
        return 0.5
    return float(np.clip(numeric, STRICT_SCORE_MIN, STRICT_SCORE_MAX))

@dataclass
class GraderResult:
    score: float
    task_id: str
    trial_outcome: str
    breakdown: dict


def efficacy_grader(session_state: dict) -> GraderResult:
    stop = session_state["stop_reason"]
    enrolled = session_state["total_enrolled"]
    max_p = session_state["task"]["max_patients"]
    best_p = session_state.get("best_pvalue", 1.0)
    budget_consumed = session_state.get("budget_consumed", enrolled)
    efficiency = 1.0 - min(0.99, budget_consumed / max_p)  # Cap at 0.99 to prevent efficiency=1.0

    if stop == "success":
        score = 0.60 + 0.39 * efficiency  # Reduced from 0.40 to guarantee score < 1.0
    elif stop == "safety_stop":
        score = 0.15
    elif stop == "futility":
        score = 0.05
    else:
        score = 0.30 * max(0.001, 1.0 - best_p / 0.05)

    return GraderResult(
        score=strict_score(score),
        task_id="task_1",
        trial_outcome=stop or "budget_exhausted",
        breakdown={
            "reached_significance": stop == "success",
            "patients_used": enrolled,
            "max_patients": max_p,
            "efficiency": round(efficiency, 3),
            "best_pvalue": round(best_p, 4),
            "interims_run": session_state.get("interim_number", 0),
        },
    )


def tradeoff_grader(session_state: dict) -> GraderResult:
    stop = session_state["stop_reason"]
    enrolled = session_state["total_enrolled"]
    max_p = session_state["task"]["max_patients"]
    best_p = session_state.get("best_pvalue", 1.0)
    unsafe_patients = session_state.get("unsafe_arm_patients", 0)

    if stop == "success":
        eff = 1.0 - min(0.99, enrolled / max_p)  # Cap at 0.99
        efficacy_score = 0.60 + 0.19 * eff  # Reduced from 0.20 to guarantee sum < 1.0
    else:
        efficacy_score = 0.25 * max(0.001, 1.0 - best_p / 0.10)

    safety_penalty = min(0.30, unsafe_patients * 0.01)
    safety_score = max(0.001, 0.40 - safety_penalty)

    score = efficacy_score + safety_score
    return GraderResult(
        score=strict_score(score),
        task_id="task_2",
        trial_outcome=stop or "budget_exhausted",
        breakdown={
            "efficacy_score": round(efficacy_score, 3),
            "safety_score": round(safety_score, 3),
            "unsafe_arm_patients": unsafe_patients,
            "reached_significance": stop == "success",
        },
    )


def efficiency_grader(session_state: dict) -> GraderResult:
    stop = session_state["stop_reason"]
    enrolled = session_state["total_enrolled"]
    max_p = session_state["task"]["max_patients"]
    best_p = session_state.get("best_pvalue", 1.0)
    best_posterior = session_state.get("best_posterior", 0.5)
    patient_eff = 1.0 - min(
        0.99,  # Cap at 0.99 to prevent patient_eff=1.0
        session_state.get("budget_consumed", enrolled) / max_p,
    )

    if stop == "success":
        score = 0.50 + 0.29 * patient_eff + 0.19 * best_posterior  # Adjusted coefficients to guarantee score < 1.0
    elif stop == "futility":
        score = 0.20
    else:
        score = 0.15 + 0.20 * max(0.001, 1.0 - best_p / 0.10)

    return GraderResult(
        score=strict_score(score),
        task_id="task_3",
        trial_outcome=stop or "budget_exhausted",
        breakdown={
            "reached_significance": stop == "success",
            "patients_used": enrolled,
            "budget": max_p,
            "efficiency": round(patient_eff, 3),
            "best_pvalue": round(best_p, 4),
            "best_posterior_prob": round(best_posterior, 3),
        },
    )
