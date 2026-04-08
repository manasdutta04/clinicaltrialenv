TASKS = {
    "task_1": {
        "task_id": "task_1",
        "difficulty": "easy",
        "name": "Phase II dose-finding — effective drug",
        "description": (
            "A promising oncology drug with a clear dose-response signal. "
            "Find the optimal dose, reach statistical significance (p<0.05) "
            "using as few patients as possible."
        ),
        "true_params": {
            "baseline": 0.10, "emax": 0.45, "ed50": 30.0,
            "hill": 1.5, "ae_baseline": 0.05, "ae_slope": 0.15
        },
        "doses": {"low": 10.0, "mid": 30.0, "high": 60.0},
        "max_patients": 200,
        "ae_stopping_threshold": 0.30,
        "min_interims_before_stop": 2,
        "target_pvalue": 0.05,
        "grader": "efficacy_grader"
    },
    "task_2": {
        "task_id": "task_2",
        "difficulty": "medium",
        "name": "Phase II — efficacy vs safety tradeoff",
        "description": (
            "A drug with good efficacy at high dose but dangerous adverse events. "
            "High dose works best but hits the AE threshold. "
            "Find the optimal arm balancing efficacy and safety."
        ),
        "true_params": {
            "baseline": 0.10, "emax": 0.50, "ed50": 20.0,
            "hill": 2.0, "ae_baseline": 0.05, "ae_slope": 0.35
        },
        "doses": {"low": 10.0, "mid": 25.0, "high": 50.0},
        "max_patients": 250,
        "ae_stopping_threshold": 0.25,
        "min_interims_before_stop": 3,
        "target_pvalue": 0.05,
        "grader": "tradeoff_grader"
    },
    "task_3": {
        "task_id": "task_3",
        "difficulty": "hard",
        "name": "Rare disease trial — weak signal, tiny budget",
        "description": (
            "A drug for a rare disease with a modest effect size (Emax=0.20). "
            "Only 150 patients available. The agent must allocate extremely "
            "efficiently, drop futile arms early, and squeeze every bit of "
            "statistical power from a tiny sample."
        ),
        "true_params": {
            "baseline": 0.12, "emax": 0.20, "ed50": 25.0,
            "hill": 1.2, "ae_baseline": 0.04, "ae_slope": 0.10
        },
        "doses": {"low": 15.0, "mid": 30.0, "high": 45.0},
        "max_patients": 150,
        "ae_stopping_threshold": 0.30,
        "min_interims_before_stop": 2,
        "target_pvalue": 0.05,
        "grader": "efficiency_grader"
    }
}
