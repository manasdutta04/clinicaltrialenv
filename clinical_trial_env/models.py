"""
Data models for ClinicalTrialEnv.
Uses the exact same base classes as openenv-core's template.
"""
from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class TrialAction(Action):
    """Agent action: how to design the next interim analysis cohort."""
    n_next_cohort: int = Field(default=20, description="Patients to enroll (5-100)")
    allocation_control: float = Field(default=0.25, description="Allocation weight for control arm")
    allocation_low: float = Field(default=0.25, description="Allocation weight for low-dose arm")
    allocation_mid: float = Field(default=0.25, description="Allocation weight for mid-dose arm")
    allocation_high: float = Field(default=0.25, description="Allocation weight for high-dose arm")
    stop_for_success: bool = Field(default=False, description="Signal early stop if p<0.05")
    stop_for_futility: bool = Field(default=False, description="Signal early stop for futility")
    drop_arm: Optional[str] = Field(default=None, description="Arm to drop: 'low', 'mid', 'high', or null")
    inclusion_criteria_strictness: float = Field(default=0.5, description="Strictness of patient inclusion [0.0 = relaxed, 1.0 = highly strict]")

    def model_post_init(self, __context):
        """Normalize allocation weights to sum to 1.0 and clamp cohort size."""
        self.n_next_cohort = max(5, min(100, self.n_next_cohort))
        self.inclusion_criteria_strictness = max(0.0, min(1.0, self.inclusion_criteria_strictness))
        total = (self.allocation_control + self.allocation_low +
                 self.allocation_mid + self.allocation_high)
        if total > 0:
            object.__setattr__(self, 'allocation_control', self.allocation_control / total)
            object.__setattr__(self, 'allocation_low', self.allocation_low / total)
            object.__setattr__(self, 'allocation_mid', self.allocation_mid / total)
            object.__setattr__(self, 'allocation_high', self.allocation_high / total)


class TrialObservation(Observation):
    """Observed state of the adaptive clinical trial returned after each action."""
    # Trial progress
    interim_number: int = Field(default=0)
    total_patients_enrolled: int = Field(default=0)
    budget_remaining: int = Field(default=0)
    enrollment_rate: float = Field(default=0.9999)
    population_heterogeneity: float = Field(default=0.5)

    # Per-arm observed response rates
    control_response_rate: float = Field(default=0.0001)
    low_response_rate: float = Field(default=0.0001)
    mid_response_rate: float = Field(default=0.0001)
    high_response_rate: float = Field(default=0.0001)

    # Per-arm adverse event rates
    control_ae_rate: float = Field(default=0.0001)
    low_ae_rate: float = Field(default=0.0001)
    mid_ae_rate: float = Field(default=0.0001)
    high_ae_rate: float = Field(default=0.0001)

    # Per-arm patient counts
    n_control: int = Field(default=0)
    n_low: int = Field(default=0)
    n_mid: int = Field(default=0)
    n_high: int = Field(default=0)

    # Statistical signals (scipy Fisher's exact test)
    p_value_low: float = Field(default=0.9999)
    p_value_mid: float = Field(default=0.9999)
    p_value_high: float = Field(default=0.9999)

    # Bayesian posteriors P(arm > control)
    prob_low_beats_control: float = Field(default=0.5)
    prob_mid_beats_control: float = Field(default=0.5)
    prob_high_beats_control: float = Field(default=0.5)

    # Power estimate
    estimated_power: float = Field(default=0.0001)

    # Active arm flags
    low_active: bool = Field(default=True)
    mid_active: bool = Field(default=True)
    high_active: bool = Field(default=True)

    # Stopping flags
    any_arm_significant: bool = Field(default=False)
    futility_flag: bool = Field(default=False)

    # Episode metadata — OpenEnv requires done and reward in Observation
    task_id: str = Field(default="task_1")
    stop_reason: Optional[str] = Field(default=None)
