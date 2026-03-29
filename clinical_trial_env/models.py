from typing import Optional
from openenv.core import Action as BaseAction, Observation as BaseObservation


class TrialAction(BaseAction):
    """Agent action: how to allocate patients in the next cohort."""
    n_next_cohort: int = 20
    allocation_control: float = 0.25
    allocation_low: float = 0.25
    allocation_mid: float = 0.25
    allocation_high: float = 0.25
    stop_for_success: bool = False
    stop_for_futility: bool = False
    drop_arm: Optional[str] = None

    def model_post_init(self, __context):
        """Pydantic v2 equivalent of __post_init__: normalize allocations."""
        self.n_next_cohort = max(5, min(100, self.n_next_cohort))
        total = (self.allocation_control + self.allocation_low +
                 self.allocation_mid + self.allocation_high)
        if total > 0:
            object.__setattr__(self, 'allocation_control', self.allocation_control / total)
            object.__setattr__(self, 'allocation_low', self.allocation_low / total)
            object.__setattr__(self, 'allocation_mid', self.allocation_mid / total)
            object.__setattr__(self, 'allocation_high', self.allocation_high / total)


class TrialObservation(BaseObservation):
    """Observed state of the adaptive clinical trial."""
    # Trial progress
    interim_number: int = 0
    total_patients_enrolled: int = 0
    budget_remaining: int = 0

    # Per-arm observed response rates (hidden from agent — must infer)
    control_response_rate: float = 0.0
    low_response_rate: float = 0.0
    mid_response_rate: float = 0.0
    high_response_rate: float = 0.0

    # Per-arm adverse event rates
    control_ae_rate: float = 0.0
    low_ae_rate: float = 0.0
    mid_ae_rate: float = 0.0
    high_ae_rate: float = 0.0

    # Per-arm patient counts
    n_control: int = 0
    n_low: int = 0
    n_mid: int = 0
    n_high: int = 0

    # Statistical signals (computed by scipy)
    p_value_low: float = 1.0
    p_value_mid: float = 1.0
    p_value_high: float = 1.0

    # Bayesian posteriors P(arm > control)
    prob_low_beats_control: float = 0.5
    prob_mid_beats_control: float = 0.5
    prob_high_beats_control: float = 0.5

    # Power estimate
    estimated_power: float = 0.0

    # Active arm flags
    low_active: bool = True
    mid_active: bool = True
    high_active: bool = True

    # Flags
    any_arm_significant: bool = False
    futility_flag: bool = False

    # Episode metadata
    task_id: str = "task_1"
    stop_reason: Optional[str] = None
