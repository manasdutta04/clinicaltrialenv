"""
ClinicalTrialEnvironment — synchronous, stateful per-session environment.
Follows the exact pattern from openenv-core's scaffold template.
"""
from uuid import uuid4
import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .graders import efficacy_grader, tradeoff_grader, efficiency_grader
from .patient_simulator import PatientSimulator
from .session_store import _completed_sessions
from .statistics import TrialStatistics
from .tasks import TASKS

try:
    from models import TrialAction, TrialObservation
except ImportError:
    from clinical_trial_env.models import TrialAction, TrialObservation


class ClinicalTrialEnvironment(Environment):
    """
    OpenEnv-compliant adaptive clinical trial design environment.

    Each WebSocket session gets its own instance (SUPPORTS_CONCURRENT_SESSIONS=True).
    The agent acts as trial statistician, making sequential decisions about patient
    enrollment, dose allocation, and early stopping.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.task = None
        self.simulator: PatientSimulator = None
        self.stats: TrialStatistics = TrialStatistics()
        self.arm_data: dict = {"control": [], "low": [], "mid": [], "high": []}
        self.dropped_arms: set = set()
        self.interim_number: int = 0
        self.total_enrolled: int = 0
        self.episode_active: bool = False
        self.stop_reason: str = None
        self.unsafe_arm_patients: int = 0
        self.budget_consumed: float = 0.0
        self.last_strictness: float = 0.5
        self.enroll_cost_multiplier: float = 1.0

        # Auto-initialize so step() works even before explicit reset()
        self.reset("task_1")

    def reset(self, task_id: str = "task_1") -> TrialObservation:
        """Initialize a new trial episode."""
        if task_id not in TASKS:
            task_id = "task_1"

        self.task = TASKS[task_id]
        self.simulator = PatientSimulator(
            self.task["true_params"],
            seed=np.random.randint(0, 99999)
        )
        self.arm_data = {"control": [], "low": [], "mid": [], "high": []}
        self.dropped_arms = set()
        self.interim_number = 0
        self.total_enrolled = 0
        self.episode_active = True
        self.stop_reason = None
        self.unsafe_arm_patients = 0
        self.budget_consumed = 0.0
        self.last_strictness = 0.5
        self.enroll_cost_multiplier = 1.0
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return TrialObservation(
            task_id=task_id,
            budget_remaining=self.task["max_patients"],
            done=False,
            reward=0.0,
        )

    def step(self, action: TrialAction) -> TrialObservation:
        """Execute one interim analysis period."""
        # Safety guard: if task is somehow None, reset with default
        if self.task is None:
            self.reset("task_1")
        if not self.episode_active:
            obs = self._build_observation()
            obs.done = True
            obs.reward = 0.0
            return obs

        # Handle arm drop
        if action.drop_arm and action.drop_arm in ("low", "mid", "high"):
            self.dropped_arms.add(action.drop_arm)

        # Compute per-arm cohort sizes
        n = max(5, action.n_next_cohort)
        arm_sizes = {
            "control": max(1, round(action.allocation_control * n)),
            "low":     0 if "low"  in self.dropped_arms else max(1, round(action.allocation_low  * n)),
            "mid":     0 if "mid"  in self.dropped_arms else max(1, round(action.allocation_mid  * n)),
            "high":    0 if "high" in self.dropped_arms else max(1, round(action.allocation_high * n)),
        }

        doses = self.task["doses"]
        ae_threshold = self.task["ae_stopping_threshold"]
        strictness = action.inclusion_criteria_strictness
        
        self.last_strictness = strictness
        self.enroll_cost_multiplier = 1.0 + (strictness ** 2) * 5.0

        for arm, count in arm_sizes.items():
            if count == 0:
                continue
            if arm == "control":
                cohort = self.simulator.enroll_control(count, strictness=strictness)
            else:
                cohort = self.simulator.enroll_cohort(count, doses[arm], arm, strictness=strictness)
            self.arm_data[arm].append(cohort)
            self.total_enrolled += count
            self.budget_consumed += count * self.enroll_cost_multiplier

            cumulative_ae = (sum(c.adverse_events for c in self.arm_data[arm]) /
                             max(1, sum(c.n_enrolled for c in self.arm_data[arm])))
            if cumulative_ae > ae_threshold * 0.8:
                self.unsafe_arm_patients += count

        self.interim_number += 1
        self._state.step_count = self.interim_number

        obs = self._build_observation()

        # Forced safety/budget stops
        forced_stop = self._check_forced_stops(obs)
        if forced_stop:
            self.stop_reason = forced_stop
            self.episode_active = False
            reward = float(self._grade_score())
            obs.stop_reason = forced_stop
            obs.done = True
            obs.reward = reward
            self._mark_completed()
            return obs

        min_i = self.task["min_interims_before_stop"]

        if action.stop_for_success and self.interim_number >= min_i and obs.any_arm_significant:
            self.stop_reason = "success"
            self.episode_active = False
            reward = float(self._grade_score())
            obs.stop_reason = "success"
            obs.done = True
            obs.reward = reward
            self._mark_completed()
            return obs

        if action.stop_for_futility and self.interim_number >= min_i:
            self.stop_reason = "futility"
            self.episode_active = False
            reward = float(self._grade_score())
            obs.stop_reason = "futility"
            obs.done = True
            obs.reward = reward
            self._mark_completed()
            return obs

        if self.budget_consumed >= self.task["max_patients"]:
            self.stop_reason = "budget_exhausted"
            self.episode_active = False
            reward = float(self._grade_score())
            obs.stop_reason = "budget_exhausted"
            obs.done = True
            obs.reward = reward
            self._mark_completed()
            return obs

        # Continuing step reward
        obs.done = False
        obs.reward = float(self._step_reward(obs))
        return obs

    @property
    def state(self) -> State:
        return self._state

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_observation(self) -> TrialObservation:
        def rate(cohorts):
            r = sum(c.responders for c in cohorts)
            n = sum(c.n_enrolled for c in cohorts)
            return r / n if n > 0 else 0.0

        def ae(cohorts):
            a = sum(c.adverse_events for c in cohorts)
            n = sum(c.n_enrolled for c in cohorts)
            return a / n if n > 0 else 0.0

        def count(cohorts):
            return sum(c.n_enrolled for c in cohorts)

        ctrl = self.arm_data["control"]
        low  = self.arm_data["low"]
        mid  = self.arm_data["mid"]
        high = self.arm_data["high"]

        p_low  = self.stats.compute_pvalue(low, ctrl)
        p_mid  = self.stats.compute_pvalue(mid, ctrl)
        p_high = self.stats.compute_pvalue(high, ctrl)

        prob_low  = self.stats.compare_posteriors(low, ctrl)  if low  else 0.5
        prob_mid  = self.stats.compare_posteriors(mid, ctrl)  if mid  else 0.5
        prob_high = self.stats.compare_posteriors(high, ctrl) if high else 0.5

        fut_low  = self.stats.futility_check(low, ctrl)  if "low"  not in self.dropped_arms else True
        fut_mid  = self.stats.futility_check(mid, ctrl)  if "mid"  not in self.dropped_arms else True
        fut_high = self.stats.futility_check(high, ctrl) if "high" not in self.dropped_arms else True
        all_futile = fut_low and fut_mid and fut_high

        n_best = max(count(low), count(mid), count(high))
        best_r = max(rate(low), rate(mid), rate(high))
        ctrl_r = rate(ctrl)
        power = self.stats.compute_power(max(1, n_best), best_r, ctrl_r) if n_best > 0 else 0.0

        return TrialObservation(
            task_id=self.task["task_id"] if self.task else "task_1",
            interim_number=self.interim_number,
            total_patients_enrolled=self.total_enrolled,
            budget_remaining=int(max(0, self.task["max_patients"] - self.budget_consumed)),
            enrollment_rate=round(1.0 / self.enroll_cost_multiplier, 4),
            population_heterogeneity=round(0.05 + ((1.0 - self.last_strictness) * 0.15), 4),
            control_response_rate=round(rate(ctrl), 4),
            low_response_rate=round(rate(low), 4),
            mid_response_rate=round(rate(mid), 4),
            high_response_rate=round(rate(high), 4),
            control_ae_rate=round(ae(ctrl), 4),
            low_ae_rate=round(ae(low), 4),
            mid_ae_rate=round(ae(mid), 4),
            high_ae_rate=round(ae(high), 4),
            n_control=count(ctrl),
            n_low=count(low),
            n_mid=count(mid),
            n_high=count(high),
            p_value_low=round(p_low, 4),
            p_value_mid=round(p_mid, 4),
            p_value_high=round(p_high, 4),
            prob_low_beats_control=round(prob_low, 4),
            prob_mid_beats_control=round(prob_mid, 4),
            prob_high_beats_control=round(prob_high, 4),
            estimated_power=round(power, 4),
            low_active="low" not in self.dropped_arms,
            mid_active="mid" not in self.dropped_arms,
            high_active="high" not in self.dropped_arms,
            any_arm_significant=any(p < 0.05 for p in [p_low, p_mid, p_high]),
            futility_flag=all_futile,
            done=False,
            reward=0.0,
        )

    def _mark_completed(self) -> None:
        """Persist the last completed environment per task for grading."""
        if self.task:
            _completed_sessions[self.task["task_id"]] = self

    def _check_forced_stops(self, obs: TrialObservation) -> str | None:
        ae_thresh = self.task["ae_stopping_threshold"]
        for arm, ae_rate in [("low", obs.low_ae_rate),
                              ("mid", obs.mid_ae_rate),
                              ("high", obs.high_ae_rate)]:
            if arm not in self.dropped_arms and ae_rate > ae_thresh:
                return "safety_stop"
        if self.budget_consumed >= self.task["max_patients"]:
            return "budget_exhausted"
        return None

    def _grade_score(self) -> float:
        result = self.grade()
        return float(result.score)

    def _step_reward(self, obs: TrialObservation) -> float:
        best_prob = max(obs.prob_low_beats_control,
                        obs.prob_mid_beats_control,
                        obs.prob_high_beats_control)
        ae_thresh = self.task["ae_stopping_threshold"]
        max_ae = max(obs.low_ae_rate, obs.mid_ae_rate, obs.high_ae_rate)
        ae_penalty = -0.03 * max(0.0, max_ae / ae_thresh - 0.7)
        sig_bonus = 0.05 if obs.any_arm_significant else 0.0
        reward = 0.02 * best_prob + ae_penalty + sig_bonus
        return float(np.clip(reward, -0.1, 0.15))

    def grade(self):
        """Run the appropriate task grader."""
        session_state = {
            "stop_reason": self.stop_reason,
            "total_enrolled": self.total_enrolled,
            "task": self.task,
            "best_pvalue": min(
                self.stats.compute_pvalue(self.arm_data["low"], self.arm_data["control"]),
                self.stats.compute_pvalue(self.arm_data["mid"], self.arm_data["control"]),
                self.stats.compute_pvalue(self.arm_data["high"], self.arm_data["control"])
            ) if any(self.arm_data[a] for a in ["low", "mid", "high"]) else 1.0,
            "best_posterior": max(
                self.stats.compare_posteriors(self.arm_data["low"], self.arm_data["control"]) if self.arm_data["low"] else 0.5,
                self.stats.compare_posteriors(self.arm_data["mid"], self.arm_data["control"]) if self.arm_data["mid"] else 0.5,
                self.stats.compare_posteriors(self.arm_data["high"], self.arm_data["control"]) if self.arm_data["high"] else 0.5,
            ),
            "unsafe_arm_patients": self.unsafe_arm_patients,
            "interim_number": self.interim_number,
            "arm_data": self.arm_data,
            "budget_consumed": self.budget_consumed,
        }
        grader_map = {
            "efficacy_grader": efficacy_grader,
            "tradeoff_grader": tradeoff_grader,
            "efficiency_grader": efficiency_grader,
        }
        fn = grader_map[self.task["grader"]]
        return fn(session_state)
