from openenv.core import Environment
from openenv.core.client_types import StepResult
from openenv.core import State
import uuid, numpy as np
from .patient_simulator import PatientSimulator
from .statistics import TrialStatistics
from .tasks import TASKS
from .graders import efficacy_grader, tradeoff_grader, efficiency_grader
from models import TrialAction, TrialObservation


class ClinicalTrialEnvironment(Environment):
    """
    OpenEnv-compliant server-side environment for adaptive clinical trial design.

    One instance per WebSocket connection (OpenEnv creates a new env per session).
    No global state — fully isolated.
    """

    def __init__(self):
        self.session_id: str = str(uuid.uuid4())
        self.task = None
        self.simulator: PatientSimulator = None
        self.stats: TrialStatistics = TrialStatistics()

        # Accumulated cohort data per arm
        self.arm_data: dict = {"control": [], "low": [], "mid": [], "high": []}
        self.dropped_arms: set = set()

        # Episode tracking
        self.interim_number: int = 0
        self.total_enrolled: int = 0
        self.episode_active: bool = False
        self.stop_reason: str = None
        self.prev_best_acc: float = 0.0
        self.unsafe_arm_patients: int = 0

    async def reset(self, task_id: str = "task_1") -> TrialObservation:
        """
        Initialize a new trial episode.
        Called when agent sends reset command.
        Returns initial (zeroed) observation.
        """
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
        self.prev_best_acc = 0.0
        self.unsafe_arm_patients = 0

        return TrialObservation(
            task_id=task_id,
            budget_remaining=self.task["max_patients"]
        )

    async def step(self, action: TrialAction) -> StepResult:
        """
        Execute one interim analysis period.
        Agent enrolls a cohort, receives updated statistics + reward.
        """
        if not self.episode_active:
            obs = self._build_observation()
            return StepResult(observation=obs, reward=0.0, done=True)

        # Handle arm drop request
        if action.drop_arm and action.drop_arm in ("low", "mid", "high"):
            self.dropped_arms.add(action.drop_arm)

        # Compute per-arm cohort sizes from allocation ratios
        n = max(5, action.n_next_cohort)
        arm_sizes = {
            "control": max(1, round(action.allocation_control * n)),
            "low":     0 if "low" in self.dropped_arms else max(1, round(action.allocation_low * n)),
            "mid":     0 if "mid" in self.dropped_arms else max(1, round(action.allocation_mid * n)),
            "high":    0 if "high" in self.dropped_arms else max(1, round(action.allocation_high * n)),
        }

        # Enroll patients per arm
        doses = self.task["doses"]
        for arm, count in arm_sizes.items():
            if count == 0:
                continue
            if arm == "control":
                cohort = self.simulator.enroll_control(count)
            else:
                cohort = self.simulator.enroll_cohort(count, doses[arm], arm)
            self.arm_data[arm].append(cohort)
            self.total_enrolled += count

            # Track unsafe arm patients (for task_2 grader)
            ae_threshold = self.task["ae_stopping_threshold"]
            cumulative_ae = (sum(c.adverse_events for c in self.arm_data[arm]) /
                             max(1, sum(c.n_enrolled for c in self.arm_data[arm])))
            if cumulative_ae > ae_threshold * 0.8:
                self.unsafe_arm_patients += count

        self.interim_number += 1

        # Build observation with current stats
        obs = self._build_observation()

        # Check forced stopping rules (override agent)
        forced_stop = self._check_forced_stops(obs)
        if forced_stop:
            self.stop_reason = forced_stop
            self.episode_active = False
            obs.stop_reason = forced_stop
            reward = self._terminal_reward()
            return StepResult(observation=obs, reward=reward, done=True)

        # Check agent-requested stops
        min_i = self.task["min_interims_before_stop"]
        if action.stop_for_success and self.interim_number >= min_i:
            if obs.any_arm_significant:
                self.stop_reason = "success"
                self.episode_active = False
                obs.stop_reason = "success"
                reward = self._terminal_reward()
                return StepResult(observation=obs, reward=reward, done=True)

        if action.stop_for_futility and self.interim_number >= min_i:
            self.stop_reason = "futility"
            self.episode_active = False
            obs.stop_reason = "futility"
            reward = self._terminal_reward()
            return StepResult(observation=obs, reward=reward, done=True)

        # Check budget
        if self.total_enrolled >= self.task["max_patients"]:
            self.stop_reason = "budget_exhausted"
            self.episode_active = False
            obs.stop_reason = "budget_exhausted"
            reward = self._terminal_reward()
            return StepResult(observation=obs, reward=reward, done=True)

        # Continuing — compute dense shaping reward
        reward = self._step_reward(obs)
        return StepResult(observation=obs, reward=reward, done=False)

    async def state(self) -> State:
        return State(
            episode_id=self.session_id,
            step_count=self.interim_number,
            metadata={
                "task": self.task["task_id"] if self.task else None,
                "total_enrolled": self.total_enrolled,
                "episode_active": self.episode_active,
                "stop_reason": self.stop_reason
            }
        )

    def _build_observation(self) -> TrialObservation:
        """Build full TrialObservation from current accumulated data."""
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

        # p-values
        p_low  = self.stats.compute_pvalue(low, ctrl)
        p_mid  = self.stats.compute_pvalue(mid, ctrl)
        p_high = self.stats.compute_pvalue(high, ctrl)

        # Posteriors
        prob_low  = self.stats.compare_posteriors(low, ctrl)  if low  else 0.5
        prob_mid  = self.stats.compare_posteriors(mid, ctrl)  if mid  else 0.5
        prob_high = self.stats.compare_posteriors(high, ctrl) if high else 0.5

        # Futility
        fut_low  = self.stats.futility_check(low, ctrl)  if "low"  not in self.dropped_arms else True
        fut_mid  = self.stats.futility_check(mid, ctrl)  if "mid"  not in self.dropped_arms else True
        fut_high = self.stats.futility_check(high, ctrl) if "high" not in self.dropped_arms else True
        all_futile = fut_low and fut_mid and fut_high

        # Best arm for power estimate
        best_p = max(prob_low, prob_mid, prob_high)
        n_best = max(count(low), count(mid), count(high))
        ctrl_r = rate(ctrl)
        best_r = max(rate(low), rate(mid), rate(high))
        power = self.stats.compute_power(
            max(1, n_best), best_r, ctrl_r
        ) if n_best > 0 else 0.0

        return TrialObservation(
            task_id=self.task["task_id"] if self.task else "task_1",
            interim_number=self.interim_number,
            total_patients_enrolled=self.total_enrolled,
            budget_remaining=max(0, self.task["max_patients"] - self.total_enrolled),
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
            futility_flag=all_futile
        )

    def _check_forced_stops(self, obs: TrialObservation) -> str | None:
        """Forced stops the agent cannot override."""
        ae_thresh = self.task["ae_stopping_threshold"]
        for arm, ae_rate in [("low", obs.low_ae_rate),
                              ("mid", obs.mid_ae_rate),
                              ("high", obs.high_ae_rate)]:
            if arm not in self.dropped_arms and ae_rate > ae_thresh:
                return "safety_stop"
        if self.total_enrolled >= self.task["max_patients"]:
            return "budget_exhausted"
        return None

    def _terminal_reward(self) -> float:
        """Compute final episode reward based on outcome."""
        result = self.grade()
        return float(result.score)

    def _step_reward(self, obs: TrialObservation) -> float:
        """Dense shaping reward each step — keeps agent learning between episodes."""
        # Reward allocation toward best posterior arm
        best_prob = max(obs.prob_low_beats_control,
                        obs.prob_mid_beats_control,
                        obs.prob_high_beats_control)
        alloc_bonus = 0.02 * best_prob

        # Penalize AE rate approaching threshold
        ae_thresh = self.task["ae_stopping_threshold"]
        max_ae = max(obs.low_ae_rate, obs.mid_ae_rate, obs.high_ae_rate)
        ae_penalty = -0.03 * max(0.0, max_ae / ae_thresh - 0.7)

        # Bonus for any arm becoming significant
        sig_bonus = 0.05 if obs.any_arm_significant else 0.0

        reward = alloc_bonus + ae_penalty + sig_bonus
        return float(np.clip(reward, -0.1, 0.15))

    def grade(self) -> "GraderResult":
        """Run the appropriate grader for this task."""
        session_state = {
            "stop_reason": self.stop_reason,
            "total_enrolled": self.total_enrolled,
            "task": self.task,
            "best_pvalue": min(
                self.stats.compute_pvalue(self.arm_data["low"], self.arm_data["control"]),
                self.stats.compute_pvalue(self.arm_data["mid"], self.arm_data["control"]),
                self.stats.compute_pvalue(self.arm_data["high"], self.arm_data["control"])
            ) if any(self.arm_data[a] for a in ["low","mid","high"]) else 1.0,
            "best_posterior": max(
                self.stats.compare_posteriors(self.arm_data["low"], self.arm_data["control"]) if self.arm_data["low"] else 0.5,
                self.stats.compare_posteriors(self.arm_data["mid"], self.arm_data["control"]) if self.arm_data["mid"] else 0.5,
                self.stats.compare_posteriors(self.arm_data["high"], self.arm_data["control"]) if self.arm_data["high"] else 0.5,
            ),
            "unsafe_arm_patients": self.unsafe_arm_patients,
            "interim_number": self.interim_number,
            "arm_data": self.arm_data
        }
        grader_map = {
            "efficacy_grader": efficacy_grader,
            "tradeoff_grader": tradeoff_grader,
            "efficiency_grader": efficiency_grader
        }
        fn = grader_map[self.task["grader"]]
        return fn(session_state)
