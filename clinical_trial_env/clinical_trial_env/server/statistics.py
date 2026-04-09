import numpy as np
from scipy import stats
from scipy.stats import beta as beta_dist
from dataclasses import dataclass
from typing import List

OBS_FLOAT_MIN = 0.0001
OBS_FLOAT_MAX = 0.9999

@dataclass
class BetaPosterior:
    alpha: float      # 1 + total_responders
    beta: float       # 1 + total_non_responders
    mean: float
    ci_lower: float   # 2.5th percentile
    ci_upper: float   # 97.5th percentile
    prob_better_than: float  # P(this arm > reference mean)


class TrialStatistics:
    """
    All statistical computations. Uses real scipy — no approximations.
    """

    def compute_posterior(self, cohorts: list) -> BetaPosterior:
        """
        Beta-Binomial conjugate update.
        Prior: Beta(1,1) — uniform, we know nothing.
        After observing r responders out of n:
        Posterior: Beta(1+r, 1+(n-r))
        """
        total_r = sum(c.responders for c in cohorts)
        total_n = sum(c.n_enrolled for c in cohorts)
        alpha = 1 + total_r
        beta_param = 1 + (total_n - total_r)
        dist = beta_dist(alpha, beta_param)
        return BetaPosterior(
            alpha=alpha,
            beta=beta_param,
            mean=float(dist.mean()),
            ci_lower=float(dist.ppf(0.025)),
            ci_upper=float(dist.ppf(0.975)),
            prob_better_than=0.5  # filled in compare_posteriors
        )

    def compare_posteriors(self, treatment_cohorts: list,
                           control_cohorts: list,
                           n_samples: int = 10000) -> float:
        """
        P(treatment response rate > control response rate) via Monte Carlo.
        Draw n_samples from each posterior, compute fraction where treatment > control.
        """
        ctrl_post = self.compute_posterior(control_cohorts)
        trt_post = self.compute_posterior(treatment_cohorts)
        rng = np.random.RandomState(42)
        ctrl_samples = rng.beta(ctrl_post.alpha, ctrl_post.beta, n_samples)
        trt_samples = rng.beta(trt_post.alpha, trt_post.beta, n_samples)
        return float(np.mean(trt_samples > ctrl_samples))

    def compute_pvalue(self, treatment_cohorts: list,
                       control_cohorts: list) -> float:
        """
        Two-sided Fisher's exact test on responder counts.
        Returns p-value. Uses scipy.stats.fisher_exact.
        """
        if not treatment_cohorts or not control_cohorts:
            return OBS_FLOAT_MAX
        trt_r = sum(c.responders for c in treatment_cohorts)
        trt_n = sum(c.n_enrolled for c in treatment_cohorts)
        ctrl_r = sum(c.responders for c in control_cohorts)
        ctrl_n = sum(c.n_enrolled for c in control_cohorts)
        if trt_n == 0 or ctrl_n == 0:
            return OBS_FLOAT_MAX
        table = [[trt_r, trt_n - trt_r],
                 [ctrl_r, ctrl_n - ctrl_r]]
        _, pvalue = stats.fisher_exact(table, alternative='two-sided')
        return float(np.clip(pvalue, OBS_FLOAT_MIN, OBS_FLOAT_MAX))

    def compute_power(self, n_per_arm: int,
                      p_treatment: float, p_control: float,
                      alpha: float = 0.05) -> float:
        """
        Prospective power for two-proportion z-test.
        Uses scipy.stats.norm.
        """
        if n_per_arm < 2:
            return OBS_FLOAT_MIN
        effect = abs(p_treatment - p_control)
        p_pooled = (p_treatment + p_control) / 2
        se = np.sqrt(2 * p_pooled * (1 - p_pooled) / n_per_arm)
        if se == 0:
            return OBS_FLOAT_MIN
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z = effect / se - z_alpha
        power = float(stats.norm.cdf(z))
        return float(np.clip(power, OBS_FLOAT_MIN, OBS_FLOAT_MAX))

    def futility_check(self, treatment_cohorts: list,
                       control_cohorts: list,
                       min_effect: float = 0.10) -> bool:
        """
        Predictive probability of success (PPoS).
        Returns True (futile) if P(treatment effect > min_effect | data) < 0.10.
        """
        if not treatment_cohorts:
            return False
        post = self.compute_posterior(treatment_cohorts)
        ctrl_post = self.compute_posterior(control_cohorts)
        rng = np.random.RandomState(0)
        n = 10000
        trt_s = rng.beta(post.alpha, post.beta, n)
        ctrl_s = rng.beta(ctrl_post.alpha, ctrl_post.beta, n)
        prob_meaningful = float(np.mean((trt_s - ctrl_s) > min_effect))
        return prob_meaningful < 0.10
