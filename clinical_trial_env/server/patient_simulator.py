import numpy as np
from dataclasses import dataclass

@dataclass
class CohortResult:
    arm: str
    dose: float
    n_enrolled: int
    responders: int
    adverse_events: int

    @property
    def response_rate(self) -> float:
        return self.responders / self.n_enrolled if self.n_enrolled > 0 else 0.0

    @property
    def ae_rate(self) -> float:
        return self.adverse_events / self.n_enrolled if self.n_enrolled > 0 else 0.0


class PatientSimulator:
    """
    Simulates patient responses using the Emax pharmacological model (Hill equation).
    This is the FDA/ICH standard model for dose-response relationships.

    TRUE PARAMETERS ARE HIDDEN — agent never sees these, must infer from data.

    Emax model (Hill equation):
        response_rate(dose) = baseline + (Emax × dose^hill) / (ED50^hill + dose^hill)

    Adverse event model:
        ae_rate(dose) = ae_baseline + ae_slope × (dose / max_dose)^2
    """

    def __init__(self, true_params: dict, seed: int = None):
        """
        true_params = {
            "baseline": 0.10,      # placebo response rate
            "emax": 0.45,          # max drug effect above baseline
            "ed50": 30.0,          # dose giving 50% of max effect
            "hill": 1.5,           # steepness of curve
            "ae_baseline": 0.05,   # background AE rate
            "ae_slope": 0.15       # dose-dependent AE increase
        }
        """
        self.params = true_params
        self.rng = np.random.RandomState(seed)
        self.max_dose = 60.0  # reference for AE model

    def true_response_rate(self, dose: float) -> float:
        p = self.params
        if dose == 0:
            return p["baseline"]
        numerator = p["emax"] * (dose ** p["hill"])
        denominator = (p["ed50"] ** p["hill"]) + (dose ** p["hill"])
        rate = p["baseline"] + numerator / denominator
        return float(np.clip(rate, 0.01, 0.99))

    def true_ae_rate(self, dose: float) -> float:
        p = self.params
        rate = p["ae_baseline"] + p["ae_slope"] * ((dose / self.max_dose) ** 2)
        return float(np.clip(rate, 0.01, 0.99))

    def enroll_cohort(self, n_patients: int, dose: float, arm: str) -> CohortResult:
        """
        Simulate n_patients at a given dose.
        Individual patient variability: ±5% noise on true response probability.
        Returns CohortResult with observed counts.
        """
        true_p = self.true_response_rate(dose)
        true_ae = self.true_ae_rate(dose)

        # Individual variability — each patient has slightly different response prob
        patient_probs = np.clip(
            self.rng.normal(true_p, 0.05, n_patients), 0.01, 0.99
        )
        responders = int(sum(self.rng.binomial(1, p) for p in patient_probs))
        adverse_events = int(self.rng.binomial(n_patients, true_ae))

        return CohortResult(
            arm=arm,
            dose=dose,
            n_enrolled=n_patients,
            responders=responders,
            adverse_events=adverse_events
        )

    def enroll_control(self, n_patients: int) -> CohortResult:
        return self.enroll_cohort(n_patients, dose=0.0, arm="control")
