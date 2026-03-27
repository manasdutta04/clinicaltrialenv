from openenv.core.client import EnvClient
from .models import TrialAction, TrialObservation


class ClinicalTrialEnv(EnvClient):
    """
    Client for ClinicalTrialEnv.
    Usage:
        async with ClinicalTrialEnv(base_url="https://your-hf-space.hf.space") as env:
            obs = await env.reset("task_1")
            result = await env.step(TrialAction(n_next_cohort=20))
    """
    action_class = TrialAction
    observation_class = TrialObservation
