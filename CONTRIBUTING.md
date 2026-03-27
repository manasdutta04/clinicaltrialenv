# Contributing to ClinicalTrialEnv

First off, thank you for considering contributing to ClinicalTrialEnv! It's people like you that make scalable AI and RL environments for healthcare so incredible.

## How Can I Contribute?

### Reporting Bugs
If you find a bug in the environment simulation, grading logic, or statistical engines:
1. Ensure the bug was not already reported by searching our GitHub issues.
2. Open a new issue with a clear title and detailed description. Provide code samples that reproduce the bug.

### Suggesting Enhancements
Is there a way to make the OpenEnv integration faster? A new task difficulty you'd like to see?
1. Open an issue detailing your suggested enhancement.
2. Explain why this would be useful to other RL developers in the Meta PyTorch Hackathon.

### Pull Requests
1. Fork the repo and create your branch from `main`.
2. Ensure you have installed the development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. If you've added code that should be tested, add tests to cover your changes.
4. Ensure the test suite passes (`pytest`).
5. Update the documentation (`README.md` or `openenv.yaml`) if applicable.
6. Issue that pull request!

## Environment Development Guidelines
* **Do not expose true parameters:** Ensure `true_params` (Emax, ED50, hill constants) are never exposed via the API or observation space. RL agents must strictly infer these from cohort results.
* **Use strictly SciPy:** Any new statistical measurements must use exact SciPy functions (like `scipy.stats.fisher_exact`) rather than manual approximations to maintain the medical fidelity of the environment.
* **OpenEnv Spec:** Maintain strict compatibility with `openenv.core`. WebSocket interactions run the full simulation.

## Code of Conduct
Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.
