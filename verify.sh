#!/bin/bash
# 1. Syntax check all Python files
python -c "import ast; [ast.parse(open(f).read()) for f in ['inference.py','clinical_trial_env/server/app.py','clinical_trial_env/server/graders.py']]; print('All syntax OK')"

# 2. YAML check
python -c "import yaml; yaml.safe_load(open('openenv.yaml')); print('YAML OK')"
python -c "import yaml; yaml.safe_load(open('clinical_trial_env/openenv.yaml')); print('clinical_trial_env/YAML OK')"

# 3. Check inference.py has correct env var pattern
python -c "
src = open('inference.py').read()
assert 'HF_TOKEN = os.getenv(\"HF_TOKEN\")' in src or \"HF_TOKEN = os.getenv('HF_TOKEN')\" in src, 'HF_TOKEN must have NO default'
assert 'API_BASE_URL = os.getenv' in src, 'API_BASE_URL missing'
assert 'MODEL_NAME = os.getenv' in src, 'MODEL_NAME missing'
assert '[START]' in src, '[START] log missing'
assert '[STEP]' in src, '[STEP] log missing'
assert '[END]' in src, '[END] log missing'
assert 'from openai import OpenAI' in src, 'OpenAI client missing'
print('inference.py checklist: ALL PASS')
"
