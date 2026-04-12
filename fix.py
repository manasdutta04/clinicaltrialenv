import os
import glob
import re

graders_files = glob.glob('**/graders.py', recursive=True)
for f in graders_files:
    with open(f, 'r') as file:
        content = file.read()
    content = content.replace("STRICT_SCORE_MIN = 0.01", "STRICT_SCORE_MIN = 0.05")
    content = content.replace("STRICT_SCORE_MAX = 0.95", "STRICT_SCORE_MAX = 0.93")
    
    with open(f, 'w') as file:
        file.write(content)

inference_files = glob.glob('**/inference.py', recursive=True)
for f in inference_files:
    with open(f, 'r') as file:
        content = file.read()
    content = content.replace("STRICT_SCORE_MIN = 0.01", "STRICT_SCORE_MIN = 0.05")
    content = content.replace("STRICT_SCORE_MAX = 0.95", "STRICT_SCORE_MAX = 0.93")
    
    with open(f, 'w') as file:
        file.write(content)

print("done")
