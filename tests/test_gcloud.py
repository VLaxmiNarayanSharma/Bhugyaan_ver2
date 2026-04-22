import subprocess
import requests

# Test using "gcloud" (without extension)
try:
    subprocess.run(["gcloud", "--version"], check=True)
    print("gcloud command found using 'gcloud'")
except FileNotFoundError:
    print("gcloud command NOT found using 'gcloud'")

# Test using "gcloud.cmd" explicitly
try:
    subprocess.run(["gcloud.cmd", "--version"], check=True)
    print("gcloud command found using 'gcloud.cmd'")
except FileNotFoundError:
    print("gcloud command NOT found using 'gcloud.cmd'")
