# Location: notebooks/kaggle_gpu_runner_cell.py
# Paste this into a single Kaggle notebook cell (GPU accelerator + Internet ON).
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
GH_RUNNER_TOKEN = secrets.get_secret("GH_RUNNER_TOKEN")
GH_REPO_URL = secrets.get_secret("GH_REPO_URL")

os.system("mkdir -p /kaggle/working/actions-runner")
os.chdir("/kaggle/working/actions-runner")
os.system(
    "curl -o runner.tar.gz -L "
    "https://github.com/actions/runner/releases/latest/download/actions-runner-linux-x64-2.319.1.tar.gz "
    "&& tar xzf runner.tar.gz"
)
os.system(
    f"./config.sh --url {GH_REPO_URL} --token {GH_RUNNER_TOKEN} "
    "--labels self-hosted,gpu,kaggle --name kaggle-gpu-runner "
    "--work _work --unattended --ephemeral"
)
os.system("./run.sh")