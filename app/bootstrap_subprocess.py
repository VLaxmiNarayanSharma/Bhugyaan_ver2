"""Patch subprocess so Windows finds gcloud.cmd when EE libraries invoke `gcloud`."""
import subprocess

_original_run = subprocess.run


def patched_run(args, **kwargs):
    if args and args[0].lower() == "gcloud":
        args = list(args)
        args[0] = "gcloud.cmd"
    return _original_run(args, **kwargs)


subprocess.run = patched_run
