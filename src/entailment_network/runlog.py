"""Optional Weights & Biases logging for experiment scripts.

No-op unless `wandb` is importable AND credentials are reachable, via either:
  - WANDB_API_KEY env var, or
  - a Kaggle UserSecretsClient secret named "WANDB_API_KEY".

WANDB_PROJECT env var overrides the default project name. Everything here is
safe to call unconditionally — get_wandb() returns None and log()/finish()
become no-ops when logging is unavailable.
"""

import os

_run = None
_tried = False


def get_wandb(project=None, name=None, config=None):
    """Initialize a wandb run once; returns the run or None."""
    global _run, _tried
    if _tried:
        return _run
    _tried = True
    try:
        import wandb  # noqa: F401
    except Exception:
        return None
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        try:
            from kaggle_secrets import UserSecretsClient
            key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception:
            key = None
    if not key:
        return None
    os.environ.setdefault("WANDB_API_KEY", key)
    os.environ.setdefault("WANDB_PROJECT", project or "trace-rag")
    project = project or os.environ.get("WANDB_PROJECT", "trace-rag")
    try:
        _run = wandb.init(project=project, name=name, config=config)
    except Exception:
        _run = None
    return _run


def log(metrics):
    if _run is not None:
        try:
            _run.log(metrics)
        except Exception:
            pass


def finish():
    global _run
    if _run is not None:
        try:
            _run.finish()
        except Exception:
            pass
    _run = None
