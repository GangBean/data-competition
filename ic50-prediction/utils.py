import wandb

from functools import wraps

def log_wandb(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        cfg = args[0]
        if cfg.wandb:
            wandb.init(
                project=cfg.project,
                config=dict(cfg),
                notes=cfg.notes,
                tags=cfg.tags,
            )
        result = func(*args, **kwargs)
        if cfg.wandb:
            wandb.finish()
        return result
    return wrapper
