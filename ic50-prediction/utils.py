import wandb

from functools import wraps

from datetime import datetime
import pytz


def log_wandb(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        cfg = args[0]
        run_time: str = datetime.now().astimezone(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
        run_name: str = f'[{cfg.model_name}]{run_time}'
        cfg['run_name'] = run_name
        if cfg.wandb:
            wandb.init(
                project=cfg.project,
                name=run_name,
                config=dict(cfg),
                notes=cfg.notes,
                tags=cfg.tags,
            )
        result = func(*args, **kwargs)
        if cfg.wandb:
            wandb.finish()
        return result
    return wrapper
