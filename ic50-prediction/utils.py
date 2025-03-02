import os
import random
import wandb

from datetime import datetime
from functools import wraps
from loguru import logger

import pytz
import numpy as np
import torch

def selected_features(cfg) -> list[str]:
    features = [feature for feature, flag in cfg.features.items() if flag == 1]
    if len(features) == 0:
        raise ValueError(f"[Utils] feature는 1개 이상 선택해야 합니다: {len(features)} 개")
    return features

def set_seed(seed: int):
    logger.info(f"[utils] set seed as {seed}...")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

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

def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)
