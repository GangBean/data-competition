import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from dataset import DataPreprocess, IC50Dataset
from trainers import Trainer
from utils import log_wandb

from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from datetime import datetime
import pytz


@hydra.main(version_base=None, config_path="./", config_name="train_config")
@log_wandb
def run(cfg: DictConfig):
    logger.info("[Train] start train...")

    logger.info("[Train] 1. preprocess data...")
    preprocess = DataPreprocess(cfg.data_dir)

    logger.info("[Train] 2. split data...")
    train_df, valid_df, test_df = preprocess.split()
    train_data = IC50Dataset(train_df, train=True)
    valid_data = IC50Dataset(valid_df, train=True)
    test_data = IC50Dataset(test_df, train=False)

    logger.info("[Train] 3. prepare dataloader...")
    train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size)
    valid_dataloader = DataLoader(valid_data, batch_size=cfg.batch_size)
    test_dataloader = DataLoader(test_data)

    logger.info("[Train] 4. prepare trainer...")
    trainer = Trainer(cfg)

    logger.info("[Train] 5. run trainer...")
    trainer.run(train_dataloader, valid_dataloader)

    logger.info("[Train] 6. inference trainer...")
    trainer.load_best_model()
    submission = trainer.evaluate(test_dataloader)

    sample_df = pd.read_csv('./data/sample_submission.csv')
    sample_df['IC50_nM'] = np.array(submission).reshape(-1)

    output_name = datetime.now(pytz.timezone("Asia/Seoul"))
    sample_df.to_csv(f'./data/submissions/{output_name}.csv', index=False)

    logger.info("[Train] end train...")

if __name__ == '__main__':
    run()
