import hydra

from torch.utils.data import DataLoader

from dataset import (
    DataPreprocess, IC50Dataset,
    SimpleDNNPreprocess, SimpleDNNDataset,
)
from trainers import Trainer
from utils import log_wandb, set_seed

from omegaconf import DictConfig
from loguru import logger


@hydra.main(version_base=None, config_path="./", config_name="train_config")
@log_wandb
def run(cfg: DictConfig):
    set_seed(cfg.seed)

    logger.info("[Train] start train...")
    logger.info("[Train] 1. preprocess data...")
    if cfg.model_name in ('dnn', ):
        preprocess = SimpleDNNPreprocess(cfg.data_dir)
    else:
        preprocess = DataPreprocess(cfg.data_dir)

    logger.info("[Train] 2. split data...")
    train_df, valid_df, test_df = preprocess.split()
    if cfg.model_name in ('dnn', ):
        train_data = SimpleDNNDataset(train_df, train=True)
        valid_data = SimpleDNNDataset(valid_df, train=True)
        test_data = SimpleDNNDataset(test_df, train=False)
    else:
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
    trainer.inference(submission)
    logger.info("[Train] end run...")

if __name__ == '__main__':
    run()
