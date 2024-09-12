import hydra

from torch.utils.data import DataLoader

from dataset import (
    DataPreprocess, IC50Dataset,
    SimpleDNNPreprocess, SimpleDNNDataset,
    XGBoostPreprocess, XGBoostDataset
)
from trainers import Trainer, XGBTrainer
from utils import (
    log_wandb, set_seed, selected_features
)

from omegaconf import DictConfig
from loguru import logger

import numpy as np
import wandb

def run_fold(cfg, fold, fold_dataset, test_df, features):
    logger.info(f"[Train]_{fold + 1} 2. split data...")
    if cfg.model_name in ('dnn', ):
        train_data = SimpleDNNDataset(fold_dataset['train_df'], train=True)
        valid_data = SimpleDNNDataset(fold_dataset['valid_df'], train=True)
        test_data = SimpleDNNDataset(test_df, train=False)
    elif cfg.model_name in ('xgb', ):
            train_data = XGBoostDataset(fold_dataset['train_df'], features, train=True)
            valid_data = XGBoostDataset(fold_dataset['valid_df'], features, train=True)
            test_data = XGBoostDataset(test_df, features, train=False)
    else:
        train_data = IC50Dataset(fold_dataset['train_df'], train=True)
        valid_data = IC50Dataset(fold_dataset['valid_df'], train=True)
        test_data = IC50Dataset(test_df, train=False)

    logger.info(f"[Train]_{fold + 1} 3. prepare dataloader...")
    if cfg.model_name in ('xgb', ):
        train_dataloader = train_data()
        valid_dataloader = valid_data()
        test_dataloader = test_data()
    else:
        train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size)
        valid_dataloader = DataLoader(valid_data, batch_size=cfg.batch_size)
        test_dataloader = DataLoader(test_data)

    logger.info(f"[Train]_{fold + 1} 4. prepare trainer...")
    if cfg.model_name in ('xgb', ):
        trainer = XGBTrainer(cfg, fold=fold)
    else:
        trainer = Trainer(cfg, fold=fold)

    logger.info(f"[Train]_{fold + 1} 5. run trainer...")
    best_valid_loss, best_valid_score = trainer.run(train_dataloader, valid_dataloader)

    logger.info(f"[Train]_{fold + 1} 6. inference trainer...")
    trainer.load_best_model()
    return trainer.evaluate(test_dataloader), best_valid_loss, best_valid_score

@hydra.main(version_base=None, config_path="./", config_name="train_config")
@log_wandb
def run(cfg: DictConfig):
    set_seed(cfg.seed)

    logger.info("[Train] start train...")
    logger.info("[Train] 1. preprocess data...")
    features: list[str] = selected_features(cfg)
    
    if cfg.model_name in ('dnn', ):
        preprocess = SimpleDNNPreprocess(cfg.data_dir, features)
    elif cfg.model_name in ('xgb', ):
        preprocess = XGBoostPreprocess(cfg.data_dir, features)
    else:
        preprocess = DataPreprocess(cfg.data_dir, features)

    if cfg.k_fold >= 2:
        k_folded_datasets, test_df = preprocess.k_fold_split(k_fold=cfg.k_fold, seed=cfg.seed)

        submissions, valid_losses, valid_scores = [], [], []
        for fold in range(cfg.k_fold):
            submission, best_valid_loss, best_valid_score = run_fold(cfg, fold, k_folded_datasets[fold], test_df, features)
            submissions.append(submission)
            valid_losses.append(best_valid_loss)
            valid_scores.append(best_valid_score)

        logger.info(f"[Output] K-fold loss: {np.mean(best_valid_loss):.4f} / score: {np.mean(best_valid_score):.4f}")
        if cfg.wandb:
            wandb.log({
                'k-fold loss': np.mean(best_valid_loss),
                'k-fold score': np.mean(best_valid_score),
            })
        
        if cfg.model_name in ('xgb', ):
            trainer = XGBTrainer(cfg)
        else:
            trainer = Trainer(cfg)
        trainer.inference(np.mean(submissions, axis=0))

    else:
        logger.info("[Train] 2. split data...")
        train_df, valid_df, test_df = preprocess.split(valid_ratio=cfg.valid_ratio)
        if cfg.model_name in ('dnn', ):
            train_data = SimpleDNNDataset(train_df, train=True)
            valid_data = SimpleDNNDataset(valid_df, train=True)
            test_data = SimpleDNNDataset(test_df, train=False)
        elif cfg.model_name in ('xgb', ):
            train_data = XGBoostDataset(train_df, features, train=True)
            valid_data = XGBoostDataset(valid_df, features, train=True)
            test_data = XGBoostDataset(test_df, features, train=False)
        else:
            train_data = IC50Dataset(train_df, train=True)
            valid_data = IC50Dataset(valid_df, train=True)
            test_data = IC50Dataset(test_df, train=False)

        logger.info("[Train] 3. prepare dataloader...")
        if cfg.model_name in ('xgb', ):
            train_dataloader = train_data()
            valid_dataloader = valid_data()
            test_dataloader = test_data()
        else:
            train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size)
            valid_dataloader = DataLoader(valid_data, batch_size=cfg.batch_size)
            test_dataloader = DataLoader(test_data)

        logger.info("[Train] 4. prepare trainer...")
        if cfg.model_name in ('xgb', ):
            trainer = XGBTrainer(cfg)
        else:
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
