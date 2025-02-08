import yaml
import logging
import random
import wandb

from datetime import datetime
from src.datas.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def load_config(config_path: str = 'config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # Setup
    setup_logging()
    config = load_config()
    logger = logging.getLogger(__name__)
    
    # Initialize wandb
    wandb.init(
        project="credit-default-prediction",
        config=config,
        name=f"xgb_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    # Set random seed
    set_seed(config['train']['random_state'])

    # Load data
    logger.info("Loading data...")
    data_loader = DataLoader(config)
    train_df, test_df = data_loader.load_data()

    # Feature engineering
    logger.info("Processing features...")
    feature_engineer = FeatureEngineer(config)
    train_processed, test_processed = feature_engineer.process_features(train_df, test_df)

    # Prepare data for CV
    logger.info("Preparing data for cross-validation...")
    X = train_processed.drop(columns=[config['data']['label_column']])
    y = train_processed[config['data']['label_column']]
    
    # Initialize CV
    n_splits = config['train'].get('n_splits', 5)  # CV 폴드 수, config에서 지정하거나 기본값 5
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config['train']['random_state']
    )
    
    # CV 학습 및 예측
    logger.info(f"Starting {n_splits}-fold cross validation...")
    test_preds = np.zeros(len(test_processed))
    val_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"Training fold {fold}/{n_splits}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBClassifier(**config['model']['params'])
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        
        # Validation score
        val_score = model.score(X_val, y_val)
        val_scores.append(val_score)
        logger.info(f"Fold {fold} validation score: {val_score:.4f}")
        
        # Log metrics to wandb
        wandb.log({
            f"fold_{fold}_val_score": val_score,
            "fold": fold,
        })
        
        # Predict test data
        test_preds += model.predict_proba(test_processed)[:, 1] / n_splits

    mean_val_score = np.mean(val_scores)
    std_val_score = np.std(val_scores)
    logger.info(f"Mean validation score: {mean_val_score:.4f} ± {std_val_score:.4f}")
    
    # Log final metrics to wandb
    wandb.log({
        "mean_val_score": mean_val_score,
        "std_val_score": std_val_score,
    })

    # Save results
    logger.info("Saving results...")
    submit = pd.read_csv(config['data']['submission_path'])
    submit['채무 불이행 확률'] = test_preds
    submit.to_csv(f"{config['data']['output_path']}_{datetime.now()}.csv", encoding='UTF-8-sig', index=False)

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
