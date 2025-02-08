import yaml
import logging
import random
import wandb
import os

from datetime import datetime
from src.datas.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # Set additional seeds for libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def load_config(config_path: str = 'config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_model(model_config):
    """모델 설정에 따라 적절한 모델을 반환합니다."""
    model_type = model_config['type'].lower()
    random_state = model_config.get('random_state', 42)  # default random_state
    logging.info(f"[Model] {model_type} is selected..")
    
    if model_type == 'xgboost':
        params = model_config['xgboost'].copy()
        params['random_state'] = random_state
        params['seed'] = random_state  # XGBoost specific seed
        return XGBClassifier(**params)
    elif model_type == 'lightgbm':
        params = model_config['lightgbm'].copy()
        params['seed'] = random_state  # LightGBM specific seed
        params['deterministic'] = True  # LightGBM 결과 재현성 보장
        return LGBMClassifier(**params)
    elif model_type == 'catboost':
        params = model_config['catboost'].copy()
        params['random_seed'] = random_state
        return CatBoostClassifier(**params, verbose=False)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def plot_feature_importance(feature_names, importance_values, title):
    """Feature importance를 시각화합니다."""
    # Sort features by importance
    indices = np.argsort(importance_values)[::-1]
    
    # Plot top 20 features
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(min(20, len(feature_names))), 
            importance_values[indices][:20])
    plt.xticks(range(min(20, len(feature_names))), 
               [feature_names[i] for i in indices][:20], 
               rotation=45, ha='right')
    plt.tight_layout()
    
    # Log plot to wandb
    wandb.log({title: wandb.Image(plt)})
    plt.close()

def main():
    # Set random seed
    config = load_config()
    random_state = config['train']['random_state']
    set_seed(random_state)
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize wandb with fixed random state
    run_name = f"{config['model']['type'].lower()}_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="credit-default-prediction",
        config=config,
        name=run_name,
        settings=wandb.Settings(start_method="thread"),
        # wandb도 random seed 고정
        job_type=f"seed_{random_state}"
    )
    
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
    feature_importance = np.zeros(len(X.columns))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"Training fold {fold}/{n_splits}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = get_model(config['model'])
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(X_train, y_train, eval_set=eval_set)
        
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
        
        # Calculate feature importance
        if config['model']['type'].lower() == 'xgboost':
            importance = model.feature_importances_
        elif config['model']['type'].lower() == 'lightgbm':
            importance = model.feature_importances_
        elif config['model']['type'].lower() == 'catboost':
            importance = model.feature_importances_
        
        feature_importance += importance / n_splits
        
        # Log individual fold feature importance
        plot_feature_importance(
            X.columns, 
            importance,
            f"Fold {fold} Feature Importance"
        )

    mean_val_score = np.mean(val_scores)
    std_val_score = np.std(val_scores)
    logger.info(f"Mean validation score: {mean_val_score:.4f} ± {std_val_score:.4f}")
    
    # Log final metrics to wandb
    wandb.log({
        "mean_val_score": mean_val_score,
        "std_val_score": std_val_score,
    })

    # Plot average feature importance across all folds
    plot_feature_importance(
        X.columns,
        feature_importance,
        "Average Feature Importance"
    )
    
    # Save feature importance to CSV
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    feature_importance_df.to_csv(
        f"./data/feature_importance_{run_name}.csv",
        encoding='UTF-8-sig',
        index=False
    )

    # Save results
    logger.info("Saving results...")
    submit = pd.read_csv(config['data']['submission_path'])
    submit['채무 불이행 확률'] = test_preds
    submit.to_csv(f"./data/{run_name}.csv", encoding='UTF-8-sig', index=False)

    # 매핑 확인
    feature_engineer.print_ordinal_mapping()

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
