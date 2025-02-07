import yaml
import logging
from src.datas.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config(config_path: str = 'config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # Setup
    setup_logging()
    config = load_config()
    logger = logging.getLogger(__name__)

    # Load data
    logger.info("Loading data...")
    data_loader = DataLoader(config)
    train_df, test_df = data_loader.load_data()

    # Feature engineering
    logger.info("Processing features...")
    feature_engineer = FeatureEngineer(config)
    train_processed, test_processed = feature_engineer.process_features(train_df, test_df)

    # Split data
    logger.info("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        train_processed.drop(columns=[config['data']['label_column']]),
        train_processed[config['data']['label_column']],
        test_size=config['train']['test_size'],
        random_state=config['train']['random_state']
    )

    # Train model
    logger.info("Training model...")
    model = XGBClassifier(**config['model']['params'])
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    # Make predictions
    logger.info("Making predictions...")
    preds = model.predict_proba(test_processed)[:, 1]

    # Save results
    logger.info("Saving results...")
    submit = pd.read_csv(config['data']['submission_path'])
    submit['채무 불이행 확률'] = preds
    submit.to_csv(config['data']['output_path'], encoding='UTF-8-sig', index=False)

if __name__ == "__main__":
    main()
