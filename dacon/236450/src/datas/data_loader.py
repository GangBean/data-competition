import pandas as pd
from typing import Tuple, Dict
import logging

class DataLoader:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data"""
        try:
            train_df = pd.read_csv(self.config['data']['train_path']).drop(columns=['UID'])
            test_df = pd.read_csv(self.config['data']['test_path']).drop(columns=['UID'])
            
            self._validate_data(train_df, test_df)
            
            return train_df, test_df
        
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Validate data integrity"""
        # Check for missing values
        train_missing = train_df.isnull().sum()
        test_missing = test_df.isnull().sum()
        
        if train_missing.sum() > 0 or test_missing.sum() > 0:
            self.logger.warning(f"Missing values found in data")
            self.logger.warning(f"Train missing:\n{train_missing[train_missing > 0]}")
            self.logger.warning(f"Test missing:\n{test_missing[test_missing > 0]}")
            raise ValueError

        # Check data types
        if not all(train_df[col].dtype == test_df[col].dtype 
                  for col in test_df.columns if col in train_df.columns):
            self.logger.warning("Data type mismatch between train and test data")
            raise ValueError