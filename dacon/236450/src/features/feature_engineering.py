import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Dict, Tuple
import logging

class FeatureEngineer:
    def __init__(self, config: Dict):
        self.config = config
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.logger = logging.getLogger(__name__)

    def process_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process all features"""
        # Get column lists
        categorical_cols = self.config['features']['categorical_columns']
        numerical_cols = list(set(train_df.columns) - set(categorical_cols) - {self.config['data']['label_column']})

        # Encode categorical features
        train_encoded, test_encoded = self._encode_categorical_features(
            train_df[categorical_cols], 
            test_df[categorical_cols]
        )

        # Combine features
        train_processed = pd.concat([
            train_df[numerical_cols],
            train_encoded,
            train_df[self.config['data']['label_column']]
        ], axis=1)

        test_processed = pd.concat([
            test_df[numerical_cols],
            test_encoded
        ], axis=1)

        return train_processed, test_processed

    def _encode_categorical_features(self, train_cat: pd.DataFrame, test_cat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features using OneHotEncoder"""
        self.encoder.fit(train_cat)
        
        train_encoded = pd.DataFrame(
            self.encoder.transform(train_cat),
            columns=self.encoder.get_feature_names_out(train_cat.columns)
        )
        
        test_encoded = pd.DataFrame(
            self.encoder.transform(test_cat),
            columns=self.encoder.get_feature_names_out(test_cat.columns)
        )
        
        return train_encoded, test_encoded
    