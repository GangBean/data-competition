import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from typing import Dict, Tuple
import logging

class FeatureEngineer:
    def __init__(self, config: Dict):
        self.config = config
        self.nominal_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Get ordinal feature categories from config
        ordinal_config = self.config['features'].get('ordinal_columns', {})
        ordinal_categories = []
        self.ordinal_feature_names = []
        
        # Extract feature names and their categories
        for item in ordinal_config:
            if isinstance(item, dict):
                # Get the feature name (first and only key)
                feature_name = list(item.keys())[0]
                self.ordinal_feature_names.append(feature_name)
                # Get the categories (value is a list)
                ordinal_categories.append(item[feature_name])
            else:
                # If no categories specified, use default feature name
                self.ordinal_feature_names.append(item)
                ordinal_categories.append(None)
        
        self.ordinal_encoder = OrdinalEncoder(
            categories=ordinal_categories if any(cat is not None for cat in ordinal_categories) else None,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        self.logger = logging.getLogger(__name__)
        
        # Default features remain the same
        self.default_nominal = [
            '주거 형태',          
            '대출 목적',            
            '대출 상환 기간'                
        ]
        
        self.default_ordinal = [
            '현재 직장 근속 연수'           
        ]

    def process_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process all features"""
        # Get column lists with defaults
        nominal_cols = sorted(self.config['features'].get('nominal_columns', self.default_nominal))
        ordinal_cols = sorted(self.ordinal_feature_names if self.ordinal_feature_names else self.default_ordinal)
        numerical_cols = sorted(list(set(train_df.columns) - set(nominal_cols) - set(ordinal_cols) - {self.config['data']['label_column']}))

        # Encode nominal features
        train_nominal, test_nominal = self._encode_nominal_features(
            train_df[nominal_cols], 
            test_df[nominal_cols]
        ) if nominal_cols else (pd.DataFrame(), pd.DataFrame())

        # Encode ordinal features
        train_ordinal, test_ordinal = self._encode_ordinal_features(
            train_df[ordinal_cols],
            test_df[ordinal_cols]
        ) if ordinal_cols else (pd.DataFrame(), pd.DataFrame())

        # Combine features
        train_processed = pd.concat([
            train_df[numerical_cols],
            train_nominal,
            train_ordinal,
            train_df[self.config['data']['label_column']]
        ], axis=1)

        test_processed = pd.concat([
            test_df[numerical_cols],
            test_nominal,
            test_ordinal
        ], axis=1)

        return train_processed, test_processed

    def _encode_nominal_features(self, train_cat: pd.DataFrame, test_cat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode nominal features using OneHotEncoder"""
        self.nominal_encoder.fit(train_cat)
        
        train_encoded = pd.DataFrame(
            self.nominal_encoder.transform(train_cat),
            columns=self.nominal_encoder.get_feature_names_out(train_cat.columns)
        )
        
        test_encoded = pd.DataFrame(
            self.nominal_encoder.transform(test_cat),
            columns=self.nominal_encoder.get_feature_names_out(test_cat.columns)
        )
        
        return train_encoded, test_encoded

    def _encode_ordinal_features(self, train_cat: pd.DataFrame, test_cat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode ordinal features using OrdinalEncoder"""
        self.ordinal_encoder.fit(train_cat)
        
        train_encoded = pd.DataFrame(
            self.ordinal_encoder.transform(train_cat),
            columns=train_cat.columns
        )
        
        test_encoded = pd.DataFrame(
            self.ordinal_encoder.transform(test_cat),
            columns=test_cat.columns
        )
        
        return train_encoded, test_encoded

    def get_ordinal_mapping(self) -> Dict:
        """Get the mapping of ordinal features"""
        if not hasattr(self.ordinal_encoder, 'categories_'):
            raise ValueError("Ordinal encoder has not been fitted yet.")
        
        mapping = {}
        for feature_idx, feature_name in enumerate(self.ordinal_encoder.feature_names_in_):
            categories = self.ordinal_encoder.categories_[feature_idx]
            mapping[feature_name] = {
                category: idx for idx, category in enumerate(categories)
            }
        return mapping

    def print_ordinal_mapping(self):
        """Print the mapping of ordinal features in a readable format"""
        try:
            mapping = self.get_ordinal_mapping()
            print("\n=== Ordinal Feature Mappings ===")
            for feature, value_mapping in mapping.items():
                print(f"\n{feature}:")
                for original, encoded in value_mapping.items():
                    print(f"  {original} -> {encoded}")
        except ValueError as e:
            print(f"Error: {e}")
    