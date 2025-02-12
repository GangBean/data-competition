import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder,
    MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler, Normalizer,
)
from typing import Dict, Tuple
import logging

from .features import create_feature, create_statistical_features, create_mixup_features, pca

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
        
        # Default features remain the same
        self.default_nominal = [
            '주거 형태',          
            '대출 목적',            
            '대출 상환 기간'                
        ]
        
        self.default_ordinal = [
            '현재 직장 근속 연수'
        ]

        self.scaler = self._scaler(self.config['train']['scaler'])
    
        self.logger = logging.getLogger(__name__)

    def _scaler(self, type: str) -> any:
        """Return scaler based on config type
        
        Args:
            type (str): Type of scaler ('minmax', 'standard', 'robust', 'maxabs', 'normalizer')
            
        Returns:
            Scaler object from sklearn.preprocessing
        """
        scaler_map = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'maxabs': MaxAbsScaler(),
            'normalizer': Normalizer()
        }
        
        return scaler_map.get(type.lower())

    def _select_feature(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        ordinal_keys = [list(d.keys())[0] if isinstance(d, dict) else d for d in self.config['features']['ordinal_columns']]
        selected_cols = sorted(list(set(self.config['features']['nominal_columns']) |
            set(ordinal_keys) |
            set(self.config['features']['numerical_discrete']) |
            set(self.config['features']['numerical_continuous'])))
        
        logging.info(f"[Before Filtering] {len(selected_cols)} / train: {len(train_df.columns)} 개")
        
        # 1. Importance based filtering
        if self.config['train']['importance']['use']:
            importance_type = self.config['train']['importance']['type']
            importance = self.config['importance']  # Feature Importance Dictionary
            selected_cols = set(selected_cols)  # 기존 선택된 컬럼들

            if importance_type == "rank":
                rank_threshold = self.config['train']['importance'].get('top-k', 20)  # 상위 N개 선택
                top_features = sorted(importance.keys(), key=importance.get, reverse=True)[:rank_threshold]
                selected_cols = selected_cols.intersection(set(top_features))

            elif importance_type == "threshold":
                importance_threshold = self.config['train']['importance'].get('threshold', 1.0)  # 임계값 설정
                filtered_features = [f for f, imp in importance.items() if imp >= importance_threshold]
                selected_cols = selected_cols.intersection(set(filtered_features))

            else:
                raise ValueError(f"[Importance Type] 사용 불가한 importance type 입니다 (rank, threshold): {importance_type}")
            
            selected_cols = sorted(list(selected_cols), key=importance.get, reverse=True)

        # 2. Group based filtering
        if self.config['train']['group']['use']:
            groups_in_use: list[str] = self.config['train']['group']['value']
            groups: dict = self.config['groups']
            features_in_groups = set()
            for group in groups_in_use:
                features_in_groups = features_in_groups | set(groups.get(group, []))

            # logging.info(f"[Group features] {features_in_groups} / {len(features_in_groups)} 개")
            # logging.info(f"[누락된 Group feature] {set(selected_cols) - features_in_groups}")
            
            selected_cols = sorted(list(set(selected_cols).intersection(features_in_groups)))

        if self.config['exclude']:
            selected_cols = sorted(list(set(selected_cols) - set(self.config['exclude'])))

        logging.info(f"[Selected Features]: {selected_cols} -> {len(selected_cols)} / {len(train_df.columns)} 개")

        train_df.drop(columns=[col for col in train_df.columns if col != self.config['data']['label_column'] and col not in selected_cols], inplace=True)
        test_df.drop(columns=[col for col in test_df.columns if col != self.config['data']['label_column'] and col not in selected_cols], inplace=True)

    def process_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process all features"""
        create_feature(train_df, test_df)
        # logging.info(f"[After create]: {train_df.columns} / {len(train_df.columns)} 개")

        self._select_feature(train_df, test_df)

        mixup_cols = create_mixup_features(train_df, test_df) # mixup 은 모두 nominal로 처리
        # logging.info(f"[After mixup create]: {train_df.columns} / {len(train_df.columns)} 개 / {mixup_cols} / {len(mixup_cols)} 개")

        create_statistical_features(train_df, test_df)
        # logging.info(f"[After statistical create]: {train_df.columns} / {len(train_df.columns)} 개")
        # Get column lists with defaults
        nominal_cols = sorted(list((set(self.config['features'].get('nominal_columns', self.default_nominal)) | set(mixup_cols)).intersection(set(train_df.columns))))
        ordinal_cols = sorted(list(set(self.ordinal_feature_names if self.ordinal_feature_names else self.default_ordinal).intersection(set(train_df.columns))))
        numerical_cols = sorted(list(set(train_df.columns) - set(nominal_cols) - set(ordinal_cols) - {self.config['data']['label_column']}))

        logging.info(f"개수: 전체: {len(train_df.columns)} / nominal: {len(nominal_cols)} / ordinal: {len(ordinal_cols)} / numerical: {len(numerical_cols)}")
        # logging.info(f"비교: {(set(nominal_cols) | set(ordinal_cols) | set(numerical_cols)) - set(train_df.columns)}")
        
        # Encode nominal features
        train_nominal, test_nominal = self._encode_nominal_features(
            train_df[nominal_cols], 
            test_df[nominal_cols]
        ) if nominal_cols else (pd.DataFrame(), pd.DataFrame())
        # logging.info(f"[after encoding nomianl]: {train_nominal}")

        # Encode ordinal features
        train_ordinal, test_ordinal = self._encode_ordinal_features(
            train_df[ordinal_cols],
            test_df[ordinal_cols]
        ) if ordinal_cols else (pd.DataFrame(), pd.DataFrame())

        # Scaling numerical features
        train_numeric, test_numeric = self._scaling_numerical_features(
            train_df[numerical_cols],
            test_df[numerical_cols]
        ) if numerical_cols else (pd.DataFrame(), pd.DataFrame())

        # Combine features
        train_processed = pd.concat([
            train_numeric,
            train_nominal,
            train_ordinal,
            train_df[self.config['data']['label_column']]
        ], axis=1)

        test_processed = pd.concat([
            test_numeric,
            test_nominal,
            test_ordinal
        ], axis=1)

        logging.info(f"[Final Features]: {train_processed.columns} / {len(train_processed.columns)} 개")
        train_processed, test_processed = pca(train_processed, test_processed, self.config["train"]["pca_num"])

        return train_processed, test_processed
    
    def _scaling_numerical_features(self, train_num: pd.DataFrame, test_num: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Dataframe copy
        train_num = train_num.copy()
        test_num = test_num.copy()

        # fill NaN with mean
        train_num.fillna(train_num.mean(), inplace=True)
        test_num.fillna(test_num.mean(), inplace=True)

        # drop all NaN cols
        train_num.dropna(axis=1, inplace=True)
        test_num = test_num[train_num.columns]
        
        train_scaled = pd.DataFrame(
            self.scaler.fit_transform(train_num),
            columns=train_num.columns
        )
        
        test_scaled = pd.DataFrame(
            self.scaler.transform(test_num),
            columns=test_num.columns
        )

        return train_scaled, test_scaled

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
    