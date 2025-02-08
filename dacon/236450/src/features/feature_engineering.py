import numpy as np
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
        # Create new features
        train_df['자산'] = train_df['연간 소득'] + train_df['현재 대출 잔액']
        test_df['자산'] = test_df['연간 소득'] + test_df['현재 대출 잔액']

        train_df['대출 이자율'] = train_df['월 상환 부채액'] / train_df['현재 대출 잔액'] * 100
        test_df['대출 이자율'] = test_df['월 상환 부채액'] / test_df['현재 대출 잔액'] * 100

        train_df['대출 비율'] = train_df['월 상환 부채액'] / train_df['자산'] * 100
        test_df['대출 비율'] = test_df['월 상환 부채액'] / test_df['자산'] * 100
        
        train_df['신용 비율'] = train_df['현재 미상환 신용액'] / train_df['자산'] * 100
        test_df['신용 비율'] = test_df['현재 미상환 신용액'] / test_df['자산'] * 100

        for col in ['주거 형태', '대출 목적', '대출 상환 기간']:
            mean_default_rate = train_df.groupby(col)['채무 불이행 여부'].mean()
            train_df[f"{col} 채무 불이행률"] = train_df[col].map(mean_default_rate)
            test_df[f"{col} 채무 불이행률"] = test_df[col].map(mean_default_rate)

        high_risk_categories = train_df.groupby('대출 목적')['채무 불이행 여부'].mean().sort_values(ascending=False).head(3).index
        train_df['대출 목적_고위험'] = train_df['대출 목적'].apply(lambda x: 1 if x in high_risk_categories else 0)
        test_df['대출 목적_고위험'] = test_df['대출 목적'].apply(lambda x: 1 if x in high_risk_categories else 0)

        # 기존 매핑을 유지하면서 숫자로 변환
        employment_mapping = {
            '1년 미만': 0, '1년': 1, '2년': 2, '3년': 3, '4년': 4,
            '5년': 5, '6년': 6, '7년': 7, '8년': 8, '9년': 9, '10년 이상': 10
        }
        train_df['근속 연수'] = train_df['현재 직장 근속 연수'].map(employment_mapping)
        test_df['근속 연수'] = test_df['현재 직장 근속 연수'].map(employment_mapping)

        train_df['장기 근속 여부'] = (train_df['근속 연수'] >= 5).astype(int)
        test_df['장기 근속 여부'] = (test_df['근속 연수'] >= 5).astype(int)

        train_df['최근 연체 있음'] = (train_df['마지막 연체 이후 경과 개월 수'] < 6).astype(int)
        test_df['최근 연체 있음'] = (test_df['마지막 연체 이후 경과 개월 수'] < 6).astype(int)

        train_df['DTI'] = train_df['월 상환 부채액'] / train_df['연간 소득']
        test_df['DTI'] = test_df['월 상환 부채액'] / test_df['연간 소득']

        train_df['LTV'] = train_df['현재 대출 잔액'] / train_df['자산']
        test_df['LTV'] = test_df['현재 대출 잔액'] / test_df['자산']

        train_df['로그 신용 점수'] = np.log1p(train_df['신용 점수'])
        test_df['로그 신용 점수'] = np.log1p(test_df['신용 점수'])

        # train_df['신용 점수 등급'] = pd.cut(train_df['신용 점수'], bins=[0, 500, 600, 700, 800, 900], labels=[1, 2, 3, 4, 5])
        # test_df['신용 점수 등급'] = pd.cut(test_df['신용 점수'], bins=[0, 500, 600, 700, 800, 900], labels=[1, 2, 3, 4, 5])

        train_df['신용 문제율'] = train_df['신용 문제 발생 횟수'] / (train_df['신용 거래 연수'] + 1)
        test_df['신용 문제율'] = test_df['신용 문제 발생 횟수'] / (test_df['신용 거래 연수'] + 1)

        train_df['파산 경험 여부'] = (train_df['개인 파산 횟수'] > 0).astype(int)
        test_df['파산 경험 여부'] = (test_df['개인 파산 횟수'] > 0).astype(int)

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
    