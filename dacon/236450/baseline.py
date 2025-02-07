import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings(action='ignore')

# 학습/평가 데이터 로드
train_df = pd.read_csv('./train.csv').drop(columns=['UID'])
test_df = pd.read_csv('./test.csv').drop(columns=['UID'])

label_col = "채무 불이행 여부"
whole_feature_col = set(train_df.drop(columns=label_col).columns)

# basic feature list
basic_categorical_col = set(["주거 형태", "현재 직장 근속 연수", "대출 목적", "대출 상환 기간"])
basic_numerical_col = whole_feature_col - basic_categorical_col

# feature engineering
create_feature_tasks = set()

# TO-DO: feature engineering task 추가 default input 정의 필수
created_feature_col = set([task(train_df=train_df, test_df=test_df) for task in create_feature_tasks])
created_categorical_col = set()
created_numerical_col = created_feature_col - created_categorical_col

# selected feature list
selected_categorical_col = sorted(list(set(basic_categorical_col | created_categorical_col)))
selected_numerical_col = sorted(list(set(basic_numerical_col | created_numerical_col)))

# OneHotEncoder 초기화
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# 훈련 데이터에 대해 인코더 학습
encoder.fit(train_df[selected_categorical_col])

# 훈련 데이터와 테스트 데이터 변환
train_encoded = encoder.transform(train_df[selected_categorical_col])
test_encoded = encoder.transform(test_df[selected_categorical_col])

# One-hot encoding 결과를 데이터프레임으로 변환
train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(selected_categorical_col))
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(selected_categorical_col))

train_numerical_df = train_df[selected_numerical_col].reset_index(drop=True)
test_numerical_df = test_df[selected_numerical_col].reset_index(drop=True)

train_label_df = train_df[label_col].reset_index(drop=True)

# 인코딩된 결과를 원래 데이터에 적용
train_df = pd.concat([train_numerical_df, train_encoded_df, train_label_df], axis=1)
test_df = pd.concat([test_numerical_df, test_encoded_df], axis=1)


X_train, X_val, y_train, y_val = train_test_split(
    train_df.drop(columns=['채무 불이행 여부']), 
    train_df['채무 불이행 여부'], 
    test_size=0.2, 
    random_state=42
)

# XGBoost 모델 학습
model: XGBClassifier = XGBClassifier(
    n_estimators=100,  # 트리 개수
    max_depth=5,       # 최대 깊이
    learning_rate=0.15, # 학습률
    random_state=42,
    use_label_encoder=False,  # 경고 방지
    eval_metric="auc",        # 평가 지표 설정
)

# 학습 및 Validation 성능 모니터링
eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=True,  # 학습 로그 출력
)

# 채무 불이행 '확률'을 예측합니다.
preds = model.predict_proba(test_df)[:,1]
submit = pd.read_csv('./sample_submission.csv')

# 결과 저장
submit['채무 불이행 확률'] = preds
submit.to_csv('./submission.csv', encoding='UTF-8-sig', index=False)