import numpy as np
import pandas as pd
import logging

def create_feature(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
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

    train_df['신용 문제율'] = train_df['신용 문제 발생 횟수'] / (train_df['신용 거래 연수'] + 1)
    test_df['신용 문제율'] = test_df['신용 문제 발생 횟수'] / (test_df['신용 거래 연수'] + 1)

    train_df['파산 경험 여부'] = (train_df['개인 파산 횟수'] > 0).astype(int)
    test_df['파산 경험 여부'] = (test_df['개인 파산 횟수'] > 0).astype(int)

    train_df['부채 대비 신용 점수'] = train_df['월 상환 부채액'] / (train_df['신용 점수'] + 1)
    test_df['부채 대비 신용 점수'] = test_df['월 상환 부채액'] / (test_df['신용 점수'] + 1)

    train_df['총 부채 비율'] = (train_df['현재 대출 잔액'] + train_df['현재 미상환 신용액']) / (train_df['자산'] + 1)
    test_df['총 부채 비율'] = (test_df['현재 대출 잔액'] + test_df['현재 미상환 신용액']) / (test_df['자산'] + 1)

    train_df['월 소득 대비 부채 비율'] = train_df['월 상환 부채액'] / (train_df['연간 소득'] / 12 + 1)
    test_df['월 소득 대비 부채 비율'] = test_df['월 상환 부채액'] / (test_df['연간 소득'] / 12 + 1)

    train_df['신용-부채 변화율'] = train_df['신용 비율'] - train_df['대출 비율']
    test_df['신용-부채 변화율'] = test_df['신용 비율'] - test_df['대출 비율']

    train_df['근속 연수 대비 소득'] = train_df['연간 소득'] / (train_df['근속 연수'] + 1)
    test_df['근속 연수 대비 소득'] = test_df['연간 소득'] / (test_df['근속 연수'] + 1)

    train_df['연체 발생 빈도'] = train_df['신용 문제 발생 횟수'] / (train_df['신용 거래 연수'] + 1)
    test_df['연체 발생 빈도'] = test_df['신용 문제 발생 횟수'] / (test_df['신용 거래 연수'] + 1)

    train_df['최근 연체 위험 지표'] = train_df['마지막 연체 이후 경과 개월 수'] / (train_df['신용 문제 발생 횟수'] + 1)
    test_df['최근 연체 위험 지표'] = test_df['마지막 연체 이후 경과 개월 수'] / (test_df['신용 문제 발생 횟수'] + 1)

    train_df['금융 위험 지수'] = (
        train_df['대출 비율'] +
        train_df['신용 비율'] +
        train_df['DTI'] +
        train_df['총 부채 비율'] +
        train_df['연체 발생 빈도']
    )
    test_df['금융 위험 지수'] = (
        test_df['대출 비율'] +
        test_df['신용 비율'] +
        test_df['DTI'] +
        test_df['총 부채 비율'] +
        test_df['연체 발생 빈도']
    )

    train_df['상환 속도'] = train_df['월 상환 부채액'] / (train_df['자산'] + 1)
    test_df['상환 속도'] = test_df['월 상환 부채액'] / (test_df['자산'] + 1)

    train_df['자산 부족 정도'] = train_df['자산'] / train_df['연간 소득']
    test_df['자산 부족 정도'] = test_df['자산'] / test_df['연간 소득']

    train_df['재정 스트레스 지표'] = (train_df['DTI'] * 0.5 + train_df['LTV'] * 0.3) / (train_df['신용 점수'] + 1)
    test_df['재정 스트레스 지표'] = (test_df['DTI'] * 0.5 + test_df['LTV'] * 0.3) / (test_df['신용 점수'] + 1)

    train_df['단기 부채 위험'] = train_df['최근 연체 있음'] * train_df['월 상환 부채액']
    test_df['단기 부채 위험'] = test_df['최근 연체 있음'] * test_df['월 상환 부채액']

    train_df['주거 형태_소득'] = train_df['주거 형태 채무 불이행률'] * train_df['연간 소득']
    test_df['주거 형태_소득'] = test_df['주거 형태 채무 불이행률'] * test_df['연간 소득']

    train_df['과거 신용 위험 지표'] = train_df['신용 문제율'] * (1 + train_df['파산 경험 여부'])
    test_df['과거 신용 위험 지표'] = test_df['신용 문제율'] * (1 + test_df['파산 경험 여부'])

    # 대출 상환 기간을 숫자로 매핑
    loan_term_mapping = {
        '단기 상환': 3,
        '장기 상환': 10
    }
    train_df['잔액 대비 상환 속도'] = train_df['현재 대출 잔액'] / (train_df['대출 상환 기간'].map(loan_term_mapping) + 1)
    test_df['잔액 대비 상환 속도'] = test_df['현재 대출 잔액'] / (test_df['대출 상환 기간'].map(loan_term_mapping) + 1)

    train_df['부채 증가 위험'] = train_df['대출 비율'] * (1 / (train_df['연간 소득'] + 1))
    test_df['부채 증가 위험'] = test_df['대출 비율'] * (1 / (test_df['연간 소득'] + 1))

    train_df['대출 목적_부채 부담'] = train_df['대출 목적 채무 불이행률'] * train_df['월 상환 부채액']
    test_df['대출 목적_부채 부담'] = test_df['대출 목적 채무 불이행률'] * test_df['월 상환 부채액']

    train_df['신용 위험'] = (1 / (train_df['신용 점수'] + 1)) * train_df['최근 연체 있음']
    test_df['신용 위험'] = (1 / (test_df['신용 점수'] + 1)) * test_df['최근 연체 있음']

    income_savings_rate = 0.4  # 소득에서 저축 가능한 비율 (40%로 가정)
    train_df['자산2'] = (train_df['연간 소득'] * income_savings_rate) + train_df['최대 신용한도'] - train_df['현재 대출 잔액']
    test_df['자산2'] = (test_df['연간 소득'] * income_savings_rate) + test_df['최대 신용한도'] - test_df['현재 대출 잔액']

    train_df['자산3'] = (train_df['연간 소득'] * income_savings_rate) - train_df['현재 대출 잔액'] * (1 + train_df['금융 위험 지수'])
    test_df['자산3'] = (test_df['연간 소득'] * income_savings_rate) - test_df['현재 대출 잔액'] * (1 + test_df['금융 위험 지수'])

    train_df['자산4'] = train_df['연간 소득'] + (train_df['부채 대비 신용 점수'] * train_df['최대 신용한도'])
    test_df['자산4'] = test_df['연간 소득'] + (test_df['부채 대비 신용 점수'] * test_df['최대 신용한도'])

    # feature #5
    # 월 소득 대비 대출 상환액 비율
    train_df['월 소득 대비 대출 상환액 비율'] = train_df['월 상환 부채액'] / (train_df['연간 소득'] / 12)
    test_df['월 소득 대비 대출 상환액 비율'] = test_df['월 상환 부채액'] / (test_df['연간 소득'] / 12)
    
    # 최대 신용한도 대비 사용 비율
    train_df['최대 신용한도 대비 사용 비율'] = train_df['현재 미상환 신용액'] / (train_df['최대 신용한도'] + 1)
    test_df['최대 신용한도 대비 사용 비율'] = test_df['현재 미상환 신용액'] / (test_df['최대 신용한도'] + 1)
    
    # 대출 금액 대비 연체 위험
    train_df['대출 금액 대비 연체 위험'] = train_df['현재 대출 잔액'] / (train_df['마지막 연체 이후 경과 개월 수'] + 1)
    test_df['대출 금액 대비 연체 위험'] = test_df['현재 대출 잔액'] / (test_df['마지막 연체 이후 경과 개월 수'] + 1)
    
    # 신용 점수 변화율
    train_df['신용 점수 변화율'] = train_df['로그 신용 점수'] / (train_df['신용 거래 연수'] + 1)
    test_df['신용 점수 변화율'] = test_df['로그 신용 점수'] / (test_df['신용 거래 연수'] + 1)
    
    # 신용 위험 가중치
    train_df['신용 위험 가중치'] = (1 / (train_df['신용 점수'] + 1)) * train_df['신용 문제 발생 횟수']
    test_df['신용 위험 가중치'] = (1 / (test_df['신용 점수'] + 1)) * test_df['신용 문제 발생 횟수']
    
    # 부채 비율 변화율
    train_df['부채 비율 변화율'] = train_df['총 부채 비율'] - train_df['대출 비율']
    test_df['부채 비율 변화율'] = test_df['총 부채 비율'] - test_df['대출 비율']
    
    # LTV 변화율
    train_df['LTV 변화율'] = train_df['LTV'] / (train_df['신용 거래 연수'] + 1)
    test_df['LTV 변화율'] = test_df['LTV'] / (test_df['신용 거래 연수'] + 1)
    
    # 신용 한도 대비 신용 사용률 변화
    train_df['신용 한도 대비 신용 사용률 변화'] = train_df['최대 신용한도'] / (train_df['신용 거래 연수'] + 1)
    test_df['신용 한도 대비 신용 사용률 변화'] = test_df['최대 신용한도'] / (test_df['신용 거래 연수'] + 1)
    
    # 최근 연체율
    train_df['최근 연체율'] = train_df['신용 문제 발생 횟수'] / (train_df['마지막 연체 이후 경과 개월 수'] + 1)
    test_df['최근 연체율'] = test_df['신용 문제 발생 횟수'] / (test_df['마지막 연체 이후 경과 개월 수'] + 1)
    
    # 신용 한도 대비 부채 부담
    train_df['신용 한도 대비 부채 부담'] = (train_df['현재 미상환 신용액'] + train_df['현재 대출 잔액']) / (train_df['최대 신용한도'] + 1)
    test_df['신용 한도 대비 부채 부담'] = (test_df['현재 미상환 신용액'] + test_df['현재 대출 잔액']) / (test_df['최대 신용한도'] + 1)
    
    # 월 저축 가능액
    train_df['월 저축 가능액'] = (train_df['연간 소득'] / 12) - train_df['월 상환 부채액']
    test_df['월 저축 가능액'] = (test_df['연간 소득'] / 12) - test_df['월 상환 부채액']
    
    # 연체 빈도 가중치
    train_df['연체 빈도 가중치'] = train_df['신용 문제 발생 횟수'] / (train_df['마지막 연체 이후 경과 개월 수'] + 1)
    test_df['연체 빈도 가중치'] = test_df['신용 문제 발생 횟수'] / (test_df['마지막 연체 이후 경과 개월 수'] + 1)
    
    # 연체 심각도 지수
    train_df['연체 심각도 지수'] = (train_df['최근 연체 있음'] * 2 + train_df['신용 문제 발생 횟수']) / (train_df['신용 거래 연수'] + 1)
    test_df['연체 심각도 지수'] = (test_df['최근 연체 있음'] * 2 + test_df['신용 문제 발생 횟수']) / (test_df['신용 거래 연수'] + 1)
    
    # 연체 후 회복력
    train_df['연체 후 회복력'] = train_df['마지막 연체 이후 경과 개월 수'] / (train_df['신용 문제 발생 횟수'] + 1)
    test_df['연체 후 회복력'] = test_df['마지막 연체 이후 경과 개월 수'] / (test_df['신용 문제 발생 횟수'] + 1)

    # Nominal Column 값별 Numerical 변수의 통계값 추가
    nominal_cols = ['주거 형태', '대출 목적', '대출 상환 기간']
    numerical_cols = ['연간 소득', '현재 대출 잔액', '월 상환 부채액', '신용 점수']

    for cat_col in nominal_cols:
        stats = train_df.groupby(cat_col)[numerical_cols].agg(['mean', 'std', 'min', 'max'])
        stats.columns = [f'{cat_col}_{num_col}_{agg}' for num_col, agg in stats.columns]

        train_df.loc[:, stats.columns] = train_df.merge(stats, on=cat_col, how='left')[stats.columns]
        test_df.loc[:, stats.columns] = test_df.merge(stats, on=cat_col, how='left')[stats.columns]
