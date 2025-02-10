#!/bin/bash

# Feature 그룹 정의
feature_groups=$(cat <<EOF
주거: 주거 형태, 주거 형태 채무 불이행률
신용: 신용 점수, 로그 신용 점수, 신용 거래 연수, 신용 문제율, 신용 문제 발생 횟수, 개설된 신용계좌 수, 부채 대비 신용 점수, 총 부채 비율, 월 소득 대비 부채 비율, 신용-부채 변화율, 신용 비율, DTI, LTV
대출: 대출 목적, 대출 상환 기간, 대출 상환 기간 채무 불이행률, 현재 대출 잔액, 현재 미상환 신용액, 월 상환 부채액, 대출 이자율, 대출 비율, 잔액 대비 상환 속도, 상환 속도, 대출 목적_고위험, 대출 목적_부채 부담, 대출 목적 채무 불이행률
직장/소득: 연간 소득, 현재 직장 근속 연수, 근속 연수, 장기 근속 여부, 근속 연수 대비 소득
파산/연체: 최근 연체 있음, 최근 연체 위험 지표, 연체 발생 빈도, 마지막 연체 이후 경과 개월 수, 개인 파산 횟수, 파산 경험 여부
금융위험: 재정 스트레스 지표, 단기 부채 위험, 과거 신용 위험 지표, 부채 증가 위험, 금융 위험 지수, 자산 부족 정도, 신용 위험
자산: 자산, 체납 세금 압류 횟수, 최대 신용한도, 자산2, 자산3, 자산4
EOF
)

# 기본 실행 커맨드
BASE_COMMAND="./run.sh --wandb"

# 그룹별로 반복 실행
echo "$feature_groups" | while IFS=: read -r group features; do
    IFS=',' read -ra feature_list <<< "$features"  # Feature 리스트 생성

    for feature in "${feature_list[@]}"; do
        feature=$(echo "$feature" | xargs)  # 공백 제거
        DESC="${group} - ${feature} 제외"
        echo "실행 중: ${DESC}..."
        
        # 실제 실행 커맨드
        $BASE_COMMAND --desc "${DESC}" --exclude "${feature}"
        
        echo "완료: ${DESC}"
        echo "--------------------------------"
    done
done
