#!/bin/bash

# 제외할 피처 목록
features_to_remove=(
    "월 소득 대비 대출 상환액 비율"
    "최대 신용한도 대비 사용 비율"
    "대출 금액 대비 연체 위험"
    "신용 점수 변화율"
    "신용 위험 가중치"
    "부채 비율 변화율"
    "LTV 변화율"
    "신용 한도 대비 신용 사용률 변화"
    "최근 연체율"
    "신용 한도 대비 부채 부담"
    "월 저축 가능액"
    "연체 빈도 가중치"
    "연체 심각도 지수"
    "연체 후 회복력"
)

# 원본 실행 스크립트
base_command="./run.sh --wandb"

# 모든 피처를 제거하면서 실행
for feature in "${features_to_remove[@]}"; do
    echo "실행 중: ${feature} 제외"
    $base_command --desc "${feature} 제외" --exclude "$feature"
done

echo "모든 실험 실행 완료!"

