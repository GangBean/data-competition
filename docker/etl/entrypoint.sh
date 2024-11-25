#!/bin/bash

# Airflow 데이터베이스 초기화
airflow db init

# 관리자 사용자 생성
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com

# Airflow 웹서버 실행
airflow webserver
