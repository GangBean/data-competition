# 1. 공식 Python 이미지를 기반으로 설정
FROM python:3.11-slim

RUN pip install poetry
# # 2. Poetry 설치
# RUN curl -sSL https://install.python-poetry.org | python3 -

# # 3. Poetry 경로를 환경 변수에 추가
# ENV PATH="/root/.local/bin:${PATH}"

# 4. 작업 디렉토리 설정
WORKDIR /app

# 5. Poetry의 프로젝트 의존성 파일 복사
COPY ./be/pyproject.toml ./be/poetry.lock* /app/

# 6. 의존성 설치
RUN poetry install --no-root --no-dev

# 7. 프로젝트 소스 파일 복사
# COPY ./be /app/

# 8. FastAPI 앱 실행 명령 설정
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
