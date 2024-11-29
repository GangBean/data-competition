-- public 스키마 생성 (존재하지 않는 경우)
CREATE SCHEMA IF NOT EXISTS public;

-- 프로젝트 테이블
CREATE TABLE IF NOT EXISTS public.project (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    period DATERANGE NOT NULL,
    goal TEXT,
    status TEXT CHECK (status IN ('PREPARING', 'IN-PROGRESS', 'FINISHED')) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT DEFAULT 'admin',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by TEXT DEFAULT 'admin'
);

-- 가설 테이블
CREATE TABLE IF NOT EXISTS public.hypothesis (
    id SERIAL PRIMARY KEY,
    project_id INT,
    name TEXT NOT NULL,
    background TEXT,
    content TEXT,
    goal TEXT,
    status TEXT CHECK (status IN ('PREPARING', 'IN-PROGRESS', 'FINISHED')) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT DEFAULT 'admin',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by TEXT DEFAULT 'admin'
);

-- 실험 테이블
CREATE TABLE IF NOT EXISTS public.experiment (
    id SERIAL PRIMARY KEY,
    project_id INT,
    name TEXT NOT NULL,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    train_start_time TIMESTAMP,
    train_end_time TIMESTAMP,
    input_data TEXT,
    input_features BIGINT[] NOT NULL,
    metric TEXT,
    model TEXT,
    inference_data TEXT,
    result TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT DEFAULT 'admin',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by TEXT DEFAULT 'admin'
);

-- 데이터셋 테이블
CREATE TABLE IF NOT EXISTS public.datasets (
    id SERIAL PRIMARY KEY,
    name TEXT,
    location TEXT NOT NULL,
    description TEXT,
    version TEXT NOT NULL,
    type TEXT CHECK (type IN ('train', 'valid', 'test')) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT DEFAULT 'admin',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by TEXT DEFAULT 'admin'
);

-- 피처 테이블
CREATE TABLE IF NOT EXISTS public.feature (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    location TEXT NOT NULL,
    source_data TEXT,
    type TEXT CHECK (type IN ('DISCRETE', 'CONTINUOUS', 'ORDINAL', 'NOMINAL')) NOT NULL,
    extract_method TEXT,
    statistics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT DEFAULT 'admin',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by TEXT DEFAULT 'admin'
);

-- 모델 테이블
CREATE TABLE IF NOT EXISTS public.model (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    feature_set BIGINT[] NOT NULL,
    information JSONB,
    status TEXT CHECK (status IN ('INIT', 'TRAINING', 'COMPLETED', 'ERROR')) NOT NULL,
    train_start_time TIMESTAMP,
    train_end_time TIMESTAMP,
    hyperparameters JSONB,
    training_time INTERVAL,
    error_time TIMESTAMP,
    error_message TEXT,
    best_state JSONB,
    storage_location TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT DEFAULT 'admin',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by TEXT DEFAULT 'admin'
);
