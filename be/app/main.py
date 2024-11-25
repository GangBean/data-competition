from fastapi import FastAPI
import psycopg2

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI"}

@app.get("/db")
def get_data_from_db():
    # PostgreSQL 연결 및 간단한 쿼리 예시
    conn = psycopg2.connect(
        dbname="mydatabase", user="user", password="password", host="postgres"
    )
    cur = conn.cursor()
    cur.execute("SELECT 'Hello from PostgreSQL' AS message;")
    result = cur.fetchone()
    cur.close()
    conn.close()
    return {"message": result[0]}
