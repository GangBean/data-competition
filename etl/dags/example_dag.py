from airflow import DAG
from airflow.operators.python import PythonOperator
import psycopg2
from datetime import datetime

def fetch_data_from_postgres():
    conn = psycopg2.connect(
        dbname="mydatabase", user="user", password="password", host="postgres"
    )
    cur = conn.cursor()
    cur.execute("SELECT 'Hello from Airflow and PostgreSQL' AS message;")
    result = cur.fetchone()
    cur.close()
    conn.close()
    print(result)

with DAG(
    'example_dag',
    default_args={'owner': 'airflow'},
    description='An example DAG',
    schedule_interval=None,
    start_date=datetime(2024, 11, 24),
    catchup=False,
) as dag:
    task = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_data_from_postgres
    )
