import streamlit as st
import requests

st.title('Streamlit & FastAPI Example')

# FastAPI에서 데이터 가져오기
response = requests.get("http://fastapi:8000/db")
data = response.json()

st.write("Data from FastAPI and PostgreSQL:")
st.write(data)
