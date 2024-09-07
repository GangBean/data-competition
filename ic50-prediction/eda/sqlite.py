import pandas as pd
import sqlite3

if __name__ == '__main__':
    DATABASE_URL = "../data/chembl_34/chembl_34_sqlite/chembl_34.db"
    with sqlite3.connect(DATABASE_URL) as db:
        while True:
            print(">> ")
            sql = input()
            if sql == 'd':
                break
            cursor = db.execute(sql)
            df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
            print(df)
            cursor.close()
