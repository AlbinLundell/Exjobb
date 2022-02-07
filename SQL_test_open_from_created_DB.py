
import sqlite3
import pandas as pd

connection = sqlite3.connect("Dailytask.db")

df = pd.read_sql_query("select * from daily_tasks;", connection)

print(df.head())


