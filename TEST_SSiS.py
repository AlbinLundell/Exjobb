import sqlite3
import pandas as pd

conn = sqlite3.connect('LA.db')

#cursor = conn.cursor()

#sql_file = open("data_learning_analytics_20210202.sql")
#sql_as_string = sql_file.read()
#cursor.executescript(sql_as_string)

#for row in cursor.execute("SELECT * FROM elever;"):
#    print(row)

with open('data_learning_analytics_20210202.sql', 'r', encoding="utf-8") as sql_file:
    conn.executescript(sql_file.read())


conn.close()
