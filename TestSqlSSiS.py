
import sqlite3
import pandas as pd

conn = sqlite3.connect('LA.db')

cursor = conn.cursor()

sql_file = open("learning_analytics_elever.sql")
sql_as_string = sql_file.read()
cursor.executescript(sql_as_string)

for row in cursor.execute("SELECT * FROM elever;"):
    print(row)

