# Testa programmera lite SQL

import sqlite3
import pandas as pd

conn = sqlite3.connect("flightsdata.db")

data = conn.execute("select * from airlines")
print(data.description)

# convert to pandas frame
df = pd.read_sql_query("select * from airlines limit 5;", conn)


print(df.head())
print(df["country"])