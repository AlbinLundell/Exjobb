import sqlite3
import pandas as pd

conn = sqlite3.connect("learning_analytics.db")

#data = conn.execute("select * from elever")
#print(data.description)

# convert to pandas frame
df = pd.read_sql_query("select * from elever", conn)


print(df.head())