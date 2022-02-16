"""
Correlation matrix för;
Kurs: Matte1C
Input tables: Betyg, Canvas, diagnoser, frånvaro
Antal elever: 12 stycken
Binary classification: E-F
Datum (Canvas):
"""


# ----------------------------------- Modules and connect to DB --------------------------------------------
from sqlalchemy import create_engine        # Läser mellan server och Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

#                                    user:password          Borde vara samma/namn på DB
db_connection_str = 'mysql+pymysql://root:LearningAnalytics@localhost:3306/learning_analytics'
db_connection = create_engine(db_connection_str)
pd.set_option('display.max_columns', None)



# --------------------------- Initial Queries -----------------------------------------------------

# -------------------------------- Betyg Table --------------------------------------
df_betyg = pd.read_sql_query('SELECT * FROM betyg '
                             'WHERE kurs = "MATMAT01C" ', con = db_connection)
# Clean duplicates of students
df_betyg_1 = df_betyg.drop_duplicates(subset=['slumpkod'])

# --------------------------------- Canvas Table -----------------------------------------------
#
# ----------- Read the table -------------------------
df_canvas = pd.read_sql_query('SELECT * FROM canvas '
                              'WHERE kurs LIKE "MATMAT01%%" AND datum LIKE "%%09-15" ', con = db_connection )
df_canvas_4weeks2 = df_canvas.drop_duplicates(subset = ['slumpkod'])
# print(df_canvas_4weeks)

# ------------------- Create two new columns ------------------
# Måste först byta till "to_datetime", droppar "datum" för har inför en annany ny column
df_canvas_4weeks2['datum_DateTime'] = pd.to_datetime(df_canvas_4weeks2.loc[:, ('datum')], format='%Y-%m-%d')
df_canvas_4weeks2 = df_canvas_4weeks2.drop(columns = ["datum"])

# Skapar en kolumn som heter "average_page_views", baserat på alla elever i klassen
df_canvas_4weeks2['Average_page_views'] = df_canvas_4weeks2.groupby('kurs')['page_views'].transform('mean')
# skapa kolumn baserat på klassen participations,
df_canvas_4weeks2['Average_participation'] = df_canvas_4weeks2.groupby('kurs')['participations'].transform('mean')

# Create a new column, each student has a page_view_factor
df_canvas_4weeks2['page_view_factor'] = df_canvas_4weeks2['page_views']/df_canvas_4weeks2['Average_page_views']
# Create a new column, each student has a participations_factor
df_canvas_4weeks2['participation_factor'] = df_canvas_4weeks2['participations']/df_canvas_4weeks2['Average_participation']



#  ------------------------ Franvarotable ---------------------------------------
sql_query_franvaro = 'SELECT * ' \
            'FROM franvaro ' \
            'WHERE kurs = "MATMAT01c" ' \
            'GROUP BY(slumpkod) ' \
            'ORDER BY Läsår '
df_franvaro = pd.read_sql_query(sql_query_franvaro, con = db_connection)
#print(df_franvaro)

# ----------------------------- Diagnostable ---------------------------------------
sql_query_diagnos = 'SELECT * ' \
                    'FROM diagnoser ' \
                    'GROUP BY(slumpkod) ' \
                    'ORDER BY id '
df_diagnos = pd.read_sql_query(sql_query_diagnos, con = db_connection)
# print(df_diagnos)

# --------------------------- Merge tables -------------------------------
# Merge Betyg, Canvas, frånvaro, diagnos
df_merge = df_betyg_1.merge(
    df_canvas_4weeks2.merge(
        df_franvaro.merge(
            df_diagnos, how ='inner', on = 'slumpkod'),
        how ='inner', on = 'slumpkod')
    , how ='inner', on = 'slumpkod')

# Droppa onödiga variabler
df_merge_1 = df_merge.drop(columns = ['id_x', 'slumpkod', 'kurs', 'id_y', 'kurs_x', 'start', 'slut', 'status', 'max_page_views',
                                      'max_participations', 'tardiness_breakdown_floating', 'tardiness_breakdown_total',
                                      'kurs_y', 'Lektionstid', 'Läsår', 'Engelska Vocabulary', 'Engelska Grammar', 'Engelska Reading',
                                      'Svenska diktamen Stanine', 'Average_page_views', 'Average_participation'])


# Måste ersätta betyg med siffror
# Här måste koden se annorlunda ut om vi ska göra om alla ["A", "B", "C", "D", "E"] till "P"
values = ["A", "B", "C", "D", "-"]

# Droppar alla element i values lista
df_dropped = df_merge_1[df_merge_1.betyg.isin(values) == False]
# Byter ut F mot 0 och E mot 1
df_dropped['betyg'] = df_dropped['betyg'].str.replace('F', '0')
df_dropped['betyg'] = df_dropped['betyg'].str.replace('E', '1')
df_dropped["betyg"] = df_dropped['betyg'].astype(int)

# Drop even more
df_more_dropped = df_dropped.drop(columns = ['tardiness_breakdown_late', 'tardiness_breakdown_missing', 'tardiness_breakdown_on_time'])

print(df_more_dropped)
# Har bara tolv elever kvar som har E eller F

# ------------------------- Matrix -------------------------
corrMatrix = df_more_dropped.corr()
# Använder Pearson metoden
# print(corrMatrix)
sn.heatmap(corrMatrix, annot=True)
plt.show()


