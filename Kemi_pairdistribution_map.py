"""
Example code for printing a pairplot matrix
Here with Kemi variables!

"""

from sqlalchemy import create_engine        # Läser mellan server och Python
import pandas as pd
#                                    user:password          Borde vara samma/namn på DB
db_connection_str = 'mysql+pymysql://root:LearningAnalytics@localhost:3306/learning_analytics'
db_connection = create_engine(db_connection_str)
pd.set_option('display.max_columns', None)

# -------------------------------------- SQL queries -----------------------------------------------------

# ------------ Läs in betyg data ------------------------
# Betyg the student has in KEMI
# DOnt have to remove duplicates, they disepear later
df_betyg_kemi = pd.read_sql('SELECT * FROM betyg WHERE kurs = "KEMKEM01" ', con = db_connection)

# -------------- Läs in frånvaro ----------------------

df_franvaro = pd.read_sql('SELECT * FROM franvaro WHERE kurs = "KEMKEM01" ', con = db_connection)


# ----------------------- Läs in canvas data ---------------------------
# Väljer alla kurser som börjar med "KEMKEM"
# Väljer alla datum som slutar med "09-15" (Kollar en månad in)
# Väljer datum på hösten eftersom det generar mer data (dock kanske vi inte har slutbetyg på 21or?)
# Skapa en average per klass page views
# Skapa ny kolumn som är floattal = pageviews / average pageview

# ------------------------ Canvas ---------------------------------------
# --------------- Page views -------- Participations ----------- and tardiness on time ----------------------
df_canvas_kemi = pd.read_sql('SELECT slumpkod, kurs, page_views, participations, tardiness_breakdown_on_time, datum FROM canvas '
                             'WHERE kurs LIKE "KEMKEM%%" AND datum LIKE "%%12-15" ', con = db_connection)
# Droppa eventeulla dubletter
df_canvas_kemi_2 = df_canvas_kemi.drop_duplicates(subset = ['slumpkod'])

# Måste först byta till "to_datetime", droppar "datum" för har inför en annany ny column
df_canvas_kemi_2['datum_DateTime'] = pd.to_datetime(df_canvas_kemi_2['datum'], format='%Y-%m-%d')
df_canvas_kemi_3 = df_canvas_kemi_2.drop(columns = ["datum"])

# Skapar en kolumn som heter "average_page_views", baserat på alla elever i klassen
df_canvas_kemi_3['Average_page_views'] = df_canvas_kemi_3.groupby('kurs')['page_views'].transform('mean')
# skapa kolumn baserat på klassen participations,
df_canvas_kemi_3['Average_participation'] = df_canvas_kemi_3.groupby('kurs')['participations'].transform('mean')

# Create a new column, each student has a page_view_factor
df_canvas_kemi_3['page_view_factor'] = df_canvas_kemi_3['page_views']/df_canvas_kemi_3['Average_page_views']
# Create a new column, each student has a participations_factor
df_canvas_kemi_3['participation_factor'] = df_canvas_kemi_3['participations']/df_canvas_kemi_3['Average_participation']

# Drop unecesary columns
df_canvas_kemi_4 = df_canvas_kemi_3.drop(columns = ['page_views', 'kurs', 'datum_DateTime', 'Average_page_views',
                                                    'Average_participation', 'participations'])
# print(df_canvas_kemi_4)

# ------------------ ---- Merge ------- -------------------
# merge betyg with tardiness_on_time, page_view_factor and participation_factor
df_merge = df_betyg_kemi.merge(df_canvas_kemi_4, how='inner', on='slumpkod')
# And dropp unecessary columns
df_merge_dropped = df_merge.drop(columns = ['id', 'kurs'])

df_merge_2 = df_merge_dropped.merge(df_franvaro, how = 'inner', on = 'slumpkod')
#print(df_merge_2)

# ------------------------------ DT prepp -----------------------

values = ["-", "A", "B", "C", "D"]
# Drop rows that contain any value of betyg that is "-"
df_DT_clean = df_merge_2[df_merge_2.betyg.isin(values) == False]
# Drop NaN values
df_DT_clean_2 = df_DT_clean.dropna()
df_DT_clean_3 = df_DT_clean_2.drop(columns = ['slumpkod', 'Lektionstid', 'id', 'Läsår','kurs'])

print(df_DT_clean_3)

# --------------------- PLot -----------------------

import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df_DT_clean_3, hue = "betyg", diag_kind = "hist", markers = ["o", "s"])
plt.show()


