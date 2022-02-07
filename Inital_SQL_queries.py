# ------------------------------------------------------------------------------------------
# Importera moduler
from sqlalchemy import create_engine        # Läser mellan server och Python
import pandas as pd
#                                    user:password          Borde vara samma/namn på DB
db_connection_str = 'mysql+pymysql://root:LearningAnalytics@localhost:3306/learning_analytics'
db_connection = create_engine(db_connection_str)
pd.set_option('display.max_columns', None)      # Options
# pd.set_option('display.max_rows', None)

# ------------------------------------------------------------------------------------------

# 701 elever
df_elever = pd.read_sql('SELECT * FROM elever', con=db_connection)                     # Elev table
print(df_elever)

#df_diagnoser = pd.read_sql('SELECT * FROM diagnoser', con = db_connection)             # Diagnos table
#print(df_diagnoser.head())

# df_betyg = pd.read_sql('select * from betyg', con = db_connection)                     # Betyg Table
# print(df_betyg)

# df_unika_betyg = pd.read_sql('select distinct kurs from betyg', con = db_connection)    # Ger samtliga unika kurskoder
# print(df_unika_betyg.to_string())

# df_betyg_kemi = pd.read_sql('select * from betyg where kurs = "KEMKEM01" ', con = db_connection)    # Alla elevers betyg i Kemi
# print(df_betyg_kemi)

# 649 betyg i matte1C.
df_betyg_matte1 = pd.read_sql('select * from betyg where kurs = "MATMAT01c" order by slumpkod ', con = db_connection)    # Alla elevers betyg i matte 1
print(df_betyg_matte1)
# Testar så att elev inte ligger inne fler än gång, dvs det finns bara en slumpkod!
# Fortfarande 649 betyg i matte1C.
#df_betyg_matte1_distinct = pd.read_sql('select distinct slumpkod from betyg where kurs = "MATMAT01c" ', con = db_connection)
#print(df_betyg_matte1_distinct)
# Ta bort dem som har betyg "-" ???

# df_franvaro_kemi = pd.read_sql('select kurs, lektionstid, frånvarotid, id from franvaro where kurs = "MATMAT01c" ', con = db_connection) # Alla elevers frånvaro i Kemi
# print(df_franvaro_kemi)

df_franvaro_matte1 = pd.read_sql('select * from franvaro where kurs = "MATMAT01c" order by slumpkod ', con = db_connection) # Alla elevers frånvaro i Matte 1
print(df_franvaro_matte1)
# Resultat: 858 elever

# Ibland ligger två slumpkoder inne per elev!
# Använder distinct för att få fram att, resultat = 793 elever unika elever
df_franvaro_matte1_wo_duplicate = pd.read_sql('SELECT DISTINCT slumpkod from franvaro where kurs = "MATMAT01c" ', con = db_connection) # Alla unika elevers slumpid som finns i frånvaro
print(df_franvaro_matte1_wo_duplicate)

# Sortera table efter första slumpkoden som registreras på eleven
# Välj alla kolumner, från frånvaro, men enbatr Matte 1c. Group by och order by ger att vi sorterar
# dem i siffer-bokstavsordning enligt slumpkod samt väljer bara ut ett distinct värde.
sql_query = 'SELECT * ' \
            'FROM franvaro ' \
            'WHERE kurs = "MATMAT01c" ' \
            'GROUP BY(slumpkod) ' \
            'ORDER BY Läsår '
df_franvaro_matte1_sorterad = pd.read_sql(sql_query, con = db_connection)
print("--------------------- Matte1, en slumpkod med frånvaro och närvaro  -----------------------")
print(df_franvaro_matte1_sorterad)

# Resultat med utan
# Försök till att joina två tables. Men något blir fel för kemi!!!
'''
# SELECT betyg.slumpkod, franvaro.slumpkod,
sql_string = 'SELECT betyg.kurs, betyg.betyg, franvaro.lektionstid, franvaro.frånvarotid  ' \
             'FROM franvaro ' \
             'INNER JOIN betyg ' \
             'ON franvaro.slumpkod = betyg.slumpkod '\
             'WHERE betyg.kurs = "KEMKEM01" '

df_franvaro_betyg = pd.read_sql(sql_string, con = db_connection)
print(df_franvaro_betyg)
# df_canvas = pd.read_sql('select * from canvas', con = db_connection)                  # Canvas table
# print(df_canvas.head())
'''

# Försök till att joina två tables. Men något blir fel för matte1!!!
'''
sql_string = 'SELECT betyg.kurs, betyg.betyg, franvaro.lektionstid, franvaro.frånvarotid  ' \
             'FROM betyg ' \
             'LEFT OUTER JOIN franvaro ' \
             'ON betyg.slumpkod = franvaro.slumpkod '\
             'WHERE betyg.kurs = "MATMAT01c" '

df_franvaro_betyg = pd.read_sql(sql_string, con = db_connection)
print(df_franvaro_betyg)
'''
'''
sql_string = 'SELECT elever.id, elever.slumpkod, betyg.slumpkod, betyg.betyg, franvaro.frånvarotid ' \
             'FROM elever ' \
             'INNER JOIN betyg ' \
             'ON elever.slumpkod = betyg.slumpkod ' \
             'INNER JOIN franvaro ' \
             'ON elever.slumpkod = franvaro.slumpkod ' \
             'WHERE betyg.kurs = "MATMAT01c" '
df_kolla_elever = pd.read_sql(sql_string, con = db_connection)
print(df_kolla_elever)
'''

# Använder pandas merge på slumpkod!
# tidigare df_franvaro_matte1_sorterad är städad.
# Vi får ut 646 elever, men det var 649 elever som hade ett betyg i matte 1. Tre elever finns
# således ej i frånvaro table
print('--------------------------------- Merged ------------------------')
df_combined = df_betyg_matte1.merge(df_franvaro_matte1_sorterad, how = 'inner', on = 'slumpkod')
df_combined_2 = df_combined.drop(columns=['id_x', 'id_y', 'kurs_y', 'Läsår'])
print(df_combined_2)
# df1.merge(df2, how='inner', on='a')

