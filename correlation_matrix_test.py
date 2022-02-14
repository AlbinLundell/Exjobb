from sqlalchemy import create_engine
import pymysql
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


db_connection_str = 'mysql+pymysql://root:StockholmStad@localhost:3306/learning_analytics'
db_connection = create_engine(db_connection_str)

dfcc = pd.read_sql('SHOW columns FROM nationella_prov', con=db_connection)
#df = pd.read_sql('SELECT betyg, kurs FROM betyg where kurs="KEMKEM01"', con=db_connection)
df2= pd.read_sql('SELECT `Totalt betyg NP` AS TBNP, slumpkod FROM nationella_prov where kurs = "ENGENG05"',con=db_connection)
df5 = pd.read_sql('SELECT participations_level, page_views_level, tardiness_breakdown_missing, tardiness_breakdown_on_time, tardiness_breakdown_late, slumpkod, '
                  'kurs, datum FROM canvas WHERE kurs = "ENGENG05-TE19A" AND datum = "2020-06-11" OR kurs = '
                  '"ENGENG05-TE19B" AND datum = "2020-06-11" OR kurs = "ENGENG05-TE19C" AND datum = '
                  '"2020-06-11" OR kurs = "ENGENG05-TE19D" AND datum = "2020-06-11"', con=db_connection)
df3= pd.read_sql('SELECT betyg, slumpkod FROM betyg where kurs = "ENGENG05" ',con=db_connection)
df4=pd.read_sql('SELECT frånvarotid, slumpkod FROM franvaro where kurs = "ENGENG05"', con=db_connection)
pd.set_option('display.max_columns', None)
print(df5)


df2['TBNP'] = df2['TBNP'].str.replace('F', '0')
counter=5
for i in ["A", "B", "C", "D","E"]:
    counter_str=str(counter)
    df2['TBNP'] = df2['TBNP'].str.replace(i, counter_str)
    counter=counter-1
df2.TBNP=pd.to_numeric(df2.TBNP, errors='coerce')



df3['betyg'] = df3['betyg'].str.replace('F', '0')
#df3['betyg'] = df3['betyg'].str.replace('E', '1')
counter=5
for i in ["A", "B", "C", "D","E"]:
    counter_str=str(counter)
    df3['betyg'] = df3['betyg'].str.replace(i, counter_str)
    counter=counter-1
df3.betyg=pd.to_numeric(df3.betyg, errors='coerce')


df_combined = df3.merge(df4, how = 'inner', on = 'slumpkod')

print(df_combined)
#df_combined=df_combined.merge(df4, how = 'inner', on ='slumpkod')
#print(df_combined)
df_combined=df_combined.merge(df5, how = 'inner', on ='slumpkod')
print(df_combined)


corrMatrix = df_combined.corr()
print (corrMatrix)
sn.heatmap(corrMatrix, annot=True)
plt.show()