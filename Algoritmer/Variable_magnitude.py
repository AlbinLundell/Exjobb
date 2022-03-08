""" ----------------------- Import Modules ----------------------------------------- """
from sqlalchemy import create_engine        # Läser mellan server och Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings
from sklearn.metrics import plot_confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sn

""" ------------------------- Functions ------------------------------------------ """


""" Returns a coneection to the DB """
# Output: MySQL connector variable
def connection_to_db():
    db_connection_str = 'mysql+pymysql://root:StockholmStad@localhost:3306/learning_analytics'
    db_connection = create_engine(db_connection_str)
    return db_connection

# Input: Two df to be merged and list of variables
# Output: New df
def merge_function(df_1, df_2, non_remove_variables ):
    df_new = df_1.merge(df_2, how = 'inner', on = 'slumpkod')
    df_new_with_dropped = df_new.loc[:, df_new.columns.isin(non_remove_variables)]         # Drop all variables apart from them in the list
    # Skriv en try, except, som klagar på att variabel som ska finnas kvar inte ens finns kvar!
    return df_new_with_dropped

def pageviews_participation_factor(dataframe):
    # Måste först byta till "to_datetime", droppar "datum" för har inför en annany ny column

    dataframe['datum_DateTime'] = pd.to_datetime(dataframe.loc[:, ('datum')], format='%Y-%m-%d')
    dataframe_2 = dataframe.drop(columns = ["datum"])

    # Skapar en kolumn som heter "average_page_views", baserat på alla elever i klassen
    dataframe_2['Average_page_views'] = dataframe_2.groupby('kurs')['page_views'].transform('median')
    # skapa kolumn baserat på klassen participations,
    dataframe_2['Average_participation'] = dataframe_2.groupby('kurs')['participations'].transform('median')

    # Create a new column, each student has a page_view_factor
    dataframe_2['page_view_factor'] = dataframe_2['page_views']/dataframe_2['Average_page_views']
    # Create a new column, each student has a participations_factor
    dataframe_2['participation_factor'] = dataframe_2['participations']/dataframe_2['Average_participation']
    dataframe_3 = dataframe_2.replace([np.inf, -np.inf], np.nan)                                                     # replace all inf with Nan
    return dataframe_3

def tardiness_factor(dataframe_5,type):
    #Med type väljer man om man vill köra på median eller mean. Problemet med median är att många värden försvinner

    # skapa kolumn baserat på klassens saknade inlämningar,
    dataframe_5['Average_saknade_inlämningar'] = dataframe_5.groupby('kurs')[
        'tardiness_breakdown_missing'].transform(type)
    # skapa kolumn baserat på klassens inlämningar i tid,
    dataframe_5['Average_on_time_inlämningar'] = dataframe_5.groupby('kurs')[
        'tardiness_breakdown_on_time'].transform(type)
    # skapa kolumn baserat på klassens sena inlämningar,
    dataframe_5['Average_late_inlämningar'] = dataframe_5.groupby('kurs')[
        'tardiness_breakdown_late'].transform(type)

    # Create a new column, each student has a missing_factor
    dataframe_5['missing_factor'] = dataframe_5['tardiness_breakdown_missing'] / dataframe_5[
        'Average_saknade_inlämningar']
    # Create a new column, each student has an on_time_factor
    dataframe_5['on_time_factor'] = dataframe_5['tardiness_breakdown_on_time'] / dataframe_5[
        'Average_on_time_inlämningar']
    # Create a new column, each student has a late_factor
    dataframe_5['late_factor'] = dataframe_5['tardiness_breakdown_late'] / dataframe_5[
        'Average_late_inlämningar']
    dataframe_6 = dataframe_5.replace([np.inf, -np.inf], np.nan)                                                     # replace all inf with Nan
    return dataframe_6

#Skapar en ny dataframe där
def quantity_maker(df):
    columns = list(df)

    columns.remove('id')
    columns.remove('slumpkod')
    #print(columns)
    quantity=[]
    for col_name in columns:
        variable = df[col_name]
        len_var=len(variable)
        variable_na=variable.dropna()
        len_var_na=len(variable_na)
        #print(col_name,len_var,len_var_na)
        quantity.append(len_var_na)
    #print(quantity)
    data = {'col_name':columns,
            'quantity':quantity}
    df_new = pd.DataFrame(data)
    return(df_new)

""" ------------------------ Main -------------------------- """

""" ---- Settings ------ """
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

db_con = connection_to_db() # Create connection to database, Använd denna i alla SQL_queries!

""" ------------------- Canvas data! ---------------------------------------------------------------------- """
""" Byt variabler HÄR, DATUM och KURS """
df_canvas_1 = pd.read_sql_query('SELECT * FROM canvas '
                              'WHERE kurs LIKE "FYSFYS01a%%" '
                              'AND datum LIKE "%%02-15" ', con = db_con)
df_canvas_2 = pageviews_participation_factor(df_canvas_1)                         # Update Canvas table, add two columns
df_canvas=tardiness_factor(df_canvas_2,'median')
""" --------------------------- Read other tables -------------------------------------------------
sql_query = 'SELECT betyg.slumpkod, betyg.betyg, franvaro.Frånvarotid' \
           'FROM betyg ' \
           'INNER JOIN franvaro ON franvaro.slumpkod = betyg.slumpkod ' \
           'WHERE betyg.kurs = "FYSFYS01a" ' \
           'GROUP BY(betyg.slumpkod)'"""

df_betyg= pd.read_sql_query('SELECT slumpkod, betyg FROM betyg WHERE kurs="FYSFYS01a"', con = db_con)
df_franvaro=pd.read_sql_query('SELECT Frånvarotid, slumpkod FROM franvaro WHERE kurs="FYSFYS01a"', con = db_con)
df_betyg_franvaro= df_betyg.merge(df_franvaro, how = 'inner', on = 'slumpkod')

df_diagnoser= pd.read_sql_query('SELECT * FROM diagnoser WHERE NOT `Engelska totalt` = "NULL"', con = db_con)
df_betyg_franvaro_diagnoser =df_betyg_franvaro.merge(df_diagnoser, how = 'inner', on = 'slumpkod')


""" --------------------------------- merge tables ----------------------------------------------------------- """
# List of variables, not to drop!
variables_tokeep_list = ['betyg', 'Frånvarotid', 'Matematik', 'page_view_factor', 'participation_factor',
                         'missing_factor','late_factor','on_time_factor','Engelska totalt',
                        'Svenska rättstavning', 'Svenska bokstavskedjor Stanine',
                         'Svenska ordkedjor Stanine','Svenska meningskedjor Stanine']
df_merged = merge_function(df_betyg_franvaro_diagnoser, df_canvas, variables_tokeep_list)         # merge and keep variables above.
#print(df_merged)

diagnos_quant=quantity_maker(df_betyg_franvaro_diagnoser)
canvas_quant=quantity_maker(df_canvas)

print(diagnos_quant)
print(canvas_quant)


#print(Matematik_na)
#print(len(Matematik_na))