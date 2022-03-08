# Python code returning a dataframe over the number of
# Have to improve by adding the course and the data as a parameter
import pandas as pd
import Algoritmer.RandomForrest

# lägg till variabler som datum och betyg
def df_data(kurs, datum):

    db_con = Algoritmer.RandomForrest.connection_to_db()
    # Variable list that is later added to the DF
    variable_list = ["Betyg", "Frånvaro", "Diagnoser", "Diagnoser P-F", "Diagnoser E-F", "Canvas", "Canvas P-F", "Canvas E-F"]
    number_list = []        # list of the number od DP that is later added to the DF

    # Read betyg data
    sql_query = ()

    df_betyg = pd.read_sql_query(f'SELECT * FROM betyg '
                                 f'WHERE kurs = "{kurs}" '
                                 f'GROUP BY(betyg.slumpkod) ', con = db_con)
    number_list.append(len(df_betyg))       # Append it list
    # Read frånvaro data
    df_franvaro = pd.read_sql_query(f'SELECT * from franvaro '
                                    f'WHERE kurs = "{kurs}" '
                                    f'GROUP BY(franvaro.slumpkod) ', con = db_con)
    df_betyg_franvaro = df_betyg.merge(df_franvaro, how = 'inner', on = 'slumpkod')         # Merge franvaro and betyg data
    df_betyg_franvaro_2 = df_betyg_franvaro.dropna()                                        # Drop Na values
    number_list.append(len(df_betyg_franvaro_2))        # Append value to list
    # Read diagnos data
    df_diagnos = pd.read_sql_query('SELECT * from diagnoser '
                                   'GROUP BY(diagnoser.slumpkod) ', con = db_con)
    # HÄR SKULLE jag vilja skapa module till en lista
    drop_columns_list = ['Svenska rättstavning', 'Svenska diktamen Stanine']      # Denna måste uppdateras för hand nu
    df_diagnos_2 = df_diagnos.drop(columns = drop_columns_list)
    df_diagnos_2_droppedna = df_diagnos_2.dropna()                                  # Drop rows that still have NaN values
    df_betyg_franvaro_diagnos = df_betyg_franvaro.merge(df_diagnos_2_droppedna, how='inner', on='slumpkod')      # Merge betyg, frånvaro, diagnos
    number_list.append(len(df_betyg_franvaro_diagnos))
    # Diagnoser for P-F
    df_betyg_franvaro_diagnos_P = Algoritmer.RandomForrest.remove_grades(df_betyg_franvaro_diagnos, ["-"])
    number_list.append(len(df_betyg_franvaro_diagnos_P))
    # Diagnoser for E-F
    df_betyg_franvaro_diagnos_E = Algoritmer.RandomForrest.remove_grades(df_betyg_franvaro_diagnos, ["A", "B", "C", "D", "-"])
    number_list.append(len(df_betyg_franvaro_diagnos_E))
    df_betyg_franvaro_diagnos_droppdup = df_betyg_franvaro_diagnos.drop(columns = ["id_x"])

    # Read Canvas table
    # Change date - to see if it differ any?
    # remove one character from the kurs string
    # Måste ange hur string ska skrivas för datum!
    kurs_wo_a = kurs.replace('a', '')
    df_canvas_1 = pd.read_sql_query(f'SELECT * FROM canvas '
                                  f'WHERE kurs LIKE "{kurs_wo_a}%%" '
                                  f'AND datum LIKE "%%{datum}" ', con = db_con)
    df_canvas = Algoritmer.RandomForrest.pageviews_participation_factor(df_canvas_1)
    df_canvas_dropna = df_canvas.dropna()
    df_betyg_franvaro_diagnos_canvas = df_betyg_franvaro_diagnos_droppdup.merge(df_canvas_dropna, how = 'inner', on ='slumpkod')
    number_list.append(len(df_betyg_franvaro_diagnos_canvas))
    df_betyg_franvaro_diagnos_canvas_P = Algoritmer.RandomForrest.remove_grades(df_betyg_franvaro_diagnos_canvas, ["-"])
    number_list.append(len(df_betyg_franvaro_diagnos_canvas_P))
    df_betyg_franvaro_diagnos_canvas_E = Algoritmer.RandomForrest.remove_grades(df_betyg_franvaro_diagnos_canvas, ["A", "B", "C", "D", "-"])
    number_list.append(len(df_betyg_franvaro_diagnos_canvas_E))

    # crete a data dict
    data = {'Variable': variable_list, 'Antal': number_list}
    # create a df
    df = pd.DataFrame(data)
    return df

# NOTIS: Jag tar inte Canvas variabler som evenentuellt är dåliga.
# NOTIS: Jag har inte lagt till andra factors än page_views och participation
# Notis: men dropna används.

"""
DF      Variabel    Antal 
1       Betyg       XXX 
2       Frånvaro    

"""
