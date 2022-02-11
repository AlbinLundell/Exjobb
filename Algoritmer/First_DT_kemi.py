"""
Code that imports data from SQL,
combine different tables
save it within a pandas dataframe
Har inte lagt in denna i excel dokument då det är mycket test än så länge
create a DT, variables here:
- Ämne: Kemi
- Features: frånvarotid, page_views_average
- Target: betyg - P, F
Kommentar: Får ganska bra accuracy
- MEN fortsatt lite dålig data.
- Problem med frånvarotid kvarstår,
"""
# ----------------------------------------------- Modules and intial DB connection -------------------------
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

# ----------- Läs in frånvaro data -----------------
# Läser in alla elever enligt läsår, group by slumpkod innebär att det inte kommer in samma student som blivit registerad
# på kursen flera gånger. Order by läsår gör att det enbart är första året som eleven läser kursen
sql_query_kemi = 'SELECT * ' \
            'FROM franvaro ' \
            'WHERE kurs = "KEMKEM01" ' \
            'GROUP BY(slumpkod) ' \
            'ORDER BY Läsår '

# Frånvaro på kemi elever
df_franvaro_kemi = pd.read_sql(sql_query_kemi, con = db_connection)


# ------------- Läs in canvas data --------------
# Väljer alla kurser som börjar med "KEMKEM"
# Väljer alla datum som slutar med "09-15" (Kollar en månad in)
# Väljer datum på hösten eftersom det generar mer data (dock kanske vi inte har slutbetyg på 21or?)
# Skapa en average per klass page views
# Skapa ny kolumn som är floattal = pageviews / average pageview

df_canvas_kemi = pd.read_sql('SELECT slumpkod, kurs, page_views, datum FROM canvas '
                             'WHERE kurs LIKE "KEMKEM%%" AND datum LIKE "%%09-15" ', con = db_connection)
# Droppa eventeulla dubletter
df_canvas_kemi_2 = df_canvas_kemi.drop_duplicates(subset = ['slumpkod'])

# Måste först byta till "to_datetime"
df_canvas_kemi_2['datum_DateTime'] = pd.to_datetime(df_canvas_kemi_2['datum'], format='%Y-%m-%d')
df_canvas_kemi_3 = df_canvas_kemi_2.drop(columns = ["datum"])

# Skapar en kolumn som heter "average", baserat på alla elever i klassen
df_canvas_kemi_3['Average_page_views'] = df_canvas_kemi_3.groupby('kurs')['page_views'].transform('mean')

# Create a new column that use a column index!
df_canvas_kemi_3['page_view_factor'] = df_canvas_kemi_3['page_views']/df_canvas_kemi_3['Average_page_views']

df_canvas_kemi_4 = df_canvas_kemi_3.drop(columns = ['page_views', 'kurs', 'datum_DateTime', 'Average_page_views'])

# print(df_canvas_kemi_4)

# --------------------------------- Merge dataframes  ------------------------------

# merge of betyg and franvaro
df_merge_betyg_franvaro = df_betyg_kemi.merge(df_franvaro_kemi, how ='inner', on='slumpkod')
df_merge_betyg_franvaro_2 = df_merge_betyg_franvaro.drop(columns = ['id_x', 'kurs_x', 'id_y', 'kurs_y', 'Lektionstid', 'Läsår'])

# merge betyg/franvaro med Canvas aktivitet
df_merge_betyg_franvaro_canvas = df_merge_betyg_franvaro.merge(df_canvas_kemi_4, how = 'inner', on = 'slumpkod')
df_merge_betyg_franvaro_canvas_2 = df_merge_betyg_franvaro_canvas.drop(columns = ['id_x', 'slumpkod', 'kurs_x', 'id_y', 'kurs_y', 'Läsår', 'Lektionstid'])

# ------------------------------ DT prepp -----------------------

values = ["-"]
# Drop rows that contain any value of betyg that is "-"
df_DT_clean = df_merge_betyg_franvaro_canvas_2[df_merge_betyg_franvaro_canvas_2.betyg.isin(values) == False]
# Drop NaN values
df_DT_clean_2 = df_DT_clean.dropna()

# Replace grades ["A", "B", "C", "D", "E"]  with an "P"
for i in ["A", "B", "C", "D", "E"]:
    df_DT_clean_2['betyg'] = df_DT_clean_2['betyg'].str.replace(i, 'P')

print(df_DT_clean_2)
# ----------------------------- DT ---------------------------------------
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

# function input: dataframe, Column to be changed
def encode_target(df, target_column):
    '''Add column to df with integers for the target.
    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.
    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    '''
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)
    return (df_mod, targets)

# Function that create a target column in DF, that is df_DT
df_DT, targets = encode_target(df_DT_clean_2, "betyg")
# Features are frånvarotid, mattediagnos, page_view_factor
# Targets are P(0), F(1)

features = list(df_DT.columns[1:3])      # Store the two features as a list
print(features)
X = df_DT[features]                       # Store feature data
y = df_DT["Target"]                     # Represented against target data

# Train/test split data!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
dt = DecisionTreeClassifier(criterion = 'gini', min_samples_split = 5, min_samples_leaf = 2, random_state=10, max_depth=5)
# criterion can be etiher gini or entropy
# max_depth, maximum lvl at which the algoritm is stops running
# The minimum number of samples required to split an internal node, deafault = 2. Has to be geq 2.
# random_state controls the randomness of the estimator.


dt.fit(X_train, y_train)

print(dt.predict(X_test))
print(dt.score(X_test, y_test))



# ------------------------------ Plot the DT -----------------------------------------------------
import matplotlib.pyplot as plt
from sklearn import tree

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)

tree.plot_tree(dt,feature_names = features, class_names=targets,filled = True)
fig.savefig('DT2_kemi.png')