"""
Code that imports data from SQL,
combine different tables
save it within a pandas dataframe
create a DT, variables here:
- Ämne: Matte1C
- Features: diagnospoäng, frånvarotid
- Target: betyg - E, F
"""

# ------------------ Import modules ----------------------------------------------
from sqlalchemy import create_engine        # Läser mellan server och Python
import pandas as pd

# -------------------------------- Connect to data database --------------------------------------
#                                    user:password          Borde vara samma/namn på DB
db_connection_str = 'mysql+pymysql://root:LearningAnalytics@localhost:3306/learning_analytics'
db_connection = create_engine(db_connection_str)

# ------------------------------------- SQL queries ----------------------------------------------
# Elev table
# 701 elever
df_elever = pd.read_sql('SELECT * FROM elever', con=db_connection)

# Diagnos table
# 686 elever
df_diagnoser = pd.read_sql('SELECT slumpkod, Matematik AS mattediagnos FROM diagnoser', con = db_connection)

# Alla elevers betyg i matte1C
# 649 elever.
df_betyg_matte1 = pd.read_sql('select * from betyg where kurs = "MATMAT01c" order by slumpkod ', con = db_connection)

# Alla elevers frånvaro i Matte1C
# 858 elever
# df_franvaro_matte1 = pd.read_sql('select * from franvaro where kurs = "MATMAT01c" order by slumpkod ', con = db_connection)


# Sortera table efter Läsår som registreras på eleven
# Välj alla kolumner, från frånvaro, men enbart Matte 1c. Group by och order by ger att vi sorterar
# dem i siffer-bokstavsordning enligt läsår samt väljer bara ut ett distinct värde.
sql_query = 'SELECT * ' \
            'FROM franvaro ' \
            'WHERE kurs = "MATMAT01c" ' \
            'GROUP BY(slumpkod) ' \
            'ORDER BY Läsår '
df_franvaro_matte1_sorterad = pd.read_sql(sql_query, con = db_connection)
# 793 elever


# Använder pandas merge på slumpkod!
# tidigare df_franvaro_matte1_sorterad är städad.
# Vi får ut 646 elever, men det var 649 elever som hade ett betyg i matte 1. Tre elever finns således ej i frånvaro table

# merge between betyg and frånvaro on slumpid
df_combined = df_betyg_matte1.merge(df_franvaro_matte1_sorterad, how = 'inner', on = 'slumpkod')
# Drop unneccesary values
df_combined_2 = df_combined.drop(columns=['id_x', 'id_y', 'kurs_y', 'Läsår'])
# merge between betyg-frånvaro och diagnosvärde.
df_combined_3 = df_combined_2.merge(df_diagnoser, how = 'inner', on = 'slumpkod')
# There are 548 students who also have a diagnosis value
print(df_combined_3)
# Drop more uneccesary columns
df_combined_prep_plot = df_combined_3.drop(columns = ['slumpkod', 'kurs_x'])

# ------------------ DT cleaning -----------------------------------
# Define values
values = ["A", "B", "C", "D", "-"]
# Drop rows that contain any value of betyg that is not E or F.
df_DT_clean = df_combined_prep_plot[df_combined_prep_plot.betyg.isin(values) == False]
df_DT_clean_2 = df_DT_clean.drop(columns = "Lektionstid") # Droppa även lektionstid som column
df_DT_clean_3 = df_DT_clean_2.dropna()                  # Drop NaN values
print(df_DT_clean_3)

# ----------------------------- DT ---------------------------------------
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

# function input: dataframe, Column to be changed
def encode_target(df, target_column):
    """Add column to df with integers for the target.
    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.
    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)
    return (df_mod, targets)

# Function that create a target column in DF, that is df_DT
df_DT, targets = encode_target(df_DT_clean_3, "betyg")
# Features are frånvarotid and mattediagnos
# Targets are E(0), F(1)

features = list(df_DT.columns[1:3])      # Store the two features as a list
X = df_DT[features]                       # Store feature data
y = df_DT["Target"]                     # Represented against target data

# Train/test split data!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

dt = DecisionTreeClassifier(criterion = 'gini', min_samples_split = 2, random_state=10, max_depth=5)
# criterion can be etiher gini or entropy
# max_depth, maximum lvl at which the algoritm is stops running
# The minimum number of samples required to split an internal node, deafault = 2. Has to be geq 2.
# random_state controls the randomness of the estimator.


dt.fit(X_train, y_train)

print(dt.predict(X_test))
print(dt.score(X_test, y_test))

import matplotlib.pyplot as plt
from sklearn import tree


# ------------------------------ Plot the DT -----------------------------------------------------
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)

tree.plot_tree(dt,feature_names = features, class_names=targets,filled = True)
fig.savefig('DT.png')


