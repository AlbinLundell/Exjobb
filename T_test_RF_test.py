"""
T_test to try some variables
T-test verkar vara baserad på Normalfördelning

Elever i fysik1 som har E respektive F

"""


# ----------------------------------- Modules and connect to DB --------------------------------------------
from sqlalchemy import create_engine        # Läser mellan server och Python
import pandas as pd
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

#                                    user:password          Borde vara samma/namn på DB
db_connection_str = 'mysql+pymysql://root:LearningAnalytics@localhost:3306/learning_analytics'
db_connection = create_engine(db_connection_str)
pd.set_option('display.max_columns', None)


# ---------------------------------------- SQL ---------------------------------------------------

df_betyg = pd.read_sql_query('SELECT * FROM betyg '
                             'WHERE kurs = "FYSFYS01a" ', con = db_connection)
df_betyg_1 = df_betyg.drop_duplicates(subset=['slumpkod'])
#print(df_betyg_1)

df_franvaro = pd.read_sql_query('SELECT * FROM franvaro '
                                'WHERE kurs = "FYSFYS01a "'
                                'GROUP BY(slumpkod) '
                                'ORDER BY id', con = db_connection)
#print(df_franvaro)


# -------------------- SQL inner join try -------------------------------
# ----------------- df with both frånvaro and betyg ---------------
sql_query = 'SELECT betyg.betyg, franvaro.Frånvarotid, diagnoser.Matematik ' \
           'FROM betyg ' \
           'INNER JOIN franvaro ON franvaro.slumpkod = betyg.slumpkod ' \
           'INNER JOIN diagnoser ON diagnoser.slumpkod = franvaro.slumpkod ' \
           'WHERE betyg.kurs = "FYSFYS01a" ' \
           'GROUP BY(betyg.slumpkod)'

df_betyg_franvaro_mattediagnos = pd.read_sql_query(sql_query, con = db_connection)
#print(df_betyg_franvaro_mattediagnos)

# ----------------- Clean all grades apart from E and F ---------------------
df_EF = df_betyg_franvaro_mattediagnos.loc[df_betyg_franvaro_mattediagnos["betyg"].isin(["E","F"])]
#print(df_EF)
# ----------------- Clean from NaN values ------------------
df_EF_r = df_EF.dropna()

# ---------------------- T-test -------------------------------------------
# ---------- SQL query for matte diagnos --------------------

"""
print(df_EF.groupby("betyg")["Matematik"].describe())
df_E = df_EF.loc[df_EF["betyg"] == "E"]
df_F = df_EF.loc[df_EF["betyg"] == "F"]
print(df_F)

print(stats.levene(df_E['Matematik'], df_F['Matematik']))
df_E['Matematik'].plot(kind="hist")
df_F['Matematik'].plot(kind="hist", title="Matematik")

plt.xlabel("Length (units)")
plt.show()
"""

# ---------------------------- RF --------------------------------

df_F = df_EF_r.loc[df_EF_r["betyg"] == "F"]
print(df_F.loc[df_F["Matematik"] >= 30])


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# Assign features to X
# assign response / classifier variable / targets to y
def encode_target(df, target_column):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)
    return (df_mod, targets)
df_EF_2, old_targets = encode_target(df_EF_r, "betyg")
y = df_EF_2["Target"]
print(df_EF_2)
X = df_EF_2.drop(columns = ["betyg", "Target"])
print(X)                            # Features
print(y)                            # Target
# 23 st som har F
# 120 som har E


# ----------------- train - test - split ---------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
# print train and test sizes.
print(X_train.shape, X_test.shape)

# ----------------------------- RF algortihm -----------------------------
# create classfier algorthm
classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)
# train the tree
classifier_rf.fit(X_train, y_train)


# -------------- Hyper parameter tuning ------------------------------------

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
params = {
    'max_depth': [2,3,5,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10,25,30,50,100,200]
}
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy")
grid_search.fit(X_train, y_train)
print(grid_search.best_score_)
rf_best = grid_search.best_estimator_
print(rf_best)


# ----------------- Plot ----------------------
from sklearn.tree import plot_tree
plt.figure(figsize=(20,5))
plot_tree(rf_best.estimators_[5], feature_names = X.columns, class_names=['F','E'], filled=True)
plt.show()




