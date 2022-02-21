"""
Random forest för Fysik med variablerna och dess importance för denna algortihm.
Frånvarotid                    0.413126
Matematik                      0.158718
tardiness_breakdown_missing    0.073941
page_view_factor               0.194738
participation_factor           0.159477

Datum från Canvas 02-15

Confusion matrix är inte så bra
Men accuracy är bra.
För få med F


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

# -------------------------------- Betyg Table --------------------------------------
df_betyg = pd.read_sql_query('SELECT * FROM betyg '
                             'WHERE kurs = "FYSFYS01a" ', con = db_connection)
# Clean duplicates of students
df_betyg_1 = df_betyg.drop_duplicates(subset=['slumpkod'])
#print(df_betyg_1)

# --------------------------------- Canvas Table -----------------------------------------------
#
# ----------- Read the table -------------------------
df_canvas = pd.read_sql_query('SELECT * FROM canvas '
                              'WHERE kurs LIKE "FYSFYS01%%" AND datum LIKE "%%02-15" ', con = db_connection )
df_canvas_1month = df_canvas.drop_duplicates(subset = ['slumpkod'])
# print(df_canvas_1month)

# ------------------- Create two new columns ------------------
# Måste först byta till "to_datetime", droppar "datum" för har inför en annany ny column
df_canvas_1month['datum_DateTime'] = pd.to_datetime(df_canvas_1month.loc[:, ('datum')], format='%Y-%m-%d')
df_canvas_1month2 = df_canvas_1month.drop(columns = ["datum"])

# Skapar en kolumn som heter "average_page_views", baserat på alla elever i klassen
df_canvas_1month2['Average_page_views'] = df_canvas_1month2.groupby('kurs')['page_views'].transform('mean')
# skapa kolumn baserat på klassen participations,
df_canvas_1month2['Average_participation'] = df_canvas_1month2.groupby('kurs')['participations'].transform('mean')

# Create a new column, each student has a page_view_factor
df_canvas_1month2['page_view_factor'] = df_canvas_1month2['page_views']/df_canvas_1month2['Average_page_views']
# Create a new column, each student has a participations_factor
df_canvas_1month2['participation_factor'] = df_canvas_1month2['participations']/df_canvas_1month2['Average_participation']



#  ------------------------ Franvarotable ---------------------------------------
sql_query_franvaro = 'SELECT * ' \
            'FROM franvaro ' \
            'WHERE kurs = "FYSFYS01a" ' \
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
#print(df_diagnos)

# --------------------------- Merge tables -------------------------------
# Merge Betyg, Canvas, frånvaro, diagnos
df_merge = df_betyg_1.merge(
    df_diagnos.merge(
        df_franvaro.merge(
            df_canvas_1month2, how ='inner', on = 'slumpkod'),
        how ='inner', on = 'slumpkod')
    , how ='inner', on = 'slumpkod')


# Droppa onödiga variabler
df_merge_1 = df_merge.drop(columns = ['id_x', 'slumpkod', 'kurs', 'id_y', 'kurs_x', 'start', 'slut', 'status', 'max_page_views',
                                      'max_participations', 'tardiness_breakdown_floating', 'tardiness_breakdown_total',
                                      'kurs_y', 'Lektionstid', 'Läsår', 'Engelska Vocabulary', 'Engelska Grammar', 'Engelska Reading',
                                      'Svenska diktamen Stanine', 'Average_page_views', 'Average_participation', 'Engelska totalt',
                                      'Svenska rättstavning','Svenska bokstavskedjor Stanine', 'Svenska ordkedjor Stanine',
                                      'Svenska meningskedjor Stanine', 'page_views', 'page_views_level', 'participations', 'participations_level',
                                      'tardiness_breakdown_late', 'tardiness_breakdown_on_time', 'datum_DateTime'])

#print(df_merge_1)

# Måste ersätta betyg med siffror
# Här måste koden se annorlunda ut om vi ska göra om alla ["A", "B", "C", "D", "E"] till "P"
values = ["A", "B", "C", "D", "-"]
#values = ["-"]

# Droppar alla element i values lista
df_dropped = df_merge_1[df_merge_1.betyg.isin(values) == False]
df_dropped_na = df_dropped.dropna()
print(df_dropped)

# ---------------------------- RF --------------------------------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn.model_selection import GridSearchCV


# Assign features to X
# assign response / classifier variable / targets to y
def encode_target(df, target_column):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)
    return (df_mod, targets)
df_EF_2, old_targets = encode_target(df_dropped_na, "betyg")
y = df_EF_2["Target"]
print(df_EF_2)
X = df_EF_2.drop(columns = ["betyg", "Target"])
print(X)                            # Features
print(y)                            # Target

# ----------------------- Data info -------------------------------------
# 23 st som har F
# 120 som har E
# 114 i train data
# 29 st i test data


# ----------------- train - test - split ---------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
# print train and test sizes.
print(X_train.shape, X_test.shape)

# ----------------------------- RF algortihm -----------------------------
# create classfier algorthm
classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)
# train the tree
classifier_rf.fit(X_train, y_train)

# OOB score
print(f'OOB score: {classifier_rf.oob_score_}')
# Accuracy from test set.
y_pred = classifier_rf.predict(X_test)
print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')

# skriv om till en dataframe för
df_test_pred = y_test.to_frame(name='test_value')
df_test_pred["pred"] = y_pred
print(df_test_pred)

# ------------------------------ What order of importance do we variable have ---------------------
# https://www.datacamp.com/community/tutorials/random-forests-classifier-python
feature_imp = pd.Series(classifier_rf.feature_importances_, index=["Frånvarotid", "Matematik", "tardiness_breakdown_missing",
                                                                   "page_view_factor", "participation_factor"])
print(feature_imp)

# --------------------------------------- Hypertuning ------------------------------

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

# ----------------------------------- Confustion matrix ------------------------------------

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(classifier_rf, X_test, y_test, display_labels = ["F", "E"])

plt.show()