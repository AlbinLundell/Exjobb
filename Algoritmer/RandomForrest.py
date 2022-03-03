

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
    db_connection_str = 'mysql+pymysql://root:LearningAnalytics@localhost:3306/learning_analytics'
    db_connection = create_engine(db_connection_str)
    return db_connection

# Create two columns, one for page view factor and one for participation.
# Input: df
# Output: Updated df
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

# Input: Two df to be merged and list of variables
# Output: New df
def merge_function(df_1, df_2, non_remove_variables ):
    df_new = df_1.merge(df_2, how = 'inner', on = 'slumpkod')
    df_new_with_dropped = df_new.loc[:, df_new.columns.isin(non_remove_variables)]         # Drop all variables apart from them in the list
    # Skriv en try, except, som klagar på att variabel som ska finnas kvar inte ens finns kvar!
    return df_new_with_dropped

# Input: df
# Output: df with F-P grades and grades with out interest gone.
def remove_grades(df, grades_list):
    df_dropped = df[df.betyg.isin(grades_list) == False]
    df_dropped['betyg'] = df_dropped['betyg'].replace(to_replace=['A', 'B', 'C', 'D', 'E'], value='P')      # Replace values with P, depening on grades_list
    df_na_dropped = df_dropped.dropna()                                                                    # Remove NaN values rows.
    return df_na_dropped

# Input: df, string to be target column
# Output: X = features set, y = target set, targets = list of targets.
# Function rewrite target column to integer classifier
def encode_target(df, target_column):
    df_mod_sorted = df.copy()          # create a copy df
    # df_mod_sorted = df_mod.sort_values(by = 'betyg', ascending = False)     # Sort the dateframe, we want F before E (And P before E)
    targets = df_mod_sorted[target_column].unique()    # define targets
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod_sorted["Target"] = df_mod_sorted[target_column].replace(map_to_int)     # Create target column
    y = df_mod_sorted["Target"]                                             # Create features and label series and df
    X = df_mod_sorted.drop(columns = ["betyg", "Target"])
    return (X, y, targets)

# Input: Train and test data
# Output: Print statements about the OOB and accuracy
def random_forrest(X_train, X_test, y_train, y_test):
    print("------------------------RF --------------------------")
    n_F_train = y_train.value_counts()[0]
    n_E_train = y_train.value_counts()[1]
    n_F_test = y_test.value_counts()[0]
    n_E_test = y_test.value_counts()[1]

    print(f'Train datapunkter: {y_train.shape} varav (F = {n_F_train}, E = {n_E_train}) \n'
          f'Test datapunkter: {y_test.shape} varav (F = {n_F_test}, E = {n_E_test})')          # Print the shape of the data set
    classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)   # create classfier algorthm
    classifier_rf.fit(X_train, y_train)         # train the RF algortihm

    print(f'OOB score: {classifier_rf.oob_score_}')                     # OOB score
    y_pred = classifier_rf.predict(X_test)                              # Accuracy from test set.
    # print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')        # Print the accuracy

    print(classification_report(y_test, y_pred))

    #skriv om till en dataframe för att jämföra pred med test
    #df_test_pred = y_test.to_frame(name='test_value')
    #df_test_pred["pred"] = y_pred
    #print(df_test_pred)

def random_forrest_SMOTE(X_train, X_test, y_train, y_test):
    print("------------------------RF with SMOTE --------------------------")
    bsmote = BorderlineSMOTE(random_state = 42, kind = 'borderline-2', k_neighbors = 3)
    #X_res_sm, y_res_sm = sm.fit_resample(X_train, y_train)    # Apply it on the train data
    #X_res_1, y_res_1 = bsmote1.fit_resample(X_train, y_train)
    X_res, y_res = bsmote.fit_resample(X_train, y_train)

    print(X_res.shape, X_test.shape)          # Print the shape of the data set
    classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)   # create classfier algorthm
    classifier_rf.fit(X_res, y_res)         # train the RF algortihm

    print(f'OOB score: {classifier_rf.oob_score_}')                     # OOB score
    y_pred = classifier_rf.predict(X_test)                              # Accuracy from test set.
    #print(f'Accuracy: {metrics.accuracy_score(y_res, y_pred)}')        # Print the accuracy
    print(classification_report(y_test, y_pred))

    # skriv om till en dataframe för att jämföra pred med test
    #df_test_pred = y_test.to_frame(name='test_value')
    #df_test_pred["pred"] = y_pred
    #print(df_test_pred)

# Input:
# Output: print statements + classifier
def random_forrest_SMOTE_HYPER(X_train, X_test, y_train, y_test):
    print("------------------------RF with SMOTE and HYPER--------------------------")

    bsmote = BorderlineSMOTE(random_state = 42, kind = 'borderline-2', k_neighbors = 3)
    #X_res_sm, y_res_sm = sm.fit_resample(X_train, y_train)    # Apply it on the train data
    #X_res_1, y_res_1 = bsmote1.fit_resample(X_train, y_train)
    X_res, y_res = bsmote.fit_resample(X_train, y_train)

    n_F_train = y_res.value_counts()[0]
    n_E_train = y_res.value_counts()[1]
    n_F_test = y_test.value_counts()[0]
    n_E_test = y_test.value_counts()[1]
    print(f'Train datapunkter: {y_train.shape} varav (F = {n_F_train}, E = {n_E_train}) \n'
          f'Test datapunkter: {y_test.shape} varav (F = {n_F_test}, E = {n_E_test})')          # Print the shape of the data set

    # create classfier algorthm, NOW with hypertuned parameters.
    # NOTE hypertuning parameters might differ for E-F and P-F and what class is given.
    # 'max_depth': None, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 0.5, 'n_estimators': 50
    classifier_rf = RandomForestClassifier(random_state = 42, n_jobs=-1, max_depth = None, max_features = None, min_samples_leaf = 1,
                                           min_samples_split = 0.5, n_estimators = 50, oob_score=True)
    classifier_rf.fit(X_res, y_res)         # train the RF algortihm

    print(f'OOB score: {classifier_rf.oob_score_}')                     # OOB score
    y_pred = classifier_rf.predict(X_test)                              # Accuracy from test set.
    #print(f'Accuracy: {metrics.accuracy_score(y_res, y_pred)}')        # Print the accuracy
    print(classification_report(y_test, y_pred))

    # skriv om till en dataframe för att jämföra pred med test
    #df_test_pred = y_test.to_frame(name='test_value')
    #df_test_pred["pred"] = y_pred
    #print(df_test_pred)

    return classifier_rf

def hypertune_param(X_train, X_test, y_train, y_test):
    n_samples = len(X_train.index)             # Måste ange dessa för hand, kalla på side i dataframe
    n_features = 5
    params = {'n_estimators': [10, 20, 50],
              'max_depth': [None, 2, 4, 5],
              'min_samples_split': [2, 3, 4, 5, 10, 20],
              'min_samples_leaf': [1, 2, 5],
              'max_features': [None, 'sqrt', 'auto', 'log2', 0.3, 0.5, n_features//2]
             }

    # CV only works on accuracy, hade varit intreesant att CV på någon annan parameter
    rf_classifier_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=params, n_jobs=-1, cv=4, verbose=1)
    # CV parameter beror på vår intiala split i train / test. Om 0.8 --> 4, 8, 12; Om 0.75 --> 3, 6, 9
    rf_classifier_grid.fit(X_train, y_train)

    print('Best Parameters : ',rf_classifier_grid.best_params_)
    cross_val_results = pd.DataFrame(rf_classifier_grid.cv_results_)

    print('Train Accuracy : %.3f'%rf_classifier_grid.best_estimator_.score(X_train, y_train))
    print('Test Accurqacy : %.3f'%rf_classifier_grid.best_estimator_.score(X_test, y_test))
    print('Best Accuracy Through Grid Search : %.3f'%rf_classifier_grid.best_score_)
    print('Number of Various Combinations of Parameters Tried : %d', len(cross_val_results))

    # cross_val_results_sorted = cross_val_results.sort_values(by = 'rank_test_score')
    #print(cross_val_results_sorted.head()) ## Printing first few results.
    # Hitta hypertune parameters och Corssvalidation  -- > Jämför med ny accuracy

# Input: Classifier algoritm, testdata, labels
# En Confusion matrix plot
def confusion_matrix(classifier_rf, X_test, y_test, labels):
    plot_confusion_matrix(classifier_rf, X_test, y_test, display_labels = labels)
    plt.show()

# input: the orginal df, before manipulated. RF classifier object!
# Output: Confusion Matrix + classfication_report illustrating how many students got classified as F instead.
def higher_grades_test(df, classifier_rf):
    # Plocka ut D,C,B,A elever och påvisa om det är någon skillnad i vilket betyg de får!

    grades_list_extra = ["F", "E", "-"]       # Remove these grades from df
    df_higher_grades = df[df.betyg.isin(grades_list_extra) == False]
    df_higher_grades["betyg"] = df_higher_grades["betyg"].replace(to_replace = ['A', 'B', 'C', 'D'], value = 'P')   # A, B, C, D = P now
    df_higher_grades_drop = df_higher_grades.dropna()                                                               # Drop Nan values

    X_test, y_test, targets_list = encode_target(df_higher_grades_drop, "betyg")
    # Vi måste byta y_test till 1 från 0 eftersom vi inte har några F med så blir det fel klassificerng.

    # Kolla vilket som är det första värdet!
    if targets_list[0] == "P":
        test_int = 1
    else:
        test_int = 0
    y_test_correct = y_test.replace(to_replace = 0, value = test_int)

    y_pred = classifier_rf.predict(X_test)                              # Accuracy from test set.

    df_test_pred = y_test_correct.to_frame(name='test_value')
    df_test_pred["pred"] = y_pred
    print(df_test_pred)

    print(classification_report(y_test_correct, y_pred))
    confusion_matrix(classifier_rf, X_test, y_test_correct, ["F", "P"])



""" ------------------------ Main -------------------------- """

""" ---- Settings ------ """
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

db_con = connection_to_db() # Create connection to database, Använd denna i alla SQL_queries!

""" ------------------- Canvas data! ---------------------------------------------------------------------- """
""" Byt variabler HÄR, DATUM och KURS """
df_canvas_1 = pd.read_sql_query('SELECT * FROM canvas '
                              'WHERE kurs LIKE "FYSFYS01%%" '
                              'AND datum LIKE "%%02-15" ', con = db_con)
df_canvas = pageviews_participation_factor(df_canvas_1)                         # Update Canvas table, add two columns

""" --------------------------- Read other tables -------------------------------------------------"""
sql_query = 'SELECT betyg.slumpkod, betyg.betyg, franvaro.Frånvarotid, diagnoser.Matematik ' \
           'FROM betyg ' \
           'INNER JOIN franvaro ON franvaro.slumpkod = betyg.slumpkod ' \
           'INNER JOIN diagnoser ON diagnoser.slumpkod = franvaro.slumpkod ' \
           'WHERE betyg.kurs = "FYSFYS01a" ' \
           'GROUP BY(betyg.slumpkod)'
df_betyg_franvaro_diagnoser = pd.read_sql_query(sql_query, con = db_con)

""" --------------------------------- merge tables ----------------------------------------------------------- """
# List of variables, not to drop!
variables_tokeep_list = ['betyg', 'Frånvarotid', 'Matematik', 'page_view_factor', 'participation_factor', 'tardiness_breakdown_missing']
df_merged = merge_function(df_betyg_franvaro_diagnoser, df_canvas, variables_tokeep_list)         # merge and keep variables above.

""" ----------------------------- Clean and Transform data -----------------------------------------------------"""
# With Only E and F
grades_list = ["A", "B", "C", "D", "-"]
df_improved = remove_grades(df_merged, grades_list)      # Remove grades above!

# With P and F
grades_list_2 = ["-"]
df_improved_2 = remove_grades(df_merged, grades_list_2)      # Remove grades above!

X, y, targets = encode_target(df_improved, "betyg")               # E-F: Create X, y dfs, and target string list.
X_2, y_2, targets_2 = encode_target(df_improved_2, "betyg")       # P-F: Create X, y dfs, and target string list.

# Vill skriva ut hur många som har E och F för set

# ------------------ Train - Test split data --------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)         # Train and test data
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, train_size=0.8, random_state=42 )

# ----------------------------------- RF algorithm --------------------------------------------------------
random_forrest(X_train, X_test, y_train, y_test)                # RF function that create, split, train and return statements about OOB and accuracy.
# random_forrest(X_train_2, X_test_2, y_train_2, y_test_2)        # RF function that create, split, train and return statements about OOB and accuracy.

# ------------------------------ Apply SMOTE  and RF again -------------------------------------------------------------
# Depending on size we choose for train data, we have to change k_neighbors parameter.
# Det finns olika SMOTE algortimer, man skulle även här kunna genomföra hypertuning av parameters och på så sätt finna, vilken som är bäst!
# For now, Borderline 2 is the best!
random_forrest_SMOTE(X_train, X_test, y_train, y_test)                  # RF function that create, train and return statements about OOB and accuracy AND apply SMOTE
# random_forrest_SMOTE(X_train_2, X_test_2, y_train_2, y_test_2)          # RF function that create, train and return statements about OOB and accuracy AND apply SMOTE

# -------------------------------------- Hypertuning parameters --------------------------------
# Behöver bara köra denna en gång. Därefter, vet vi ju parametrar
# Intressant att jämföra med olika kurser och mellan olika dataset.
# hypertune_param(X_train, X_test, y_train, y_test)
# Ex, för F-P i fysik så förändras en hyper tune parameter.

# ----------------------------------- Apply RF again but now with hypertuned parameters ---------------------------

rf_classifier_1 = random_forrest_SMOTE_HYPER(X_train, X_test, y_train, y_test)
rf_classifier_2 = random_forrest_SMOTE_HYPER(X_train_2, X_test_2, y_train_2, y_test_2)

# hypertune_param(X_train_2, X_test_2, y_train_2, y_test_2)

# confusion_matrix(rf_classifier_1, X_test, y_test, targets)
confusion_matrix(rf_classifier_2, X_test_2, y_test_2, targets_2)



# Some testing
# higher_grades_test(df_merged, rf_classifier_1)





# Man önskar testa hur många som har D-A, och hur dessa klassificeras?
# Hur förändras algoritmens värde med fler datapunkter (och då tyvärr färre variabler.)
# Hur förändras algoritmen med tiden?
# Betyg från Matte1C kanske man också kan lägga in?
# Måste sortera data så det alltid kommer en F först!
# Men vill inte sortera enligt ascending för då förändras algoritm.
# Varför: 1: enklare att göra print statement, 2: confusion matrix blir återkommande enhetlig.
# Måste se till att båda ligger längst fram i bägge fall!
