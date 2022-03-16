

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
from sklearn.feature_selection import SelectKBest, chi2
import seaborn as sn
from sklearn.inspection import permutation_importance


""" ------------------------- Functions ------------------------------------------ """


""" Returns a coneection to the DB """
# Output: MySQL connector variable
def connection_to_db():
    db_connection_str = 'mysql+pymysql://root:StockholmStad@localhost:3306/learning_analytics'
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

    # skapa kolumn baserat på klassens saknade inlämningar,
    dataframe_2['Average_saknade_inlämningar'] = dataframe_2.groupby('kurs')['tardiness_breakdown_missing'].transform('median')
    # skapa kolumn baserat på klassens inlämningar i tid,
    dataframe_2['Average_on_time_inlämningar'] = dataframe_2.groupby('kurs')['tardiness_breakdown_on_time'].transform('median')
    # skapa kolumn baserat på klassens sena inlämningar,
    dataframe_2['Average_late_inlämningar'] = dataframe_2.groupby('kurs')['tardiness_breakdown_late'].transform('median')

    # Create a new column, each student has a missing_factor
    dataframe_2['missing_factor'] = dataframe_2['tardiness_breakdown_missing']/dataframe_2['Average_saknade_inlämningar']
    # Create a new column, each student has an on_time_factor
    dataframe_2['on_time_factor'] = dataframe_2['tardiness_breakdown_on_time']/dataframe_2['Average_on_time_inlämningar']
    # Create a new column, each student has a late_factor
    dataframe_2['late_factor'] = dataframe_2['tardiness_breakdown_late']/dataframe_2['Average_late_inlämningar']

    dataframe_3 = dataframe_2.replace([np.inf, -np.inf], np.nan)                                                     # replace all inf with Nan
    return dataframe_3

def read_sql(kurs, con):
    # Hade varit nice att ha en lista här med vilka kurs man inte ska läsa in!
    sql_query = 'SELECT betyg.slumpkod, betyg.betyg, franvaro.Frånvarotid, diagnoser.Matematik, ' \
                'diagnoser.`Svenska ordkedjor Stanine`, diagnoser.`Svenska bokstavskedjor Stanine`,' \
                'diagnoser.`Svenska meningskedjor Stanine`, diagnoser.`Engelska totalt` ' \
                'FROM betyg ' \
                'INNER JOIN franvaro ON franvaro.slumpkod = betyg.slumpkod ' \
                'INNER JOIN diagnoser ON diagnoser.slumpkod = franvaro.slumpkod ' \
                f'WHERE betyg.kurs = "{kurs}" ' \
                'GROUP BY(betyg.slumpkod)'
    df_betyg_franvaro_diagnoser = pd.read_sql_query(sql_query, con = con)

    return df_betyg_franvaro_diagnoser

def read_sql_betyg(kurs, datum, con):
    kurs_update = kurs[:8]      # Jag tror alla kurser har 8 tecken! DUBBELKOLLA!
    df_canvas_1 = pd.read_sql_query(f'SELECT * FROM canvas '
                              f'WHERE kurs LIKE "{kurs_update}%%" '
                              f'AND datum LIKE "%%{datum}" ', con = con)
    return df_canvas_1

# Input: Two df to be merged and list of variables
# Output: New df

# Call this if you only would like to return variables
def remove_variables(df, non_remove_variables):
    df_new_with_dropped = df.loc[:, df.columns.isin(non_remove_variables)]         # Drop all variables apart from them in the list
    return df_new_with_dropped

# Call this is you like to both merge and remove variables.
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

def grades_to_numbers(df,col_name):
    grades_list = ["A", "B", "C", "D", "E", "F"]
    df_changed = df[df[col_name].isin(["-"]) == False]
    counter=5
    for i in grades_list:
        counter_str=str(counter)
        df_changed[col_name] = df_changed[col_name].replace(to_replace=[i], value=counter_str)
        counter=counter-1
    df_changed[col_name] = df_changed[col_name].astype(int)
    df_changed=df_changed.dropna()
    return df_changed




    # Input: df, string to be target column
# Output: X = features set, y = target set, targets = list of targets.
# Function rewrite target column to integer classifier
def encode_target(df, target_column):
    df_mod_sorted = df.copy()          # create a copy df
    # df_mod_sorted = df_mod.sort_values(by = 'betyg', ascending = False)     # Sort the dateframe, we want F before E
    # Om denna är F så måste vi byta två rader

    # Om första raden har har ett P betyg så måste vi byta till F längst fram!
    betyg = df_mod_sorted.iloc[0]["betyg"]
    if betyg == "P":
        index_list = df_mod_sorted.index.tolist()       # Skriv om alla index till en list
        ind_list_2 = df_mod_sorted.index[df_mod_sorted["betyg"] == "F"].tolist()        # Plocka ut första elementet med F
        ind_2 = ind_list_2[0]
        # Droppa index för första F betyg
        index_list.remove(ind_2)
        index_list.insert(0, ind_2)
        df_mod_sorted_2 = df_mod_sorted.reindex(index_list)
    else:
        df_mod_sorted_2 = df_mod_sorted


    targets = df_mod_sorted_2[target_column].unique()    # define targets
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod_sorted_2["Target"] = df_mod_sorted_2[target_column].replace(map_to_int)     # Create target column
    y = df_mod_sorted_2["Target"]                                             # Create features and label series and df
    X = df_mod_sorted_2.drop(columns = ["betyg", "Target"])
    return (X, y, targets)

def remove_percentage(df):
    df_copy = df.copy()
    df_copy['Engelska totalt'] = df_copy['Engelska totalt'].str.replace('%', '')
    df_copy_2 = df_copy.dropna()
    df_copy_3 = df_copy_2.loc[df_copy_2['Engelska totalt'] != 'NULL']
    # print(df_copy_3)
    df_copy_3['Engelska totalt'] = df_copy_3['Engelska totalt'].astype(int)
    return df_copy_3

# Gör ett Chi_2 test. Hur bra är egentligen chi_2 test???
def kscore(X,y):
    select = SelectKBest(chi2, k = 3).fit(X,y)
    print(select.scores_, select.pvalues_)

# Input: Train and test data
# Output: Print statements about the OOB and accuracy
def random_forrest(X_train, X_test, y_train, y_test):
    print("------------------------RF --------------------------")
    n_F_train = y_train.value_counts()[0]
    n_E_train = y_train.value_counts()[1]
    n_F_test = y_test.value_counts()[0]
    n_E_test = y_test.value_counts()[1]
    print(f'Train datapunkter: {int(n_F_train) + int(n_E_train)} varav (F = {n_F_train}, E = {n_E_train}) \n'
          f'Test datapunkter: {int(n_F_test) + int(n_E_test)} varav (F = {n_F_test}, E = {n_E_test}) \n '
          f'Total antal datapunkter: {int(n_F_test) + int(n_E_test) + int(n_F_train) + int(n_E_train)}')         # Print the shape of the data set

    classifier_rf = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=5, max_features=None, n_estimators=100, oob_score=True)   # create classfier algorthm
    classifier_rf.fit(X_train, y_train)         # train the RF algortihm

    print(f'OOB score: {classifier_rf.oob_score_}')                     # OOB score
    y_pred = classifier_rf.predict(X_test)                              # Accuracy from test set.
    # print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')        # Print the accuracy

    print(classification_report(y_test, y_pred))

    return classifier_rf
    #skriv om till en dataframe för att jämföra pred med test
    #df_test_pred = y_test.to_frame(name='test_value')
    #df_test_pred["pred"] = y_pred
    #print(df_test_pred)

def random_forrest_SMOTE(X_train, X_test, y_train, y_test):
    print("------------------------RF with SMOTE --------------------------")
    # Om felkod, testa att byta k_neighbors, m_neighbours
    # Kolla different smote parametrar
    bsmote = BorderlineSMOTE(random_state = 1, kind = 'borderline-2', k_neighbors = 2, m_neighbors = 7)
    #X_res_sm, y_res_sm = sm.fit_resample(X_train, y_train)    # Apply it on the train data
    #X_res_1, y_res_1 = bsmote1.fit_resample(X_train, y_train)
    X_res, y_res = bsmote.fit_resample(X_train, y_train)

    n_F_train = y_res.value_counts()[0]
    n_E_train = y_res.value_counts()[1]
    n_F_test = y_test.value_counts()[0]
    n_E_test = y_test.value_counts()[1]
    print(f'Train datapunkter: {int(n_F_train) + int(n_E_train)} varav (F = {n_F_train}, E = {n_E_train}) \n'
          f'Test datapunkter: {int(n_F_test) + int(n_E_test)} varav (F = {n_F_test}, E = {n_E_test}) \n '
          f'Total antal datapunkter: {int(n_F_test) + int(n_E_test) + int(n_F_train) + int(n_E_train)}')

    print(X_res.shape, X_test.shape)          # Print the shape of the data set
    classifier_rf = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)   # create classfier algorthm
    classifier_rf.fit(X_res, y_res)         # train the RF algortihm

    print(f'OOB score: {classifier_rf.oob_score_}')                     # OOB score
    y_pred = classifier_rf.predict(X_test)                              # Accuracy from test set.
    #print(f'Accuracy: {metrics.accuracy_score(y_res, y_pred)}')        # Print the accuracy
    print(classification_report(y_test, y_pred))

    return classifier_rf

    # skriv om till en dataframe för att jämföra pred med test
    #df_test_pred = y_test.to_frame(name='test_value')
    #df_test_pred["pred"] = y_pred
    #print(df_test_pred)

# Input:
# Output: print statements + classifier
def random_forrest_SMOTE_HYPER(X_train, X_test, y_train, y_test):
    print("------------------------RF with SMOTE and HYPER--------------------------")

    bsmote = BorderlineSMOTE(random_state = 1, kind = 'borderline-2', k_neighbors = 2, m_neighbors = 7)
    #X_res_sm, y_res_sm = sm.fit_resample(X_train, y_train)    # Apply it on the train data
    #X_res_1, y_res_1 = bsmote1.fit_resample(X_train, y_train)
    X_res, y_res = bsmote.fit_resample(X_train, y_train)

    n_F_train = y_res.value_counts()[0]
    n_E_train = y_res.value_counts()[1]
    n_F_test = y_test.value_counts()[0]
    n_E_test = y_test.value_counts()[1]
    print(f'Train datapunkter: {int(n_F_train) + int(n_E_train)} varav (F = {n_F_train}, E = {n_E_train}) \n'
          f'Test datapunkter: {int(n_F_test) + int(n_E_test)} varav (F = {n_F_test}, E = {n_E_test}) \n '
          f'Total antal datapunkter: {int(n_F_test) + int(n_E_test) + int(n_F_train) + int(n_E_train)}')          # Print the shape of the data set

    # create classfier algorthm, NOW with hypertuned parameters.
    # NOTE hypertuning parameters might differ for E-F and P-F and what class is given.

    # Best Parameters : {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 10}
    # Ska vi lägga in Max depth och max_features ändå???
    classifier_rf = RandomForestClassifier(random_state = 1, n_jobs=-1, max_depth = None, max_features = 'sqrt', min_samples_leaf = 1,
                                           min_samples_split = 2, n_estimators = 10, oob_score=True)
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
    n_features = 4
    params = {'n_estimators': [10, 20, 50, 100],
              'max_depth': [None, 2, 4, 5, 7],
              'min_samples_split': [2, 3, 4, 5, 10, 20],
              'min_samples_leaf': [1, 2, 5, 10],
              'max_features': [None, 'sqrt', 'auto', 'log2', 0.3, 0.5, n_features//2, 2, 4, 5, 6]
             }

    # CV only works on accuracy, hade varit intreesant att CV på någon annan parameter
    rf_classifier_grid = GridSearchCV(RandomForestClassifier(random_state=1), param_grid=params, n_jobs=-1, cv=4, verbose=1)
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

# Function that returns the importance of the feature based on the classifier algoritm.
def features(classifier, variables_list, X_test, y_test):
    # Måste också skriva in listan i vilken ordning attributen förekommer
    feature_imp = pd.Series(classifier.feature_importances_, index = variables_list)
    perm_importance = permutation_importance(classifier, X_test, y_test)
    print(feature_imp)

# input: the orginal df, before manipulated. RF classifier object!
# Output: Confusion Matrix + classfication_report illustrating how many students got classified as F instead.
def higher_grades_test(df, classifier_rf):
    #Vill jämföra df_higher_grades med om de fick F
    # Det borde gå att merge på index?

    # Plocka ut D,C,B,A elever och påvisa om det är någon skillnad i vilket betyg de får!

    grades_list_extra = ["F", "E", "-"]       # Remove these grades from df
    df_higher_grades = df[df.betyg.isin(grades_list_extra) == False]
    df_higher_grades_copy = df_higher_grades.copy()                   # create a copy that will be used later
    df_higher_grades_copy_drop = df_higher_grades_copy.dropna()
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
    # Detta behövs vid confusion matrisen!

    y_pred = classifier_rf.predict(X_test)                              # Accuracy from test set.
    df_test_pred = y_test_correct.to_frame(name='test_value')           # Sammanfoga till en ny df som jämför test med pred.
    df_test_pred["pred"] = y_pred
    # print(df_higher_grades_copy_drop)
    # print(df_test_pred)
    print(classification_report(y_test_correct, y_pred))                    # Print some info
    confusion_matrix(classifier_rf, X_test, y_test_correct, ["F", "P"])     # create confusion matrix

    # Create a plot and df illustrate how many students with a unique grade got classfied as F.
    df_merged_higher_grades_test_pred = df_higher_grades_copy_drop.merge(df_test_pred, how = 'inner', left_index = True, right_index = True)
    # Group by betyg, hur många som med varje betyg fick ett särskilt
    test = df_merged_higher_grades_test_pred.groupby(['betyg', 'pred']).size().unstack(fill_value = 0)
    test_1 = test.rename(columns = {0:"F"})
    test_2 = test_1.rename(columns = {1:"P"})
    print(test_2)
    test_2.plot.bar(xlabel = "betyg", ylabel = "antal elever")
    plt.legend(loc = 'upper left')
    plt.title('What students of grade A, B, C, D got classfied as')
    plt.show()





# ------------------------ Main --------------------------

# ---- Settings ------
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

db_con = connection_to_db() # Create connection to database, Använd denna i alla SQL_queries!

# ------------------- Canvas data! ----------------------------------------------------------------------
df_canvas_1 = read_sql_betyg("MATMAT04" ,'02-15' ,db_con)        # read canvas table from SQL
#Byt variabler HÄR, DATUM och KURS
df_canvas = pageviews_participation_factor(df_canvas_1)                         # Update Canvas table, add factorised columns

# --------------------------- Read other tables -------------------------------------------------
# Lägg till vilka variabler du vill select här!
df_betyg_franvaro_diagnoser = read_sql("MATMAT04", db_con)

#---------------------------- Grades from earlier courses--------------------------------------------------

df_betygMA2=pd.read_sql_query('SELECT betyg AS betyg_MA2, slumpkod FROM betyg '
                             'WHERE kurs = "MATMAT02c" ', con = db_con)
df_betygMA1=pd.read_sql_query('SELECT betyg AS betyg_MA1, slumpkod FROM betyg '
                             'WHERE kurs = "MATMAT01c" ', con = db_con)
df_betygMA3=pd.read_sql_query('SELECT betyg AS betyg_MA3, slumpkod FROM betyg '
                             'WHERE kurs = "MATMAT03c" ', con = db_con)
df_numbers_MA2 = grades_to_numbers(df_betygMA2,"betyg_MA2")
df_numbers_MA1= grades_to_numbers(df_betygMA1,"betyg_MA1")
df_numbers_MA3= grades_to_numbers(df_betygMA3, "betyg_MA3")
df_tidigare_betyg= df_numbers_MA1.merge(df_numbers_MA2, how = 'inner', on = 'slumpkod')
df_tidigare_betyg= df_tidigare_betyg.merge(df_numbers_MA3, how = 'inner', on = 'slumpkod')
# Clean duplicates of students
df_earlier_grades=df_tidigare_betyg.drop_duplicates(subset="slumpkod")

print(df_earlier_grades)

# --------------------------------- merge tables -----------------------------------------------------------
# List of variables, not to drop!
# Add variables to keep here:
# Måste ändra MED CANVAS DATA
# MED CANVAS: 'betyg', 'Frånvarotid', 'Matematik', 'page', 'page_view_factor, 'participation_factor', 'on_time_factor''
# 'Svenska ordkedjor Stanine', 'Svenska bokstavskedjor Stanine', 'Svenska meningskedjor Stanine','Engelska totalt'
# Utan CANVAS: 'betyg', 'Frånvarotid', 'Matematik',
# 'Svenska ordkedjor Stanine', 'Svenska bokstavskedjor Stanine', 'Svenska meningskedjor Stanine','Engelska totalt'

variables_tokeep_list = ['slumpkod','betyg', 'Frånvarotid', 'Matematik', 'page_view_factor','Svenska ordkedjor Stanine',
                         'Svenska bokstavskedjor Stanine', 'Svenska meningskedjor Stanine','Engelska totalt']
more_variables_to_keep = ['betyg', 'Frånvarotid', 'Matematik','Svenska ordkedjor Stanine', 'Svenska bokstavskedjor Stanine',
                          'Svenska meningskedjor Stanine','Engelska totalt',
                          'betyg_MA1','betyg_MA2','betyg_MA3','page_view_factor']
# Måste skapa en variables to keep function och en merge funktion
# Bara om vi ska ha med Canvasdata
df_merged_0 = merge_function(df_betyg_franvaro_diagnoser, df_canvas, variables_tokeep_list)
# merge and keep variables above.
df_merged = merge_function(df_merged_0,df_earlier_grades,more_variables_to_keep)
print(df_merged_0)
# Om vi INTE har canvas data
# df_merged = remove_variables(df_betyg_franvaro_diagnoser, variables_tokeep_list)

# ----------------------------- Clean and Transform data -----------------------------------------------------
df_cleaned = remove_percentage(df_merged)       # Clean % for engelska total )
# ------------ With Only E and F ----------------
grades_list = ["A", "B", "C", "D", "-"]
df_improved = remove_grades(df_cleaned, grades_list)      # Remove grades above!

X, y, targets = encode_target(df_improved, "betyg")               # E-F: Create X, y dfs, and target string list.

# ------------------ With P and F -------------------
# grades_list_2 = ["-"]
# df_improved_2 = remove_grades(df_merged, grades_list_2)      # Remove grades above!
# X_2, y_2, targets_2 = encode_target(df_improved_2, "betyg")       # P-F: Create X, y dfs, and target string list.

# ------------------ Train - Test split data --------------------------------------------------------------
# randomstate i ordning: 42, 1, 2
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)         # Train and test data
# X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, train_size=0.8, random_state=42)

# ----------------------------------- RF algorithm --------------------------------------------------------
just_rf_classifier = random_forrest(X_train, X_test, y_train, y_test)                # RF function that create, split, train and return statements about OOB and accuracy.
features(just_rf_classifier, list(X_train), X_test, y_test)                     # Skriver ut features lista!

# random_forrest(X_train_2, X_test_2, y_train_2, y_test_2)        # RF function that create, split, train and return statements about OOB and accuracy.

# ------------------------------ Apply SMOTE  and RF again -------------------------------------------------------------
# Depending on size we choose for train data, we have to change k_neighbors parameter.
# Det finns olika SMOTE algortimer, man skulle även här kunna genomföra hypertuning av parameters och på så sätt finna, vilken som är bäst!
# For now, Borderline 2 is the best!

# we remove the worst values and therefore the features has to eb dropped
# Do this by hand!
X_train_dropped = X_train.drop(columns = ['Frånvarotid','Svenska bokstavskedjor Stanine','Svenska meningskedjor Stanine','betyg_MA3','betyg_MA1'])
X_test_dropped = X_test.drop(columns = ['Frånvarotid','Svenska bokstavskedjor Stanine','Svenska meningskedjor Stanine','betyg_MA3','betyg_MA1'])
remove_targets = np.array(['Frånvarotid','Svenska bokstavskedjor Stanine','Svenska meningskedjor Stanine','betyg_MA3','betyg_MA1'])
new_targets = np.setdiff1d(targets, remove_targets)

# --------------- RF again, not with less variables -----------------------
just_rf_classifier = random_forrest(X_train_dropped, X_test_dropped, y_train, y_test)                # RF function that create, split, train and return statements about OOB and accuracy.
features(just_rf_classifier, list(X_train_dropped), X_test_dropped, y_test)

# -------------------------- SMOTE RF --------------------------------
rf_Smote_classifier = random_forrest_SMOTE(X_train_dropped, X_test_dropped, y_train, y_test)                  # RF function that create, train and return statements about OOB and accuracy AND apply SMOTE
features(rf_Smote_classifier, list(X_train_dropped), X_test_dropped, y_test)

# random_forrest_SMOTE(X_train_2, X_test_2, y_train_2, y_test_2)          # RF function that create, train and return statements about OOB and accuracy AND apply SMOTE

# -------------------------------------- Hypertuning parameters --------------------------------
# Behöver bara köra denna en gång. Därefter, vet vi ju parametrar
# Intressant att jämföra med olika kurser och mellan olika dataset.

# AVKOMMENTERA HÄR
hypertune_param(X_train_dropped, X_test_dropped, y_train, y_test)

# ----------------------------------- Apply RF again but now with hypertuned parameters ---------------------------

rf_classifier_hyper_smote = random_forrest_SMOTE_HYPER(X_train_dropped, X_test_dropped, y_train, y_test)
features(rf_classifier_hyper_smote, list(X_train_dropped), X_test_dropped, y_test)


# rf_classifier_2 = random_forrest_SMOTE_HYPER(X_train_2, X_test_2, y_train_2, y_test_2)
# hypertune_param(X_train_2, X_test_2, y_train_2, y_test_2)
# confusion_matrix(rf_classifier_1, X_test, y_test, targets)

# ------------ CONFUSION MATRIX --------------------------

#confusion_matrix(rf_classifier_2, X_test_2, y_test_2, targets_2)
confusion_matrix(rf_classifier_hyper_smote, X_test_dropped, y_test, new_targets)
# Some testing
#higher_grades_test(df_merged, rf_classifier_1)

# ------------------- Tankar / att göra --------------------------
# Man önskar testa hur många som har D-A, och hur dessa klassificeras?
# Hur förändras algoritmens värde med fler datapunkter (och då tyvärr färre variabler.)
# Hur förändras algoritmen med tiden?
# Betyg från Matte1C kanske man också kan lägga in?
# Måste sortera data så det alltid kommer en F först!
# Men vill inte sortera enligt ascending för då förändras algoritm.
# Varför: 1: enklare att göra print statement, 2: confusion matrix blir återkommande enhetlig.
# Måste se till att båda ligger längst fram i bägge fall!
# Ändra i encode target, vilken som blir F och vilken som blir E



