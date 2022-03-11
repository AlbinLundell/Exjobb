
# -------------- Modules ------------------------------
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

# ----- Conection to DB -------------
def connection_to_db():
    db_connection_str = 'mysql+pymysql://root:LearningAnalytics@localhost:3306/learning_analytics'
    db_connection = create_engine(db_connection_str)
    return db_connection

def read_sql(db_con):
    # Jag vill läsa in från betyg table.
    # Groupby kurser.
    # Plocka ut de som har F, typ groupby betyg == F
    # Räkna de som har betyg
    """
    DF  Kurs    Antal
    0   Fysik   XXX
    1   Matte   XXX
    2
    ...
    """
    sql_query = 'SELECT COUNT(slumpkod) AS Antal, kurs ' \
                'FROM betyg ' \
                'WHERE betyg = "F" ' \
                'GROUP BY kurs ' \
                'ORDER BY Antal DESC'

    df = pd.read_sql_query(sql_query, con = db_con)
    return df
# Skapa en graf över detta!
# Vi kan kolla mer på Matte 4, Kemi 1,

# -------- Main -------------
pd.set_option('display.max_rows', None)
db_con = connection_to_db()
df = read_sql(db_con)
# df = df.rename(columns={"Category":"Pet"})
df = df.rename(columns={"Antal":"Number of students"})
df = df.rename(columns={"kurs":"Course"})

df_more5 = df.loc[df["Number of students"] >= 5]
print(df_more5)

sn.barplot(x = 'Number of students', y = 'Course', data = df, palette = 'flare',
            capsize = 0.05,
            saturation = 8,
            errcolor = 'gray', errwidth = 2,
            ci = 'sd'
            ).set(title = 'Number of students with grade F in each course')

# df.plot(x = 'kurs', y = 'Antal', kind = 'bar')
plt.show()