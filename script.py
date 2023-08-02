import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline


# Preprocess application_train.csv
def application_train(num_rows=None, nan_as_category=False):
    # Read data
    df = pd.read_csv('./application_train.csv', nrows=num_rows)
    print("Train samples: {}".format(len(df)))
    # Remove applications with XNA CODE_GENDER
    df = df[df['CODE_GENDER'] != 'XNA']
    # NaN values for DAYS_EMPLOYED: 365 243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    return df


# Assigning a weight to FN and FP
def cost(actual, pred, TN_val=0, FN_val=10, TP_val=0, FP_val=1):
    matrix = confusion_matrix(actual, pred)
    TN = matrix[0, 0]
    FN = matrix[1, 0]
    FP = matrix[0, 1]
    TP = matrix[1, 1]
    total_cost = TP * TP_val + TN * TN_val + FP * FP_val + FN * FN_val
    return total_cost


# Metrics
def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    AUC = roc_auc_score(actual, pred)
    f1 = f1_score(actual, pred)
    bank_cost = cost(actual, pred)
    return f1, AUC, accuracy, bank_cost


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    # Split the data into training and test sets. (0.75, 0.25) split.
    df_id = application_train(10000)
    df = df_id.drop(["SK_ID_CURR"], axis=1)
    train, test = train_test_split(df)

    # The predicted column is "TARGET" (0 or 1)
    train_x = train.drop(["TARGET"], axis=1)
    test_x = test.drop(["TARGET"], axis=1)
    train_y = train[["TARGET"]]
    test_y = test[["TARGET"]]

    # Imbalanced class analysis
    num_0 = df.TARGET.value_counts()[0]
    num_1 = df.TARGET.value_counts()[1]
    percentage_0 = num_0 / (num_0 + num_1) * 100
    percentage_1 = num_1 / (num_0 + num_1) * 100
    print("\nImbalanced class analysis: ")
    print(" - Percentage of 0: {}".format(percentage_0))
    print(" - Percentage of 1: {}".format(percentage_1))

    # Pipeline that aggregates preprocessing steps (encoder + scaler + model)

    ct = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse=False), make_column_selector(dtype_include=object)),
        (StandardScaler(with_mean=False), make_column_selector(dtype_exclude=object)))

    steps = [("t", ct),
             ("model", LGBMClassifier())]
    pipe = Pipeline(steps)
    pipe.fit(train_x, train_y)

    # GridSearchCV that allows to choose the best model for the problem
    # I add the param class_weight="balanced" to deal with imbalanced class
    param_grid = {"model": [LGBMClassifier(class_weight="balanced"),
                            LogisticRegression(class_weight="balanced"),
                            DecisionTreeClassifier(class_weight="balanced"),
                            RandomForestClassifier(class_weight="balanced"),
                            SVC(class_weight="balanced")]}

    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring="f1")
    grid.fit(train_x, train_y)
    print("\nGridSearchCV: ")
    print("    Best score ", grid.best_score_, "using ", grid.best_params_)

