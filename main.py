
import pandas as pd
import numpy as np
import os
from joblib import dump, load

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, plot_roc_curve, f1_score
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV, cross_val_predict

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline

import mlflow
import argparse
import sys

# def parse_input():
#     parser = argparse.ArgumentParser(description='Model Scoring')
#     parser.add_argument('data', type=str, help='path where csv to score is saved')
#     parser.add_argument('model', type=str, help='path where model binary is saved')
#     args = parser.parse_args()
#     print(f'>> Path to data : {args.path_to_data}')
#     print(f'>> Path to model : {args.path_to_model}')
#     return args.path_to_data, args.path_to_model

def parse_input():
    path_to_data = sys.argv[ 1 ]
    path_to_model = sys.argv[ 2 ]
    return path_to_data, path_to_model

def data_clearance(path_to_csv):
    df = pd.read_csv(path_to_csv)
    features = [ 'Age', 'Fare', 'Embarked', 'Sex', 'Pclass' ]
    X = df[ features ]
    y = df[ 'Survived' ]
    print(f'DataFrame of {df.shape}\n')
    print(f'X of shape : {X.shape}\n')
    return X, y

def score_model(path_to_model, X, y):
    current_model = load(path_to_model)
    print(f'Model loaded ...')
    predLabels = current_model.predict(X)
    print(f'>> Accuracy score : {accuracy_score(y,predLabels)}\n')
    print(f'>> RoCAuC score : {roc_auc_score(y,predLabels)}\n')
    print(f'>> Confusion matrix : {confusion_matrix( y, predLabels)}')

if __name__ == '__main__':
    path_to_csv, path_to_model = parse_input()
    X, y = data_clearance(path_to_csv)
    score_model(path_to_model, X, y)
