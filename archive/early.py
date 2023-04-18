import os
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer, ColumnTransformer,make_column_selector
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler, PowerTransformer, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

PATH = 'https://gist.githubusercontent.com/Takfes/ba8e651ac4ddbae1846b8f9471a1efad/raw/068a33d07d3fe2fb12f7e186c736812ed3f4b8d5/credit_customer_churn.csv'

TARGET = 'Target'
TEST_SIZE = 0.33

#%% Load data
raw = pd.read_csv(PATH).\
    iloc[:,:-2].\
        set_index('CLIENTNUM').\
            rename_axis('Client_id').\
                rename(columns={'Attrition_Flag':'Target'}).\
                    assign(**{'Target' : lambda x : x['Target'].map({'Existing Customer':0,'Attrited Customer':1})})

X = raw.drop(TARGET, axis=1)
y = raw[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=1990, stratify = y)
[x.shape for x in [X_train, X_test, y_train, y_test]]

def make_education_level(df):
    dc = df.copy()
    education_level_dict = {'Unknown': 0,'Uneducated': 1,
                            'High School': 2,'College': 3,
                            'Graduate': 4,'Post-Graduate': 5,
                            'Doctorate': 6}
    dc['Education_Level'] = dc['Education_Level'].map(education_level_dict)
    return dc

def make_card_category(df):
    dc = df.copy()
    card_category_dict = {'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
    dc['Card_Category'] = dc['Card_Category'].map(card_category_dict)
    return dc

def make_income_category(df):
    dc = df.copy()
    income_category_dict = {'Unknown': 0,
                          'Less than $40K': 1,
                          '$40K - $60K': 2,
                          '$60K - $80K': 3,
                          '$80K - $120K': 4,
                          '$120K +': 5,
                          }
    dc['Income_Category'] = dc['Income_Category'].map(income_category_dict)
    return dc

def ftft(func):
    return FunctionTransformer(func, validate=False)

trs_education_level = ftft(make_education_level)
trs_card_category = ftft(make_card_category)
trs_income_category = ftft(make_income_category)

preprocessor = ColumnTransformer(
    transformers=[
        ('education_level', trs_education_level, ['Education_Level']),
        ('card_category', trs_card_category, ['Card_Category']),
        ('income_category',trs_income_category, ['Income_Category']),
        ('marital_status', OneHotEncoder(drop='first'),['Marital_Status']),
        ('gender', OneHotEncoder(drop='first'),['Gender']),
        # ('credit_limit', ce.WOEEncoder(),['Credit_Limit'])
        ],
        remainder='passthrough')

md = pd.DataFrame(preprocessor.fit_transform(raw,y))
md.shape
md.head()

X_train.dtypes

woe = ce.WOEEncoder()
woe.fit_transform(X_train.Education_Level,y_train)


df = prepare_data(raw)
X = df.drop(TARGET,axis=1)
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=1990, stratify = y)

numeric_features = ['Age', 'Fare', 'SibSp', 'Parch', 'Parch_Sibsp']
categorical_features = ['Embarked', 'Sex', 'Pclass']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

pipe = Pipeline([('preprocessor', preprocessor),('RF',RandomForestClassifier())])
cross_val_score(pipe, X, y, scoring = 'roc_auc', cv = 10)


def fn_embarked(df):
    dataframe = df.copy()
    embarked_map = {'C' : 0, 'Q' : 1, 'S' : 2}
    dataframe['Embarked'] = dataframe['Embarked'].map(embarked_map)
    return dataframe

def fn_parch_sibsp(df):
    dataframe = df.copy()
    dataframe['Parch_Sibsp'] = dataframe['Parch'] + dataframe['SibSp']
    return dataframe

trns_emabrked = FunctionTransformer(fn_embarked, validate=False)
trns_parch_sibsp = FunctionTransformer(fn_parch_sibsp, validate=False)

pp1 = Pipeline(memory=None,
    steps=[
        ('embarked', trns_emabrked),
        ('parch_sibsp', trns_parch_sibsp)
    ], verbose=False)

pp1.fit_transform(raw)