

import pandas as pd
import numpy as np
import os

from joblib import dump, load

model_to_save = gs.best_estimator_
dump(model_to_save, 'titanic_pipe.pkl')

os.chdir(r'C:\Users\Takis\Google Drive\_projects_\_python_examples_')
os.getcwd()

df = pd.read_csv('http://bit.ly/kaggletrain')
df.shape
df.dtypes
df.isnull().sum()
df.to_csv('titanic.csv')

features = ['Age', 'Fare', 'Embarked', 'Sex', 'Pclass']

X = df[features]
y = df['Survived']

y.value_counts()
y.value_counts(normalize=True)

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 ,random_state = 1990)
[x.shape for x in [X_train, X_test, y_train, y_test]]

y_test.value_counts()

numeric_features = ['Age', 'Fare']
categorical_features = ['Embarked', 'Sex', 'Pclass']

# ==============================================
# PREPROCESSING
# ==============================================

# ==============================================
# PATH 1 : make_pipeline, make_column_transformer
# ==============================================

# pipe_num = make_pipeline(
#     (KNNImputer(n_neighbors=3)),
#     (StandardScaler())
# )
#
# pipe_cat = make_pipeline(
#     (SimpleImputer(strategy='most_frequent')),
#     (OneHotEncoder(handle_unknown='ignore'))
# )
#
# prepro = make_column_transformer(
#     (pipe_num, numeric_features),
#     (pipe_cat, categorical_features),
#     remainder='passthrough'
# )
#
# prepro.fit_transform(X_train)
#
# dir(prepro)
# prepro.get_params()

# ==============================================
# PATH 2 : Pipeline , ColumnTransformer
# ==============================================

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



# --------------------------------------------------------------------------------------------
# Create pipeline and column transformer objects for ease of use
# --------------------------------------------------------------------------------------------

# def build_pipeline(pipe_blocks):
#     myPipeline = Pipeline(steps=[(k,v) for k,v in pipe_blocks.items()])
#     return myPipeline
#
# def what_params(object):
#     params = {k : None for k, v in object.get_params().items() if "__" in k}
#     # import json
#     # with open('params.json', 'w') as outfile:
#     #     json.dump(params, outfile)
#     # pd.DataFrame().from_dict(params,orient='index').to_excel('params.xlsx')
#     return params
#
# def build_column_transformer(features_to_pipes):
#     myColumnTransformer = ColumnTransformer(transformers=[(k,v[0],v[1]) for k,v in features_to_pipes.items()])
#     return myColumnTransformer
#
# pipe1_blocks = {'imputer' : SimpleImputer(), 'scaler' : StandardScaler()}
# pipe2_blocks = {'encoder' : OneHotEncoder(), 'scaler' : StandardScaler()}
#
# pipe1 = build_pipeline(pipe1_blocks)
# pipe2 = build_pipeline(pipe2_blocks)
#
# params1 = what_params(pipe1)
# params1.update({'imputer__strategy':'mean'})
#
# params2 = what_params(pipe2)
# params2.update({'encoder__handle_unknown':'ignore'})
# params2.update({'encoder__drop':'first'})
#
# features_to_pipes = {
#                     'num' : ( pipe1 , numeric_features),
#                     'cat' : ( pipe2 , categorical_features)
#                      }
#
# transformer1 = build_column_transformer(features_to_pipes)
# params = what_params(transformer1)
# params.update({'cat__encoder__drop':'first',
#                'cat__encoder__handle_unknown':'ignore'})

# preprocessor.set_params(num__imputer__strategy='median')
# preprocessor.set_params(**pipe_params)
# preprocessor.get_params()

# encodersDict is somewhere else in the script.... :(
# pipes_list = encodersDict
# aa = [ build_pipeline({k:v}) for k,v in pipes_list.items() ]

# --------------------------------------------------------------------------------------------
# ML IMPORTS
# --------------------------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV, cross_val_predict

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, plot_roc_curve, f1_score


# --------------------------------------------------------------------------------------------
# SET THE BASELINE
# --------------------------------------------------------------------------------------------

# from sklearn.dummy import DummyClassifier
#
# dummy_tunning = { 'baselineModel__strategy' : ['stratified','most_frequent','prior','uniform'] }
# scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score), 'Recall' : 'recall'}
#
# baseline = GridSearchCV(estimator = Pipeline([('preprocessor',preprocessor),('baselineModel',DummyClassifier())]),
#                         param_grid = dummy_tunning,
#                         # scoring='accuracy',
#                         scoring = scoring, refit='Recall', # use refit to return the best estimator according to refit criterion
#                         cv=10)
#
# baseline.fit(X_train,y_train)

# baseline.cv_results_
# baseline.best_params_
# baseline.best_estimator_
# baseline.best_score_


# --------------------------------------------------------------------------------------------
# Custom Transformer : OneHotEncoder with Drop One functionality
# Still unclear on how to use a custom transformer with ColumnTransformer ; how to pass arguments into it
# --------------------------------------------------------------------------------------------
#
# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin
#
# class ohe_drop1(BaseEstimator, TransformerMixin):
#
#     def __init__(self, drop_one = True, drop_initial=True, return_array = True):
#         self.drop_one = drop_one
#         self.return_array = return_array
#         self.drop_initial = drop_initial
#
#     def transform(self, X, *_):
#
#         Xy = X.copy()
#
#         # encode the appropriate levels defined during fit
#         for v in self.encoding_levels:
#             Xy[ self.colname + '_' + str(v) ] = np.where(Xy[ self.colname ] == v, 1, 0)
#
#         # drop staring column if needed
#         if self.drop_initial:
#             result = Xy.drop(self.colname, axis=1)
#         else :
#             result = Xy
#
#         if self.return_array:
#             result = result.values
#
#         return result
#
#     def fit(self, X, colname):
#
#         # check input does not contains NaNs
#         # print(X[colname].isnull().sum())
#         if X[colname].isnull().sum()!=0 :
#             raise ValueError('Ensure input does not contain NaN values')
#
#         self.colname = colname
#         self.levels = X[ self.colname ].value_counts().index.tolist()
#
#         # track levels to be encoded into binary columns
#         if self.drop_one:
#             self.reference_level = self.levels[ :1 ]
#             self.encoding_levels = self.levels[ 1: ]
#         else:
#             self.reference_level = None
#             self.encoding_levels = self.levels
#         return self
#
#
# X = df[features]
# X.isnull().sum()
# del b
# oo = ohe_drop1(drop_initial=False,drop_one=False,return_array=False)
# b = oo.fit_transform( X , colname = 'Sex')
# b.isnull().sum()
# X.Sex.value_counts()
# b['Sex_male'].sum()
# # oo.fit( X , colname = 'Sex' )
# # a = oo.transform( X )
# # a
#
# oo.levels
# oo.reference_level
# oo.encoding_levels
#
#
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', ohe_drop1(drop_initial=False,drop_one=False,return_array=False)) ])
#
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features) ])

# preprocessor = make_column_transformer(
#             (numeric_transformer, numeric_features),
#             (ohe_drop1(drop_initial=False,drop_one=False,return_array=False), categorical_features))

# --------------------------------------------------------------------------------------------
# One Hot Encoder Drop One functionality
# --------------------------------------------------------------------------------------------
#
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', OneHotEncoder(drop='first', sparse=True)) ])
#
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features) ])
#
# preprocessor.fit_transform(X).shape
#
# # dir(preprocessor)
# # preprocessor.named_transformers_['cat']
# # dir(preprocessor.named_transformers_['cat'])
# # preprocessor.named_transformers_['cat'].steps[1][1]
#
# dir(preprocessor.named_transformers_['cat'].steps[1][1])
# preprocessor.named_transformers_['cat'].steps[1][1].get_params()
# preprocessor.named_transformers_['cat'].steps[1][1].get_feature_names()
# preprocessor.named_transformers_['cat'].steps[1][1].categories_
# preprocessor.named_transformers_['cat'].steps[1][1].drop_idx_
#

# --------------------------------------------------------------------------------------------
# Category encoder experimentations
# --------------------------------------------------------------------------------------------
#
# import category_encoders as ce
#
# ohe = ce.one_hot.OneHotEncoder(drop_invariant=True)
# dir(ohe)
# ff = ohe.fit_transform(X['Embarked'])
# ohe.get_feature_names()
# type(ff)
# ff.columns
#
#
# X.columns
# tt = ohe.fit_transform(X, cols = ['Embarked','Sex'], use_cat_names=True)
# tt.columns
# X.dtypes
#
# ohe.get_feature_names()
# X.Pclass.value_counts()
# X.isnull().sum()
#
# # --------------------------------------------------------------------------------------------
# # ATTACH DIFFERENT ENCODING METHODS TO OUR PIPELINE
# # Cross Product between Encoders and Different models
# # --------------------------------------------------------------------------------------------
#
# # Encoder Names
# encoder_names = [
#                 'BackwardDifferenceEncoder',
#                 'BaseNEncoder',
#                 'BinaryEncoder',
#                 'CatBoostEncoder',
#                 'HashingEncoder',
#                 'HelmertEncoder',
#                 'JamesSteinEncoder',
#                 'OneHotEncoder',
#                 'LeaveOneOutEncoder',
#                 'MEstimateEncoder',
#                 'OrdinalEncoder',
#                 'PolynomialEncoder',
#                 'SumEncoder',
#                 'TargetEncoder',
#                 'WOEEncoder'
#                 ]
#
# # Encoder Objects
# encoder_list = [ ce.backward_difference.BackwardDifferenceEncoder(),
#                  ce.basen.BaseNEncoder(),
#                  ce.binary.BinaryEncoder(),
#                  ce.cat_boost.CatBoostEncoder(),
#                  ce.hashing.HashingEncoder(),
#                  ce.helmert.HelmertEncoder(),
#                  ce.james_stein.JamesSteinEncoder(),
#                  ce.one_hot.OneHotEncoder(),
#                  ce.leave_one_out.LeaveOneOutEncoder(),
#                  ce.m_estimate.MEstimateEncoder(),
#                  ce.ordinal.OrdinalEncoder(),
#                  ce.polynomial.PolynomialEncoder(),
#                  ce.sum_coding.SumEncoder(),
#                  ce.target_encoder.TargetEncoder(),
#                  ce.woe.WOEEncoder()
#                  ]
#
# # Encoders Dictionary
# encodersDict = { x : y for x,y in zip(encoder_names, encoder_list)}
#
# # Define Models Dictionary
# # Model Names List
# modelsNames = ['LogisticRegression','RandomForestClassifier','AdaBoost',
#                'KNeighborsClassifier','SVC','LinearDiscriminantAnalysis',
#                'GaussianNB', 'XGBClassifier', 'LGBMClassifier',
#                'CatBoostClassifier']
# # Models List
# modelsList = [LogisticRegression(), RandomForestClassifier(), AdaBoostClassifier(),
#               KNeighborsClassifier(), SVC(), LinearDiscriminantAnalysis(),
#               GaussianNB(), XGBClassifier(), LGBMClassifier(),
#               CatBoostClassifier(verbose=0)]
#
# # Models Dictionary
# modelsDict = {x:y for x,y in zip(modelsNames, modelsList)}
#
# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 ,random_state = 1990)
# [x.shape for x in [X_train, X_test, y_train, y_test]]
#
# y_test.value_counts()
#
# # Preparatory work for pipeline construction
# numeric_features = [ 'Age', 'Fare' ]
# categorical_features = [ 'Embarked', 'Sex', 'Pclass' ]
#
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler()) ])
#
#
#
# import time
#
# start = time.time()
# configs_cnt = 0
# # Initiate dataframe to maintain results
# settingsdf = pd.DataFrame({ k : [] for k in ['EncoderName', 'ModelName', 'MeanAUC', 'AUC_est_std']})
#
# for encName, encObj in encodersDict.items():
#
#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('encoder', encObj)])
#
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, numeric_features),
#             ('cat', categorical_transformer, categorical_features)])
#
#     pipesDict = { x : Pipeline([ ('preprocessor', preprocessor), (x, y) ]) for x, y in modelsDict.items()}
#
#     for name, pipe in pipesDict.items():
#
#         configs_cnt += 1
#         print(f'Current Pipeline Setting :\n {encName} == > {name}')
#
#         try:
#             scores = cross_val_score(pipe, X, y, scoring='roc_auc', cv=10)
#
#             settingsdf = settingsdf.append(
#                 {'EncoderName': encName, 'ModelName': name, 'MeanAUC': scores.mean(), 'AUC_est_std': 2 * scores.std()},
#                 ignore_index=True)
#
#             print(f'Roc-Auc : {scores.mean():.3f} :: (+/-) {2 * scores.std():.3f}')
#
#         except Exception as e:
#
#             settingsdf = settingsdf.append(
#                 {'EncoderName': encName, 'ModelName': name, 'MeanAUC': None, 'AUC_est_std': None},
#                 ignore_index=True)
#
#             print(f'Current Pipeline Setting :\n {encName} == > {name} FAILED during training!')
#
#         print(f'\n{50*"_"} ')
#
# end = time.time()
# print(f'Process took {end-start} and tested {configs_cnt} configurations')
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # ScatterPlot for Performance
# settingsdf.isnull().sum()
# settingsdf.plot(kind='scatter',x='MeanAUC',y='AUC_est_std',color='red')
# plt.show()
#
# # Tabular Exploration
# settingsdf.sort_values(by=['MeanAUC','AUC_est_std'], ascending=False).head(20)
#
# settingsdf.groupby('ModelName').mean().sort_values(by=['MeanAUC'], ascending=False)
# settingsdf.groupby('EncoderName').mean().sort_values(by=['MeanAUC'], ascending=False)
# settingsdf.groupby('EncoderName').max().sort_values(by=['MeanAUC'], ascending=False)
#
# # Examine Model
# chart = sns.boxplot(x="ModelName", y="MeanAUC", data=settingsdf)
# chart.set_xticklabels(
#                         chart.get_xticklabels(),
#                         rotation=45,
#                         horizontalalignment='right',
#                         fontweight='light'
#                     )
# plt.show()
#
# # Examine Encoders
# chart = sns.boxplot(x="EncoderName", y="MeanAUC", data=settingsdf)
# chart.set_xticklabels(
#                         chart.get_xticklabels(),
#                         rotation=45,
#                         horizontalalignment='right',
#                         fontweight='light'
#                     )
# plt.show()


# --------------------------------------------------------------------------------------------
# ML CLASSIFICATION MODELS
# --------------------------------------------------------------------------------------------

# Model Names List
modelsNames = ['LogisticRegression','RandomForestClassifier','AdaBoost',
               'KNeighborsClassifier','SVC','LinearDiscriminantAnalysis',
               'GaussianNB', 'XGBClassifier', 'LGBMClassifier',
               'CatBoostClassifier']
# Models List
modelsList = [LogisticRegression(), RandomForestClassifier(), AdaBoostClassifier(),
              KNeighborsClassifier(), SVC(), LinearDiscriminantAnalysis(),
              GaussianNB(), XGBClassifier(), LGBMClassifier(),
              CatBoostClassifier(verbose=0)]

# Models Dictionary
modelsDict = {x:y for x,y in zip(modelsNames, modelsList)}

# Build dictionary of pipelines :: 'ModelName' (key) : Pipeline() (value)
# pipesDict = {x : make_pipeline(preprocessor, y) for x, y in modelsDict.items()}
# pipesDict = {x : make_pipeline(preprocessor, y) for x, y in modelsDict.items()}
pipesDict = { x : Pipeline([('preprocessor', preprocessor),(x,y)]) for x, y in modelsDict.items()}

# cross_val_score tracks only one metric
for name, pipe in pipesDict.items():
    print(f'training {name}')
    scores = cross_val_score(pipe, X, y, scoring = 'roc_auc', cv = 10)
    print(f'Roc-Auc : {scores.mean():.3f} :: (+/-) {2 * scores.std():.3f}')

# # cross_validate can track multiple metrics
# for name, pipe in pipesDict.items():
#     print(f'\n>> training {name}  :: \n')
#     scores = cross_validate(pipe, X, y, cv=5, scoring=('accuracy', 'f1', 'recall', 'precision', 'roc_auc'), return_train_score=True)
#     for k, v in scores.items():
#         print(f'{k} : {v.mean():.3f}')
#     print(50 * '-')



# --------------------------------------------------------------------------------------------
# GRIDSEARCHCV
# --------------------------------------------------------------------------------------------

# Model Names List
modelsNames = ['LogisticRegression','RandomForestClassifier','SVC']
# Models Dictionary
modelsList = [ LogisticRegression(), RandomForestClassifier(), SVC()]
# Models Dictionary
modelsDict = {x : y for x,y in zip(modelsNames, modelsList)}
# Build dictionary of pipelines :: 'ModelName' (key) : Pipeline() (value)
pipesDict = {x : Pipeline([('preprocessor',preprocessor),(x,y)]) for x, y in modelsDict.items()}

# for k,v in pipesDict.items():
#     print(what_params(v))

# Build a dict with grids
gridsDict_ = {
    # 'DummyClassifier' :
    # {
    #     'strategy' : ['stratified','most_frequent','prior','uniform']
    # },
    'LogisticRegression':
    {
        'penalty': [ 'l1', 'l2' ],
        'C': [6, 8],
        'solver': [ 'liblinear' ]
    },
    'RandomForestClassifier':
        {
            'criterion': [ 'gini', 'entropy' ],
            # 'min_samples_leaf': param_range,
            # 'max_depth': param_range,
            # 'min_samples_split': param_range
        }
    ,
    'SVC':
        {
            'kernel': [ 'linear', 'rbf' ],
            'C': [100, 10, 1.0, 0.5, 0.1],
        }
}

# Re-Build the dict of grids, by altering the keys in the dicts so as to match the estimator name
_gridsDict_ = { k : { "__".join([k,kk]) : vv for kk,vv in v.items()} for k,v in gridsDict_.items()}

# Adjust this to get a variety of metrics during CV
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score), 'Recall' : 'recall'}

# Build a dictionary of GridSearchCV objects
gridsDict = {name : GridSearchCV(estimator=pipe,
                                 param_grid=_gridsDict_[name],
                                 # scoring='accuracy',
                                 scoring=scoring, refit='AUC', # use refit to return the best estimator according to refit criterion
                                 cv=10,
                                 return_train_score=True)
             for name, pipe in pipesDict.items()}

# --------------------------------------------------------------------------------------------
# TRAIN PIPES WITH GRIDS
# --------------------------------------------------------------------------------------------

best_acc = 0.0
best_clf = 0
best_gs = ''

for idx, (name, gs) in enumerate(gridsDict.items()):
    print(75 * '=')
    print('\nEstimator: %s' % name)
    # Fit grid search
    gs.fit(X_train, y_train)
    # Best params
    print(50 * '-')
    print('Best params: %s' % gs.best_params_)
    # Best training data accuracy
    print(50 * '-')
    print('Best score (accoding to what was set as refit parameter) : %.3f' % gs.best_score_)
    # Predict on test data with best params
    y_pred = gs.predict(X_test)
    # Test data accuracy of model with best params
    # print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))

    # Print Param report :
    print(50 * '-')
    for param, auc, acc, rec in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_AUC'], gs.cv_results_['mean_test_Accuracy'], gs.cv_results_['mean_test_Recall']):
        print(f'param {param}:: \n>> auc : {auc:.3f}, \n>> accuracy : {acc:.3f}, \n>> recall : {rec:.3f}')

    # Print Best params :
    print(50 * '-')
    print(f'Best params : {gs.best_params_}')

    # Classification Report :
    print(50 * '-')
    print(f'Classification Report : \n{classification_report(y_test, y_pred)}')

    # Confusion Matrix :
    print(50 * '-')
    print(f'Confusion Matrix : \n{pd.DataFrame(confusion_matrix(y_test, y_pred))}')

    # Track best (highest test accuracy) model
    if roc_auc_score(y_test, y_pred) > best_acc:
        best_acc = roc_auc_score(y_test, y_pred)
        best_gs = gs
        best_clf = name

print('\nClassifier with best test set accuracy: %s' % name)

# type(gs)
# dir(gs)
# gs.multimetric_
# gs.classes_
# gs.cv_results_
# pd.DataFrame(gs.cv_results_).to_clipboard()
# gs.cv_results_.keys()
# gs.best_params_
# gs.best_estimator_
# gs.best_score_
# type(gs.best_estimator_)



# --------------------------------------------------------------------------------------------
# HPO WITH RandomizedSearchCV
# --------------------------------------------------------------------------------------------

# from sklearn.model_selection import RandomizedSearchCV
#
# rf_clf_ppl = pipesDict[ 'RandomForestClassifier' ]
#
# scoring = {
#                 'AveragePrecision': 'average_precision',
#                 'Accuracy': 'accuracy',
#                 'F1': 'f1_weighted',
#                 'Recall': 'recall',
#                 'Precision': 'precision',
#                 'ROC': 'roc_auc',
#                 'NLL': 'neg_log_loss'
#                 }
#
# params =  {
#             'RandomForestClassifier__criterion': [ 'gini', 'entropy' ],
#             'RandomForestClassifier__min_samples_leaf': np.arange(4,40),
#             'RandomForestClassifier__max_depth': np.arange(5,40),
#             'RandomForestClassifier__min_samples_split': np.arange(10,30),
#             'RandomForestClassifier__max_features': [ 'sqrt', 'log2', 'auto' ]
#         }
#
# rs = RandomizedSearchCV(rf_clf_ppl, param_distributions=params,
#                         n_iter=100, scoring = scoring, n_jobs=-1,
#                         cv=10, return_train_score=True, refit = 'Accuracy')
#
# rs.fit(X_train,y_train)
#
# rs.cv_results_['mean_test_Accuracy'].mean()
#
# rs.best_estimator_
# rs.best_params_


# --------------------------------------------------------------------------------------------
# HPO WITH OPTUNA :: RANDOM FOREST SIMPLE
# --------------------------------------------------------------------------------------------

# import optuna
#
# def objective(trial):
#
#     rf_clf_ppl = pipesDict[ 'RandomForestClassifier' ]
#
#     params = {
#             'RandomForestClassifier__min_samples_leaf': trial.suggest_int('RandomForestClassifier__min_samples_leaf', 10, 200),
#             'RandomForestClassifier__criterion': trial.suggest_categorical('RandomForestClassifier__criterion',['gini', 'entropy']),
#             'RandomForestClassifier__max_depth': trial.suggest_int('RandomForestClassifier__max_depth', 5, 40),
#             'RandomForestClassifier__min_samples_split': trial.suggest_int('RandomForestClassifier__min_samples_split', 10, 30),
#             'RandomForestClassifier__max_features': trial.suggest_categorical('RandomForestClassifier__max_features',[ 'sqrt', 'log2', 'auto' ])
#         }
#
#     rf_clf_ppl.set_params(**params)
#
#     scoring = {
#                 'AveragePrecision': 'average_precision',
#                 'Accuracy': 'accuracy',
#                 'F1': 'f1_weighted',
#                 'Recall': 'recall',
#                 'Precision': 'precision',
#                 'ROC': 'roc_auc',
#                 'NLL': 'neg_log_loss'
#                 }
#
#     print(params)
#
#     scores = cross_validate(rf_clf_ppl, X_train, y_train,
#                             cv=10,
#                             scoring=scoring,
#                             return_train_score=True
#                             )
#
#     print(pd.DataFrame(scores).iloc[:,2:].transpose().mean(axis=1))
#
#     return - np.mean(scores['test_Accuracy'])
#
#
# study = optuna.create_study()
# # study.optimize(objective, n_trials=50)
# study.optimize(objective, timeout=60)
#
# dir(study)
# study.best_params
# study.best_value
# study.optimize()
# study.trials_dataframe().to_clipboard()


# --------------------------------------------------------------------------------------------
# HPO OPTUNA : XGBOOST
# --------------------------------------------------------------------------------------------

#
# def objective(trial):
#
#     clf_ppl = pipesDict[ 'XGBClassifier' ]
#
#     params = {
#             'XGBClassifier__learning_rate': trial.suggest_uniform('XGBClassifier__learning_rate', 0.05, 1.0) ,
#             'XGBClassifier__colsample_bytree': trial.suggest_uniform('XGBClassifier__colsample_bytree', 0.1, 0.9),
#             'XGBClassifier__max_depth': trial.suggest_int('XGBClassifier__max_depth', 3, 30),
#             'XGBClassifier__min_child_weight': trial.suggest_int('XGBClassifier__min_child_weight', 1, 15),
#             'XGBClassifier__gamma': trial.suggest_uniform('XGBClassifier__gamma', 0.05, 1.0)
#         }
#
#     clf_ppl.set_params(**params)
#
#     scoring = {
#                 'AveragePrecision': 'average_precision',
#                 'Accuracy': 'accuracy',
#                 'F1': 'f1_weighted',
#                 'Recall': 'recall',
#                 'Precision': 'precision',
#                 'ROC': 'roc_auc',
#                 'NLL': 'neg_log_loss'
#                 }
#
#     scores = cross_validate(clf_ppl, X_train, y_train,
#                             cv=10,
#                             scoring=scoring,
#                             return_train_score=True
#                             )
#
#     print(pd.DataFrame(scores).iloc[:,2:].transpose().mean(axis=1))
#
#     overfit_ban = 1
#
#     return - (
#                 scores[ 'test_AveragePrecision' ].mean() / abs(overfit_ban * np.mean(scores[ 'test_AveragePrecision' ] - scores[ 'train_AveragePrecision' ])) + \
#                 scores[ 'test_Accuracy' ].mean() / abs(overfit_ban * np.mean(scores[ 'test_Accuracy' ] - scores[ 'train_Accuracy' ])) + \
#                 scores[ 'test_F1' ].mean() / abs(overfit_ban * np.mean(scores[ 'test_F1' ] - scores[ 'train_F1' ])) + \
#                 scores[ 'test_Recall' ].mean() / abs(overfit_ban * np.mean(scores[ 'test_Recall' ] - scores[ 'train_Recall' ])) + \
#                 scores[ 'test_Precision' ].mean() / abs(overfit_ban * np.mean(scores[ 'test_Precision' ] - scores[ 'train_Precision' ])) + \
#                 scores[ 'test_ROC' ].mean() / abs(overfit_ban * np.mean(scores[ 'test_ROC' ] - scores[ 'train_ROC' ])) - \
#                 scores[ 'test_NLL' ].mean() / abs(overfit_ban * np.mean(scores[ 'test_NLL' ] - scores[ 'train_NLL' ]))
#               )
#
#
# study = optuna.create_study()
# # study.optimize(objective, n_trials=100)
# study.optimize(objective, timeout=60)
#
# study.trials_dataframe()
# study.best_params

# {'XGBClassifier__learning_rate': 0.4350043854173714,
#  'XGBClassifier__colsample_bytree': 0.1959631648501437,
#  'XGBClassifier__max_depth': 13,
#  'XGBClassifier__min_child_weight': 12,
#  'XGBClassifier__gamma': 0.5283289321245073}


# --------------------------------------------------------------------------------------------
# ANOMALY DETECTION APPROACH
# --------------------------------------------------------------------------------------------

# from sklearn.ensemble import IsolationForest
# from sklearn.svm import OneClassSVM
#
# import optuna
# from sklearn.metrics import recall_score, precision_score
#
#
# # Model Names List
# modelsNames = ['IsolationForest','OneClassSVM']
# # Models List
# modelsList = [IsolationForest(),OneClassSVM()]
# # Models Dictionary
# modelsDict = {x:y for x,y in zip(modelsNames, modelsList)}
#
# # Build dictionary of pipelines :: 'ModelName' (key) : Pipeline() (value)
# pipesDict = {x : Pipeline([('preprocessor', preprocessor), (x, y)]) for x, y in modelsDict.items()}
#
# # Build a dict with grids
# gridsDict_ = {
#     'IsolationForest' :
#         {
#             'n_estimators': np.arange(400,600,100),
#             'contamination': [0.3, 0.4],
#             'max_features': np.arange(5, 10, 2),
#             # 'bootstrap': [True, False]
#         }
#     ,
#     'OneClassSVM':
#         {
#             'kernel': [ 'linear', 'rbf', 'poly', 'sigmoid' ]
#         }
# }
#
# # Re-Build the dict of grids, by altering the keys in the dicts so as to match the estimator name
# _gridsDict_ = { k : { "__".join([k,kk]) : vv for kk,vv in v.items()} for k,v in gridsDict_.items()}
#
# # Adjust this to get a variety of metrics during CV
# scoring = {
#     'AveragePrecision': 'average_precision',
#     'Accuracy': 'accuracy',
#     'F1': 'f1_weighted',
#     'Recall': make_scorer(recall_score, average='weighted'),
#     'Precision': make_scorer(precision_score, average='weighted'),
#     'ROC': 'roc_auc'
# }

# STACK OVERFLOW EXAMPLE
# # https://stackoverflow.com/questions/56078831/isolation-forest-parameter-tuning-with-gridsearchcv
# f1sc = make_scorer(f1_score, average='weighted')
#
# # Build a dictionary of GridSearchCV objects
# gridsDict = {name : GridSearchCV(estimator=pipe,
#                                  param_grid=_gridsDict_[name],
#                                  # scoring='accuracy',
#                                  scoring=f1sc,
#                                  refit=True,
#                                  cv=10,
#                                  return_train_score=True)
#              for name, pipe in pipesDict.items()}
#
# mm = gridsDict['IsolationForest']
# mm.fit(X_train, y_train)
#
# ypreds = mm.predict(X_test)
# ypreds[ypreds==1] = 0
# ypreds[ypreds!=0] = 1
#
# pd.crosstab(y_test,ypreds)


#
# def objective(trial):
#
#     clf_ppl = pipesDict[ 'IsolationForest' ]
#
#     params = {
#             'IsolationForest__contamination': trial.suggest_uniform('IsolationForest__contamination', 0.05, 0.4) ,
#             'IsolationForest__n_estimators': trial.suggest_int('IsolationForest__n_estimators', 200, 800),
#             'IsolationForest__max_features': trial.suggest_int('IsolationForest__max_features', 3, 5),
#         }
#
#     clf_ppl.set_params(**params)
#
#     scoring = {
#                 'AveragePrecision': 'average_precision',
#                 'Accuracy': 'accuracy',
#                 'F1': 'f1_weighted',
#                 'Recall': make_scorer(recall_score, average='weighted'),
#                 'Precision': make_scorer(precision_score, average='weighted'),
#                 'ROC': 'roc_auc'
#                 }
#
#     y_train_anomalus = y_train.copy()
#     y_train_anomalus[ y_train_anomalus == 1 ] = -1
#     y_train_anomalus[ y_train_anomalus == 0 ] = 1
#
#     scores = cross_validate(clf_ppl, X_train, y_train_anomalus,
#                             cv=10,
#                             scoring=scoring,
#                             return_train_score=True
#                             )
#
#     print(pd.DataFrame(scores).iloc[:,2:].transpose().mean(axis=1))
#
#     overfit_ban = 1
#
#     return - (
#                 1 * scores[ 'test_AveragePrecision' ].mean() / abs(overfit_ban * np.mean(scores[ 'test_AveragePrecision' ] - scores[ 'train_AveragePrecision' ])) + \
#                 1 * scores[ 'test_Accuracy' ].mean() / abs(overfit_ban * np.mean(scores[ 'test_Accuracy' ] - scores[ 'train_Accuracy' ])) + \
#                 2 * scores[ 'test_F1' ].mean() / abs(overfit_ban * np.mean(scores[ 'test_F1' ] - scores[ 'train_F1' ])) + \
#                 1 * scores[ 'test_Recall' ].mean() / abs(overfit_ban * np.mean(scores[ 'test_Recall' ] - scores[ 'train_Recall' ])) + \
#                 1 * scores[ 'test_Precision' ].mean() / abs(overfit_ban * np.mean(scores[ 'test_Precision' ] - scores[ 'train_Precision' ])) + \
#                 20 * scores[ 'test_ROC' ].mean() / abs(overfit_ban * np.mean(scores[ 'test_ROC' ] - scores[ 'train_ROC' ]))
#               )
#
#
# study = optuna.create_study()
# # study.optimize(objective, n_trials=100)
# study.optimize(objective, timeout=300)
#
# study.trials_dataframe().to_clipboard()
# study.best_params
#
#
#
# ypreds = mm.predict(X_test)
# ypreds[ypreds==1] = 0
# ypreds[ypreds!=0] = 1
#
# pd.crosstab(y_test,ypreds)
#
#
#
#
#
#
#
#
# best_acc = 0.0
# best_clf = 0
# best_gs = ''
#
# for idx, (name, gs) in enumerate(gridsDict.items()):
#     print(75 * '=')
#     print()
#     print('\nEstimator: %s' % name)
#     # Fit grid search
#     gs.fit(X_train, y_train)
#     # Best params
#     print(50 * '-')
#     print('Best params: %s' % gs.best_params_)
#     # Best training data accuracy
#     print(50 * '-')
#     print('Best score (accoding to what was set as refit parameter) : %.3f' % gs.best_score_)
#     # Predict on test data with best params
#     y_pred = gs.predict(X_test)
#
#     # Print Param report :
#     # print(50 * '-')
#     # for param, auc, acc, rec in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_AUC'], gs.cv_results_['mean_test_Accuracy'], gs.cv_results_['mean_test_Recall']):
#     #     print(f'param {param}:: \n>> auc : {auc:.3f}, \n>> accuracy : {acc:.3f}, \n>> recall : {rec:.3f}')
#
#     # Print Best params :
#     print(50 * '-')
#     print(f'Best params : {gs.best_params_}')
#
#     # Classification Report :
#     print(50 * '-')
#     print(f'Classification Report : \n{classification_report(y_test, y_pred)}')
#
#     # Confusion Matrix :
#     print(50 * '-')
#     print(f'Confusion Matrix : \n{pd.DataFrame(confusion_matrix(y_test, y_pred))}')
#
#     # Track best (highest test accuracy) model
#     if accuracy_score(y_test, y_pred) > best_acc:
#         best_acc = accuracy_score(y_test, y_pred)
#         best_gs = gs
#         best_clf = name
#
#
# print('\nClassifier with best test set accuracy: %s' % name)
#
#
# type(gs)
# dir(gs)
# gs.multimetric_
# gs.classes_
# gs.cv_results_
# pd.DataFrame(gs.cv_results_).to_clipboard()
# gs.cv_results_.keys()
# gs.best_params_
# gs.best_estimator_
# gs.best_score_
# type(gs.best_estimator_)



# --------------------------------------------------------------------------------------------
# GRIDSEARCH RESULTS TO DATAFRAME
# --------------------------------------------------------------------------------------------
#
# aa = pd.DataFrame(gs.cv_results_).set_index('params')
#
# # SELECTION METHOD 1
# column_names_selection = ['mean_test_AUC','mean_train_AUC',
#                           'mean_test_Accuracy','mean_train_Accuracy',
#                           'mean_test_Recall','mean_train_Recall']
#
# # SELECTION METHOD 2
# # mask = pd.Series(aa.columns).str.contains("|".join(["mean"])).tolist()
# mask = pd.Series(aa.columns).str.contains("mean").tolist()
# bb = aa.loc[:,mask]
# bb.to_clipboard()
#
# # SELECTION METHOD 3
# cv_results = aa.filter(like='mean', axis=1).mean(axis=0).to_frame(name='Metrics')
#
# # --------------------------------------------------------------------------------------------
# # CREATE RESULTS REPORT
# # --------------------------------------------------------------------------------------------
#
# import os
# from datetime import datetime
# from mdutils.mdutils import MdUtils
# import matplotlib.pyplot as plt
# import scikitplot as skplt
#
# def create_directory(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
# # User Input
# directory_artifacts = 'track_experiment_artifacts'
# directory_reports = 'track_experiment_reports'
# experiment_name = 'experiment_rf'
#
# artifacts_list = [
#     'classification_report_html',
#     'classification_report_csv',
#     'roc_png']
#
#
# def track_experiment(experiment_name,directory_artifacts,directory_reports):
#
#     # Adjust User Input
#     directory_execution = os.getcwd()
#     timetag = datetime.now().isoformat(timespec='seconds').replace("-", "").replace("T", "").replace(":", "")
#     experiment_name = experiment_name + timetag
#
#     # Create directory to save artifacts based on experiment name
#     experiment_directory = os.path.join(directory_execution, directory_artifacts, experiment_name)
#     create_directory(experiment_directory)
#
#     # Create dict with pathfiles for later use
#     artifact_paths = {k: os.path.join(directory_execution, directory_artifacts, experiment_directory, k + "." + k.rpartition("_")[ 2 ]) for k
#                       in artifacts_list}
#
#     # classification_report_html
#     pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().to_html(artifact_paths['classification_report_html'])
#
#     # classification_report_csv
#     pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().to_csv(artifact_paths['classification_report_csv'])
#
#     # Create directory to save reports based on directory_reports
#     reports_directory = os.path.join(directory_execution, directory_reports)
#     create_directory(reports_directory)
#
#     # Create MD Report
#     mdFile = MdUtils(file_name=os.path.join(directory_execution, directory_reports, experiment_name), title= experiment_name)
#     mdFile.new_header(level=1, title='Overview')
#
#     # ----------- Add a Table ::  Classification Report
#     mdFile.new_header(level=2, title='Classification Report : ')
#
#     clf_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().reset_index()
#     list_of_strings = clf_report.columns.tolist()
#     for i, qq in clf_report.iterrows():
#         list_of_strings.extend([ str(round(vv, 4)) if not isinstance(vv, str) else vv for vv in qq.values ])
#     mdFile.new_line()
#     mdFile.new_table(columns=clf_report.shape[ 1 ], rows=clf_report.shape[ 0 ] + 1, text=list_of_strings,
#                      text_align='center')
#
#     mdFile.create_md_file()
#
#     # ----------- Add a png :: confusion_matrix
#     mdFile.new_header(level=2, title='confusion_matrix : ')
#
#     plot_name = 'confusion_matrix'
#     plot_path = os.path.join(directory_execution, directory_artifacts, experiment_name, plot_name + '.png')
#     preds = gs.best_estimator_.predict(X_test)
#     skplt.metrics.plot_confusion_matrix(y_true=y_test, y_pred=preds).get_figure().savefig(plot_path)
#     content_string = f"![{plot_name}]({plot_path})"
#     mdFile.new_paragraph(content_string)
#
#     # ----------- Add a png : plot_precision_recall
#     # mdFile.new_header(level=2, title='plot_precision_recall : ')
#     #
#     # plot_name = 'plot_precision_recall'
#     # plot_path = os.path.join(directory_execution, directory_artifacts, experiment_name, plot_name + '.png')
#     # probas = gs.best_estimator_.predict_proba(X_test)
#     # skplt.metrics.plot_precision_recall(y_true=y_test, y_probas=probas)
#     # content_string = f"![{plot_name}]({plot_path})"
#     # mdFile.new_paragraph(content_string)
#
#     mdFile.create_md_file()
#
#
# track_experiment(experiment_name,directory_artifacts,directory_reports)
#



# --------------------------------------------------------------------------------------------
# CROSS VALIDATE FUNCTIONS
# --------------------------------------------------------------------------------------------

# # USAGE CROSS_VAL_SCORE
# scores = cross_val_score(gs.best_estimator_, X, y, scoring=make_scorer(roc_auc_score), cv=10)
# print(f'Accuracy : {scores.mean():.2f} :: (+/-) {2 * scores.std():.2f}')
#
# # USAGE CROSS_VALIDATE
# scores = cross_validate(gs.best_estimator_, X, y, cv=5, scoring=('accuracy', 'f1', 'recall', 'precision', 'roc_auc'), return_train_score=True)
# for k, v in scores.items():
#     print(f'{k} : {v.mean():.3f}')


# --------------------------------------------------------------------------------------------
# MLFLOW
# --------------------------------------------------------------------------------------------

'''
______________________________________________________________________________________________
run while in terminal :
mlflow ui
OR
mlflow server                                     
______________________________________________________________________________________________
'''

import mlflow
from datetime import datetime
import pickle
from joblib import dump, load
import matplotlib.pyplot as plt
import shutil

best_acc = 0.0
best_clf = 0
best_gs = ''

# xdf = pd.DataFrame(gs.cv_results_, index=json.dumps(gs.cv_results_['params'])).filter(like='mean',axis=1)
xdf = pd.DataFrame(gs.cv_results_).filter(like='mean',axis=1)
xdf['params'] = gs.cv_results_['params']

# for idx, row in xdf.iterrows():
#     print(row)
#     for dd in gs.cv_results_['params']:
#         for k,v in dd.items():
#         mlflow.log_param(k,v)
#
# mlflow.log_metric('best_score', gs.best_score_)

mlflow.start_run()
mlflow.end_run()
mlflow.delete_run('23b57fb5161745fabd53879a5718dc27')

dir(mlflow)

mlflow.set_tracking_uri('file:///C:/Users/Takis/Google%20Drive/_projects_/_python_examples_/mlruns')
mlflow.get_artifact_uri()
mlflow.active_run().info
mlflow.search_runs()


mlflow_artifacts_path = r'C:\Users\Takis\Google Drive\_projects_\_python_examples_\mlflow_artifacts'
shutil.rmtree(mlflow_artifacts_path)
os.makedirs(mlflow_artifacts_path)

mlflow.set_experiment('malakismeno')
mlflow.get_experiment_by_name('malakismeno')
mlflow.get_experiment('6')
mlflow.delete_experiment('4')

mlflow.active_run().info
mlflow.get_run(mlflow.active_run().info.run_id).data

for idx, (name, gs) in enumerate(gridsDict.items()):

    timetag = datetime.now().isoformat(timespec='seconds')
    modeltag = timetag + "_" + name
    modelfolder = timetag + "_" + name + ".txt"

    with mlflow.start_run(nested=True):

        gs.fit(X_train, y_train)
        mlflow.log_metric('best score',gs.best_score_)
        y_pred = gs.predict(X_test)

        # Print Param report :
        for param, auc, acc, rec in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_AUC'], gs.cv_results_['mean_test_Accuracy'], gs.cv_results_['mean_test_Recall']):
            print(f'param {param}:: \n>> auc : {auc:.3f}, \n>> accuracy : {acc:.3f}, \n>> recall : {rec:.3f}')

        # Print Best params :
        print(f'Best params : {gs.best_params_}')
        for k, v in gs.best_params_.items():
            mlflow.log_param(k,v)

        # GridSearchCV results
        temp_artifact = pd.DataFrame(gs.cv_results_)
        temp_filename = os.path.join(mlflow_artifacts_path,'gridsearchcv.html')
        temp_artifact.to_html(temp_filename)

        # Confusion Matrix :
        temp_artifact = pd.DataFrame(confusion_matrix(y_test, y_pred))
        temp_filename = os.path.join(mlflow_artifacts_path,'confusion_matrix.html')
        temp_artifact.to_html(temp_filename)

        # Classification Report :
        temp_artifact = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        temp_filename = os.path.join(mlflow_artifacts_path, 'classification_report.html')
        temp_artifact.to_html(temp_filename)

        # ROC Curve
        plot_roc_curve(gs.best_estimator_, X_test, y_test)
        temp_filename = os.path.join(mlflow_artifacts_path,'roc.png')
        plt.savefig(temp_filename)

        # Save model
        temp_filename = 'model.joblib'
        temp_filename = temp_filename.replace('-',"_").replace(':',"_")
        temp_filename = os.path.join(mlflow_artifacts_path, temp_filename)
        dump(gs.best_estimator_, temp_filename)

        # Save artifacts folder
        mlflow.log_artifacts(mlflow_artifacts_path)

        if roc_auc_score(y_test, y_pred) > best_acc:
            best_acc = roc_auc_score(y_test, y_pred)
            best_gs = gs
            best_clf = name


# for idx, (name, gs) in enumerate(gridsDict.items()):
#
#     timetag = datetime.now().isoformat(timespec='seconds')
#     modeltag = timetag + "_" + name
#     modelfolder = timetag + "_" + name + ".txt"
#
#     mlflow.start_run(run_name=name)
#
#     gs.fit(X_train, y_train)
#     mlflow.log_metric('best score',gs.best_score_)
#     y_pred = gs.predict(X_test)
#
#     # Print Param report :
#     for param, auc, acc, rec in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_AUC'], gs.cv_results_['mean_test_Accuracy'], gs.cv_results_['mean_test_Recall']):
#         print(f'param {param}:: \n>> auc : {auc:.3f}, \n>> accuracy : {acc:.3f}, \n>> recall : {rec:.3f}')
#
#     # Print Best params :
#     print(f'Best params : {gs.best_params_}')
#     for k, v in gs.best_params_.items():
#         mlflow.log_param(k,v)
#
#     # GridSearchCV results
#     temp_artifact = pd.DataFrame(gs.cv_results_)
#     temp_filename = os.path.join(mlflow_artifacts_path,'gridsearchcv.html')
#     temp_artifact.to_html(temp_filename)
#
#     # Confusion Matrix :
#     temp_artifact = pd.DataFrame(confusion_matrix(y_test, y_pred))
#     temp_filename = os.path.join(mlflow_artifacts_path,'confusion_matrix.html')
#     temp_artifact.to_html(temp_filename)
#
#     # Classification Report :
#     temp_artifact = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
#     temp_filename = os.path.join(mlflow_artifacts_path, 'classification_report.html')
#     temp_artifact.to_html(temp_filename)
#
#     # ROC Curve
#     plot_roc_curve(gs.best_estimator_, X_test, y_test)
#     temp_filename = os.path.join(mlflow_artifacts_path,'roc.png')
#     plt.savefig(temp_filename)
#
#     # Save model
#     temp_filename = 'model.joblib'
#     temp_filename = temp_filename.replace('-',"_").replace(':',"_")
#     temp_filename = os.path.join(mlflow_artifacts_path, temp_filename)
#     dump(gs.best_estimator_, temp_filename)
#
#     # Save artifacts folder
#     mlflow.log_artifacts(mlflow_artifacts_path)
#
#     # End current run
#     mlflow.end_run()
#
#     if roc_auc_score(y_test, y_pred) > best_acc:
#         best_acc = roc_auc_score(y_test, y_pred)
#         best_gs = gs
#         best_clf = name


# --------------------------------------------------------------------------------------------
# HELPER FUNCTION TO CLEAR RUNS
# --------------------------------------------------------------------------------------------

def clear_mlflow_logs():
    mlfdf = mlflow.search_runs()
    for ee in mlfdf.run_id.unique():
        mlflow.delete_run(ee)
    print('MLFlow metadata was successfully erased')
    print(mlflow.search_runs())

clear_mlflow_logs()

mlflow.search_runs()
