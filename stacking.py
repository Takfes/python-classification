# ---------------------------------------------
# Imports
# ---------------------------------------------
import pandas as pd
import numpy as np
from joblib import dump, load
import os

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier

# from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score, plot_roc_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV, cross_val_predict, KFold

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler
import category_encoders as ce

# ---------------------------------------------
# Set directories
# ---------------------------------------------
os.chdir(r'C:\Users\Takis\Google Drive\_projects_\python-ml')
# os.chdir(r'/home/takis/Desktop/sckool/ml_flow_project')
os.getcwd()

# ---------------------------------------------
# Load data
# ---------------------------------------------
# df = pd.read_csv('http://bit.ly/kaggletrain')
df = pd.read_csv('titanic.csv')
df.shape
df.dtypes
df.isnull().sum()

# ---------------------------------------------
# Split data
# ---------------------------------------------

numeric_features = ['Age', 'Fare']
categorical_features = ['Embarked', 'Sex', 'Pclass']
features = numeric_features + categorical_features

X = df[features]
y = df['Survived']

y.value_counts()
y.value_counts(normalize=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 ,random_state = 1990)
[x.shape for x in [X_train, X_test, y_train, y_test]]

y_test.value_counts()


# ---------------------------------------------
# List preprocessing components
# ---------------------------------------------

num_trs_1 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler_standard', StandardScaler())])

num_trs_2 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler_robust',RobustScaler())])

cat_trs_1 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('enc_ohe_do', OneHotEncoder(drop='first'))])

cat_trs_2 = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('enc_ohe', OneHotEncoder())])


# ---------------------------------------------
# List hyperparameter grids
# ---------------------------------------------

# Logistic Regression parameters
lr_params = {
    'penalty' : 'l2',
    'C':0.5
}

# KNN Parameters
knn_params = {
    'n_neighbors':8
}

# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     #'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.03,
    'probability' : True
    }

# Define hyperparams list
paramsList = [lr_params, knn_params, rf_params, et_params,
                  ada_params,gb_params, svc_params]


# ---------------------------------------------
# List models
# ---------------------------------------------

modelsList = [LogisticRegression(), KNeighborsClassifier(),
              RandomForestClassifier(), ExtraTreesClassifier(),
              AdaBoostClassifier(),GradientBoostingClassifier(),SVC()]


# ---------------------------------------------
# Process loop
# ---------------------------------------------

cv_scores_list = []
list_X_train = []
list_X_test = []
base_learners = {}

for idx, (model,param) in enumerate(zip(modelsList,paramsList)):
    for num in [num_trs_1]:
        for cat in [cat_trs_1]:
#     for num in [num_trs_1,num_trs_2]:
#         for cat in [cat_trs_1,cat_trs_2]:
        
            # ---------------------------------------------
            # Messaging
            # ---------------------------------------------
            print(f'training : {type(model).__name__}... \n')
            print(f'hyperparams : {param} \n')
            print(f'numeric preprocessing : {dir(num.named_steps)} \n')
            print(f'categorical preprocessing : {dir(cat.named_steps)} \n')            
        
        
            # ---------------------------------------------
            # Define preprocessing pipeline component
            # ---------------------------------------------
            preprocessor = ColumnTransformer(transformers=[
                ('num', num, numeric_features),
                ('cat', cat, categorical_features)])
            
            
            # ---------------------------------------------
            # Define the entire pipeline Preprocessor & Model & Hyperparams
            # ---------------------------------------------
            temp_pipe = Pipeline([('prepr',preprocessor),('clf',model.set_params(**param))])
            base_pipe = Pipeline([('prepr',preprocessor),('clf',model.set_params(**param))])

            # ---------------------------------------------
            # fit data
            # ---------------------------------------------
            temp_pipe.fit(X_train,y_train)
            y_pred_labels = temp_pipe.predict(X_test)
            y_pred_probas = temp_pipe.predict_proba(X_test)[:,1]
            
            base_learners[type(model).__name__] = (y_pred_labels,y_pred_probas)
            
            print(f'Accuracy : {accuracy_score(y_test, y_pred_labels)}')
            print(f'F1 score : {f1_score(y_test, y_pred_labels)}')
            print(f'Rrecision : {precision_score(y_test, y_pred_labels)}')
            print(f'Recall : {recall_score(y_test, y_pred_labels)}')
            print(f'ROC : {roc_auc_score(y_test, y_pred_labels)}')

#             [round(fun(y_test, y_pred_labels),2) for fun in [accuracy_score,f1_score,precision_score]]
            
            # ---------------------------------------------
            # Scoring
            # ---------------------------------------------
#             scoring = {
#                 'AveragePrecision': 'average_precision',
#                 'Accuracy': 'accuracy',
#                 'F1': 'f1_weighted',
#                 'Recall': 'recall',
#                 'Precision': 'precision',
#                 'ROC': 'roc_auc',
#                 'NLL': 'neg_log_loss'
#                 }
    
    
            # ---------------------------------------------
            # cross_val_score
            # ---------------------------------------------
#             scores = cross_val_score(temp_pipe, X_train, y_train, scoring = 'roc_auc', cv = 3)
#             print(f'Roc-Auc : {scores.mean():.3f} :: (+/-) {2 * scores.std():.3f}')
#             print(50*'-')
            
    
            # ---------------------------------------------
            # cross_validate :: can track multiple metrics
            # ---------------------------------------------
#             scores = cross_validate(temp_pipe, X_train, y_train, cv=3, scoring=('accuracy', 'f1', 'recall', 'precision', 'roc_auc'), return_train_score=True)
            
    
            # ---------------------------------------------
            # track cv results
            # ---------------------------------------------
#             temp_df = pd.DataFrame(scores)
#             temp_df['model'] = type(model).__name__
#             temp_df['num'] = '->'.join(dir(num.named_steps))
#             temp_df['cat'] = '->'.join(dir(cat.named_steps))
            
#             cv_scores_list.append(temp_df)
    
    
            # ---------------------------------------------
            # RandomizedGridSearch
            # ---------------------------------------------
#             rs = RandomizedSearchCV(temp_pipe, param_distributions=param,
#                         n_iter=10, scoring = scoring, n_jobs=-1,
#                         cv=3, return_train_score=True, refit = 'ROC')
            
#             rs.fit(X_train,y_train)
    
    
            # ---------------------------------------------
            # perform Kfold cross val
            # ---------------------------------------------
            NFOLDS = 5
            kf = KFold(n_splits= NFOLDS)
            
            ntrain = X_train.shape[0]
            ntest = X_test.shape[0]
            
            oof_train = np.zeros((ntrain,))
            oof_test = np.zeros((ntest,))
            oof_test_skf = np.empty((NFOLDS, ntest))
            
            for i, (train_index, test_index) in enumerate(kf.split(X_train)):
                
                print(f'Base model CV training : {i} \n')
                # where to train
                x_tr = X_train.iloc[train_index,:]
                # target variable for train
                y_tr = y_train.iloc[train_index]
                # where to score
                x_te = X_train.iloc[test_index,:]
                
                base_pipe.named_steps['clf'].set_params(**param)
                base_pipe.fit(x_tr, y_tr)
                
                # gradually compile what will be a column in the stacked_X_train ; one column per candidate model
                oof_train[test_index] = base_pipe.predict_proba(x_te)[:,1].tolist()
                # in terms of test, we will derive as many predictions in the test set as the folds
                # ie for every fitted model on the training folds, predict the entire test dataframe 
                oof_test_skf[i, :] = base_pipe.predict_proba(X_test)[:,1].tolist()

            # average across the predictions derived from the various kfold-trained models to derive one column in the stacked_X_test set
            oof_test[:] = oof_test_skf.mean(axis=0)
            
            assert X_train.shape[0] == oof_train.shape[0]
            assert X_test.shape[0] == oof_test.shape[0]
            
            # track the new columns (predicted probas) across the various models
            list_X_train.append(oof_train)
            list_X_test.append(oof_test)


# ---------------------------------------------
# Develop meta model
# ---------------------------------------------

stacked_X_train = pd.concat([pd.Series(x) for x in list_X_train],axis=1)
stacked_X_test = pd.concat([pd.Series(x) for x in list_X_test],axis=1)
            
meta_columns = [type(x).__name__ for x in modelsList]
stacked_X_train.columns = meta_columns
stacked_X_test.columns = meta_columns

meta_grid = {'n_estimators': [100,300,500,700],
             'max_depth': [6,8,10],
             'learning_rate': [0.002,0.05,0.1],
             'colsample_bytree': [0.5,0.8],
             'gamma': [1,2,3],
             'min_child_weight': [1,2,3],
             'subsample': [0.6,0.8,1]}

metamodel = XGBClassifier()           

scoring = {
    'AveragePrecision': 'average_precision',
    'Accuracy': 'accuracy',
    'F1': 'f1_weighted',
    'Recall': 'recall',
    'Precision': 'precision',
    'ROC': 'roc_auc',
    'NLL': 'neg_log_loss'
    }

rs = RandomizedSearchCV(estimator = metamodel, param_distributions = meta_grid, n_iter = 20, scoring = scoring, refit = 'ROC', cv = 5)
rs.fit(stacked_X_train,y_train)

metafit = rs.best_estimator_

y_meta_labels = metafit.predict(stacked_X_test)
y_meta_probas = metafit.predict_proba(stacked_X_test)[:,1]

print(f'Accuracy : {accuracy_score(y_test, y_meta_labels)}')
print(f'F1 score : {f1_score(y_test, y_meta_labels)}')
print(f'Rrecision : {precision_score(y_test, y_meta_labels)}')
print(f'Recall : {recall_score(y_test, y_meta_labels)}')
print(f'ROC : {roc_auc_score(y_test, y_meta_labels)}')


# ---------------------------------------------
# Evaluate individually trained base models
# ---------------------------------------------

for k,(v,p) in base_learners.items():
    
    print(f'Evaluate base model : {k}')

    print(f'Accuracy : {accuracy_score(y_test, v)}')
    print(f'F1 score : {f1_score(y_test, v)}')
    print(f'Rrecision : {precision_score(y_test, v)}')
    print(f'Recall : {recall_score(y_test, v)}')
    print(f'ROC : {roc_auc_score(y_test, v)}')
    print()

    
# ---------------------------------------------
# Evaluation plots
# ---------------------------------------------
    
%matplotlib inline
import matplotlib.pyplot as plt    
from sklearn.metrics import precision_recall_curve

# Define which clf to use
y_scores=base_learners['ExtraTreesClassifier'][1]
# y_score=y_meta_probas

prec, rec, tre = precision_recall_curve(y_test, y_scores)

def plot_prec_recall_vs_tresh(precisions, recalls, thresholds):
    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(thresholds, precisions[:-1], label="Precisions")
    plt.plot(thresholds, recalls[:-1], "#424242", label="Recalls")
    plt.ylabel('Level of Precision and Recall', fontsize=12)
    plt.title('Precision and Recall Scores as a function of the decision threshold', fontsize=12)
    plt.xlabel('Thresholds', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.ylim([0,1])
    plt.axvline(x=0.47, linewidth=3, color='#0B3861')

plot_prec_recall_vs_tresh(prec, rec, tre)
plt.show()
    


from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

# Define which clf to use
y_scores=base_learners['ExtraTreesClassifier'][1]
# y_score=y_meta_probas

clf_y, clf_x = calibration_curve(y_test, y_scores, n_bins=20)

def plot_calibration_curve(clf_y, clf_x):
    fig, ax = plt.subplots()
    # only these two lines are calibration curves
    plt.plot(clf_x,clf_y, marker='o', linewidth=1, label='clf')
#     plt.plot(clf_x,clf_y, marker='o', linewidth=1, label='clf')
    # reference line, legends, and axis labels
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle('Calibration plot')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability in each bin')
    plt.legend()
    plt.show()
    

# ---------------------------------------------
# Probability Calibration
# ---------------------------------------------

# Explicitly fit metamodel classifier based on optimal params from grid search
metamodel_wparams = metamodel.set_params(**rs.best_params_)
metamodel_wparams.fit(stacked_X_train.values,y_train)

# Calibrate metamodel
from sklearn.calibration import CalibratedClassifierCV

clbr1 = CalibratedClassifierCV(metamodel_wparams, method='sigmoid', cv='prefit')
clbr1.fit(stacked_X_test.values,y_test)

clbr2 = CalibratedClassifierCV(metamodel_wparams, method='isotonic', cv='prefit')
clbr2.fit(stacked_X_test.values,y_test)

def plot_calibration_curve_multi(modellist):
    fig, ax = plt.subplots()
    for i,m in enumerate(modellist):
        y_scores = m.predict_proba(stacked_X_test.values)[:,1]
        clf_y, clf_x = calibration_curve(y_test, y_scores, n_bins=20)
        plt.plot(clf_y, clf_x, marker='o', linewidth=1, label='clf'+str(i))
    
    # reference line, legends, and axis labels
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle('Calibration plot')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability in each bin')
    plt.legend()
    plt.show()

plot_calibration_curve_multi([metamodel_wparams,clbr1,clbr2])
