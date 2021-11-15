
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from xverse.ensemble import VotingSelector


# ---------------------------------------------
# sklearn model wrapper
# ---------------------------------------------

def my_RandomForestClassifier(option='explicit'):
    
    if option in ['estimator','params','param_grid','excplicit']:
        raise(f"option {option} not in 'estimator','params','param_grid','excplicit'")
    
    name = 'RandomForestClassifier'
    
    estimator = RandomForestClassifier()
    
    params = {
        'n_jobs': -1,
        'n_estimators': 500,
        'warm_start': True, 
         #'max_features': 0.2,
        'max_depth': 6,
        'min_samples_leaf': 2,
        'max_features' : 'sqrt',
        'verbose': 0
    }
    
    param_grid = {
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600]
    }

    
    if option == 'estimator':
        return estimator
    elif option == 'params':
        return estimator.set_params(**params)
    elif option == 'param_grid':
        return estimator.set_params(**param_grid)
    else :
        return name, estimator, params, param_grid

    
## USAGE #########################################

from modelaki import * 
name,model,pg,ppg = my_RandomForestClassifier()
my_RandomForestClassifier('params')
my_RandomForestClassifier('param_grid')

## OTHER CLASSIFIERS TO BE DONE : ###############
    
# def my_AdaBoostClassifier():
    
#     estimator = AdaBoostClassifier()
    
#     params = {
#         'n_estimators': 500,
#         'learning_rate' : 0.75
#     }
    
#     param_grid = {
#     }

#     return estimator, params, param_grid


# def my_ExtraTreesClassifier():
    
#     estimator = ExtraTreesClassifier()
    
#     params = {
#             'n_jobs': -1,
#             'n_estimators':500,
#             #'max_features': 0.5,
#             'max_depth': 8,
#             'min_samples_leaf': 2,
#             'verbose': 0
#     }
    
#     param_grid = {
#     }

#     return estimator, params, param_grid


# def my_GradientBoostingClassifier():
    
#     estimator = GradientBoostingClassifier()
    
#     params = {
#             'n_estimators': 500,
#              #'max_features': 0.2,
#             'max_depth': 5,
#             'min_samples_leaf': 2,
#             'verbose': 0
#     }
    
#     param_grid = {
#     }

#     return estimator, params, param_grid



# ---------------------------------------------
# Custom Transformers to extend pipeline components
# ---------------------------------------------


###### BRING IN DATA FOR EXPERIMENTATION #####
    
import pandas as pd    
raw = pd.read_csv('titanic.csv')
raw.shape

numeric_features = ['Age', 'Fare']
categorical_features = ['Embarked', 'Sex', 'Pclass']
target = ['Survived']
features = numeric_features + categorical_features

df = raw[features+target].dropna()

X = df[features]
y = df['Survived']

#############################################

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline

# ---------------------------------------------
# Select and return only a subset of columns
# ---------------------------------------------

class ColumnSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names ):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self._feature_names ] 
    
# ---------------------------------------------
# Provide your preprocessing function in here
# ---------------------------------------------    

class CustomTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, custom_function):
        pass
    
    def transform(self, X, *_):
        Xy = X.copy()
        '''
        fill in code to do stuff here
        return custom_function(Xy)
        '''
        return custom_function(Xy)

    def fit(self, X, colname):
        return self

# ---------------------------------------------
# Select top X most important features
# ---------------------------------------------
    
class FeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self,max_features = None, override_features = None, minimum_votes = 0):
        self.max_features = max_features
        self.minimum_votes = minimum_votes
        
    def transform(self, X, *_):
        return X[self.selected_features]

    def fit(self, X, colname):
        
        if (not self.max_features) or (self.max_features > X.shape[1]) or (self.max_features < 0):
            print(f'"max_feature" was not set (or not set correctly) ; process will continue with all the avaiable features')
            self.max_features = X.shape[1]
        
        voting_selector = VotingSelector(minimum_votes=self.minimum_votes)
        voting_selector.fit(X, y)
        selected_features = voting_selector.feature_votes_.Variable_Name.head(self.max_features).to_list()
        
        self.votes = voting_selector.feature_votes_
        self.importances = voting_selector.feature_importances_
        self.selected_features = selected_features
        
        return self

### USAGE

vs = FeatureSelector(8)
vs.fit(X,y)
vs.votes
vs.importances
vs.selected_features    
vs.fit_transform(X,y)

    
    
# ---------------------------------------------
# Generate pairwise interactions across the most important features
# ---------------------------------------------    

class InteractionMaker(BaseEstimator, TransformerMixin):
    
    def __init__(self, top_n_features = 5):
        self.top_n_features = top_n_features
    
    def num_to_woe(colname):
        
        woe = WOE()
        woe_col = woe.fit_transform(X[[colname]], y).iloc[:,0]
        mapper = {woe_value:colname+'_bin_'+str(idx) for idx,woe_value in enumerate(woe.woe_bins.get(colname).values())}    
        
        return woe_col.map(mapper)

    
    def transform(self, X, *_):
        
        Xy = X.copy()
        self.newcols = []
        
        for col1,col2 in self.interactions_list:
    
            temp_col_name = col1 + '_' + col2

            if (col1 in categorical_features) and (col2 in categorical_features): 
                Xy[temp_col_name] = Xy[col1] + '_' + Xy[col2]
            elif (col1 in numerical_features) and (col2 in numerical_features):
                Xy[temp_col_name] = num_to_woe(col1) + '_' + num_to_woe(col2)
            elif (col1 in categorical_features) and (col2 in numerical_features):
                Xy[temp_col_name] = Xy[col1] + '_' + num_to_woe(col2)
            elif (col1 in numerical_features) and (col2 in categorical_features):
                Xy[temp_col_name] = num_to_woe(col1) + '_' + Xy[col2]

            self.newcols.append(temp_col_name)

        ohe = OneHotEncoder(drop='first', sparse = False)
        ct = ColumnTransformer(transformers=[('ohe_enc_intr', ohe, newcols)], remainder='passthrough')
        Xy = ct.fit_transform(Xy)
        
        return Xy

    
    def fit(self, X, colname):
        
        vs = FeatureSelector(self.top_n_features)
        vs.fit(X,y)
        
        self.votes = vs.votes
        self.importances = vs.importances
        self.selected_features = selection = vs.selected_features
        
        self.numerical_features = list(X[selection].select_dtypes(include=[np.number]).columns)
        self.categorical_features = list(X[selection].select_dtypes(include=["category","object"]).columns)
        self.interactions_list = list(itertools.combinations(selection,2))
        
        return self  

### USAGE
    
X.shape
im = InteractionMaker(13)
gg = im.fit_transform(X,y)

X.shape
gg.shape
im.votes

im.numerical_features
im.categorical_features
im.selected_features
im.interactions_list


###### Make interactions #####

# import numpy as np
# import itertools
# from xverse.transformer import WOE

# def num_to_woe(colname):
#     woe = WOE()
#     woe_col = woe.fit_transform(X[[colname]], y).iloc[:,0]
#     mapper = {woe_value:colname+'_bin_'+str(idx) for idx,woe_value in enumerate(woe.woe_bins.get(colname).values())}
#     return woe_col.map(mapper)

# # usage
# num_to_woe('Age')
# # useful
# # clf.iv_df clf.woe_df clf.woe_bins

# selection = vs.selected_features
# numerical_features = list(X[selection].select_dtypes(include=[np.number]).columns)
# categorical_features = list(X[selection].select_dtypes(include=["category","object"]).columns)
# interactions_list = list(itertools.combinations(selection,2))

# X.shape # (712, 5)
# X.columns
# len(interactions_list) # 10

# newcols = []

# for col1,col2 in interactions_list:
    
#     temp_col_name = col1 + '_' + col2
    
#     if (col1 in categorical_features) and (col2 in categorical_features): 
#         X[temp_col_name] = X[col1] + '_' + X[col2]
#     elif (col1 in numerical_features) and (col2 in numerical_features):
#         X[temp_col_name] = num_to_woe(col1) + '_' + num_to_woe(col2)
#     elif (col1 in categorical_features) and (col2 in numerical_features):
#         X[temp_col_name] = X[col1] + '_' + num_to_woe(col2)
#     elif (col1 in numerical_features) and (col2 in categorical_features):
#         X[temp_col_name] = num_to_woe(col1) + '_' + X[col2]
    
#     newcols.append(temp_col_name)

# ohe = OneHotEncoder(drop='first', sparse = False)
# ct = ColumnTransformer(transformers=[('ohe_enc_intr', ohe, newcols)], remainder='passthrough')
# dt = ct.fit_transform(X)
# dt.shape

    
# Check how to pass parameters across different components
# Then implement the following : 
# custom function to create kpis => feature selection => interactions (set proper type) => columns selector => num/cat pipelines