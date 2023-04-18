import numpy as np
import pandas as pd
from functools import partial
from sklearn.experimental import enable_halving_search_cv
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    cross_val_predict,
    RandomizedSearchCV,
    HalvingGridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from xgboost import XGBClassifier
import category_encoders as ce

SEED = 1990
TEST_RATIO = 1 / 4

data = pd.read_pickle("classification_dataset.pkl")
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_RATIO, random_state=SEED, stratify=y
)

pipen = Pipeline([("numeric", StandardScaler())])
pipec = Pipeline([("categorical", ce.CatBoostEncoder())])

prepro = ColumnTransformer(
    [
        ("num", pipen, make_column_selector(dtype_include=np.number)),
        ("cat", pipec, make_column_selector(dtype_include=object)),
    ]
)

model = Pipeline([("prp", prepro), ("clf", XGBClassifier())])

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "f1": make_scorer(f1_score),
    "f2": make_scorer(fbeta_score, beta=2),
    "f0.5": make_scorer(fbeta_score, beta=0.5),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "roc_auc": make_scorer(roc_auc_score),
    "pr_auc": make_scorer(average_precision_score),
    "log_loss": make_scorer(log_loss),
}

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "f1": make_scorer(f1_score),
    "f2": make_scorer(partial(fbeta_score, beta=2)),
    "f0.5": make_scorer(partial(fbeta_score, beta=0.5)),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "roc_auc": make_scorer(roc_auc_score),
    "pr_auc": make_scorer(partial(average_precision_score, average="micro")),
    "log_loss": make_scorer(log_loss),
}


def evaluate_scoring(y_true, y_pred, scoring):
    scores = {}
    for name, scorer in scoring.items():
        scores[name] = scorer(y_true, y_pred)
    return scores


params_grid = {
    "clf__n_estimators": [100, 150, 200],
    "clf__learning_rate": [0.01, 0.1, 0.5],
    "clf__max_depth": [3, 5, 7],
    "clf__min_child_weight": [1, 3, 5],
    "clf__subsample": [0.6, 0.8, 1.0],
    "clf__colsample_bytree": [0.6, 0.8, 1.0],
    "clf__gamma": [0, 0.3, 0.5],
    "clf__reg_alpha": [0.1, 0.25, 0.75],
    "clf__reg_lambda": [0.1, 0.25, 0.75],
}

cvscore1 = cross_val_score(model, X_train, y_train, scoring="f1", cv=3)

# cv : int, to specify the number of folds in a (Stratified)KFold
cvscore2 = cross_validate(
    model,
    X_train,
    y_train,
    scoring=scoring,
    cv=3,
    return_train_score=True,
)

search1 = RandomizedSearchCV(
    estimator=model,
    param_distributions=params_grid,
    n_iter=20,
    scoring=scoring,
    refit="accuracy",
    cv=3,
    verbose=1,
    return_train_score=True,
    random_state=SEED,
)

search1.fit(X_train, y_train)

search1.best_score_
search1.best_params_
search1.best_index_
search1.cv_results_
pd.DataFrame(search1.cv_results_).filter(like="mean_")

model1 = clone(model)
model1.set_params(**search1.best_params_)
model1.get_params()

model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
y_prob1 = model1.predict_proba(X_test)

evaluate_scoring(y_true=y_test, y_pred=y_pred1, scoring=scoring)
make_scorer(accuracy_score)(y_test, y_pred1)

search2 = HalvingGridSearchCV(
    estimator=model,
    param_distributions=params_grid,
    n_iter=20,
    scoring=scoring,
    refit="accuracy",
    cv=3,
    verbose=1,
    return_train_score=True,
    random_state=SEED,
)

for name, scorer in scoring.items():
    scores[name] = scorer(y_true, y_pred)
