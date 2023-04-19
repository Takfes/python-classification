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
    HalvingRandomSearchCV,
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

search2 = HalvingRandomSearchCV(
    estimator=model,
    n_candidates=200,
    param_distributions=params_grid,
    scoring="accuracy",
    refit=True,
    cv=3,
    verbose=1,
    return_train_score=True,
    random_state=SEED,
)

search2.fit(X_train, y_train)

import scikitplot as skplt
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve
from mlutils.classification import classifier_metrics, classifier_report

estim1 = search1.best_estimator_
estim2 = search2.best_estimator_

train_probs1 = estim1.predict_proba(X_train)
proba1 = train_probs1[:, 1]

train_probs2 = estim2.predict_proba(X_train)
proba2 = train_probs2[:, 1]

classifier_metrics(y_train, proba1)
classifier_metrics(y_train, proba1, threshold=0.9)
classifier_report(estim1, X_train, X_test, y_train, y_test)


def classifier_plots(y_true, probas, threshold=0.5):
    y_pred = np.where(probas[:, 1] >= threshold, 1, 0)

    plots = {}

    plots["confmat"] = skplt.metrics.plot_confusion_matrix(
        y_train, y_pred, normalize=True
    )
    plots["roc"] = skplt.metrics.plot_roc(y_true, probas)
    plots["pr"] = skplt.metrics.plot_precision_recall(y_true, probas)
    plots["cal"] = skplt.metrics.plot_calibration_curve(y_true, [probas])
    plots["ks"] = skplt.metrics.plot_ks_statistic(y_true, probas)
    plots["lift"] = skplt.metrics.plot_lift_curve(y_true, probas)
    plots["gains"] = skplt.metrics.plot_cumulative_gain(y_true, probas)

    return plots


def classifier_plots_int(y_true, probas_list, model_names=None, show=False):
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(probas_list))]

    # Create plotly figures
    roc_fig = go.Figure()
    pr_fig = go.Figure()
    cal_fig = go.Figure()

    for proba, model_name in zip(probas_list, model_names):
        # Calculate metrics
        fpr, tpr, _ = roc_curve(y_true, proba)
        precision, recall, _ = precision_recall_curve(y_true, proba)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, proba, n_bins=10
        )

        # Add traces to figures
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=model_name))
        pr_fig.add_trace(
            go.Scatter(x=recall, y=precision, mode="lines", name=model_name)
        )
        cal_fig.add_trace(
            go.Scatter(
                x=mean_predicted_value,
                y=fraction_of_positives,
                mode="lines",
                name=model_name,
            )
        )

    # Add diagonal lines
    roc_fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        yref="y",
        xref="x",
        line=dict(color="gray", dash="dash"),
    )
    cal_fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        yref="y",
        xref="x",
        line=dict(color="gray", dash="dash"),
    )

    # Update layout
    roc_fig.update_layout(title="ROC Curve")
    pr_fig.update_layout(title="Precision-Recall Curve")
    cal_fig.update_layout(title="Calibration Curve")

    plots = {"roc": roc_fig, "pr": pr_fig, "cal": cal_fig}

    if show:
        for k, v in plots.items():
            v.show()

    return plots


plots = classifier_plots_int(y_train, [proba1, proba2], show=True)


def classifier_lift_df(y_true, y_prob, n_bins=10):
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df = df.sort_values(by="y_prob", ascending=False)
    df["rank"] = np.arange(len(df)) + 1
    df["bin"] = pd.qcut(df["rank"], n_bins, labels=False)

    lift_df = (
        df.groupby("bin")
        .agg(
            positive_count=pd.NamedAgg(column="y_true", aggfunc="sum"),
            total_count=pd.NamedAgg(column="y_true", aggfunc="count"),
        )
        .reset_index()
    )

    lift_df["fraction_of_positives"] = (
        lift_df["positive_count"] / lift_df["total_count"]
    )
    overall_positive_fraction = df["y_true"].sum() / df["y_true"].count()
    lift_df["lift"] = lift_df["fraction_of_positives"] / overall_positive_fraction

    return lift_df


classifier_lift_df(y_train, proba1)


def classifier_lift_plot(y_true, probas_list, model_names=None, n_bins=10, show=False):
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(probas_list))]

    lift_dfs = {}
    lift_fig = go.Figure()

    for proba, model_name in zip(probas_list, model_names):
        lift_df = classifier_lift_df(y_true, proba, n_bins)
        lift_dfs[model_name] = lift_df

        lift_fig.add_trace(
            go.Scatter(
                x=lift_df.index, y=lift_df["lift"], mode="lines", name=model_name
            )
        )
    # add random lift
    lift_fig.add_shape(
        type="line",
        x0=0,
        x1=n_bins - 1,
        y0=1,
        y1=1,
        yref="y",
        xref="x",
        line=dict(color="gray", dash="dash"),
    )

    lift_fig.update_layout(title="Lift Curve", xaxis_title="Bin", yaxis_title="Lift")

    if show:
        lift_fig.show()

    return lift_dfs, lift_fig


lift_dfs, lift_plot = classifier_lift_plot(
    y_train, [proba1, proba2], n_bins=12, show=True
)
