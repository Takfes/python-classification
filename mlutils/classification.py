import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    classification_report,
    precision_recall_curve,
    roc_curve,
    # make_scorer,
    # f1_score,
    # confusion_matrix,
)
from sklearn.dummy import DummyClassifier
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scikitplot as skplt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from yellowbrick.classifier import discrimination_threshold, DiscriminationThreshold

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier


def classifier_param_grid(model):
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]

    if isinstance(model, AdaBoostClassifier):
        param_grid = {
            "n_estimators": [50, 100, 200, 400],
            "learning_rate": [0.01, 0.1, 0.5, 1.0],
        }
    elif isinstance(model, GradientBoostingClassifier):
        param_grid = {
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.01, 0.1, 0.5, 1.0],
            "max_depth": [3, 5, 7, 9],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    elif isinstance(model, XGBClassifier):
        param_grid = {
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.01, 0.1, 0.5, 1.0],
            "max_depth": [3, 5, 7, 9],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0.0, 0.1, 0.2, 0.3],
            "subsample": [0.5, 0.75, 1.0],
            "colsample_bytree": [0.5, 0.75, 1.0],
        }
    elif isinstance(model, LGBMClassifier):
        param_grid = {
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.01, 0.1, 0.5, 1.0],
            "num_leaves": [15, 31, 63, 127],
            "min_child_samples": [5, 10, 20, 30],
            "subsample": [0.5, 0.75, 1.0],
            "colsample_bytree": [0.5, 0.75, 1.0],
        }
    elif isinstance(model, KNeighborsClassifier):
        param_grid = {
            "n_neighbors": list(range(1, 21)),
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [10, 30, 50],
            "p": [1, 2],
        }
    elif isinstance(model, LogisticRegression):
        param_grid = {
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "C": np.logspace(-4, 4, 20),
            "fit_intercept": [True, False],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "max_iter": [100, 200, 500, 1000],
            "l1_ratio": np.linspace(0, 1, 10),
        }
    elif isinstance(model, RandomForestClassifier):
        param_grid = {
            "n_estimators": [10, 50, 100, 200, 400],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2", None],
        }
    else:
        raise ValueError("Unsupported model type: {}".format(type(model)))
    return param_grid


def classifier_confusion_matrix(y_true, y_pred, return_all=False, decimals=3):
    cmdata = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    results = {
        "norm": round(
            pd.crosstab(
                cmdata["y_true"], cmdata["y_pred"], margins=True, normalize=True
            ),
            decimals,
        ),
        "index": round(
            pd.crosstab(
                cmdata["y_true"], cmdata["y_pred"], margins=True, normalize="index"
            ),
            decimals,
        ),
        "cols": round(
            pd.crosstab(
                cmdata["y_true"], cmdata["y_pred"], margins=True, normalize="columns"
            ),
            decimals,
        ),
        "counts": round(
            pd.crosstab(
                cmdata["y_true"], cmdata["y_pred"], margins=True, normalize=False
            ),
            decimals,
        ),
    }
    if return_all:
        return results
    else:
        return results["index"]


def classifier_benchmark(X_train, X_test, y_train, y_test, strategy="stratified"):
    clfs = {
        x: DummyClassifier(strategy=x).fit(X_train, y_train)
        for x in [strategy]  # "stratified", "uniform", "most_frequent", "prior",
    }

    return classifier_report(clfs[strategy], X_train, X_test, y_train, y_test)


def classifier_metrics(y_true, proba, threshold=0.5, tables=False):
    y_pred = np.where(proba >= threshold, 1, 0)

    metrics = {}

    if tables:
        metrics["confmat"] = classifier_confusion_matrix(y_true, y_pred)
        metrics["clfreport"] = classification_report(y_true, y_pred)

    metrics["logloss"] = log_loss(y_true, proba)
    metrics["f0.5"] = fbeta_score(y_true, y_pred, beta=0.5)
    metrics["f1"] = fbeta_score(y_true, y_pred, beta=1)
    metrics["f2"] = fbeta_score(y_true, y_pred, beta=2)
    metrics["prauc"] = average_precision_score(y_true, proba)
    metrics["roc"] = roc_auc_score(y_true, proba)
    metrics["precision"] = precision_score(y_true, y_pred)
    metrics["recall"] = recall_score(y_true, y_pred)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    return metrics


def classifier_report(clf, X_train, X_test, y_train, y_test, threshold=0.5):
    metrics = {}

    # train data
    train_probs = clf.predict_proba(X_train)
    y_train_prob = train_probs[:, 1]
    metrics["train_metrics"] = classifier_metrics(
        y_train, y_train_prob, threshold=threshold, tables=False
    )

    # test data
    test_probs = clf.predict_proba(X_test)
    y_test_prob = test_probs[:, 1]
    metrics["test_metrics"] = classifier_metrics(
        y_test, y_test_prob, threshold=threshold, tables=False
    )

    return (
        pd.DataFrame(metrics)
        .assign(diff=lambda x: x.apply(lambda row: row["test"] - row["train"], axis=1))
        .assign(
            pct_diff=lambda x: x.apply(lambda row: row["diff"] / row["train"], axis=1)
        )
    )


def classifier_thresholds(clf, X, y):
    # visualizer = DiscriminationThreshold(LogisticRegression())
    # visualizer.force_model = pipeline.named_steps['classifier']
    return discrimination_threshold(clf, X, y)


def classifier_plots(y_true, probas, threshold=0.5):
    y_pred = np.where(probas[:, 1] >= threshold, 1, 0)

    plots = {}

    plots["confmat"] = skplt.metrics.plot_confusion_matrix(
        y_true, y_pred, normalize=True
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


def classifier_gain_lift(y_true, y_prob, n_bins=10):
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df = df.sort_values(by="y_prob", ascending=False)
    df["rank"] = np.arange(len(df)) + 1
    df["bin"] = pd.qcut(df["rank"], n_bins, labels=False) + 1

    lift_df = (
        df.groupby("bin")
        .agg(
            positive_count=pd.NamedAgg(column="y_true", aggfunc="sum"),
            total_count=pd.NamedAgg(column="y_true", aggfunc="count"),
        )
        .reset_index()
    )
    lift_df["positive_count_cumsum"] = lift_df["positive_count"].cumsum()
    lift_df["fraction_of_positives"] = lift_df["positive_count"] / sum(
        lift_df["positive_count"]
    )
    lift_df["gain"] = lift_df["fraction_of_positives"].cumsum()
    lift_df["lift"] = (lift_df["gain"] * 100) / (10 * lift_df["bin"])

    return lift_df


def classifier_gain_lift_plot(
    y_true, proba_list, model_names=None, n_bins=10, show=False
):
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(proba_list))]

    lift_dfs = {}
    lift_fig = go.Figure()

    for proba, model_name in zip(proba_list, model_names):
        lift_df = classifier_gain_lift(y_true, proba, n_bins)
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


def plot_decision_boundaries(X, y, clf, dim_reduction="pca"):
    if X.shape > 2:
        # dimensionality reduction step
        if dim_reduction == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
            Xreduced = reducer.fit_transform(X)
        elif dim_reduction == "pca":
            reducer = PCA(n_components=2)
            Xreduced = reducer.fit_transform(X)

    # Plot the decision boundaries
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine the minimum and maximum values for each feature
    x_min, x_max = Xreduced[:, 0].min() - 0.1, Xreduced[:, 0].max() + 0.1
    y_min, y_max = Xreduced[:, 1].min() - 0.1, Xreduced[:, 1].max() + 0.1

    # Create a meshgrid of points to plot the decision boundaries
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundaries and the training points
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(Xreduced[:, 0], Xreduced[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.show()
