import pandas as pd
import numpy as np
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
import scikitplot as skplt
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve


def confmat(y_true, y_pred, decimals=3):
    cmdata = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    return {
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


def classifier_metrics(y_true, proba, threshold=0.5, tables=False):
    y_pred = np.where(proba >= threshold, 1, 0)

    metrics = {}

    if tables:
        metrics["confmat"] = confmat(y_true, y_pred)
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
    metrics["train"] = classifier_metrics(
        y_train, y_train_prob, threshold=threshold, tables=False
    )

    # test data
    test_probs = clf.predict_proba(X_test)
    y_test_prob = test_probs[:, 1]
    metrics["test"] = classifier_metrics(
        y_test, y_test_prob, threshold=threshold, tables=False
    )

    return (
        pd.DataFrame(metrics)
        .assign(diff=lambda x: x.apply(lambda row: row["test"] - row["train"], axis=1))
        .assign(
            pct_diff=lambda x: x.apply(lambda row: row["diff"] / row["train"], axis=1)
        )
    )


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
