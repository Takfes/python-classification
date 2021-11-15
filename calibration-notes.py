# ---------------------------------------------
# Load dependencies
# ---------------------------------------------
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.axis('off')

from sklearn.datasets import make_classification

# ---------------------------------------------
# Make dataset
# ---------------------------------------------
np.random.seed(1990)
X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, weights = [0.975, 0.025])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 ,random_state = 1990)

[x.shape for x in [X_train, X_test, y_train, y_test]]
[pd.Series(x).value_counts(normalize=True) for x in [y_train, y_test]]

# ---------------------------------------------
# Fit Logistic Regression
# ---------------------------------------------

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)

# For binary classification tasks predict_proba returns a matrix containing the first class proba in the first entry,
# and the second class proba in the second entry. Since there are only two classes one is just 1 - n of the other.
# The calibration_curve implementation expects just one of these classes in an array, so we index that.
y_test_predict_proba = clf.predict_proba(X_test)[:, 1]

from sklearn.calibration import calibration_curve
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_predict_proba, n_bins=10)

# ---------------------------------------------
# Calibration plots sklearn implementation
# ---------------------------------------------

import seaborn as sns
fig, ax = plt.subplots(1, figsize=(12, 6))
plt.plot(mean_predicted_value, fraction_of_positives, 's-')
plt.plot([0, 1], [0, 1], '--', color='gray')


# ---------------------------------------------
# Calibration plots scikitplot implementation
# ---------------------------------------------

import scikitplot as skplt

from sklearn.calibration import CalibratedClassifierCV
clf_sigmoid = CalibratedClassifierCV(clf, cv='prefit', method='sigmoid')
clf_isotonic = CalibratedClassifierCV(clf, cv='prefit', method='isotonic')

clf_sigmoid.fit(X_test,y_test)
clf_isotonic.fit(X_test,y_test)

skplt.metrics.plot_calibration_curve(y_test,
                                     [m.predict_proba(X_test)[:,1] for m in [clf,clf_sigmoid,clf_isotonic]],
                                     ['clf-base','clf-sigmoid','clf-isotonic'])
