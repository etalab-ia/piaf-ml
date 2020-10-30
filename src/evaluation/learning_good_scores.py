import pandas as pd
import numpy as np

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_curve
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
from imblearn.base import BaseSampler


def plot_resampling(X, y, sampling, ax):
    X_res, y_res = sampling.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    return Counter(y_res)

def plot_decision_function(X, y, clf, ax):
    plot_step = 0.02
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], alpha=0.8, c=y, edgecolor='k')

# TODO : balance the data

def import_data():
    # Read the data
    X_full = pd.read_csv('./results/analysis_results.csv')

    X_full = X_full[X_full.level == 'theme'] # take only the theme experiment

    # find the number of retriever algo that found the fiche
    X_full['number_of_retriever'] = X_full[['score_dense_no_lemma',
           'score_sparse_no_lemma', 'score_dense_lemma', 'score_sparse_lemma']].count(axis=1)

    X_full = X_full[X_full['number_of_retriever'] > 3]
    X_full['number_of_retriever'].astype('str', copy=False)

    """
    # For testing only use 1 parameters
    X_full['prod'] = X_full['score_sparse_no_lemma'] * X_full['score_dense_no_lemma']
    X_full = X_full[['fiche_ok','prod']]
    #       'score_dense_no_lemma']]"""

    # Remove rows with missing target, separate target from predictors
    X_full.dropna(axis=0, subset=['fiche_ok'], inplace=True)
    y = X_full.fiche_ok
    X_full.drop(['fiche_ok','question', 'fiche','level','number_of_retriever'], axis=1, inplace=True)
    #X_full.drop(['fiche_ok'], axis=1, inplace=True)

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X_full, y,
                                                          train_size=0.8, test_size=0.2,
                                                          random_state=0,
                                                          stratify=y)

    return X_train, X_valid, y_train, y_valid

X_train, X_valid, y_train, y_valid = import_data()
"""
# Select categorical columns
categorical_cols = [cname for cname in X_train.columns if
                    X_train[cname].dtype == "object"]"""

# Select numerical columns
numerical_cols = [cname for cname in X_train.columns if
                  X_train[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant', fill_value=0) # TODO replace with the minimum value
scaler = StandardScaler()
polynomial_features = PolynomialFeatures(degree=3)

"""
# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])"""

smote = SMOTE(random_state=0)
random = RandomOverSampler(random_state=0)

# Define model
model = RandomForestClassifier(n_estimators=50, random_state=0)
model = LinearSVC(random_state=0)
# model = SVC(kernel='poly', degree=3)

# Bundle preprocessing and modeling code in a pipeline
# my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('model', model)
#                      ])

my_pipeline = make_pipeline(numerical_transformer, scaler,smote, model)
no_over_pip = make_pipeline(numerical_transformer, scaler,model)

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)
scores = my_pipeline.decision_function(X_valid)

print(classification_report(y_valid, preds))
print(confusion_matrix(y_valid, preds))
print(balanced_accuracy_score(y_valid, preds))

fpr, tpr, thresholds = roc_curve(y_valid, scores)
plt.plot(fpr, tpr)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
X_train, X, y_train, y = import_data()
no_over_pip.fit(X, y)
plot_decision_function(X, y, no_over_pip, ax1)
#ax1.set_title('Linear SVC with y={}'.format(Counter(y)))
X_train, X, y_train, y = import_data()
my_pipeline.fit(X, y)
plot_decision_function(X, y, my_pipeline, ax2)
ax2.set_title('Decision function for RandomOverSampler')
fig.tight_layout()
plt.show()

class FakeSampler(BaseSampler):

    _sampling_type = 'bypass'

    def _fit_resample(self, X, y):
        return X, y

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
X, X_valid, y, y_valid = import_data()
X = numerical_transformer.fit_transform(X,y)
sampler = FakeSampler()
clf = make_pipeline(sampler, LinearSVC())
plot_resampling(X, y, sampler, ax1)
ax1.set_title('Original data - y={}'.format(Counter(y)))

ax_arr = (ax2, ax3, ax4)
for ax, sampler in zip(ax_arr, (RandomOverSampler(random_state=0),
                                SMOTE(random_state=0),
                                ADASYN(random_state=0))):
    clf = make_pipeline(sampler, LinearSVC())
    clf.fit(X, y)
    plot_resampling(X, y, sampler, ax)
    ax.set_title('Resampling using {}'.format(sampler.__class__.__name__))
fig.tight_layout()
plt.show()

"""
# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X_full, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE average score:\n", scores.mean())"""

print('hello')