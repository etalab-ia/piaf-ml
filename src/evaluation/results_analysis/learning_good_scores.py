import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# TODO : balance the data

# Read the data
X_full = pd.read_csv("./output/analysis_results.csv")

X_full = X_full[X_full.level == "theme"]  # take only the theme experiment

# find the number of retriever algo that found the fiche
X_full["number_of_retriever"] = X_full[
    [
        "score_dense_no_lemma",
        "score_sparse_no_lemma",
        "score_dense_lemma",
        "score_sparse_lemma",
    ]
].count(axis=1)

X_full = X_full[X_full["number_of_retriever"] > 2]
X_full["number_of_retriever"].astype("str", copy=False)

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=["fiche_ok"], inplace=True)
y = X_full.fiche_ok
X_full.drop(["fiche_ok", "question", "fiche", "level"], axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(
    X_full, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y
)

# Select categorical columns
categorical_cols = [
    cname for cname in X_train.columns if X_train[cname].dtype == "object"
]

# Select numerical columns
numerical_cols = [
    cname for cname in X_train.columns if X_train[cname].dtype in ["int64", "float64"]
]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(
    strategy="constant", fill_value=0
)  # TODO replace with the minium value

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Define model
model = BalancedRandomForestClassifier(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

print(classification_report(y_valid, preds))
print(confusion_matrix(y_valid, preds))
print(balanced_accuracy_score(y_valid, preds))

"""
# Multiply by -1 since sklearn calculates *negative* MAE scores = -1 * cross_val_score(my_pipeline, X_full, y, cv=5,
scoring='neg_mean_absolute_error')

print("MAE average score:\n", scores.mean())
"""
