import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import plot_partial_dependence
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

if __name__ == "__main__":
    from pathlib import Path

    data = pd.read_csv(Path("./results/runs.csv"))
    y = data["reader_topk_accuracy_has_answer"]
    X = data[["k_reader_per_candidate", "k_reader_total", "k_retriever"]]

    # model = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', Ridge())])
    # model.fit(X, y)
    # results = {name:coef for name, coef in zip(model.named_steps.poly.get_feature_names(X_poly.columns), model.named_steps.linear.coef_)}
    # results = dict(sorted(results.items(), key=lambda item: abs(item[1]), reverse=True))

    poly = PolynomialFeatures(degree=2)
    model = Ridge()

    X_poly = poly.fit_transform(X)
    X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names(X.columns))

    model.fit(X_poly, y)
    model.score(X_poly, y)

    results = {name: coef for name, coef in zip(X_poly.columns, model.coef_)}
    results = dict(sorted(results.items(), key=lambda item: abs(item[1]), reverse=True))
    important_features = ["k_reader_total^2", "k_retriever", "k_reader_per_candidate^2"]
    plot_partial_dependence(model, X_poly, important_features, kind="both")

    important_features = [("k_reader_total^2", "k_retriever")]
    plot_partial_dependence(model, X_poly, important_features)
