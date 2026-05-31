import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, classification_report
from typing import Dict, Any, List, Optional


class Regression:
    @staticmethod
    def linear(data: pd.DataFrame, dv: str, predictors: List[str]) -> Dict[str, Any]:
        X = data[predictors].fillna(data[predictors].mean())
        y = data[dv].fillna(data[dv].mean())
        X_with_const = np.column_stack([np.ones(len(X)), X])

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred

        n, p = len(y), len(predictors)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
        sigma_sq = mse * n / (n - p - 1)
        se = np.sqrt(np.diag(XtX_inv * sigma_sq))
        t_values = model.intercept_ if hasattr(model, 'intercept_') else 0
        coefs = [model.intercept_] + list(model.coef_)
        t_stats = [c / s for c, s in zip(coefs, se)]
        p_values = [2 * (1 - stats.t.cdf(abs(t), n - p - 1)) for t in t_stats]

        return {
            "test": "Linear Regression",
            "r_squared": float(r2),
            "adj_r_squared": float(adj_r2),
            "rmse": float(rmse),
            "mse": float(mse),
            "mae": float(np.mean(np.abs(residuals))),
            "n": int(n),
            "predictors": p,
            "f_statistic": float((r2 / p) / ((1 - r2) / (n - p - 1))) if r2 < 1 else 0,
            "coefficients": [
                {
                    "variable": "Intercept" if i == 0 else predictors[i - 1],
                    "coefficient": float(coefs[i]),
                    "std_error": float(se[i]),
                    "t_statistic": float(t_stats[i]),
                    "p_value": float(p_values[i]),
                    "significant": bool(p_values[i] < 0.05),
                }
                for i in range(len(coefs))
            ],
            "residual_stats": {
                "min": float(residuals.min()),
                "max": float(residuals.max()),
                "mean": float(residuals.mean()),
                "std": float(residuals.std()),
            },
            "interpretation": (
                f"A linear regression was performed with {p} predictor(s). "
                f"The model was {'significant' if r2 > 0 else 'not significant'} "
                f"(R² = {r2:.4f}, adjusted R² = {adj_r2:.4f}, RMSE = {rmse:.4f}). "
                f"The model explains {r2 * 100:.1f}% of the variance in {dv}."
            ),
        }

    @staticmethod
    def logistic(data: pd.DataFrame, dv: str, predictors: List[str]) -> Dict[str, Any]:
        X = data[predictors].fillna(data[predictors].mean())
        y = data[dv]

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "test": "Logistic Regression",
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "coefficients": [
                {
                    "variable": "Intercept" if i == 0 else predictors[i - 1],
                    "coefficient": float(model.intercept_[0] if i == 0 else model.coef_[0][i - 1]),
                }
                for i in range(len(predictors) + 1)
            ],
            "interpretation": (
                f"Logistic regression model achieved {accuracy:.1%} accuracy "
                f"(precision = {precision:.3f}, recall = {recall:.3f}, F1 = {f1:.3f})."
            ),
        }

    @staticmethod
    def ridge(data: pd.DataFrame, dv: str, predictors: List[str], alpha: float = 1.0) -> Dict[str, Any]:
        X = data[predictors].fillna(data[predictors].mean())
        y = data[dv].fillna(data[dv].mean())
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        return {
            "test": "Ridge Regression",
            "alpha": alpha,
            "r_squared": float(r2),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "coefficients": [
                {"variable": predictors[i], "coefficient": float(model.coef_[i])}
                for i in range(len(predictors))
            ],
            "intercept": float(model.intercept_),
            "interpretation": f"Ridge regression (α = {alpha}) achieved R² = {r2:.4f}.",
        }

    @staticmethod
    def lasso(data: pd.DataFrame, dv: str, predictors: List[str], alpha: float = 1.0) -> Dict[str, Any]:
        X = data[predictors].fillna(data[predictors].mean())
        y = data[dv].fillna(data[dv].mean())
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        return {
            "test": "Lasso Regression",
            "alpha": alpha,
            "r_squared": float(r2),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "coefficients": [
                {"variable": predictors[i], "coefficient": float(model.coef_[i]), "zeroed": bool(abs(model.coef_[i]) < 1e-10)}
                for i in range(len(predictors))
            ],
            "intercept": float(model.intercept_),
            "features_selected": int(sum(abs(model.coef_) > 1e-10)),
            "interpretation": f"Lasso regression (α = {alpha}) achieved R² = {r2:.4f}, selecting {int(sum(abs(model.coef_) > 1e-10))}/{len(predictors)} features.",
        }

    @staticmethod
    def polynomial(data: pd.DataFrame, dv: str, predictor: str, degree: int = 2) -> Dict[str, Any]:
        X = data[[predictor]].fillna(data[predictor].mean())
        y = data[dv].fillna(data[dv].mean())

        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)

        return {
            "test": f"Polynomial Regression (degree {degree})",
            "r_squared": float(r2),
            "adj_r_squared": float(1 - (1 - r2) * (len(y) - 1) / (len(y) - degree - 1)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "coefficients": [float(c) for c in model.coef_],
            "intercept": float(model.intercept_),
            "interpretation": f"Polynomial regression (degree {degree}) achieved R² = {r2:.4f}.",
        }
