import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional


class Correlation:
    @staticmethod
    def pearson(data1: pd.Series, data2: pd.Series) -> Dict[str, Any]:
        s1, s2 = data1.dropna(), data2.dropna()
        r, p_value = stats.pearsonr(s1, s2)
        n = len(s1)
        return {
            "test": "Pearson Correlation",
            "r": float(r),
            "p_value": float(p_value),
            "n": n,
            "r_squared": float(r ** 2),
            "significant": bool(p_value < 0.05),
            "strength": Correlation._strength(r),
            "direction": "positive" if r > 0 else "negative",
            "interpretation": (
                f"Pearson correlation: r({n - 2}) = {r:.4f}, p = {p_value:.4f}, "
                f"R² = {r ** 2:.3f}. "
                f"A {Correlation._strength(r)} {Correlation._direction(r)} correlation "
                f"{'was' if p_value < 0.05 else 'was not'} statistically significant."
            ),
        }

    @staticmethod
    def spearman(data1: pd.Series, data2: pd.Series) -> Dict[str, Any]:
        s1, s2 = data1.dropna(), data2.dropna()
        rho, p_value = stats.spearmanr(s1, s2)
        n = len(s1)
        return {
            "test": "Spearman Rank Correlation",
            "rho": float(rho),
            "p_value": float(p_value),
            "n": n,
            "significant": bool(p_value < 0.05),
            "strength": Correlation._strength(rho),
            "direction": "positive" if rho > 0 else "negative",
            "interpretation": (
                f"Spearman correlation: ρ({n - 2}) = {rho:.4f}, p = {p_value:.4f}. "
                f"A {Correlation._strength(rho)} {Correlation._direction(rho)} monotonic relationship "
                f"{'was' if p_value < 0.05 else 'was not'} statistically significant."
            ),
        }

    @staticmethod
    def kendall(data1: pd.Series, data2: pd.Series) -> Dict[str, Any]:
        s1, s2 = data1.dropna(), data2.dropna()
        tau, p_value = stats.kendalltau(s1, s2)
        n = len(s1)
        return {
            "test": "Kendall's Tau",
            "tau": float(tau),
            "p_value": float(p_value),
            "n": n,
            "significant": bool(p_value < 0.05),
            "strength": Correlation._strength(tau),
            "interpretation": (
                f"Kendall's τ = {tau:.4f}, p = {p_value:.4f}. "
                f"{'Significant' if p_value < 0.05 else 'Not significant'} rank correlation."
            ),
        }

    @staticmethod
    def point_biserial(binary: pd.Series, continuous: pd.Series) -> Dict[str, Any]:
        b, c = binary.dropna(), continuous.dropna()
        r_pb, p_value = stats.pointbiserialr(b, c)
        n = len(b)
        groups = b.unique()
        group_means = {str(g): float(c[b == g].mean()) for g in groups}
        return {
            "test": "Point-Biserial Correlation",
            "r_pb": float(r_pb),
            "p_value": float(p_value),
            "n": n,
            "significant": bool(p_value < 0.05),
            "group_means": group_means,
            "interpretation": (
                f"Point-biserial correlation: r_pb = {r_pb:.4f}, p = {p_value:.4f}. "
                f"{'Significant relationship between binary and continuous variables.' if p_value < 0.05 else 'No significant relationship.'}"
            ),
        }

    @staticmethod
    def correlation_matrix(data: pd.DataFrame, method: str = "pearson") -> Dict[str, Any]:
        numeric = data.select_dtypes(include=[np.number])
        if method == "pearson":
            corr_matrix = numeric.corr(method="pearson")
        elif method == "spearman":
            corr_matrix = numeric.corr(method="spearman")
        else:
            corr_matrix = numeric.corr(method="kendall")

        columns = corr_matrix.columns.tolist()
        matrix = corr_matrix.values.tolist()
        p_values = []
        for col1 in columns:
            row_p = []
            for col2 in columns:
                if col1 == col2:
                    row_p.append(0.0)
                else:
                    _, p = stats.pearsonr(numeric[col1].dropna(), numeric[col2].dropna())
                    row_p.append(float(p))
            p_values.append(row_p)

        return {
            "method": method,
            "columns": columns,
            "matrix": [[float(v) for v in row] for row in matrix],
            "p_values": p_values,
            "significant_pairs": [
                {"var1": columns[i], "var2": columns[j], "r": float(matrix[i][j]), "p": p_values[i][j]}
                for i in range(len(columns)) for j in range(i + 1, len(columns))
                if p_values[i][j] < 0.05
            ],
        }

    @staticmethod
    def covariate(data: pd.DataFrame) -> Dict[str, Any]:
        numeric = data.select_dtypes(include=[np.number])
        cov_matrix = numeric.cov()
        return {
            "columns": cov_matrix.columns.tolist(),
            "matrix": [[float(v) for v in row] for row in cov_matrix.values.tolist()],
        }

    @staticmethod
    def _strength(r: float) -> str:
        r = abs(r)
        if r >= 0.9:
            return "very strong"
        elif r >= 0.7:
            return "strong"
        elif r >= 0.5:
            return "moderate"
        elif r >= 0.3:
            return "weak"
        else:
            return "very weak"

    @staticmethod
    def _direction(r: float) -> str:
        return "positive" if r > 0 else "negative"
