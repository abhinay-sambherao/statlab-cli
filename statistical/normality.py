import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional


class NormalityTests:
    @staticmethod
    def shapiro_wilk(data: pd.Series) -> Dict[str, Any]:
        series = data.dropna()
        if len(series) < 3 or len(series) > 5000:
            return {"error": "Shapiro-Wilk requires 3-5000 samples"}
        statistic, p_value = stats.shapiro(series)
        return {
            "test": "Shapiro-Wilk Test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "normal": bool(p_value > 0.05),
            "interpretation": (
                f"The Shapiro-Wilk test {'does not reject' if p_value > 0.05 else 'rejects'} the null hypothesis "
                f"of normality (W = {statistic:.4f}, p = {p_value:.4f}). "
                f"{'Data appears normally distributed.' if p_value > 0.05 else 'Data deviates from normal distribution.'}"
            ),
        }

    @staticmethod
    def kolmogorov_smirnov(data: pd.Series, dist: str = "norm") -> Dict[str, Any]:
        series = data.dropna()
        mu, sigma = np.mean(series), np.std(series, ddof=1)
        statistic, p_value = stats.kstest(series, dist, args=(mu, sigma))
        return {
            "test": f"Kolmogorov-Smirnov Test ({dist})",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "normal": bool(p_value > 0.05),
            "interpretation": (
                f"KS test statistic D = {statistic:.4f}, p = {p_value:.4f}. "
                f"{'Data follows normal distribution.' if p_value > 0.05 else 'Data does not follow normal distribution.'}"
            ),
        }

    @staticmethod
    def anderson_darling(data: pd.Series, dist: str = "norm") -> Dict[str, Any]:
        series = data.dropna()
        result = stats.anderson(series, dist=dist)
        significance_levels = {15: 0.576, 10: 0.656, 5: 0.787, 2.5: 0.918, 1: 1.092}
        if dist == "norm":
            significance_levels = {15: 0.576, 10: 0.656, 5: 0.787, 2.5: 0.918, 1: 1.092}

        return {
            "test": f"Anderson-Darling Test ({dist})",
            "statistic": float(result.statistic),
            "critical_values": result.critical_values.tolist(),
            "significance_levels": result.significance_level.tolist(),
            "normal": bool(result.statistic < result.critical_values[2]),
            "interpretation": (
                f"AD statistic = {result.statistic:.4f}. "
                f"At 5% significance, critical value = {result.critical_values[2]:.4f}. "
                f"{'Data appears normally distributed.' if result.statistic < result.critical_values[2] else 'Data deviates from normality.'}"
            ),
        }

    @staticmethod
    def d_agostino_pearson(data: pd.Series) -> Dict[str, Any]:
        series = data.dropna()
        if len(series) < 20:
            return {"error": "D'Agostino-Pearson requires at least 20 samples"}
        statistic, p_value = stats.normaltest(series)
        return {
            "test": "D'Agostino-Pearson Test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "normal": bool(p_value > 0.05),
            "interpretation": (
                f"K² = {statistic:.4f}, p = {p_value:.4f}. "
                f"{'Data is normally distributed.' if p_value > 0.05 else 'Data is not normally distributed.'}"
            ),
        }

    @staticmethod
    def jarque_bera(data: pd.Series) -> Dict[str, Any]:
        series = data.dropna()
        statistic, p_value = stats.jarque_bera(series)
        return {
            "test": "Jarque-Bera Test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "normal": bool(p_value > 0.05),
            "interpretation": (
                f"JB = {statistic:.4f}, p = {p_value:.4f}. "
                f"{'Data is normally distributed.' if p_value > 0.05 else 'Data is not normally distributed.'}"
            ),
        }

    @staticmethod
    def all_tests(data: pd.Series) -> Dict[str, Any]:
        results = {"column": data.name if data.name else "unknown"}
        try:
            results["shapiro_wilk"] = NormalityTests.shapiro_wilk(data)
        except Exception as e:
            results["shapiro_wilk"] = {"error": str(e)}
        try:
            results["kolmogorov_smirnov"] = NormalityTests.kolmogorov_smirnov(data)
        except Exception as e:
            results["kolmogorov_smirnov"] = {"error": str(e)}
        try:
            results["anderson_darling"] = NormalityTests.anderson_darling(data)
        except Exception as e:
            results["anderson_darling"] = {"error": str(e)}
        try:
            results["d_agostino_pearson"] = NormalityTests.d_agostino_pearson(data)
        except Exception as e:
            results["d_agostino_pearson"] = {"error": str(e)}
        try:
            results["jarque_bera"] = NormalityTests.jarque_bera(data)
        except Exception as e:
            results["jarque_bera"] = {"error": str(e)}

        conclusions = []
        for test_name, test_result in results.items():
            if isinstance(test_result, dict) and "normal" in test_result:
                conclusions.append(test_result["normal"])
        if conclusions:
            results["overall_normal"] = bool(sum(conclusions) > len(conclusions) / 2)
            results["consensus"] = (
                "Data appears normally distributed" if results["overall_normal"]
                else "Data does not appear normally distributed"
            )

        return results
