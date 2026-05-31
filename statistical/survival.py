import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from typing import Dict, Any, Optional, List


class SurvivalAnalysis:
    @staticmethod
    def kaplan_meier(data: pd.DataFrame, duration_col: str, event_col: str,
                     group_col: Optional[str] = None) -> Dict[str, Any]:
        kmf = KaplanMeierFitter()

        if group_col:
            groups = data[group_col].unique()
            results = {}
            for group in groups:
                mask = data[group_col] == group
                kmf.fit(data[mask][duration_col], event_observed=data[mask][event_col], label=str(group))
                results[str(group)] = {
                    "median_survival_time": float(kmf.median_survival_time_) if not np.isnan(kmf.median_survival_time_) else None,
                    "survival_function": {
                        "times": kmf.survival_function_.index.tolist(),
                        "values": kmf.survival_function_.iloc[:, 0].tolist(),
                    },
                    "confidence_interval": {
                        "times": kmf.confidence_interval_.index.tolist(),
                        "lower": kmf.confidence_interval_.iloc[:, 0].tolist(),
                        "upper": kmf.confidence_interval_.iloc[:, 1].tolist(),
                    },
                    "event_table": {
                        "at_risk": kmf.event_table["at_risk"].tolist(),
                        "observed": kmf.event_table["observed"].tolist(),
                        "censored": kmf.event_table["censored"].tolist(),
                    },
                }

            if len(groups) == 2:
                g1, g2 = groups[0], groups[1]
                lr_result = logrank_test(
                    data[data[group_col] == g1][duration_col],
                    data[data[group_col] == g2][duration_col],
                    data[data[group_col] == g1][event_col],
                    data[data[group_col] == g2][event_col],
                )
                results["logrank_test"] = {
                    "test_statistic": float(lr_result.test_statistic),
                    "p_value": float(lr_result.p_value),
                    "significant": bool(lr_result.p_value < 0.05),
                }

            return {
                "test": "Kaplan-Meier Survival Analysis",
                "groups": results,
                "interpretation": f"Kaplan-Meier analysis with {len(groups)} groups." + (
                    f" Log-rank p = {lr_result.p_value:.4f}." if len(groups) == 2 else ""
                ),
            }
        else:
            kmf.fit(data[duration_col], event_observed=data[event_col])
            return {
                "test": "Kaplan-Meier Survival Analysis",
                "median_survival_time": float(kmf.median_survival_time_) if not np.isnan(kmf.median_survival_time_) else None,
                "survival_function": {
                    "times": kmf.survival_function_.index.tolist(),
                    "values": kmf.survival_function_.iloc[:, 0].tolist(),
                },
                "confidence_interval": {
                    "times": kmf.confidence_interval_.index.tolist(),
                    "lower": kmf.confidence_interval_.iloc[:, 0].tolist(),
                    "upper": kmf.confidence_interval_.iloc[:, 1].tolist(),
                },
                "interpretation": f"Kaplan-Meier estimate. Median survival: {kmf.median_survival_time_:.2f}." if not np.isnan(kmf.median_survival_time_) else "Kaplan-Meier estimate computed.",
            }

    @staticmethod
    def cox_regression(data: pd.DataFrame, duration_col: str, event_col: str,
                       predictors: List[str]) -> Dict[str, Any]:
        df = data[[duration_col, event_col] + predictors].dropna()
        cph = CoxPHFitter()
        cph.fit(df, duration_col=duration_col, event_col=event_col)

        summary = cph.summary
        results = {
            "test": "Cox Proportional Hazards Regression",
            "n": len(df),
            "n_events": int(df[event_col].sum()),
            "concordance": float(cph.concordance_index_),
            "log_likelihood": float(cph.log_likelihood_),
            "partial_aic": float(cph.AIC_partial_),
        }

        coefs = []
        for predictor in predictors:
            if predictor in summary.index:
                row = summary.loc[predictor]
                coefs.append({
                    "variable": predictor,
                    "coef": float(row["coef"]),
                    "exp_coef": float(row["exp(coef)"]),
                    "se": float(row["se(coef)"]),
                    "z": float(row["z"]),
                    "p_value": float(row["p"]),
                    "ci_lower": float(row["coef lower 95%"]),
                    "ci_upper": float(row["coef upper 95%"]),
                    "significant": bool(row["p"] < 0.05),
                })
        results["coefficients"] = coefs

        results["interpretation"] = (
            f"Cox regression with {len(predictors)} predictors. "
            f"Concordance index = {cph.concordance_index_:.4f}. "
        )
        for coef in coefs:
            if coef["significant"]:
                results["interpretation"] += (
                    f"{coef['variable']}: HR = {coef['exp_coef']:.3f} "
                    f"(p = {coef['p_value']:.4f}). "
                )

        return results
