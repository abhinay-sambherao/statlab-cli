import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from typing import Dict, Any, List, Optional


class ANOVA:
    @staticmethod
    def one_way(data: pd.DataFrame, dv: str, between: str) -> Dict[str, Any]:
        groups = [g.dropna().values for _, g in data.groupby(between)[dv]]
        f_stat, p_value = stats.f_oneway(*groups)
        df_between = len(groups) - 1
        df_within = sum(len(g) for g in groups) - len(groups)

        grand_mean = np.mean([np.mean(g) for g in groups])
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_within = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups)
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        eta_sq = ss_between / (ss_between + ss_within)
        omega_sq = (ss_between - df_between * ms_within) / (ss_total := ss_between + ss_within + ms_within)

        return {
            "test": "One-Way ANOVA",
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "df_between": df_between,
            "df_within": df_within,
            "ss_between": float(ss_between),
            "ss_within": float(ss_within),
            "ms_between": float(ms_between),
            "ms_within": float(ms_within),
            "eta_squared": float(eta_sq),
            "omega_squared": float(omega_sq),
            "significant": bool(p_value < 0.05),
            "group_means": {name: float(g.mean()) for name, g in data.groupby(between)[dv]},
            "interpretation": (
                f"A one-way ANOVA revealed {'' if p_value < 0.05 else 'no '}"
                f"significant effect of {between} on {dv} "
                f"(F({df_between}, {df_within}) = {f_stat:.4f}, p = {p_value:.4f}, "
                f"η² = {eta_sq:.3f})."
            ),
        }

    @staticmethod
    def two_way(data: pd.DataFrame, dv: str, factor1: str, factor2: str) -> Dict[str, Any]:
        formula = f"{dv} ~ C({factor1}) * C({factor2})"
        model = ols(formula, data=data).fit()
        table = anova_lm(model, typ=2)

        results = {"test": "Two-Way ANOVA", "factors": [factor1, factor2], "effects": {}}
        for effect in table.index:
            row = table.loc[effect]
            p_val = row["PR(>F)"]
            results["effects"][effect] = {
                "ss": float(row["sum_sq"]),
                "df": int(row["df"]),
                "f": float(row["F"]),
                "p_value": float(row["PR(>F)"]),
                "significant": bool(p_val < 0.05),
            }

        results["interpretation"] = "Two-way ANOVA results: "
        for effect, res in results["effects"].items():
            results["interpretation"] += (
                f"{effect}: F({res['df']}, {int(model.df_resid)}) = {res['f']:.4f}, "
                f"p = {res['p_value']:.4f}; "
            )

        return results

    @staticmethod
    def repeated_measures(data: pd.DataFrame, dv: str, within: str, subject: str) -> Dict[str, Any]:
        try:
            from statsmodels.stats.anova import AnovaRM
            anova = AnovaRM(data, dv, subject, within=[within])
            result = anova.fit()
            table = result.anova_table
            effect = table.index[0]
            row = table.loc[effect]
            p_val = row["Pr > F"]
            return {
                "test": "Repeated Measures ANOVA",
                "f_statistic": float(row["F Value"]),
                "p_value": float(row["Pr > F"]),
                "df_num": int(row["Num DF"]),
                "df_den": int(row["Den DF"]),
                "significant": bool(p_val < 0.05),
                "interpretation": (
                    f"Repeated measures ANOVA: F({int(row['Num DF'])}, {int(row['Den DF'])}) = {row['F Value']:.4f}, "
                    f"p = {row['Pr > F']:.4f}. "
                    f"{'Significant within-subjects effect.' if p_val < 0.05 else 'No significant effect.'}"
                ),
            }
        except Exception as e:
            return {"error": f"Repeated measures ANOVA failed: {str(e)}"}

    @staticmethod
    def ancova(data: pd.DataFrame, dv: str, factor: str, covariate: str) -> Dict[str, Any]:
        formula = f"{dv} ~ C({factor}) + {covariate}"
        model = ols(formula, data=data).fit()
        table = anova_lm(model, typ=2)
        results = {"test": "ANCOVA", "factor": factor, "covariate": covariate, "effects": {}}
        for effect in table.index:
            row = table.loc[effect]
            p_val = row["PR(>F)"]
            results["effects"][effect] = {
                "ss": float(row["sum_sq"]),
                "df": int(row["df"]),
                "f": float(row["F"]),
                "p_value": float(row["PR(>F)"]),
                "significant": bool(p_val < 0.05),
            }
        results["interpretation"] = "ANCOVA results controlling for {covariate}: ".format(**locals())
        for effect, res in results["effects"].items():
            if res["df"] > 0:
                results["interpretation"] += f"{effect}: F({res['df']}, {int(model.df_resid)}) = {res['f']:.4f}, p = {res['p_value']:.4f}; "
        return results

    @staticmethod
    def mixed_model(data: pd.DataFrame, dv: str, between: str, within: str, subject: str) -> Dict[str, Any]:
        try:
            import pingouin as pg
            result = pg.mixed_anova(data=data, dv=dv, between=between, within=within, subject=subject)
            results = {"test": "Mixed Model ANOVA", "between": between, "within": within}
            for idx, row in result.iterrows():
                results[idx] = {
                    "f": float(row["F"]),
                    "df1": int(row["DF1"]),
                    "df2": int(row["DF2"]),
                    "p_value": float(row["p-unc"]),
                    "np2": float(row["np2"]),
                    "significant": bool(row["p-unc"] < 0.05),
                }
            return results
        except Exception as e:
            return {"error": f"Mixed ANOVA failed: {str(e)}"}

    @staticmethod
    def three_way(data: pd.DataFrame, dv: str, f1: str, f2: str, f3: str) -> Dict[str, Any]:
        formula = f"{dv} ~ C({f1}) * C({f2}) * C({f3})"
        model = ols(formula, data=data).fit()
        table = anova_lm(model, typ=2)
        results = {"test": "Three-Way ANOVA", "factors": [f1, f2, f3], "effects": {}}
        for effect in table.index:
            row = table.loc[effect]
            p_val = row["PR(>F)"]
            results["effects"][effect] = {
                "ss": float(row["sum_sq"]),
                "df": int(row["df"]),
                "f": float(row["F"]),
                "p_value": float(row["PR(>F)"]),
                "significant": bool(p_val < 0.05),
            }
        return results
