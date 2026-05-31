import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, List


class HypothesisTests:
    @staticmethod
    def one_sample_t(data: pd.Series, mu: float = 0, alternative: str = "two-sided") -> Dict[str, Any]:
        series = data.dropna()
        t_stat, p_value = stats.ttest_1samp(series, mu, alternative=alternative)
        n = len(series)
        dof = n - 1
        return {
            "test": "One Sample t-Test",
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "df": dof,
            "n": n,
            "mean": float(np.mean(series)),
            "std": float(np.std(series, ddof=1)),
            "effect_size": float(abs(np.mean(series) - mu) / np.std(series, ddof=1)),
            "ci_95": [float(c) for c in stats.t.interval(0.95, dof, loc=np.mean(series), scale=stats.sem(series))],
            "significant": bool(p_value < 0.05),
            "interpretation": (
                f"A one-sample t-test was conducted comparing the sample mean ({np.mean(series):.3f}) "
                f"to the hypothesized mean ({mu}). Results: t({dof}) = {t_stat:.4f}, p = {p_value:.4f}. "
                f"{'The result is statistically significant.' if p_value < 0.05 else 'The result is not statistically significant.'}"
            ),
        }

    @staticmethod
    def independent_t(data1: pd.Series, data2: pd.Series, alternative: str = "two-sided") -> Dict[str, Any]:
        s1, s2 = data1.dropna(), data2.dropna()
        t_stat, p_value = stats.ttest_ind(s1, s2, alternative=alternative)
        n1, n2 = len(s1), len(s2)
        dof = n1 + n2 - 2
        sp = np.sqrt(((n1 - 1) * np.var(s1, ddof=1) + (n2 - 1) * np.var(s2, ddof=1)) / dof)
        cohens_d = abs(np.mean(s1) - np.mean(s2)) / sp
        return {
            "test": "Independent Samples t-Test",
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "df": dof,
            "n1": n1,
            "n2": n2,
            "mean1": float(np.mean(s1)),
            "mean2": float(np.mean(s2)),
            "std1": float(np.std(s1, ddof=1)),
            "std2": float(np.std(s2, ddof=1)),
            "cohens_d": float(cohens_d),
            "significant": bool(p_value < 0.05),
            "interpretation": (
                f"An independent samples t-test was conducted. "
                f"Group 1 (M = {np.mean(s1):.3f}, SD = {np.std(s1, ddof=1):.3f}) "
                f"and Group 2 (M = {np.mean(s2):.3f}, SD = {np.std(s2, ddof=1):.3f}). "
                f"Results: t({dof}) = {t_stat:.4f}, p = {p_value:.4f}, d = {cohens_d:.3f}. "
                f"{'Significant difference found.' if p_value < 0.05 else 'No significant difference found.'}"
            ),
        }

    @staticmethod
    def paired_t(data1: pd.Series, data2: pd.Series, alternative: str = "two-sided") -> Dict[str, Any]:
        s1, s2 = data1.dropna(), data2.dropna()
        t_stat, p_value = stats.ttest_rel(s1, s2, alternative=alternative)
        n = len(s1)
        dof = n - 1
        diff = s1 - s2
        cohens_d = abs(np.mean(diff)) / np.std(diff, ddof=1)
        return {
            "test": "Paired Samples t-Test",
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "df": dof,
            "n": n,
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff, ddof=1)),
            "cohens_d": float(cohens_d),
            "significant": bool(p_value < 0.05),
            "interpretation": (
                f"A paired t-test was conducted comparing pre and post measurements. "
                f"Mean difference = {np.mean(diff):.3f} (SD = {np.std(diff, ddof=1):.3f}). "
                f"Results: t({dof}) = {t_stat:.4f}, p = {p_value:.4f}, d = {cohens_d:.3f}. "
                f"{'Significant difference found.' if p_value < 0.05 else 'No significant difference found.'}"
            ),
        }

    @staticmethod
    def mann_whitney(data1: pd.Series, data2: pd.Series, alternative: str = "two-sided") -> Dict[str, Any]:
        s1, s2 = data1.dropna(), data2.dropna()
        u_stat, p_value = stats.mannwhitneyu(s1, s2, alternative=alternative)
        n1, n2 = len(s1), len(s2)
        z = (u_stat - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        r = z / np.sqrt(n1 + n2)
        return {
            "test": "Mann-Whitney U Test",
            "statistic": float(u_stat),
            "p_value": float(p_value),
            "n1": n1,
            "n2": n2,
            "z_score": float(z),
            "effect_size_r": float(r),
            "significant": bool(p_value < 0.05),
            "interpretation": (
                f"Mann-Whitney U = {u_stat:.2f}, p = {p_value:.4f}, r = {r:.3f}. "
                f"{'Significant difference between groups.' if p_value < 0.05 else 'No significant difference between groups.'}"
            ),
        }

    @staticmethod
    def wilcoxon(data1: pd.Series, data2: pd.Series, alternative: str = "two-sided") -> Dict[str, Any]:
        s1, s2 = data1.dropna(), data2.dropna()
        w_stat, p_value = stats.wilcoxon(s1, s2, alternative=alternative)
        n = len(s1)
        return {
            "test": "Wilcoxon Signed-Rank Test",
            "statistic": float(w_stat),
            "p_value": float(p_value),
            "n": n,
            "significant": bool(p_value < 0.05),
            "interpretation": (
                f"Wilcoxon W = {w_stat:.2f}, p = {p_value:.4f}. "
                f"{'Significant difference found.' if p_value < 0.05 else 'No significant difference found.'}"
            ),
        }

    @staticmethod
    def friedman(*groups: pd.Series) -> Dict[str, Any]:
        clean_groups = [g.dropna() for g in groups]
        f_stat, p_value = stats.friedmanchisquare(*clean_groups)
        k = len(clean_groups)
        n = len(clean_groups[0])
        return {
            "test": "Friedman Test",
            "statistic": float(f_stat),
            "p_value": float(p_value),
            "df": k - 1,
            "n": n,
            "groups": k,
            "significant": bool(p_value < 0.05),
            "interpretation": (
                f"Friedman χ²({k - 1}) = {f_stat:.4f}, p = {p_value:.4f}. "
                f"{'Significant differences between conditions.' if p_value < 0.05 else 'No significant differences between conditions.'}"
            ),
        }

    @staticmethod
    def kruskal_wallis(*groups: pd.Series) -> Dict[str, Any]:
        clean_groups = [g.dropna() for g in groups]
        h_stat, p_value = stats.kruskal(*clean_groups)
        k = len(clean_groups)
        return {
            "test": "Kruskal-Wallis Test",
            "statistic": float(h_stat),
            "p_value": float(p_value),
            "df": k - 1,
            "groups": k,
            "significant": bool(p_value < 0.05),
            "interpretation": (
                f"Kruskal-Wallis H({k - 1}) = {h_stat:.4f}, p = {p_value:.4f}. "
                f"{'Significant differences between groups.' if p_value < 0.05 else 'No significant differences between groups.'}"
            ),
        }

    @staticmethod
    def chi_square(observed: pd.DataFrame) -> Dict[str, Any]:
        observed_array = observed.values if isinstance(observed, pd.DataFrame) else observed
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed_array)
        n = observed_array.sum()
        cramers_v = np.sqrt(chi2_stat / (n * min(observed_array.shape) - 1))
        return {
            "test": "Chi-Square Test of Independence",
            "statistic": float(chi2_stat),
            "p_value": float(p_value),
            "df": int(dof),
            "n": int(n),
            "cramers_v": float(cramers_v),
            "expected": expected.tolist(),
            "significant": bool(p_value < 0.05),
            "interpretation": (
                f"Chi-Square χ²({dof}) = {chi2_stat:.4f}, p = {p_value:.4f}, "
                f"Cramér's V = {cramers_v:.3f}. "
                f"{'Significant association between variables.' if p_value < 0.05 else 'No significant association.'}"
            ),
        }

    @staticmethod
    def binomial_test(data: pd.Series, p: float = 0.5) -> Dict[str, Any]:
        series = data.dropna()
        if series.nunique() > 2:
            return {"error": "Binomial test requires binary data"}
        unique_vals = series.unique()
        success = unique_vals[0]
        k = int((series == success).sum())
        n = len(series)
        result = stats.binomtest(k, n, p)
        return {
            "test": "Binomial Test",
            "successes": k,
            "trials": n,
            "p_value": float(result.pvalue),
            "proportion": float(k / n),
            "hypothesized_p": p,
            "ci_95": [float(c) for c in result.proportion_ci()],
            "significant": bool(result.pvalue < 0.05),
            "interpretation": (
                f"Binomial test: {k}/{n} successes ({k / n:.1%}). "
                f"p = {result.pvalue:.4f} vs hypothesized p = {p}. "
                f"{'Significantly different from hypothesized proportion.' if result.pvalue < 0.05 else 'Not significantly different.'}"
            ),
        }
