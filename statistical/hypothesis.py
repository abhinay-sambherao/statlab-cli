import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, List
import json

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _plot_to_dict(fig) -> Optional[Dict]:
    if not HAS_PLOTLY:
        return None
    return json.loads(fig.to_json())


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
    def one_sample_z(data: pd.Series, mu: float = 0, sigma: Optional[float] = None, alternative: str = "two-sided") -> Dict[str, Any]:
        series = data.dropna()
        n = len(series)
        xbar = float(np.mean(series))
        if sigma is not None:
            se = sigma / np.sqrt(n)
        else:
            se = float(np.std(series, ddof=1)) / np.sqrt(n)
        z_stat = (xbar - mu) / se

        if alternative == "two-sided":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alternative == "greater":
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            p_value = stats.norm.cdf(z_stat)

        significant = bool(p_value < 0.05)
        mean_diff = xbar - mu
        conclusion = (
            f"The null hypothesis is {'rejected' if significant else 'not rejected'}. "
            f"There is {'sufficient' if significant else 'insufficient'} evidence of a significant difference "
            f"between the sample mean ({xbar:.3f}) and the hypothesized mean ({mu}). "
            f"The mean difference is {mean_diff:.3f}."
        )

        assumption_checks = {
            "normality": "Large sample (n ≥ 30) or population is normally distributed",
            "independence": "Observations are independent",
            "known_sigma": "Population standard deviation is known" if sigma is not None else "Using sample SD as estimate (z-test is approximate)",
            "scale": "Data is continuous (interval/ratio)",
        }

        plots = {}
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=series, nbinsx=20, name="Data", opacity=0.7, histnorm="probability density"))
            kde_x = np.linspace(float(series.min()), float(series.max()), 200)
            kde_y = stats.gaussian_kde(series)(kde_x)
            fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode="lines", name="Density", line=dict(color="red")))
            fig.add_vline(x=xbar, line=dict(color="blue", width=2, dash="dash"), annotation_text=f"Sample Mean ({xbar:.2f})")
            fig.add_vline(x=mu, line=dict(color="green", width=2, dash="dot"), annotation_text=f"Hypothesized Mean ({mu})")
            fig.update_layout(title="One-Sample z-Test: Distribution of Data", xaxis_title="Value", yaxis_title="Density")
            plots["histogram"] = _plot_to_dict(fig)

        return {
            "test": "One-Sample z-Test",
            "test_type": "parametric",
            "purpose": "Tests whether the mean of a single group differs from a known/hypothesized value",
            "h0": f"μ = {mu}",
            "ha": f"μ ≠ {mu}" if alternative == "two-sided" else (f"μ > {mu}" if alternative == "greater" else f"μ < {mu}"),
            "tail_type": alternative,
            "statistic": float(z_stat),
            "statistic_name": "z",
            "p_value": p_value,
            "df": n - 1,
            "alpha": 0.05,
            "significant": significant,
            "decision": "Reject H₀" if significant else "Fail to Reject H₀",
            "n": n,
            "sample_mean": xbar,
            "hypothesized_mean": mu,
            "mean_difference": float(mean_diff),
            "standard_error": float(se),
            "effect_size": float(abs(mean_diff) / (sigma if sigma else np.std(series, ddof=1))),
            "ci_95": [float(xbar - 1.96 * se), float(xbar + 1.96 * se)],
            "assumptions": assumption_checks,
            "conclusion": conclusion,
            "plots": plots,
            "interpretation": (
                f"A one-sample z-test was conducted comparing the sample mean ({xbar:.3f}) "
                f"to the hypothesized mean ({mu}). Results: z = {z_stat:.4f}, p = {p_value:.4f}. "
                f"{'The result is statistically significant.' if significant else 'The result is not statistically significant.'}"
            ),
        }

    @staticmethod
    def two_sample_z(data1: pd.Series, data2: pd.Series, sigma1: Optional[float] = None, sigma2: Optional[float] = None, alternative: str = "two-sided") -> Dict[str, Any]:
        s1, s2 = data1.dropna(), data2.dropna()
        n1, n2 = len(s1), len(s2)
        xbar1, xbar2 = float(np.mean(s1)), float(np.mean(s2))

        if sigma1 is not None and sigma2 is not None:
            se = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
        else:
            se = np.sqrt(float(np.var(s1, ddof=1)) / n1 + float(np.var(s2, ddof=1)) / n2)

        z_stat = (xbar1 - xbar2) / se

        if alternative == "two-sided":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alternative == "greater":
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            p_value = stats.norm.cdf(z_stat)

        significant = bool(p_value < 0.05)
        mean_diff = xbar1 - xbar2
        conclusion = (
            f"The null hypothesis is {'rejected' if significant else 'not rejected'}. "
            f"There is {'sufficient' if significant else 'insufficient'} evidence of a significant difference "
            f"between the means of the two groups ({xbar1:.3f} vs {xbar2:.3f}). "
            f"The mean difference is {mean_diff:.3f}."
        )

        assumption_checks = {
            "normality": "Large samples (n₁ ≥ 30, n₂ ≥ 30) or populations are normally distributed",
            "independence": "Observations are independent within and between groups",
            "known_sigma": "Population standard deviations are known" if (sigma1 is not None and sigma2 is not None) else "Using sample SDs as estimates",
            "scale": "Data is continuous (interval/ratio)",
        }

        plots = {}
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=s1, nbinsx=20, name="Group 1", opacity=0.6, histnorm="probability density"))
            fig.add_trace(go.Histogram(x=s2, nbinsx=20, name="Group 2", opacity=0.6, histnorm="probability density"))
            fig.add_vline(x=xbar1, line=dict(color="blue", width=2, dash="dash"), annotation_text=f"Group 1 Mean ({xbar1:.2f})")
            fig.add_vline(x=xbar2, line=dict(color="red", width=2, dash="dash"), annotation_text=f"Group 2 Mean ({xbar2:.2f})")
            fig.update_layout(title="Two-Sample z-Test: Group Distributions", xaxis_title="Value", yaxis_title="Density", barmode="overlay")
            plots["histogram"] = _plot_to_dict(fig)

            fig2 = go.Figure()
            fig2.add_trace(go.Box(y=s1, name="Group 1", marker_color="blue"))
            fig2.add_trace(go.Box(y=s2, name="Group 2", marker_color="red"))
            fig2.update_layout(title="Two-Sample z-Test: Box Plot Comparison", yaxis_title="Value")
            plots["box_plot"] = _plot_to_dict(fig2)

        return {
            "test": "Two-Sample z-Test",
            "test_type": "parametric",
            "purpose": "Tests whether the means of two independent groups differ significantly",
            "h0": "μ₁ = μ₂",
            "ha": "μ₁ ≠ μ₂" if alternative == "two-sided" else (f"μ₁ > μ₂" if alternative == "greater" else "μ₁ < μ₂"),
            "tail_type": alternative,
            "statistic": float(z_stat),
            "statistic_name": "z",
            "p_value": p_value,
            "df": n1 + n2 - 2,
            "alpha": 0.05,
            "significant": significant,
            "decision": "Reject H₀" if significant else "Fail to Reject H₀",
            "n1": n1,
            "n2": n2,
            "sample_mean_1": xbar1,
            "sample_mean_2": xbar2,
            "mean_difference": float(mean_diff),
            "standard_error": float(se),
            "effect_size": float(abs(mean_diff) / np.sqrt((float(np.var(s1, ddof=1)) + float(np.var(s2, ddof=1))) / 2)),
            "ci_95": [float(mean_diff - 1.96 * se), float(mean_diff + 1.96 * se)],
            "assumptions": assumption_checks,
            "conclusion": conclusion,
            "plots": plots,
            "interpretation": (
                f"A two-sample z-test was conducted. Group 1 (M = {xbar1:.3f}) and Group 2 (M = {xbar2:.3f}). "
                f"Results: z = {z_stat:.4f}, p = {p_value:.4f}. "
                f"{'Significant difference found.' if significant else 'No significant difference found.'}"
            ),
        }

    @staticmethod
    def one_proportion_z(data: pd.Series, p0: float = 0.5, alternative: str = "two-sided") -> Dict[str, Any]:
        series = data.dropna()
        if series.nunique() > 2:
            return {"error": "Proportion z-test requires binary data"}
        if series.nunique() < 2:
            return {"error": "Proportion z-test requires at least two distinct values in data"}

        unique_vals = series.unique()
        success_val = unique_vals[0]
        x = int((series == success_val).sum())
        n = len(series)
        phat = x / n
        se = np.sqrt(p0 * (1 - p0) / n)
        z_stat = (phat - p0) / se

        if alternative == "two-sided":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alternative == "greater":
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            p_value = stats.norm.cdf(z_stat)

        significant = bool(p_value < 0.05)
        conclusion = (
            f"The null hypothesis is {'rejected' if significant else 'not rejected'}. "
            f"There is {'sufficient' if significant else 'insufficient'} evidence "
            f"that the population proportion differs from {p0}. "
            f"The sample proportion is {phat:.3f} ({int(phat * 100)}%)."
        )

        assumption_checks = {
            "random_sample": "Data is from a random sample",
            "large_sample": f"np₀ = {n * p0:.1f} ≥ 10 and n(1−p₀) = {n * (1 - p0):.1f} ≥ 10",
            "independence": "Observations are independent",
            "binary": "Outcome is binary (success/failure)",
        }

        ci = [float(phat - 1.96 * np.sqrt(phat * (1 - phat) / n)), float(phat + 1.96 * np.sqrt(phat * (1 - phat) / n))]
        plots = {}
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Observed"], y=[phat], name="Observed", marker_color="blue", width=[0.4]))
            fig.add_trace(go.Bar(x=["Hypothesized"], y=[p0], name="Hypothesized", marker_color="green", width=[0.4]))
            fig.add_trace(go.Bar(x=["CI Lower"], y=[ci[0]], name="CI Lower", marker_color="lightgray", width=[0.4], showlegend=False))
            fig.add_trace(go.Bar(x=["CI Upper"], y=[ci[1]], name="CI Upper", marker_color="lightgray", width=[0.4], showlegend=False))
            fig.add_error_bar(x=[0], y=[phat], ymin=[ci[0]], ymax=[ci[1]], name="95% CI")
            fig.update_layout(title="One-Proportion z-Test: Observed vs Hypothesized", yaxis_title="Proportion", yaxis_range=[0, 1])
            plots["proportion_bar"] = _plot_to_dict(fig)

        return {
            "test": "One-Proportion z-Test",
            "test_type": "parametric",
            "purpose": "Tests whether a population proportion differs from a hypothesized value",
            "h0": f"p = {p0}",
            "ha": f"p ≠ {p0}" if alternative == "two-sided" else (f"p > {p0}" if alternative == "greater" else f"p < {p0}"),
            "tail_type": alternative,
            "statistic": float(z_stat),
            "statistic_name": "z",
            "p_value": p_value,
            "df": n - 1,
            "alpha": 0.05,
            "significant": significant,
            "decision": "Reject H₀" if significant else "Fail to Reject H₀",
            "n": n,
            "successes": x,
            "sample_proportion": float(phat),
            "hypothesized_proportion": p0,
            "standard_error": float(se),
            "effect_size": float(abs(phat - p0) / np.sqrt(p0 * (1 - p0))),
            "ci_95": ci,
            "assumptions": assumption_checks,
            "conclusion": conclusion,
            "plots": plots,
            "interpretation": (
                f"A one-proportion z-test was conducted. Sample proportion = {phat:.3f} ({x}/{n}). "
                f"Results: z = {z_stat:.4f}, p = {p_value:.4f}. "
                f"{'Significantly different from hypothesized proportion.' if significant else 'Not significantly different.'}"
            ),
        }

    @staticmethod
    def two_proportion_z(data1: pd.Series, data2: pd.Series, alternative: str = "two-sided") -> Dict[str, Any]:
        s1, s2 = data1.dropna(), data2.dropna()
        if s1.nunique() > 2 or s2.nunique() > 2:
            return {"error": "Two-proportion z-test requires binary data in both groups"}
        if s1.nunique() < 2 or s2.nunique() < 2:
            return {"error": "Two-proportion z-test requires at least two distinct values in each group"}

        n1, n2 = len(s1), len(s2)
        x1 = int((s1 == s1.unique()[0]).sum())
        x2 = int((s2 == s2.unique()[0]).sum())
        phat1, phat2 = x1 / n1, x2 / n2

        phat_pooled = (x1 + x2) / (n1 + n2)
        se = np.sqrt(phat_pooled * (1 - phat_pooled) * (1 / n1 + 1 / n2))
        z_stat = (phat1 - phat2) / se

        if alternative == "two-sided":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alternative == "greater":
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            p_value = stats.norm.cdf(z_stat)

        significant = bool(p_value < 0.05)
        prop_diff = phat1 - phat2
        conclusion = (
            f"The null hypothesis is {'rejected' if significant else 'not rejected'}. "
            f"There is {'sufficient' if significant else 'insufficient'} evidence of a significant difference "
            f"between the two proportions ({phat1:.3f} vs {phat2:.3f}). "
            f"The difference in proportions is {prop_diff:.3f}."
        )

        assumption_checks = {
            "random_samples": "Both groups are from random samples",
            "large_samples": f"n₁p̂₁ = {n1 * phat1:.1f}, n₁(1−p̂₁) = {n1 * (1 - phat1):.1f}, n₂p̂₂ = {n2 * phat2:.1f}, n₂(1−p̂₂) = {n2 * (1 - phat2):.1f} — all ≥ 10 preferred",
            "independence": "Observations are independent within and between groups",
            "binary": "Outcome is binary in both groups",
        }

        plots = {}
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Group 1", "Group 2"], y=[phat1, phat2],
                                 marker_color=["blue", "red"], width=[0.4, 0.4],
                                 error_y=dict(type="data", array=[1.96 * np.sqrt(phat1 * (1 - phat1) / n1), 1.96 * np.sqrt(phat2 * (1 - phat2) / n2)], visible=True)))
            fig.add_hline(y=phat_pooled, line=dict(color="green", dash="dash"), annotation_text=f"Pooled proportion ({phat_pooled:.3f})")
            fig.update_layout(title="Two-Proportion z-Test: Group Proportions with 95% CI",
                              yaxis_title="Proportion", yaxis_range=[0, 1])
            plots["proportion_bar"] = _plot_to_dict(fig)

        return {
            "test": "Two-Proportion z-Test",
            "test_type": "parametric",
            "purpose": "Tests whether the proportions of two independent groups differ significantly",
            "h0": "p₁ = p₂",
            "ha": "p₁ ≠ p₂" if alternative == "two-sided" else (f"p₁ > p₂" if alternative == "greater" else "p₁ < p₂"),
            "tail_type": alternative,
            "statistic": float(z_stat),
            "statistic_name": "z",
            "p_value": p_value,
            "df": n1 + n2 - 2,
            "alpha": 0.05,
            "significant": significant,
            "decision": "Reject H₀" if significant else "Fail to Reject H₀",
            "n1": n1,
            "n2": n2,
            "successes_1": x1,
            "successes_2": x2,
            "sample_proportion_1": float(phat1),
            "sample_proportion_2": float(phat2),
            "proportion_difference": float(prop_diff),
            "standard_error": float(se),
            "effect_size_phi": float(abs(z_stat) / np.sqrt(n1 + n2)),
            "ci_95": [float(prop_diff - 1.96 * np.sqrt(phat1 * (1 - phat1) / n1 + phat2 * (1 - phat2) / n2)),
                      float(prop_diff + 1.96 * np.sqrt(phat1 * (1 - phat1) / n1 + phat2 * (1 - phat2) / n2))],
            "assumptions": assumption_checks,
            "conclusion": conclusion,
            "plots": plots,
            "interpretation": (
                f"A two-proportion z-test was conducted. Proportion 1 = {phat1:.3f} ({x1}/{n1}), "
                f"Proportion 2 = {phat2:.3f} ({x2}/{n2}). "
                f"Results: z = {z_stat:.4f}, p = {p_value:.4f}. "
                f"{'Significant difference between proportions.' if significant else 'No significant difference.'}"
            ),
        }

    @staticmethod
    def chi_square(observed: pd.DataFrame) -> Dict[str, Any]:
        observed_array = observed.values if isinstance(observed, pd.DataFrame) else observed
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed_array)
        n = observed_array.sum()
        min_dim = min(observed_array.shape)
        cramers_v = np.sqrt(chi2_stat / (n * (min_dim - 1))) if n > 0 and min_dim > 1 else 0
        significant = bool(p_value < 0.05)

        expected_flat = expected.flatten()
        observed_flat = observed_array.flatten()
        small_expected = [int(i) for i, e in enumerate(expected_flat) if e < 5]
        min_expected = float(np.min(expected_flat))
        pct_small = sum(1 for e in expected_flat if e < 5) / len(expected_flat) * 100

        assumption_checks = {
            "random_sample": "Data is from a random sample",
            "independence": "Observations are independent",
            "expected_frequencies": (
                f"{'✓ All expected frequencies ≥ 5' if pct_small == 0 else '⚠ ' + str(len(small_expected)) + ' cells (' + f'{pct_small:.0f}%) have expected frequency < 5'}"
            ),
            "min_expected": f"Minimum expected frequency = {min_expected:.2f}",
            "categorical": "Both variables are categorical (nominal or ordinal)",
        }

        conclusion = (
            f"The null hypothesis is {'rejected' if significant else 'not rejected'}. "
            f"There is {'sufficient' if significant else 'insufficient'} evidence of an association "
            f"between the two categorical variables. "
            f"The chi-square statistic is {chi2_stat:.4f} with {dof} degrees of freedom (p = {p_value:.4f}). "
            f"The strength of association (Cramér's V) is {cramers_v:.3f}."
        )

        plots = {}
        if HAS_PLOTLY:
            colors = px.colors.qualitative.Plotly
            row_labels = [str(i) for i in range(observed_array.shape[0])]
            col_labels = [str(j) for j in range(observed_array.shape[1])]
            fig_obs = go.Figure()
            for i in range(observed_array.shape[0]):
                fig_obs.add_trace(go.Bar(name=f"Observed: Row {i}", x=col_labels, y=observed_array[i],
                                          marker_color=colors[i % len(colors)]))
            fig_obs.update_layout(title="Chi-Square: Observed Frequencies", xaxis_title="Column", yaxis_title="Frequency", barmode="group")
            plots["observed_bar"] = _plot_to_dict(fig_obs)

            fig_exp = go.Figure()
            for i in range(expected.shape[0]):
                fig_exp.add_trace(go.Bar(name=f"Expected: Row {i}", x=col_labels, y=expected[i],
                                          marker_color=colors[i % len(colors)], marker_pattern_shape="/"))
            fig_exp.update_layout(title="Chi-Square: Expected Frequencies (Under Independence)",
                                  xaxis_title="Column", yaxis_title="Frequency", barmode="group")
            plots["expected_bar"] = _plot_to_dict(fig_exp)

            if observed_array.shape[0] <= 10 and observed_array.shape[1] <= 10:
                fig_heat = go.Figure(data=go.Heatmap(
                    z=observed_array.astype(float),
                    x=col_labels, y=row_labels,
                    text=observed_array.astype(str),
                    texttemplate="%{text}",
                    colorscale="Blues", showscale=True))
                fig_heat.update_layout(title="Chi-Square: Observed Frequency Heatmap",
                                       xaxis_title="Column", yaxis_title="Row")
                plots["heatmap"] = _plot_to_dict(fig_heat)

        return {
            "test": "Chi-Square Test of Independence",
            "test_type": "nonparametric",
            "purpose": "Tests whether two categorical variables are associated/independent",
            "h0": "The two variables are independent",
            "ha": "The two variables are associated (dependent)",
            "tail_type": "right-tailed",
            "statistic": float(chi2_stat),
            "statistic_name": "χ²",
            "p_value": p_value,
            "df": int(dof),
            "alpha": 0.05,
            "significant": significant,
            "decision": "Reject H₀" if significant else "Fail to Reject H₀",
            "n": int(n),
            "rows": int(observed_array.shape[0]),
            "columns": int(observed_array.shape[1]),
            "cramers_v": float(cramers_v),
            "expected": expected.tolist(),
            "assumptions": assumption_checks,
            "conclusion": conclusion,
            "plots": plots,
            "interpretation": (
                f"Chi-Square χ²({dof}) = {chi2_stat:.4f}, p = {p_value:.4f}, "
                f"Cramér's V = {cramers_v:.3f}. "
                f"{'Significant association between variables.' if significant else 'No significant association.'}"
            ),
        }

    @staticmethod
    def binomial_test(data: pd.Series, p: float = 0.5) -> Dict[str, Any]:
        series = data.dropna()
        if series.nunique() > 2:
            return {"error": "Binomial test requires binary data"}
        if series.nunique() < 2:
            return {"error": "Binomial test requires at least two distinct values in data"}
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
