import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
import base64
import io
import json


class VisualizationEngine:
    def __init__(self, dark_mode: bool = False):
        self.dark_mode = dark_mode
        self.template = "plotly_dark" if dark_mode else "plotly_white"

    def _fig_to_dict(self, fig: go.Figure) -> Dict:
        fig.update_layout(template=self.template)
        return json.loads(fig.to_json())

    def histogram(self, data: pd.Series, bins: int = 30, kde: bool = True) -> Dict:
        series = data.dropna()
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=series, nbinsx=bins, name="Histogram", opacity=0.7))
        if kde:
            kde_x = np.linspace(series.min(), series.max(), 200)
            kde_y = stats.gaussian_kde(series)(kde_x)
            fig.add_trace(go.Scatter(x=kde_x, y=kde_y * len(series) * (series.max() - series.min()) / bins,
                                      mode="lines", name="KDE", line=dict(color="red")))
        fig.update_layout(title=f"Histogram of {data.name}", xaxis_title=data.name, yaxis_title="Frequency")
        return self._fig_to_dict(fig)

    def bar_chart(self, data: pd.Series, top_n: Optional[int] = None) -> Dict:
        counts = data.value_counts()
        if top_n:
            counts = counts.head(top_n)
        fig = go.Figure([go.Bar(x=counts.index.astype(str), y=counts.values)])
        fig.update_layout(title=f"Bar Chart of {data.name}", xaxis_title=data.name, yaxis_title="Count")
        return self._fig_to_dict(fig)

    def scatter(self, x: pd.Series, y: pd.Series, color: Optional[pd.Series] = None) -> Dict:
        fig = px.scatter(x=x, y=y, color=color, trendline="ols",
                          title=f"{x.name} vs {y.name}")
        return self._fig_to_dict(fig)

    def line_chart(self, data: pd.DataFrame, x_col: str, y_cols: List[str]) -> Dict:
        fig = go.Figure()
        for col in y_cols:
            fig.add_trace(go.Scatter(x=data[x_col], y=data[col], mode="lines+markers", name=col))
        fig.update_layout(title="Line Chart", xaxis_title=x_col, yaxis_title="Value")
        return self._fig_to_dict(fig)

    def boxplot(self, data: pd.DataFrame, y_col: str, x_col: Optional[str] = None) -> Dict:
        fig = px.box(data, y=y_col, x=x_col, title=f"Boxplot of {y_col}")
        return self._fig_to_dict(fig)

    def violin(self, data: pd.DataFrame, y_col: str, x_col: Optional[str] = None) -> Dict:
        fig = px.violin(data, y=y_col, x=x_col, box=True, points="all",
                         title=f"Violin Plot of {y_col}")
        return self._fig_to_dict(fig)

    def raincloud(self, data: pd.DataFrame, y_col: str, x_col: str) -> Dict:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             vertical_spacing=0.02, row_heights=[0.5, 0.25, 0.25])
        groups = data[x_col].unique()
        colors = px.colors.qualitative.Plotly
        for i, group in enumerate(groups):
            group_data = data[data[x_col] == group][y_col].dropna()
            color = colors[i % len(colors)]
            kde = stats.gaussian_kde(group_data)
            x_kde = np.linspace(group_data.min(), group_data.max(), 100)
            y_kde = kde(x_kde)
            fig.add_trace(go.Scatter(x=y_kde, y=x_kde, fill="toself", mode="lines",
                                      name=f"{group}", line=dict(color=color),
                                      showlegend=False), row=1, col=1)
            fig.add_trace(go.Box(x=group_data, name=str(group), marker_color=color,
                                  line=dict(color=color)), row=2, col=1)
            fig.add_trace(go.Scatter(x=group_data,
                                      y=np.random.normal(0.5, 0.02, len(group_data)),
                                      mode="markers", marker=dict(color=color, size=4),
                                      showlegend=False, opacity=0.6), row=3, col=1)
        fig.update_layout(title=f"Raincloud Plot of {y_col} by {x_col}", height=600)
        return self._fig_to_dict(fig)

    def pareto(self, data: pd.Series) -> Dict:
        counts = data.value_counts().sort_values(ascending=False)
        cum_pct = np.cumsum(counts.values) / counts.sum() * 100
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=counts.index.astype(str), y=counts.values, name="Frequency"),
                       secondary_y=False)
        fig.add_trace(go.Scatter(x=counts.index.astype(str), y=cum_pct, mode="lines+markers",
                                  name="Cumulative %"), secondary_y=True)
        fig.update_layout(title="Pareto Chart")
        fig.update_yaxes(title_text="Frequency", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 110])
        return self._fig_to_dict(fig)

    def heatmap(self, data: pd.DataFrame) -> Dict:
        numeric = data.select_dtypes(include=[np.number])
        corr = numeric.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                         title="Correlation Heatmap",
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        return self._fig_to_dict(fig)

    def correlation_matrix(self, data: pd.DataFrame, method: str = "pearson") -> Dict:
        numeric = data.select_dtypes(include=[np.number])
        if method == "pearson":
            corr = numeric.corr(method="pearson")
        elif method == "spearman":
            corr = numeric.corr(method="spearman")
        else:
            corr = numeric.corr(method="kendall")
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                         colorscale="RdBu_r", zmin=-1, zmax=1,
                                         text=np.round(corr.values, 2), texttemplate="%{text}"))
        fig.update_layout(title=f"{method.capitalize()} Correlation Matrix", height=600, width=600)
        return self._fig_to_dict(fig)

    def pairplot(self, data: pd.DataFrame, color_col: Optional[str] = None) -> Dict:
        numeric = data.select_dtypes(include=[np.number])
        cols = numeric.columns[:5]
        n = len(cols)
        fig = make_subplots(rows=n, cols=n, shared_xaxes=True, shared_yaxes=True,
                             horizontal_spacing=0.02, vertical_spacing=0.02)
        colors = data[color_col] if color_col else None
        for i, col_i in enumerate(cols):
            for j, col_j in enumerate(cols):
                if i == j:
                    fig.add_trace(go.Histogram(x=numeric[col_i], showlegend=False), row=i + 1, col=j + 1)
                elif i > j:
                    scatter_data = pd.DataFrame({col_i: numeric[col_i], col_j: numeric[col_j]})
                    fig.add_trace(go.Scatter(x=numeric[col_i], y=numeric[col_j], mode="markers",
                                              marker=dict(size=3, color=colors if colors is None else colors.astype(str)),
                                              showlegend=False), row=i + 1, col=j + 1)
        fig.update_layout(title="Pairplot", height=200 * n, width=200 * n)
        return self._fig_to_dict(fig)

    def qq_plot(self, data: pd.Series) -> Dict:
        series = data.dropna()
        qq = stats.probplot(series, dist="norm")
        theoretical = qq[0][0]
        ordered = qq[0][1]
        r_value = qq[1][0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=theoretical, y=ordered, mode="markers", name="Data"))
        fit_line = np.polyfit(theoretical, ordered, 1)
        fig.add_trace(go.Scatter(x=theoretical, y=np.polyval(fit_line, theoretical),
                                  mode="lines", name="Expected", line=dict(color="red", dash="dash")))
        fig.update_layout(title=f"Q-Q Plot (R² = {r_value ** 2:.4f})",
                           xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        return self._fig_to_dict(fig)

    def kde_plot(self, data: pd.Series, group_col: Optional[pd.Series] = None) -> Dict:
        if group_col is not None:
            fig = px.density(data_frame=pd.DataFrame({data.name: data, "group": group_col}),
                              x=data.name, color="group", marginal="rug")
        else:
            fig = px.density(data_frame=pd.DataFrame({data.name: data}),
                              x=data.name, marginal="rug")
        fig.update_layout(title=f"KDE Plot of {data.name}")
        return self._fig_to_dict(fig)

    def bland_altman(self, x: pd.Series, y: pd.Series) -> Dict:
        s1, s2 = x.dropna(), y.dropna()
        diffs = s1 - s2
        means = (s1 + s2) / 2
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=means, y=diffs, mode="markers", name="Differences"))
        fig.add_hline(y=mean_diff, line=dict(color="blue", width=2), name="Mean diff")
        fig.add_hline(y=loa_upper, line=dict(color="red", dash="dash"), name="+1.96 SD")
        fig.add_hline(y=loa_lower, line=dict(color="red", dash="dash"), name="-1.96 SD")
        fig.update_layout(title="Bland-Altman Plot",
                           xaxis_title="Mean of measurements", yaxis_title="Difference")
        return self._fig_to_dict(fig)

    def roc_curve(self, y_true: pd.Series, y_score: pd.Series) -> Dict:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {roc_auc:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                  line=dict(dash="dash", color="gray")))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate", xaxis_range=[0, 1], yaxis_range=[0, 1])
        return self._fig_to_dict(fig)

    def sankey(self, data: pd.DataFrame, source_col: str, target_col: str, value_col: Optional[str] = None) -> Dict:
        labels = list(set(data[source_col].unique()) | set(data[target_col].unique()))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        source_idx = [label_to_idx[s] for s in data[source_col]]
        target_idx = [label_to_idx[t] for t in data[target_col]]
        values = data[value_col].tolist() if value_col else [1] * len(data)

        fig = go.Figure(data=[go.Sankey(node=dict(label=labels, pad=15, thickness=20),
                                         link=dict(source=source_idx, target=target_idx, value=values))])
        fig.update_layout(title="Sankey Diagram", height=500)
        return self._fig_to_dict(fig)

    def survival_curve(self, times: List[float], survival_probs: List[float],
                        ci_lower: Optional[List[float]] = None,
                        ci_upper: Optional[List[float]] = None, label: str = "Survival") -> Dict:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=survival_probs, mode="lines", name=label,
                                  line=dict(width=2)))
        if ci_lower and ci_upper:
            fig.add_trace(go.Scatter(x=times + times[::-1],
                                      y=ci_upper + ci_lower[::-1],
                                      fill="toself", fillcolor="rgba(0,100,200,0.2)",
                                      line=dict(width=0), name="95% CI"))
        fig.update_layout(title="Kaplan-Meier Survival Curve",
                           xaxis_title="Time", yaxis_title="Survival Probability")
        return self._fig_to_dict(fig)
