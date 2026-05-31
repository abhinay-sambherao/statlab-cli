import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List


class QualityControl:
    @staticmethod
    def _control_limits(data: np.ndarray, n_sigma: float = 3.0) -> Dict[str, float]:
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        return {
            "center": float(mean),
            "ucl": float(mean + n_sigma * std),
            "lcl": float(mean - n_sigma * std),
            "sigma": float(std),
        }

    @staticmethod
    def i_chart(data: pd.Series) -> Dict[str, Any]:
        values = data.dropna().values
        limits = QualityControl._control_limits(values)
        moving_ranges = np.abs(np.diff(values))

        return {
            "test": "I Chart (Individuals Chart)",
            "data": values.tolist(),
            "center_line": limits["center"],
            "ucl": limits["ucl"],
            "lcl": limits["lcl"],
            "sigma": limits["sigma"],
            "points_beyond_limits": int(np.sum((values > limits["ucl"]) | (values < limits["lcl"]))),
            "mean_moving_range": float(np.mean(moving_ranges)),
            "interpretation": (
                f"I Chart with center = {limits['center']:.3f}, "
                f"UCL = {limits['ucl']:.3f}, LCL = {limits['lcl']:.3f}. "
                f"{'Points outside control limits detected.' if np.sum((values > limits['ucl']) | (values < limits['lcl'])) > 0 else 'Process appears in control.'}"
            ),
        }

    @staticmethod
    def xbar_chart(data: pd.DataFrame, values_col: str, subgroup_col: str) -> Dict[str, Any]:
        subgroups = data.groupby(subgroup_col)[values_col]
        means = subgroups.mean().values
        ranges = subgroups.apply(lambda x: x.max() - x.min()).values

        grand_mean = np.mean(means)
        mean_range = np.mean(ranges)

        a2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
        subgroup_size = subgroups.size().iloc[0]
        a2_val = a2.get(int(subgroup_size), 0.308)

        return {
            "test": "X-Bar Chart",
            "subgroup_means": means.tolist(),
            "grand_mean": float(grand_mean),
            "mean_range": float(mean_range),
            "ucl": float(grand_mean + a2_val * mean_range),
            "lcl": float(grand_mean - a2_val * mean_range),
            "center_line": float(grand_mean),
            "subgroup_size": int(subgroup_size),
            "interpretation": f"X-Bar chart with {len(means)} subgroups of size {int(subgroup_size)}.",
        }

    @staticmethod
    def p_chart(data: pd.DataFrame, defects_col: str, sample_col: str) -> Dict[str, Any]:
        defects = data[defects_col].values
        sample_sizes = data[sample_col].values
        proportions = defects / sample_sizes
        p_bar = np.sum(defects) / np.sum(sample_sizes)

        limits = []
        for i, n in enumerate(sample_sizes):
            sigma = np.sqrt(p_bar * (1 - p_bar) / n)
            limits.append({
                "sample": i,
                "proportion": float(proportions[i]),
                "ucl": float(p_bar + 3 * sigma),
                "lcl": float(max(0, p_bar - 3 * sigma)),
                "center": float(p_bar),
            })

        return {
            "test": "P Chart (Proportion Chart)",
            "p_bar": float(p_bar),
            "total_defects": int(np.sum(defects)),
            "total_samples": int(np.sum(sample_sizes)),
            "limits": limits,
            "interpretation": f"P Chart with p̄ = {p_bar:.4f}, {len(data)} samples.",
        }

    @staticmethod
    def run_chart(data: pd.Series) -> Dict[str, Any]:
        values = data.dropna().values
        median = np.median(values)
        above_median = (values > median).astype(int)
        runs = 1
        for i in range(1, len(above_median)):
            if above_median[i] != above_median[i - 1]:
                runs += 1

        n1 = int(np.sum(above_median))
        n2 = len(above_median) - n1
        expected_runs = 1 + (2 * n1 * n2) / (n1 + n2)
        std_runs = np.sqrt(2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))) if (n1 + n2) > 0 else 0
        z = (runs - expected_runs) / std_runs if std_runs > 0 else 0

        return {
            "test": "Run Chart",
            "data": values.tolist(),
            "median": float(median),
            "n_runs": runs,
            "expected_runs": float(expected_runs),
            "z_score": float(z),
            "random_pattern": bool(abs(z) < 1.96),
            "interpretation": (
                f"Run chart with {runs} runs (expected {expected_runs:.1f}). "
                f"{'Pattern appears random.' if abs(z) < 1.96 else 'Non-random pattern detected.'}"
            ),
        }

    @staticmethod
    def i_mr_chart(data: pd.Series) -> Dict[str, Any]:
        i_chart = QualityControl.i_chart(data)
        values = data.dropna().values
        moving_ranges = np.abs(np.diff(values))
        mr_mean = np.mean(moving_ranges)
        mr_ucl = 3.267 * mr_mean
        mr_lcl = 0.0

        i_chart["moving_range_chart"] = {
            "moving_ranges": moving_ranges.tolist(),
            "center": float(mr_mean),
            "ucl": float(mr_ucl),
            "lcl": float(mr_lcl),
        }
        i_chart["test"] = "I-MR Chart"
        return i_chart

    @staticmethod
    def xbar_r_chart(data: pd.DataFrame, values_col: str, subgroup_col: str) -> Dict[str, Any]:
        xbar = QualityControl.xbar_chart(data, values_col, subgroup_col)
        subgroups = data.groupby(subgroup_col)[values_col]
        ranges = subgroups.apply(lambda x: x.max() - x.min()).values
        mean_range = np.mean(ranges)

        d3, d4 = {2: (0, 3.267), 3: (0, 2.574), 4: (0, 2.282), 5: (0, 2.114),
                   6: (0, 2.004), 7: (0.076, 1.924), 8: (0.136, 1.864),
                   9: (0.184, 1.816), 10: (0.223, 1.777)}
        subgroup_size = int(subgroups.size().iloc[0])
        d3_val, d4_val = d3.get(subgroup_size, (0.223, 1.777))

        xbar["r_chart"] = {
            "ranges": ranges.tolist(),
            "center": float(mean_range),
            "ucl": float(d4_val * mean_range),
            "lcl": float(max(0, d3_val * mean_range)),
            "subgroup_size": subgroup_size,
        }
        xbar["test"] = "X-Bar R Chart"
        return xbar

    @staticmethod
    def np_chart(data: pd.DataFrame, defects_col: str) -> Dict[str, Any]:
        defects = data[defects_col].values
        n = len(data)
        n_p_bar = np.mean(defects)
        sigma = np.sqrt(n_p_bar * (1 - n_p_bar / n))

        return {
            "test": "NP Chart (Number of Defectives)",
            "n_defects": defects.tolist(),
            "np_bar": float(n_p_bar),
            "center": float(n_p_bar),
            "ucl": float(n_p_bar + 3 * sigma),
            "lcl": float(max(0, n_p_bar - 3 * sigma)),
            "interpretation": f"NP Chart with np̄ = {n_p_bar:.2f}.",
        }

    @staticmethod
    def c_chart(data: pd.Series) -> Dict[str, Any]:
        defects = data.dropna().values
        c_bar = np.mean(defects)
        sigma = np.sqrt(c_bar)

        return {
            "test": "C Chart (Count of Defects)",
            "defects": defects.tolist(),
            "c_bar": float(c_bar),
            "center": float(c_bar),
            "ucl": float(c_bar + 3 * sigma),
            "lcl": float(max(0, c_bar - 3 * sigma)),
            "interpretation": f"C Chart with c̄ = {c_bar:.2f}.",
        }

    @staticmethod
    def u_chart(data: pd.DataFrame, defects_col: str, sample_size_col: str) -> Dict[str, Any]:
        defects = data[defects_col].values
        sizes = data[sample_size_col].values
        u_values = defects / sizes
        u_bar = np.sum(defects) / np.sum(sizes)

        limits = []
        for i, n in enumerate(sizes):
            sigma = np.sqrt(u_bar / n)
            limits.append({
                "sample": i,
                "u": float(u_values[i]),
                "ucl": float(u_bar + 3 * sigma),
                "lcl": float(max(0, u_bar - 3 * sigma)),
                "center": float(u_bar),
            })

        return {
            "test": "U Chart (Defects per Unit)",
            "u_bar": float(u_bar),
            "limits": limits,
            "interpretation": f"U Chart with ū = {u_bar:.4f}.",
        }
