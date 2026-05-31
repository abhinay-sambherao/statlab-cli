import numpy as np
from scipy import stats
from typing import Dict, Any, Optional


class PowerAnalysis:
    @staticmethod
    def sample_size_t_test(effect_size: float = 0.5, alpha: float = 0.05,
                           power: float = 0.8, alternative: str = "two-sided") -> Dict[str, Any]:
        tails = 2 if alternative == "two-sided" else 1
        z_alpha = stats.norm.ppf(1 - alpha / tails)
        z_beta = stats.norm.ppf(power)
        n = int(np.ceil((2 * (z_alpha + z_beta) ** 2) / (effect_size ** 2)))

        return {
            "test": "Sample Size for t-Test",
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power,
            "total_n": n,
            "n_per_group": int(np.ceil(n / 2)),
            "interpretation": f"Required N = {n} ({int(np.ceil(n / 2))} per group) for d = {effect_size}, α = {alpha}, power = {power}.",
        }

    @staticmethod
    def power_t_test(n: int, effect_size: float = 0.5, alpha: float = 0.05,
                     alternative: str = "two-sided") -> Dict[str, Any]:
        tails = 2 if alternative == "two-sided" else 1
        dof = n - 2
        t_crit = stats.t.ppf(1 - alpha / tails, dof)
        ncp = effect_size * np.sqrt(n / 2)
        power = 1 - stats.nct.cdf(t_crit, dof, ncp)

        return {
            "test": "Statistical Power for t-Test",
            "n": n,
            "n_per_group": int(np.ceil(n / 2)),
            "effect_size": effect_size,
            "alpha": alpha,
            "power": float(power),
            "interpretation": f"Power = {power:.4f} for N = {n}, d = {effect_size}, α = {alpha}. {'Adequate power.' if power >= 0.8 else 'Power below 0.8.'}",
        }

    @staticmethod
    def sample_size_anova(k: int, effect_size: float = 0.25, alpha: float = 0.05,
                          power: float = 0.8) -> Dict[str, Any]:
        f2 = effect_size ** 2
        u = k - 1
        v = 120
        for _ in range(100):
            lambda_ = f2 * (u + v + 1)
            f_crit = stats.f.ppf(1 - alpha, u, v)
            p = 1 - stats.ncf.cdf(f_crit, u, v, lambda_)
            prev_v = v
            if p > power:
                v = int(v * 0.9)
            else:
                v = int(v * 1.1)
            if abs(v - prev_v) <= 1:
                break

        n = int(np.ceil((u + v + 1) / k))
        total_n = n * k

        return {
            "test": "Sample Size for ANOVA",
            "k_groups": k,
            "effect_size_f": effect_size,
            "alpha": alpha,
            "power": power,
            "n_per_group": n,
            "total_n": total_n,
            "interpretation": f"Required N = {total_n} ({n} per group, {k} groups) for f = {effect_size}, α = {alpha}, power = {power}.",
        }

    @staticmethod
    def sample_size_correlation(r: float = 0.3, alpha: float = 0.05,
                                 power: float = 0.8) -> Dict[str, Any]:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        z_r = np.arctanh(r)
        n = int(np.ceil((z_alpha + z_beta) ** 2 / (z_r ** 2) + 3))

        return {
            "test": "Sample Size for Correlation",
            "expected_r": r,
            "alpha": alpha,
            "power": power,
            "total_n": n,
            "interpretation": f"Required N = {n} for r = {r}, α = {alpha}, power = {power}.",
        }

    @staticmethod
    def sample_size_proportion(p: float = 0.5, margin: float = 0.05,
                                alpha: float = 0.05) -> Dict[str, Any]:
        z = stats.norm.ppf(1 - alpha / 2)
        n = int(np.ceil(z ** 2 * p * (1 - p) / (margin ** 2)))

        return {
            "test": "Sample Size for Proportion",
            "expected_proportion": p,
            "margin_of_error": margin,
            "alpha": alpha,
            "total_n": n,
            "interpretation": f"Required N = {n} for p = {p}, margin = {margin:.1%}, α = {alpha}.",
        }

    @staticmethod
    def power_curve(n_range: range, effect_size: float = 0.5, alpha: float = 0.05) -> Dict[str, Any]:
        powers = []
        for n in n_range:
            result = PowerAnalysis.power_t_test(n, effect_size, alpha)
            powers.append(float(result["power"]))
        return {
            "test": "Power Curve",
            "n_values": list(n_range),
            "powers": powers,
            "effect_size": effect_size,
            "alpha": alpha,
            "interpretation": f"Power curve for d = {effect_size}, α = {alpha}.",
        }
