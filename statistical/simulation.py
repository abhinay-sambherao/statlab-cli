import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, List


class Simulation:
    @staticmethod
    def monte_carlo(n_simulations: int = 10000, n_samples: int = 30,
                    distribution: str = "normal", params: Optional[Dict] = None) -> Dict[str, Any]:
        if params is None:
            params = {"mean": 0, "std": 1}

        results = []
        for _ in range(n_simulations):
            if distribution == "normal":
                samples = np.random.normal(params.get("mean", 0), params.get("std", 1), n_samples)
            elif distribution == "uniform":
                samples = np.random.uniform(params.get("low", 0), params.get("high", 1), n_samples)
            elif distribution == "exponential":
                samples = np.random.exponential(params.get("scale", 1), n_samples)
            elif distribution == "binomial":
                samples = np.random.binomial(params.get("n", 10), params.get("p", 0.5), n_samples)
            elif distribution == "poisson":
                samples = np.random.poisson(params.get("lam", 5), n_samples)
            else:
                samples = np.random.normal(0, 1, n_samples)
            results.append(np.mean(samples))

        results_arr = np.array(results)
        return {
            "test": "Monte Carlo Simulation",
            "distribution": distribution,
            "n_simulations": n_simulations,
            "n_samples": n_samples,
            "params": params,
            "simulated_means": results_arr.tolist()[:1000],
            "mean_of_means": float(np.mean(results_arr)),
            "std_of_means": float(np.std(results_arr, ddof=1)),
            "percentiles": {
                "2.5": float(np.percentile(results_arr, 2.5)),
                "25": float(np.percentile(results_arr, 25)),
                "50": float(np.percentile(results_arr, 50)),
                "75": float(np.percentile(results_arr, 75)),
                "97.5": float(np.percentile(results_arr, 97.5)),
            },
            "interpretation": (
                f"Monte Carlo simulation ({n_simulations} iterations, {n_samples} samples each, "
                f"{distribution} distribution). Mean of means = {np.mean(results_arr):.4f}, "
                f"SE of means = {np.std(results_arr, ddof=1):.4f}."
            ),
        }

    @staticmethod
    def clt_demonstration(population: pd.Series, n_samples: int = 1000,
                          sample_sizes: List[int] = [5, 10, 30, 50]) -> Dict[str, Any]:
        pop = population.dropna()
        results = {}
        for n in sample_sizes:
            sample_means = [np.mean(np.random.choice(pop, size=n, replace=True)) for _ in range(n_samples)]
            results[str(n)] = {
                "sample_means": sample_means[:500],
                "mean": float(np.mean(sample_means)),
                "std": float(np.std(sample_means, ddof=1)),
                "skewness": float(stats.skew(sample_means)),
                "kurtosis": float(stats.kurtosis(sample_means, fisher=True)),
            }

        return {
            "test": "Central Limit Theorem Demonstration",
            "population_mean": float(np.mean(pop)),
            "population_std": float(np.std(pop, ddof=1)),
            "population_skewness": float(stats.skew(pop)),
            "n_simulations": n_samples,
            "results": results,
            "interpretation": (
                f"CLT demonstration with population N={len(pop)}, μ={np.mean(pop):.4f}, σ={np.std(pop, ddof=1):.4f}. "
                f"As sample size increases, sampling distribution becomes more normal."
            ),
        }

    @staticmethod
    def bootstrap(data: pd.Series, n_iterations: int = 1000,
                  statistic: str = "mean") -> Dict[str, Any]:
        series = data.dropna()
        n = len(series)
        boot_stats = []

        for _ in range(n_iterations):
            sample = np.random.choice(series, size=n, replace=True)
            if statistic == "mean":
                boot_stats.append(np.mean(sample))
            elif statistic == "median":
                boot_stats.append(np.median(sample))
            elif statistic == "std":
                boot_stats.append(np.std(sample, ddof=1))
            elif statistic == "correlation":
                boot_stats.append(np.mean(sample))

        boot_arr = np.array(boot_stats)
        return {
            "test": f"Bootstrap Simulation ({statistic})",
            "n_iterations": n_iterations,
            "n_samples": n,
            "observed_statistic": float(getattr(np, statistic)(series)) if hasattr(np, statistic) else float(np.mean(series)),
            "bootstrap_mean": float(np.mean(boot_arr)),
            "bootstrap_std": float(np.std(boot_arr, ddof=1)),
            "bias": float(np.mean(boot_arr) - getattr(np, statistic)(series)) if hasattr(np, statistic) else 0,
            "ci_95": [float(np.percentile(boot_arr, 2.5)), float(np.percentile(boot_arr, 97.5))],
            "distribution": boot_arr.tolist()[:500],
            "interpretation": (
                f"Bootstrap ({n_iterations} iterations) for {statistic}. "
                f"Observed = {getattr(np, statistic)(series):.4f}, "
                f"Bootstrap SE = {np.std(boot_arr, ddof=1):.4f}, "
                f"95% CI = [{np.percentile(boot_arr, 2.5):.4f}, {np.percentile(boot_arr, 97.5):.4f}]."
            ),
        }

    @staticmethod
    def random_sampling(data: pd.DataFrame, sample_size: int = 100,
                        method: str = "simple") -> Dict[str, Any]:
        n = len(data)
        if method == "simple":
            sample = data.sample(n=min(sample_size, n))
        elif method == "stratified":
            strat_col = data.select_dtypes(include=["object", "category"]).columns[0]
            sample = data.groupby(strat_col, group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, sample_size // data[strat_col].nunique())))
            )
        elif method == "systematic":
            step = max(1, n // sample_size)
            indices = np.arange(0, n, step)[:sample_size]
            sample = data.iloc[indices]
        else:
            sample = data.sample(n=min(sample_size, n))

        return {
            "test": f"Random Sampling ({method})",
            "method": method,
            "population_size": n,
            "sample_size": len(sample),
            "sampling_fraction": float(len(sample) / n),
            "sample": sample.to_dict(orient="records")[:20],
            "interpretation": f"{method.capitalize()} random sample of {len(sample)} from {n} observations.",
        }
