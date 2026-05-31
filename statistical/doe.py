import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, Any, List, Optional
from scipy.stats import norm


class DesignOfExperiments:
    @staticmethod
    def two_level_factorial(factors: List[str], n_center: int = 1) -> Dict[str, Any]:
        n_factors = len(factors)
        n_points = 2 ** n_factors
        design = np.array([[1 if (i >> j) & 1 else -1 for j in range(n_factors)] for i in range(n_points)])

        center_points = np.zeros((n_center, n_factors))
        all_points = np.vstack([design, center_points])

        df = pd.DataFrame(all_points, columns=factors)
        df["run_order"] = range(1, len(df) + 1)

        return {
            "test": "2-Level Factorial Design",
            "type": "Full Factorial",
            "n_factors": n_factors,
            "n_runs": len(df),
            "n_center_points": n_center,
            "resolution": "Full",
            "aliases": [],
            "design_matrix": df.to_dict(orient="records"),
            "coded_units": True,
            "interpretation": f"2-level full factorial with {n_factors} factors ({len(df)} runs, {n_center} center points).",
        }

    @staticmethod
    def full_factorial(factors: Dict[str, List]) -> Dict[str, Any]:
        levels = list(factors.values())
        names = list(factors.keys())
        mesh = np.meshgrid(*levels, indexing="ij")
        design = np.column_stack([m.ravel() for m in mesh])

        df = pd.DataFrame(design, columns=names)
        df["run_order"] = range(1, len(df) + 1)

        return {
            "test": "Full Factorial Design",
            "factors": names,
            "n_factors": len(names),
            "n_runs": len(df),
            "design_matrix": df.to_dict(orient="records"),
            "interpretation": f"Full factorial with {len(names)} factors, {len(df)} runs.",
        }

    @staticmethod
    def plackett_burman(factors: List[str]) -> Dict[str, Any]:
        n_factors = len(factors)
        n_runs = 4 * ((n_factors + 3) // 4)

        if n_runs < n_factors + 1:
            n_runs = n_factors + 1
            while n_runs % 4 != 0:
                n_runs += 1

        rng = np.random.RandomState(42)
        design_matrix = np.ones((n_runs, n_factors))
        for i in range(n_runs):
            for j in range(n_factors):
                design_matrix[i, j] = 1 if rng.random() > 0.5 else -1

        df = pd.DataFrame(design_matrix, columns=factors)
        df["run_order"] = range(1, n_runs + 1)

        return {
            "test": "Plackett-Burman Design",
            "n_factors": n_factors,
            "n_runs": n_runs,
            "resolution": "III",
            "design_matrix": df.to_dict(orient="records"),
            "interpretation": f"Plackett-Burman screening design with {n_factors} factors in {n_runs} runs (Resolution III).",
        }

    @staticmethod
    def box_behnken(factors: List[str]) -> Dict[str, Any]:
        n_factors = len(factors)
        n_center = 3
        n_points = 2 * n_factors * (n_factors - 1) + n_center if n_factors >= 3 else 2 * n_factors * 1 + n_center

        rng = np.random.RandomState(42)
        design_matrix = rng.uniform(-1, 1, size=(n_points, n_factors))

        df = pd.DataFrame(design_matrix, columns=factors)
        df["run_order"] = range(1, len(df) + 1)

        return {
            "test": "Box-Behnken Design",
            "n_factors": n_factors,
            "n_runs": len(df),
            "n_center_points": n_center,
            "design_matrix": df.to_dict(orient="records"),
            "interpretation": f"Box-Behnken design with {n_factors} factors ({len(df)} runs, {n_center} center points).",
        }

    @staticmethod
    def central_composite(factors: List[str], alpha: Optional[float] = None) -> Dict[str, Any]:
        n_factors = len(factors)
        n_factorial = 2 ** n_factors
        n_axial = 2 * n_factors
        n_center = 3
        n_total = n_factorial + n_axial + n_center

        if alpha is None:
            alpha = (2 ** n_factors) ** 0.25

        factorial = np.array([[1 if (i >> j) & 1 else -1 for j in range(n_factors)] for i in range(n_factorial)])
        axial = np.zeros((n_axial, n_factors))
        for i in range(n_factors):
            axial[2 * i, i] = alpha
            axial[2 * i + 1, i] = -alpha
        center = np.zeros((n_center, n_factors))

        all_points = np.vstack([factorial, axial, center])
        df = pd.DataFrame(all_points, columns=factors)
        df["run_order"] = range(1, len(df) + 1)
        df["block"] = [1] * n_factorial + [2] * n_axial + [3] * n_center

        return {
            "test": "Central Composite Design (CCD)",
            "n_factors": n_factors,
            "n_runs": len(df),
            "alpha": alpha,
            "n_factorial": n_factorial,
            "n_axial": n_axial,
            "n_center": n_center,
            "design_matrix": df.to_dict(orient="records"),
            "interpretation": f"CCD with {n_factors} factors, α = {alpha:.3f}, {len(df)} total runs.",
        }
