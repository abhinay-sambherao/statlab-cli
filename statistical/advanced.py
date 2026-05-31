import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from typing import Dict, Any, List, Optional


class AdvancedAnalysis:
    @staticmethod
    def pca(data: pd.DataFrame, n_components: Optional[int] = None) -> Dict[str, Any]:
        numeric = data.select_dtypes(include=[np.number]).dropna()
        if n_components is None:
            n_components = min(numeric.shape[1], 10)
        n_components = min(n_components, numeric.shape[1], numeric.shape[0])

        X = (numeric - numeric.mean()) / numeric.std()
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X)

        return {
            "test": "Principal Component Analysis (PCA)",
            "n_components": n_components,
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "cumulative_variance": float(np.cumsum(pca.explained_variance_ratio_)[-1]),
            "eigenvalues": [float(v) for v in pca.explained_variance_],
            "loadings": [
                {numeric.columns[j]: float(pca.components_[i][j]) for j in range(len(numeric.columns))}
                for i in range(n_components)
            ],
            "component_scores": components[:, :2].tolist(),
            "total_variance_explained": float(sum(pca.explained_variance_ratio_)),
            "interpretation": (
                f"PCA extracted {n_components} components explaining "
                f"{sum(pca.explained_variance_ratio_) * 100:.1f}% of total variance. "
                f"PC1 explains {pca.explained_variance_ratio_[0] * 100:.1f}% of variance."
            ),
        }

    @staticmethod
    def factor_analysis(data: pd.DataFrame, n_factors: Optional[int] = None) -> Dict[str, Any]:
        numeric = data.select_dtypes(include=[np.number]).dropna()
        if n_factors is None:
            n_factors = max(1, numeric.shape[1] // 3)
        n_factors = min(n_factors, numeric.shape[1])

        kmo_all, kmo_model = calculate_kmo(numeric)
        chi2, bartlett_p = calculate_bartlett_sphericity(numeric)

        fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
        fa.fit(numeric)
        loadings = fa.loadings_
        communalities = fa.get_communalities()
        variances = fa.get_factor_variance()

        return {
            "test": "Factor Analysis",
            "n_factors": n_factors,
            "kmo": float(kmo_model),
            "kmo_per_variable": {numeric.columns[i]: float(kmo_all[i]) for i in range(len(numeric.columns))},
            "bartlett_chi2": float(chi2),
            "bartlett_p_value": float(bartlett_p),
            "bartlett_significant": bool(bartlett_p < 0.05),
            "eigenvalues": [float(v) for v in fa.get_eigenvalues()[0][:10]],
            "loadings": [
                {numeric.columns[j]: float(loadings[j][i]) for j in range(len(numeric.columns))}
                for i in range(n_factors)
            ],
            "communalities": {numeric.columns[i]: float(communalities[i]) for i in range(len(numeric.columns))},
            "variance_explained": {
                "ss_loadings": [float(v) for v in variances[0]],
                "proportion_var": [float(v) for v in variances[1]],
                "cumulative_var": [float(v) for v in variances[2]],
            },
            "interpretation": (
                f"Factor analysis with {n_factors} factors. "
                f"KMO = {kmo_model:.3f} ({'good' if kmo_model > 0.7 else 'acceptable' if kmo_model > 0.6 else 'poor'} sampling adequacy). "
                f"Bartlett's test: χ² = {chi2:.2f}, p = {bartlett_p:.4f}."
            ),
        }

    @staticmethod
    def cronbach_alpha(data: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        if columns:
            df = data[columns]
        else:
            df = data.select_dtypes(include=[np.number])

        items = df.dropna()
        n_items = items.shape[1]
        n_subjects = items.shape[0]

        item_variances = items.var(axis=0, ddof=1)
        total_variance = items.sum(axis=1).var(ddof=1)
        alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)

        return {
            "test": "Cronbach's Alpha (Reliability Analysis)",
            "alpha": float(alpha),
            "n_items": n_items,
            "n_subjects": n_subjects,
            "interpretation": (
                f"Cronbach's α = {alpha:.4f} ({n_items} items, {n_subjects} subjects). "
                f"{'Excellent' if alpha >= 0.9 else 'Good' if alpha >= 0.8 else 'Acceptable' if alpha >= 0.7 else 'Questionable' if alpha >= 0.6 else 'Poor'} internal consistency."
            ),
        }

    @staticmethod
    def cohens_kappa(ratings1: pd.Series, ratings2: pd.Series) -> Dict[str, Any]:
        r1, r2 = ratings1.dropna(), ratings2.dropna()
        kappa = stats.cohen_kappa_score(r1, r2)
        n = len(r1)
        agreement = (r1 == r2).mean()

        return {
            "test": "Cohen's Kappa",
            "kappa": float(kappa),
            "agreement": float(agreement),
            "n": n,
            "interpretation": (
                f"Cohen's κ = {kappa:.4f}, agreement = {agreement:.1%}. "
                f"{'Almost perfect' if kappa > 0.8 else 'Substantial' if kappa > 0.6 else 'Moderate' if kappa > 0.4 else 'Fair' if kappa > 0.2 else 'Slight'} agreement."
            ),
        }

    @staticmethod
    def fleiss_kappa(data: pd.DataFrame, raters: List[str]) -> Dict[str, Any]:
        try:
            from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
            ratings = data[raters].dropna()
            agg, _ = aggregate_raters(ratings.values)
            kappa = fleiss_kappa(agg)
            return {
                "test": "Fleiss' Kappa",
                "kappa": float(kappa),
                "n_raters": len(raters),
                "n_subjects": len(ratings),
                "interpretation": f"Fleiss' κ = {kappa:.4f} for {len(raters)} raters on {len(ratings)} subjects.",
            }
        except Exception as e:
            return {"error": f"Fleiss Kappa failed: {str(e)}"}

    @staticmethod
    def intraclass_corr(data: pd.DataFrame, raters: List[str]) -> Dict[str, Any]:
        try:
            import pingouin as pg
            long_data = data[raters].melt(var_name="rater", value_name="score")
            long_data["subject"] = list(range(len(data))) * len(raters)
            icc = pg.intraclass_corr(data=long_data, targets="subject", raters="rater", ratings="score")
            icc_value = icc["ICC"].iloc[0]
            return {
                "test": "Intra-Class Correlation (ICC)",
                "icc": float(icc_value),
                "type": str(icc["Type"].iloc[0]),
                "n_raters": len(raters),
                "n_subjects": len(data),
                "interpretation": f"ICC = {icc_value:.4f} ({'Excellent' if icc_value > 0.9 else 'Good' if icc_value > 0.75 else 'Moderate' if icc_value > 0.5 else 'Poor'} reliability).",
            }
        except Exception as e:
            return {"error": f"ICC failed: {str(e)}"}
