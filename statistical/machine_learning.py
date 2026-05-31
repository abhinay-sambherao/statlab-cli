import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional


class MachineLearning:
    @staticmethod
    def kmeans(data: pd.DataFrame, n_clusters: int = 3, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        if columns:
            X = data[columns].dropna()
        else:
            X = data.select_dtypes(include=[np.number]).dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        centers = model.cluster_centers_

        sil_score = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
        ch_score = calinski_harabasz_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0
        db_score = davies_bouldin_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0

        inertia_values = []
        for k in range(1, min(11, len(X))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertia_values.append(float(km.inertia_))

        return {
            "test": "K-Means Clustering",
            "n_clusters": n_clusters,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "labels": [int(l) for l in labels],
            "cluster_sizes": {int(i): int((labels == i).sum()) for i in range(n_clusters)},
            "centers": [centers[i].tolist() for i in range(n_clusters)],
            "inertia": float(model.inertia_),
            "silhouette_score": float(sil_score),
            "calinski_harabasz_score": float(ch_score),
            "davies_bouldin_score": float(db_score),
            "elbow_inertia": inertia_values,
            "interpretation": (
                f"K-Means with k={n_clusters} achieved "
                f"silhouette score = {sil_score:.4f}, CH score = {ch_score:.2f}, "
                f"DB score = {db_score:.4f}. "
                f"{'Good cluster separation.' if sil_score > 0.5 else 'Moderate cluster separation.' if sil_score > 0.25 else 'Poor cluster separation.'}"
            ),
        }

    @staticmethod
    def hierarchical(data: pd.DataFrame, n_clusters: int = 3, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        if columns:
            X = data[columns].dropna()
        else:
            X = data.select_dtypes(include=[np.number]).dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_scaled)

        sil_score = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0

        return {
            "test": "Hierarchical Clustering",
            "n_clusters": n_clusters,
            "n_samples": len(X),
            "linkage": "ward",
            "labels": [int(l) for l in labels],
            "cluster_sizes": {int(i): int((labels == i).sum()) for i in range(n_clusters)},
            "silhouette_score": float(sil_score),
            "interpretation": f"Hierarchical clustering with {n_clusters} clusters. Silhouette score = {sil_score:.4f}.",
        }

    @staticmethod
    def dbscan(data: pd.DataFrame, eps: float = 0.5, min_samples: int = 5, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        if columns:
            X = data[columns].dropna()
        else:
            X = data.select_dtypes(include=[np.number]).dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)

        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        sil_score = silhouette_score(X_scaled[labels != -1], labels[labels != -1]) if n_clusters > 1 and len(labels[labels != -1]) > 1 else 0

        return {
            "test": "DBSCAN Clustering",
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "n_samples": len(X),
            "labels": [int(l) for l in labels],
            "silhouette_score": float(sil_score),
            "interpretation": f"DBSCAN (eps={eps}, min_samples={min_samples}) found {n_clusters} clusters and {n_noise} noise points. Silhouette = {sil_score:.4f}.",
        }

    @staticmethod
    def decision_tree(data: pd.DataFrame, dv: str, predictors: List[str]) -> Dict[str, Any]:
        X = data[predictors].fillna(data[predictors].mean())
        y = data[dv]

        model = DecisionTreeClassifier(random_state=42, max_depth=5)
        model.fit(X, y)
        y_pred = model.predict(X)

        accuracy = (y == y_pred).mean()
        cm = confusion_matrix(y, y_pred)

        importances = [
            {"feature": predictors[i], "importance": float(model.feature_importances_[i])}
            for i in range(len(predictors))
        ]

        return {
            "test": "Decision Tree (CHAID-style)",
            "accuracy": float(accuracy),
            "n_nodes": int(model.tree_.node_count),
            "max_depth": int(model.tree_.max_depth),
            "n_classes": len(np.unique(y)),
            "confusion_matrix": cm.tolist(),
            "feature_importances": sorted(importances, key=lambda x: x["importance"], reverse=True),
            "interpretation": (
                f"Decision tree with {int(model.tree_.node_count)} nodes achieved "
                f"{accuracy:.1%} accuracy. Top predictor: {importances[0]['feature']} "
                f"(importance = {importances[0]['importance']:.4f})."
            ),
        }

    @staticmethod
    def classification_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: Optional[pd.Series] = None) -> Dict[str, Any]:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        result = {
            "test": "Classification Metrics",
            "confusion_matrix": cm.tolist(),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1),
        }

        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            result["roc_auc"] = float(roc_auc)
            result["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            }

        return result
