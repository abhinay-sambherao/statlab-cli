import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional


class DescriptiveStats:
    @staticmethod
    def compute(data: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        if columns:
            df = data[columns]
        else:
            df = data.select_dtypes(include=[np.number])

        results = {}
        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            qs = series.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values

            results[col] = {
                "count": int(len(series)),
                "missing": int(data[col].isna().sum()),
                "mean": float(np.mean(series)),
                "median": float(np.median(series)),
                "mode": float(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
                "std": float(np.std(series, ddof=1)),
                "variance": float(np.var(series, ddof=1)),
                "min": float(qs[0]),
                "max": float(qs[4]),
                "range": float(qs[4] - qs[0]),
                "q1": float(qs[1]),
                "q3": float(qs[3]),
                "iqr": float(qs[3] - qs[1]),
                "skewness": float(stats.skew(series)),
                "kurtosis": float(stats.kurtosis(series, fisher=True)),
                "se": float(stats.sem(series)),
                "ci_95_lower": float(stats.t.interval(0.95, len(series) - 1, loc=np.mean(series), scale=stats.sem(series))[0]),
                "ci_95_upper": float(stats.t.interval(0.95, len(series) - 1, loc=np.mean(series), scale=stats.sem(series))[1]),
                "sum": float(np.sum(series)),
                "cv": float(np.std(series, ddof=1) / np.mean(series) * 100) if np.mean(series) != 0 else 0,
            }

        return results

    @staticmethod
    def frequency_table(data: pd.DataFrame, column: str) -> Dict[str, Any]:
        series = data[column].dropna()
        if pd.api.types.is_numeric_dtype(series):
            freq = pd.cut(series, bins=10).value_counts().sort_index()
            table = []
            for interval, count in freq.items():
                table.append({
                    "interval": str(interval),
                    "frequency": int(count),
                    "percent": float(count / len(series) * 100),
                    "cumulative": 0,
                })
            cum = 0
            for row in table:
                cum += row["percent"]
                row["cumulative"] = round(cum, 2)
        else:
            freq = series.value_counts()
            table = []
            for val, count in freq.items():
                table.append({
                    "value": str(val),
                    "frequency": int(count),
                    "percent": float(count / len(series) * 100),
                })

        return {
            "column": column,
            "total": int(len(series)),
            "table": table,
            "unique_values": int(series.nunique()),
        }

    @staticmethod
    def summary(data: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()

        return {
            "shape": list(data.shape),
            "columns": len(data.columns),
            "rows": len(data),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "datetime_columns": len(datetime_cols),
            "total_missing": int(data.isna().sum().sum()),
            "memory_usage": f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB",
            "duplicates": int(data.duplicated().sum()),
            "column_details": [
                {
                    "name": col,
                    "dtype": str(data[col].dtype),
                    "missing": int(data[col].isna().sum()),
                    "missing_pct": round(float(data[col].isna().sum() / len(data) * 100), 2),
                    "unique": int(data[col].nunique()),
                }
                for col in data.columns
            ],
        }
