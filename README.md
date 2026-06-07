<p align="center">
  <img src="https://img.shields.io/badge/STATLAB_CLI-v1.0.0-purple?style=for-the-badge&logo=python&logoColor=white" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/No_Upload_Limits-00ADD8?style=for-the-badge&logo=gnu-bash&logoColor=white" alt="No Limits">
  <br>
  <img src="https://img.shields.io/github/stars/abhinay-sambherao/statlab-cli?style=social" alt="Stars">
</p>

<div align="center">
  <h1>📊 StatLab CLI</h1>
  <h3>Statistical Analysis for Large Datasets — From Your Terminal</h3>
  <p><em>No file size limits. No browser uploads. Pure terminal power.</em></p>
  <p>
    <strong>Built by <a href="https://abhiinayy.in">Abhinay Sambherao</a></strong> •
    <a href="https://www.linkedin.com/in/abhinay-sambherao">LinkedIn</a> •
    <a href="https://abhiinayy.in">Portfolio</a>
  </p>
</div>

---

## Why StatLab CLI?

The web is great for small files. But when your dataset hits 500 MB, 1 GB, or more — uploading through a browser is impractical.

**StatLab CLI** gives you the same statistical engine used by [StatLab AI](https://github.com/abhinay-sambherao/statlab-app) as a standalone command-line tool. Process files of any size, pipe JSON output into other tools, and automate analyses in scripts and CI/CD pipelines.

## Quick Start

```bash
# Clone
git clone https://github.com/abhinay-sambherao/statlab-cli.git
cd statlab-cli

# Install
pip install -r requirements.txt

# Analyze
./statlab describe datasets/iris.csv
```

## Commands

| Command | Description |
|---------|-------------|
| `upload` | Preview dataset structure |
| `describe` | Descriptive statistics (mean, std, etc.) |
| `normality` | Shapiro-Wilk, KS, Anderson-Darling tests |
| `ttest` | Independent & paired t-tests |
| `correlation` | Pearson, Spearman, Kendall correlation |
| `anova` | One-way ANOVA |
| `regression` | Linear & multiple regression |
| `kmeans` | K-Means clustering |
| `pca` | Principal Component Analysis |
| `monte-carlo` | Monte Carlo simulation |
| `report` | Generate HTML/PDF reports |
| `batch` | Run multiple analyses from config |
| `list` | Show all commands |

## Examples

```bash
# Descriptive stats as JSON (pipeable)
./statlab describe large_dataset.csv --columns revenue cost --json > stats.json

# Normality test
./statlab normality data.csv --column age

# T-Test
./statlab ttest experiment.csv --group1 control --group2 treatment

# Multiple regression
./statlab regression data.csv --dv price --predictors size bedrooms location

# K-Means clustering
./statlab kmeans customers.csv --clusters 5 --columns age income

# PCA
./statlab pca data.csv --components 3

# Monte Carlo simulation (50,000 runs)
./statlab monte-carlo --simulations 50000 --distribution normal

# Generate PDF report from results
./statlab report results.json --format pdf -o report.pdf
```

### Batch Processing

Create a JSON config file:

```json
{
  "file": "large_sales_data.csv",
  "analyses": [
    { "type": "descriptive", "params": { "columns": ["revenue", "cost"] } },
    { "type": "regression", "params": { "dv": "revenue", "predictors": ["marketing", "price"] } },
    { "type": "kmeans", "params": { "n_clusters": 4, "columns": ["revenue", "customers"] } }
  ]
}
```

```bash
./statlab batch config.json -o results.json
```

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| CSV | `.csv` | Comma-separated |
| TSV | `.tsv` | Tab-separated |
| Excel | `.xlsx`, `.xls` | All sheets |
| JSON | `.json` | Record-oriented |
| Parquet | `.parquet` | Columnar, great for large files |
| Feather | `.feather` | Fast I/O for large data |

## Dataset Size Benchmarks

| Size | Load Time | Analysis Time |
|------|-----------|---------------|
| 1 MB | < 1s | < 1s |
| 100 MB | ~2s | ~3s |
| 1 GB | ~15s | ~20s |
| 10 GB | ~2 min | ~3 min |

*Times are approximate on a modern SSD. YMMV.*

## Requirements

- Python 3.9+
- NumPy, Pandas, SciPy, Scikit-learn, StatsModels
- Matplotlib, Seaborn (for report generation)
- openpyxl (for Excel support)

Install all: `pip install -r requirements.txt`

## Project Structure

```
statlab-cli/
├── statlab                 # Main executable (chmod +x)
├── statistical/            # Statistical engine (13 modules)
│   ├── descriptive.py      #   Descriptive statistics
│   ├── normality.py        #   Normality tests
│   ├── hypothesis.py       #   Hypothesis testing
│   ├── anova.py            #   ANOVA
│   ├── correlation.py      #   Correlation analysis
│   ├── regression.py       #   Regression analysis
│   ├── machine_learning.py #   K-Means, clustering
│   ├── advanced.py         #   PCA, factor analysis
│   ├── survival.py         #   Survival analysis
│   ├── simulation.py       #   Monte Carlo, bootstrap
│   ├── quality_control.py  #   SPC charts
│   ├── doe.py              #   Design of experiments
│   └── power_analysis.py   #   Power analysis
├── utils/
│   └── report_generator.py # HTML/PDF report generation
├── datasets/               # Example datasets
├── requirements.txt
└── README.md
```

## Testing

```bash
./statlab describe datasets/iris.csv
./statlab describe datasets/iris.csv --json
./statlab normality datasets/iris.csv --column sepal_length
```

## About the Creator

**Abhinay Sambherao** — [Portfolio](https://abhiinayy.in) • [LinkedIn](https://www.linkedin.com/in/abhinay-sambherao)

## License

MIT
