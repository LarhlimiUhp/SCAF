# SCAF-LS — Scalable Cross-Asset Forecasting (Long/Short)

SCAF-LS is a modular Python framework for financial time-series forecasting and trading-signal generation. It combines classical ML, deep learning (PyTorch), conformal prediction, Q-learning model selection, LLM-based risk overrides, and a comprehensive monitoring stack.

## Features

- **14 ML models** — Logistic Regression, Random Forest, Gradient Boosting, LightGBM (custom loss), SVM, KNN, BiLSTM, TCN, Transformer, and more
- **Walk-forward cross-validation** over 2018–2024 S&P 500 data
- **Conformal prediction** — guaranteed coverage intervals at inference time
- **Q-learning model selector** — RL agent that learns which models to activate per market regime
- **15 benchmark strategies** — buy-and-hold, momentum, RSI, Bollinger Bands, MACD, and more
- **Multi-asset support** — S&P 500, GOLD, OIL, VIX, TNX, IRX
- **Drift detection & monitoring** — Prometheus/Grafana-ready with alerting
- **LLM orchestrator** — optional risk-override layer via language model
- **Streamlit dashboard** — interactive visualisation of results
- **Offline-first** — ships with pre-downloaded CSVs; no live internet required

---

## Requirements

- Python **3.11+**
- See [`requirements.txt`](requirements.txt) for the full dependency list

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/LarhlimiUhp/SCAF.git
cd SCAF
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Optional extras:

```bash
# Deep learning (PyTorch)
pip install torch

# Live market data
pip install yfinance

# Monitoring exporters
pip install prometheus-client python-json-logger
```

Or install everything at once via the package extras:

```bash
pip install -e ".[all]"
```

---

## Running the pipeline

All modules live under `07-04-2026/`. Add it to your `PYTHONPATH` before running:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/07-04-2026   # Linux / macOS
set PYTHONPATH=%PYTHONPATH%;%CD%\07-04-2026        # Windows
```

### Minimal experiment (S&P 500, 2018–2024)

```python
import sys
sys.path.insert(0, "07-04-2026")

from pipeline.experiment import ExperimentPipeline

pipeline = ExperimentPipeline()
results = pipeline.run()
print(results["summary"])
```

### Multi-asset experiment

```python
from pipeline.multi_asset_experiment import MultiAssetExperiment

exp = MultiAssetExperiment()
results = exp.run()
```

### Ablation study

```python
from pipeline.ablation import AblationStudy

study = AblationStudy()
results = study.run()
```

### Streamlit dashboard

```bash
cd 07-04-2026
streamlit run dashboard/app.py
```

---

## Running the tests

```bash
pytest tests/ -v
# With coverage report
pytest tests/ -v --cov=07-04-2026 --cov-report=term-missing
```

---

## Project structure

```
SCAF/
├── 07-04-2026/                  # Main package root
│   ├── agents/                  # Ensemble optimisation & architecture review agents
│   ├── analysis/                # Stats, visualisation, validation reports
│   ├── benchmark/               # 15 benchmark trading strategies
│   ├── dashboard/               # Streamlit app
│   ├── data/                    # Loader, feature engineer, offline CSVs
│   │   └── csv/                 # Pre-downloaded market data (GSPC, GOLD, OIL, VIX, TNX, IRX)
│   ├── features/                # Correlation, SHAP importance, selector, optimiser
│   ├── llm/                     # LLM orchestrator & prompt templates
│   ├── models/                  # All ML/DL models + registry
│   ├── monitoring/              # Alerts, drift detection, Prometheus/Grafana
│   ├── optimization/            # Multi-agent LightGBM optimisation system
│   └── pipeline/                # Experiment, multi-asset, ablation pipelines
├── tests/                       # Pytest test suite
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Data

The repository ships with 6 pre-downloaded CSVs covering **2018-01-01 → 2024-12-31** (~1 827 daily rows each):

| File | Asset |
|---|---|
| `GSPC.csv` | S&P 500 |
| `GOLD.csv` | Gold futures |
| `OIL.csv` | Crude oil futures |
| `VIX.csv` | CBOE Volatility Index |
| `TNX.csv` | 10-year Treasury yield |
| `IRX.csv` | 13-week Treasury yield |

Live data can be fetched via `yfinance` when installed and network access is available.

---

## License

MIT
