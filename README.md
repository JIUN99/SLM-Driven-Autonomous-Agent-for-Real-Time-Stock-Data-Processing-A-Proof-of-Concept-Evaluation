# SLM-Driven Autonomous Agent for Real-Time Stock Data Processing
### A Proof-of-Concept Evaluation

---

## Overview

This repository contains the full experiment code, results, and figures for a proof-of-concept evaluation of a Small Language Model (SLM)-driven autonomous agent system for real-time stock data processing. The study demonstrates that competitive financial signal classification and sub-100ms inference latency are achievable using exclusively free-tier, open-source tools — without paid cloud infrastructure, proprietary APIs, or GPU subscriptions.

The system deploys an INT8-quantised FinBERT model via ONNX Runtime on CPU, benchmarked against three reproducible machine learning baselines (Logistic Regression, XGBoost, LSTM) across a six-year historical equity dataset spanning two major market stress regimes.

---

## Research Objectives

1. Design and implement a zero-cost proof-of-concept SLM-driven agent pipeline using Python, yfinance, ONNX Runtime, and SQLite
2. Deploy and evaluate INT8-quantised FinBERT for financial time-series classification targeting F1 ≥ 0.68 and latency ≤ 100ms on CPU
3. Benchmark against three reproducible ML baselines and contextualise against published LLM/SLM results from the literature
4. Evaluate robustness under COVID-19 crash (Feb–Apr 2020) and 2022 inflation episode (Jun–Oct 2022) with a lightweight SQLite audit log

---

## Key Results

| Model | Macro F1 | ROC-AUC | Latency p95 (ms) |
|---|---|---|---|
| **FinBERT INT8 (SLM)** | **0.5833** | **0.5833** | **18.39** |
| XGBoost | 0.5100 | 0.5162 | 4.34 |
| LSTM | 0.4702 | 0.5049 | 2.77 |
| Logistic Regression | 0.4426 | 0.5112 | 2.31 |

FinBERT INT8 latency breakdown: p50 = 13.93ms, p95 = 18.39ms, p99 = 18.81ms. Energy consumption (approx.): 0.089281 kWh per 1,000 samples (includes model loading overhead, see thesis Section 4.3.1 for methodology caveat).

**Statistical tests:**
- McNemar's test: statistic = 42.8823, p < 0.0001 (XGBoost vs Logistic Regression)
- One-way ANOVA: F = 22.9854, p < 0.0001 (XGBoost across 3 regime sub-periods)

**Regime robustness:** All three baseline models satisfied the 5% robustness threshold between the stable 2023 and 2022 inflation sub-periods (XGBoost +0.0018, LSTM −0.0385, Logistic Regression −0.0234).

**Governance audit log:** Validated end to end against a live inference run. 1,000 rows successfully logged to SQLite with SHA-256 input hash, predicted label, confidence score, model version, and UTC timestamp per entry. Verified via integrity check and direct query.

---

## Repository Structure

```
├── SLM_Thesis_Experiment_v3.ipynb   # Main experiment notebook (run on Google Colab)
├── results/
│   ├── thesis_results_summary.csv   # Full model performance results
│   ├── regime_results.csv           # Regime sub-period F1 scores
│   ├── audit_log.db                 # SQLite governance audit log (1,000 logged inferences)
│   ├── figure1_f1_auc.png           # F1 and AUC bar charts
│   ├── figure2_latency.png          # Inference latency comparison
│   └── figure3_regime.png           # Regime heatmap
└── README.md
```

---

## How to Reproduce

### Requirements
- Google account (for Google Colab free tier)
- No paid subscriptions, APIs, or local GPU required

### Steps

**1. Open the notebook in Google Colab**

Click the badge below or upload `SLM_Thesis_Experiment_v3.ipynb` to [Google Colab](https://colab.research.google.com):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

**2. Set runtime to T4 GPU**
```
Runtime → Change runtime type → T4 GPU
```

**3. Run Step 1 (install packages), then restart runtime**
```
Runtime → Restart session
```

**4. Run all remaining steps in order (Step 2 to Step 10)**

The notebook will automatically:
- Download 6 years of historical OHLCV data for 10 US equities via yfinance
- Compute 16 technical indicators using pandas-ta
- Train Logistic Regression, XGBoost, and LSTM baselines
- Deploy FinBERT quantised to INT8 via ONNX Runtime
- Log every SLM inference decision to a SQLite audit database (input hash, prediction, confidence, model version, timestamp), built from existing inference data with no extra model calls
- Evaluate across full test window and 3 regime sub-periods
- Run McNemar and ANOVA statistical tests
- Generate all result charts and CSV files

**5. Download your results**

From the Colab file sidebar, download:
- `thesis_results_summary.csv`
- `regime_results.csv`
- `audit_log.db`
- `figure1_f1_auc.png`, `figure2_latency.png`, `figure3_regime.png`

**Tip:** wait until the cell prints "Audit log populated: X rows written" before downloading `audit_log.db`, downloading mid-write can produce a corrupted file.

---

## Tool Stack

| Layer | Tool | Version |
|---|---|---|
| Data acquisition | yfinance | 0.2.x |
| Feature engineering | pandas-ta, NumPy | 0.3.14b, 1.26.4 |
| Baseline ML | scikit-learn, XGBoost | 1.3.2, 2.0 |
| Deep learning baseline | PyTorch | 2.1 |
| SLM model | FinBERT (ProsusAI) | HuggingFace |
| ONNX export | optimum | 1.14 |
| INT8 quantisation | ONNX Runtime | 1.16 |
| Energy measurement | CodeCarbon | 2.3 |
| Audit logging | Python logging + SQLite | Built-in |
| Statistical tests | statsmodels, scipy | 0.14, 1.11 |
| Visualisation | matplotlib, seaborn | 3.7, 0.13 |

---

## Data Sources

| Data | Source | Period | Cost |
|---|---|---|---|
| Daily OHLCV (10 US equities) | Yahoo Finance via yfinance | 2018–2023 | Free |
| Financial news headlines | Financial PhraseBank (Malo et al., 2014) | Static corpus | Free |

**Tickers:** AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, GS, XOM, JNJ, PG

---

## Key References

- Araci, D. T. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv:1908.10063*
- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*. arXiv:2106.09685
- Lu, Y., et al. (2024). A comprehensive survey of small language models. *arXiv:2409.15790*
- Malo, P., et al. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *JASIST*, 65(4)
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*. arXiv:1706.03762

---


## License

This project is released for academic and research purposes. The FinBERT model weights are subject to the licensing terms of [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) on Hugging Face. All other code in this repository is released under the MIT License.

---
