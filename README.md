# SLM-Driven Autonomous Agent for Real-Time Stock Data Processing

Master of Data Science — Universiti Malaya, 2025

## Overview
This repository contains the experiment code, results, and figures
for the proof-of-concept evaluation of a Small Language Model-driven
autonomous agent system for financial stock data processing.

## Requirements
- Google Colab (free tier, T4 GPU)
- Python 3.11
- All dependencies installed automatically in the notebook

## How to Run
1. Open SLM_Thesis_Experiment_v2.ipynb in Google Colab
2. Select Runtime → Change runtime type → T4 GPU
3. Run cells from top to bottom
4. Download results from /content/ when complete

## Results Summary
| Model | F1 | Latency p95 |
|---|---|---|
| FinBERT INT8 (SLM) | 0.5833 | 18.35ms |
| XGBoost | 0.5075 | 2.92ms |
| LSTM | 0.4655 | 8.58ms |
| Logistic Regression | 0.4426 | 4.25ms |

## Author
Master of Data Science Student
Universiti Malaya, Malaysia
