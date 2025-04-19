 Medium-Term Probabilistic Indicator (MTPI)

**MLTPI** is a modular, ML-powered signal engineering framework that evolves multi-indicator trading strategies based on statistical behavior and optimized portfolio logic.

---

##  Purpose

The MTPI framework is designed to:

- Identify optimal indicator settings for trend detection against a known signal profile (`manual_signal`) using machine learning.
- Combine trained indicators into a cohesive **strategy** via weighted aggregation:
  \[
  H(\alpha) = w_1 \cdot I_1 + w_2 \cdot I_2 + \ldots + w_n \cdot I_n = \text{Actual Signal}
  \]
- Optimize both indicator configurations and their respective weights to produce robust signals aligned with medium-term market movements.

---

## 🔍 Scope

- Applicable to **any asset class** (crypto, equity, commodities, etc.) where a quality training signal (manual ISP) is provided.
- Designed to extract **market alpha** through intelligent signal stacking and attenuation.
- Signal ensemble is built for **stability, clarity, and adaptability**, especially in noisy or volatile regimes.
- Encourages diversity across **oscillatory** and **perpetual** indicators with intelligent clustering and behavior-based grouping.

---

## ⚙️ Features

- Modular indicator pipeline with plug-and-play architecture
- Behavioral profiling using:
  - MAE, Sharpe, Omega, transition frequency, holding period
- Bayesian optimization for:
  - Indicator parameter tuning
  - Strategy-level weight optimization
- Multi-timeframe scaling and selection
- Strategy clustering using unsupervised learning
- Live and backtested signal simulation

---

## 🚀 Current Capabilities

- Supports multiple indicators like AGMA, QTrend, HullSuite, and more.
- Feature-based clustering for forming coherent strategy groups (S1, S2, ...).
- Generation of emergent ISP for recursive training.
- Strategy-level equity backtests and signal visualizations.

---

## 🔧 Tech Stack

- Python / Pandas / NumPy
- Scikit-learn (ML models)
- Matplotlib (signal visualization)
- Bayesian Optimization
- Modular Git-based pipeline (per indicator/strategy)

---

## 📁 Directory Structure

