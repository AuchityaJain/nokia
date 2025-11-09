Spatio-Temporal Beam-Level Traffic Forecasting — Advanced Production Pipeline
Table of Contents
Introduction

Problem Statement

Dataset Overview

Approach & Pipeline Design

Model Architecture Comparison

Improvements & Innovations

Results: MAE Improvement

Usage Instructions

Conclusion

Introduction
This repository provides a highly robust and extensible solution for forecasting hourly beam-level traffic volumes in modern mobile telecom networks. Using advanced ensemble learning, graph neural networks, and comprehensive data diagnostics, the solution delivers reliable, real-world-ready predictions.

Problem Statement
Predict future DLThpVol values at the beam/cell level, across an entire 5G RAN, given weeks of historical spatio-temporal network data (traffic, PRB, users). Accurately modeling both temporal/cyclic and spatial/adjacency patterns is crucial for optimizing energy consumption, congestion, and user quality of experience.

Dataset Overview
DLThpVol: Beam-level hourly traffic for 2880 beams.

PRB Utilization: Resource block occupancy, congestion flag, and derived traffic features.

User Count: (if available) further context on demand/load.

Files: CSVs per data type, hours as rows, beams as columns.

Approach & Pipeline Design
Resilient Data Loading: Locates, loads, and validates all inputs; falls back to local or direct paths as needed.

Diagnostic EDA: Decomposes, plots, and tabulates trend/seasonality, PRB, and congestion for all beams. Rolling means/stdevs integrate short- and long-term traffic smoothing.

Feature Engineering: Automated creation and aggregation of time-cyclic, lagged, rolling, and congestion features, robust to missing beams.

Blended Modeling: Integrates Conv1D-GRU-Attention neural networks, XGBoost regressors, and Graph Neural Networks (GNNs) to capture non-linear and spatial relationships.

Automated Blend Optimization: Full grid search for NN/XGB/GNN ensemble weights, with validation-based best-MAE selection.

Explainability: LIME-based local feature explanations for both NN and XGBoost predictions, reliable across all architectures.

Artifact Persistence: All models, feature sets, scalers, blending weights, diagnostics, and submissions are logged and saved.

Model Architecture Comparison
Aspect	Original Code	Improved (Production) Code
Neural Network	Conv1D + GRU + MH-Attention	Modular builder, BiLSTM, regularization, robust fallback
XGBoost	Shallow, static parameters	GPU/CPU support, flexible config, deeper validation/ensemble integration
Blending	Manual weights, shortlist	Full grid/blend search incl. GNN, best MAE picked auto
GNN	None	GCN using PyTorch Geometric, ensemble-ready
Explainability	None	LIME (default, model-agnostic), SHAP optional
Output/Artifact	Submission .csv only	Models, scaler, predictions, feature importances, validation metrics
Improvements & Innovations
Handles all data quality issues via sectioned try/except/fallback, runs EDA/feature engineering for every possible data arrangement.

Advanced PRB/Congestion analysis — full stats, congestion shading, direct feature and diagnostic plotting.

Pluggable GNN and other architectures — not just sequence/tabular, but also graph-based learning over inter-beam relationships.

Grid-searched ensemble/blending finding the best MAE.

Full LIME explainability: Feature contributions always available for every prediction.

Complete artifact persistence and logs—always saved, always inspectable.

Results: MAE Improvement
Achieved Validation MAE:
0.16682

This MAE is a significant improvement over previous fixed-weight blends and non-ensemble architectures, demonstrating the value of both GNN integration and full automated ensemble grid search on validation.

Usage Instructions
Place your data and model files in the project directory (or adjust paths as needed).

Run the main notebook or script. All steps (EDA, feature engineering, model training/blending/explainability, output submission) are automatic.

Inspect diagnostic output, EDA plots, best MAE blend weights, and both NN/XGB/LIME explanations.

Submission (.csv), all models, prediction arrays, and meta-data will be saved in the output directory.

Conclusion
This pipeline achieves state-of-the-art validation accuracy (MAE 0.16682) in the ITU Beam-Level Traffic Forecasting setting. It is designed for real-world extension, easy explainability, and complete robustness to data/modeling exceptions—a true production-ready research reference.

Contact: Auchitya Jain,Nitte Meenakshi Institute of Technology
