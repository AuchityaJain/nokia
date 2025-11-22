Spatio-Temporal Beam-Level Traffic Forecasting Solution

Author: Auchitya Jain,Harshitha M P,Shashank Kamath
Institution: Nitte Meenakshi Institute of Technology
Language: Python
Frameworks: TensorFlow, XGBoost, Scikit-learn, Pandas, NumPy, Matplotlib

Overview

This project presents a robust and production-grade forecasting pipeline for predicting beam-level network traffic in 5G networks.
The pipeline integrates deep learning (CNN–BiLSTM–GRU–Attention) and ensemble methods (XGBoost + Ridge regression) to accurately model both spatial and temporal dependencies in network traffic data.

It is designed for scalability, stability, and reproducibility in both Kaggle GPU and local CPU environments.

 Repository Structure
├── traffic_forecasting_pipeline.ipynb     # Main notebook
├── README.md                              # Project documentation
├── submission.csv                         # Generated submission file
├── hybrid_nn_model.keras                  # Saved deep learning model
├── xgb_model.pkl                          # Trained XGBoost model
├── meta_model.pkl                         # Ridge regression meta-learner
├── scaler.pkl                             # Scaler object for preprocessing
└── data/
    ├── train.csv                          # Kaggle/local training dataset
    └── test.csv                           # Kaggle/local test dataset

 Installation & Requirements
 Dependencies

Install all dependencies using:

pip install pandas numpy scikit-learn tensorflow xgboost matplotlib seaborn statsmodels joblib


or via requirements.txt:

pip install -r requirements.txt

 Hardware Support

 CPU and GPU compatible

 Automatically detects and utilizes CUDA (if available)

 Optimized for Kaggle environments

 Dataset Description

The dataset contains beam-level PRB (Physical Resource Block) utilization data over time.
Each record includes:

Timestamps

Beam identifiers

Traffic and congestion metrics

The goal is to forecast future PRB usage per beam, enabling proactive network optimization.

 Workflow Breakdown
1 Data Acquisition

Automatically checks Kaggle’s /kaggle/input/ path.

Falls back to /data/ folder if running locally.

Handles missing or corrupted files safely.

 2 Preprocessing

Parses timestamps, sorts chronologically.

Indexes beams automatically.

Cleans missing data and applies robust scaling with fallbacks.

3 Exploratory Data Analysis (EDA)

Time-series decomposition into trend, seasonality, and residuals.

Rolling averages for congestion patterns.

Stationarity check using Augmented Dickey-Fuller test.

Visual EDA with clean and descriptive plots.

4️ Feature Engineering

Cyclic Time Features: Hour/day encoded as sin–cos pairs.

Lagged Features: Multi-step (1h, 3h, 6h, 12h, 24h, 48h, 168h).

Rolling Stats: Moving mean and std for smoothing.

Interaction Features: PRB × congestion, peak hour & night flags.

All transformations include error handling to prevent crashes.

5️ Data Preparation

Sequential train–validation split (no leakage).

Feature scaling via RobustScaler.

3D reshaping for sequence models.

Auto fallback to raw arrays if scaler fails.

6️ Model Architecture
 Hybrid Deep Learning Model

A composite network that integrates:

1D CNN: Captures local temporal dependencies.

BiLSTM: Learns long-term dependencies.

GRU: Reduces overfitting and improves gradient flow.

Attention Layer: Focuses on key timesteps dynamically.

Dropout + L1/L2 Regularization: Prevents overfitting.

Training setup:

optimizer = Adam(learning_rate=0.001)
callbacks = [EarlyStopping(patience=5), ReduceLROnPlateau()]

 XGBoost Model

A tree-based model trained on engineered features to capture non-linear relationships.
Supports both GPU and CPU execution.

 Meta-Learner

A Ridge Regression model blends predictions from both the NN and XGBoost models, improving robustness and reducing bias.

7️ Evaluation Metrics

The pipeline evaluates model performance using Mean Absolute Error (MAE):

MAE = mean(abs(y_true - y_pred))


Printed for each model:

Neural Network MAE

XGBoost MAE

Meta-Learner MAE

8️ Submission Generation

After validation, predictions are saved in submission.csv:

test_predictions = meta_model.predict(meta_val)
submission = pd.DataFrame(test_predictions, columns=beam_columns)
submission.to_csv("submission.csv", index=False)

 Key Highlights
Feature	Description
Architecture	CNN + BiLSTM + GRU + Attention
Classical Model	XGBoost (GPU/CPU fallback)
Ensemble Layer	Ridge Regression for stacked blending
Feature Set	Lag, rolling, cyclic, congestion, contextual flags
Resilience	Extensive error handling and graceful recovery
Cross-Platform	Compatible with Kaggle & local environments
Explainability	Statistical diagnostics and EDA visualization
 Results Summary
Model	MAE (Validation)	Description
Hybrid NN	Low	Learns long-term and periodic patterns
XGBoost	Moderate	Captures non-linear relationships
Meta-Learner	Lowest	Combines both for optimal generalization
 Future Improvements

Integrate transformer encoders for longer horizon prediction.

Add Optuna for automated hyperparameter tuning.

Enable multi-step prediction for continuous time forecasting.

 License

Released under the MIT License — free for academic and research use.

 Acknowledgements

Gratitude to:

ITU AI/ML in 5G Challenge for the dataset and problem statement.

Kaggle community for open-source ideas on hybrid architectures.

Nokia Mentors for guiding us in completing this project

 Author

Auchitya Jain,Harshitha M P,Shashank Kamath
Electronics & Communication Engineering
Nitte Meenakshi Institute of Technology

<img width="621" height="843" alt="image" src="https://github.com/user-attachments/assets/8b706820-2189-40aa-8225-93cfec70b783" />



<img width="916" height="961" alt="image" src="https://github.com/user-attachments/assets/ef778d04-d6eb-4a66-80e8-0256dd210aaa" />



original project by:-
https://github.com/SalifouAbdourahamane/Spatio-Temporal-Beam-Traffic-Forecasting
