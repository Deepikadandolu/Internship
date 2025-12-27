# Intelligent Workload Estimation and Resource Allocation

This project implements an intelligent framework for workload estimation and resource allocation in Fog-Cloud computing environments. It utilizes deep learning models and metaheuristic algorithms to predict CPU usage and optimize resource allocation, ensuring Quality of Service (QoS) and minimizing Service Level Agreement (SLA) violations.

## Project Structure

*   **`rp2.ipynb`**: The main Jupyter Notebook containing the implementation of workload prediction models, error analysis, and ensemble methods.

## Key Components

### 1. Workload Prediction Models
The project implements and compares several time-series forecasting models to predict CPU workload:
*   **Holt-Winters Exponential Smoothing:** A statistical method for forecasting time series data with trend and seasonality.
*   **Bi-LSTM (Bidirectional Long Short-Term Memory):** A deep learning model capable of learning bidirectional dependencies in sequential data.
*   **OS-ELM (Online Sequential Extreme Learning Machine):** A fast and efficient learning algorithm for single-hidden layer feedforward neural networks.
*   **Ensemble Methods:**
    *   **Mean Ensemble:** Averages the predictions of base models.
    *   **Adaptive Ensemble:** Dynamically weights base models (Bi-LSTM, OS-ELM) based on their recent performance (error rates) to improve prediction accuracy.

### 2. Workload Estimation & Resource Allocation
*   **Deep Autoencoder (DAE):** Used for workload estimation/compression (referenced in the project context).
*   **Crow Search Algorithm (CSA):** A metaheuristic algorithm referenced for optimizing resource allocation decisions based on predicted workloads.

## Evaluation Metrics
The models are evaluated using the following metrics:
*   **SMAPE (Symmetric Mean Absolute Percentage Error):** Measures the accuracy of the forecast.
*   **MSE (Mean Squared Error) & RMSE (Root Mean Squared Error):** Quantify the magnitude of prediction errors.
*   **F1-Score:** Evaluates the performance of overload detection (binary classification of high workload states).
*   **SLA Violations:** Counts the number of times the predicted/actual workload exceeds the allocated capacity or safe threshold.

## Dataset
*   The project uses a dataset (`837 - Sheet1.csv`) containing timestamped CPU and Memory usage metrics.
*   Data is preprocessed and resampled (e.g., to 10-minute intervals) for training and testing.

## Results
The notebook generates visualizations comparing the actual vs. predicted CPU usage for different models. The **Adaptive Ensemble** model typically demonstrates robustness by combining the strengths of individual predictors, minimizing SLA violations and maintaining high F1-scores for overload detection.

## Requirements
*   Python 3.x
*   Jupyter Notebook
*   Libraries: `pandas`, `numpy`, `matplotlib`, `tensorflow`, `scikit-learn`, `statsmodels`

## Usage
1.  Ensure the dataset `837 - Sheet1.csv` is available (or modify the path in the notebook).
2.  Run `rp2.ipynb` to train the models and view the evaluation results and plots.
