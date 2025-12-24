# IoT Workload Prediction and Offloading Optimization

## Overview
This repository contains code for predicting IoT workload demands and optimizing task offloading decisions. The project leverages Deep Learning models (Autoencoders, LSTMs) for workload estimation and the Crow Search Algorithm (CSA) for minimizing system costs related to delay and energy.

## Key Components

### 1. Workload Prediction
The project explores different architectures to forecast CPU and memory usage:
*   **Deep Autoencoder:** Implemented to compress and reconstruct workload features, useful for anomaly detection and demand estimation.
*   **Bi-LSTM & Adaptive Ensemble:** Advanced time-series models used to predict future resource usage based on historical data.

### 2. Offloading Optimization
*   **Crow Search Algorithm (CSA):** A meta-heuristic optimization technique applied to solve the task offloading problem. It determines:
    *   **Offloading Decision:** Whether to process a task Locally, at the Fog node, or in the Cloud.
    *   **Bandwidth Allocation:** Optimal bandwidth assignment for each task.
*   **Cost Function:** The optimization goal is to minimize a composite metric of **Average Delay** and **Average Energy**.

## Files Description

*   **`raproject.ipynb`**
    *   **Dataset:** `iot_dataset_normalized.xlsx` (Synthetic IoT data).
    *   **Focus:** Preprocessing, Feature Engineering, and Deep Autoencoder implementation for workload demand estimation.

*   **`rawithbitbraindata.ipynb`**
    *   **Dataset:** `837.xlsx` (BitBrain dataset - real workload traces).
    *   **Focus:** Applying the Deep Autoencoder approach to real-world trace data.

*   **`rp2.ipynb`**
    *   **Dataset:** `837 - Sheet1.csv` (BitBrain dataset).
    *   **Focus:** Comprehensive pipeline including:
        *   Data cleaning and resampling (10-min and 1-hour intervals).
        *   Training **Holt-Winters**, **Bi-LSTM**, and **Adaptive Ensemble** models.
        *   **CSA Implementation:** Optimization of offloading decisions and bandwidth.
        *   **Evaluation:** F1-Score, SLA violation analysis, and cost convergence plots.

## Dependencies
*   Python 3.x
*   TensorFlow / Keras
*   Pandas
*   NumPy
*   Matplotlib
*   Scikit-learn

## Usage
The notebooks are originally designed for Google Colab environment (using `google.colab` drive mounting). To run locally:
1.  Comment out the Google Drive mounting code.
2.  Update the file paths to point to your local datasets.
3.  Install necessary libraries: `pip install pandas numpy matplotlib tensorflow scikit-learn`.

## Results
The project demonstrates that the **Adaptive Ensemble** model often provides robust predictions, and the **CSA** effectively converges to a lower system cost compared to initial random allocations.
