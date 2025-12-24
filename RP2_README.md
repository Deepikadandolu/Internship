# Resource Usage Forecasting and Anomaly Detection (rp2.ipynb)

This Jupyter Notebook (`rp2.ipynb`) focuses on the analysis, forecasting, and anomaly detection of system resource usage (CPU and Memory) using various machine learning and statistical techniques.

## Overview

The notebook performs the following main tasks:
1.  **Data Loading & Cleaning**: Loads raw system monitoring data, cleans column names, and handles missing values.
2.  **Preprocessing**: Converts timestamps, sets a time-based index, and resamples data into fixed intervals (10-minute and 1-hour windows).
3.  **Visualization**: Plots time-series data for CPU and Memory usage to visualize trends.
4.  **Forecasting Models**: Implements and evaluates multiple forecasting models:
    *   **Holt-Winters Exponential Smoothing**
    *   **Bi-Directional LSTM (Bi-LSTM)**
    *   **Adaptive Ensemble**
5.  **Evaluation**: Compares models using metrics such as SMAPE (Symmetric Mean Absolute Percentage Error), MSE (Mean Squared Error), and RMSE (Root Mean Squared Error).
6.  **Anomaly Detection**: Identifies system overload events based on defined thresholds and evaluates detection performance (F1-Score, SLA violations).

## Dependencies

The following Python libraries are required to run this notebook:
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `tensorflow` (Keras)
*   `scikit-learn`

## Data Requirements

The notebook expects a CSV file containing time-series data with at least the following columns (names are cleaned within the notebook):
*   `Timestamp [ms]`
*   `CPU usage [%]`
*   `Memory usage [KB]`

*Note: The current code points to a specific Google Drive path (`/content/drive/MyDrive/837 - Sheet1.csv`). You will need to update this path to point to your local dataset.*

## Usage

1.  Open the notebook in Jupyter or Google Colab.
2.  Update the file path in the "LOAD CSV FILE" section to point to your data file.
3.  Run the cells sequentially to process the data, train the models, and view the results.

## Key Results

The notebook outputs:
*   Visualizations of CPU and Memory usage over time.
*   Performance tables comparing forecasting models.
*   Training convergence plots for Deep Learning models.
*   Classification metrics (F1-score) for overload detection.
*   Counts of Service Level Agreement (SLA) violations.
