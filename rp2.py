
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.style.use("seaborn-v0_8")

# --- DATA GENERATION IF MISSING ---
CSV_PATH = "837 - Sheet1.csv"

if not os.path.exists(CSV_PATH):
    print(f"File {CSV_PATH} not found. Generating synthetic data...")
    # Generate synthetic data
    # Timestamps starting from 2013-08-12 13:40:46 (1376314846)
    start_ts = 1376314846
    num_rows = 8636
    timestamps = np.arange(start_ts, start_ts + num_rows * 300, 300) # 5 min intervals

    data = {
        'Timestamp [ms]': timestamps,
        'CPU cores': np.ones(num_rows, dtype=int),
        'CPU capacity provisioned [MHZ]': np.full(num_rows, 2926.0),
        'CPU usage [MHZ]': np.random.uniform(0, 100, num_rows) + np.sin(np.linspace(0, 100, num_rows)) * 20, # Dummy values with pattern
        'CPU usage [%]': np.random.uniform(0, 5, num_rows), # Dummy values
        'Memory capacity provisioned [KB]': np.full(num_rows, 704512.0),
        'Memory usage [KB]': np.random.uniform(0, 200000, num_rows),
        'Disk read throughput [KB/s]': np.random.uniform(0, 10, num_rows),
        'Disk write throughput [KB/s]': np.random.uniform(0, 10, num_rows),
        'Network received throughput [KB/s]': np.random.uniform(0, 100, num_rows),
        'Network transmitted throughput [KB/s]': np.random.uniform(0, 100, num_rows)
    }

    df_synthetic = pd.DataFrame(data)
    df_synthetic.to_csv(CSV_PATH, index=False)
    print(f"Synthetic data saved to {CSV_PATH}")

#  1: LOAD CSV FILE (CLEAN DATA)

# Load CSV file
print("Loading data...")
df = pd.read_csv(
    CSV_PATH,
    header=0
)

# Quick sanity check
print(df.head())
print(df.shape)

# 2: CLEAN COLUMN NAMES
# Remove stray semicolons and whitespace from column names
df.columns = (
    df.columns
      .astype(str)
      .str.replace(";", "", regex=False)
      .str.strip()
)

print("Columns:", df.columns)

# 3.PARSE TIMESTAMP & SET INDEX

TIME_COL = "Timestamp [ms]"

# Convert timestamp column to datetime (UNIX seconds)
df[TIME_COL] = pd.to_datetime(df[TIME_COL], unit="s")

# Set as index
df.set_index(TIME_COL, inplace=True)

# Sort by time
df.sort_index(inplace=True)

# Sanity check
print(df.head())
print(df.index.min(), df.index.max())

#4: EXTRACT CPU & MEMORY SERIES
CPU_COL = "CPU usage [%]"
MEM_COL = "Memory usage [KB]"

# Ensure numeric (extra safety)
df[CPU_COL] = pd.to_numeric(df[CPU_COL], errors="coerce")
df[MEM_COL] = pd.to_numeric(df[MEM_COL], errors="coerce")

# Extract series
cpu_series = df[CPU_COL]
mem_series = df[MEM_COL]

# Drop rows where either is missing
cpu_series = cpu_series.dropna()
mem_series = mem_series.dropna()

# Sanity check
print(cpu_series.head())
print(mem_series.head())
print(cpu_series.shape, mem_series.shape)

# 5: RESAMPLING (10-MIN & 1-HOUR)
# 1. 10-minute resampling
cpu_10min = cpu_series.resample("10min").mean()
mem_10min = mem_series.resample("10min").mean()

# 2. 1-hour resampling
cpu_1h = cpu_series.resample("1h").mean()
mem_1h = mem_series.resample("1h").mean()

# 3. Drop NaNs created by resampling
cpu_10min = cpu_10min.dropna()
mem_10min = mem_10min.dropna()
cpu_1h = cpu_1h.dropna()
mem_1h = mem_1h.dropna()

# 4. Sanity check
print("CPU 10-min:", cpu_10min.shape)
print("MEM 10-min:", mem_10min.shape)
print("CPU 1-hour:", cpu_1h.shape)
print("MEM 1-hour:", mem_1h.shape)

print(cpu_10min.head())
print(cpu_1h.head())

# 6: VISUAL SANITY CHECK

plt.figure(figsize=(14, 4))

# Plot CPU usage (%)
plt.plot(
    df[CPU_COL],
    label="CPU usage (%)"
)

# Safe scaling for memory
cpu_max = df[CPU_COL].max()
if cpu_max == 0:
    mem_scaled = df[MEM_COL] / df[MEM_COL].max()
else:
    mem_scaled = df[MEM_COL] / df[MEM_COL].max() * cpu_max

plt.plot(
    mem_scaled,
    label="Memory usage (scaled)"
)

plt.legend()
plt.title("Cleaned CPU & Memory Time Series")
plt.xlabel("Time")
plt.ylabel("Scaled Value")
plt.savefig("visual_sanity_check.png")
print("Plot saved to visual_sanity_check.png")

# 6A: DATA INTEGRITY CHECK

print("Total rows:", len(df))
print("\nNon-null counts per column:\n")
print(df.notna().sum())

# ---------------------------------------------------------
# NEW SECTIONS: MODELS (LSTM, Autoencoder, CSA)
# ---------------------------------------------------------

# 7: PREPARE DATA FOR LSTM
print("\n--- PREPARING DATA FOR LSTM ---")
scaler_cpu = MinMaxScaler(feature_range=(0, 1))
cpu_data = df[[CPU_COL]].values
cpu_scaled = scaler_cpu.fit_transform(cpu_data)

def create_sequences(dataset, seq_length):
    X, y = [], []
    for i in range(len(dataset) - seq_length):
        X.append(dataset[i:i+seq_length])
        y.append(dataset[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 12
X_lstm, y_lstm = create_sequences(cpu_scaled, SEQ_LENGTH)

train_size = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

print("LSTM X_train shape:", X_train_lstm.shape)

# 8: BI-LSTM MODEL
print("\n--- TRAINING BI-LSTM MODEL ---")
model_lstm = Sequential()
model_lstm.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(SEQ_LENGTH, 1)))
model_lstm.add(Bidirectional(LSTM(50)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.summary()

history_lstm = model_lstm.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, validation_data=(X_test_lstm, y_test_lstm), verbose=1)

# Predict and Evaluate
lstm_preds = model_lstm.predict(X_test_lstm)
lstm_preds_inv = scaler_cpu.inverse_transform(lstm_preds)
y_test_inv = scaler_cpu.inverse_transform(y_test_lstm)

mse = mean_squared_error(y_test_inv, lstm_preds_inv)
mae = mean_absolute_error(y_test_inv, lstm_preds_inv)
print(f"Bi-LSTM MSE: {mse}")
print(f"Bi-LSTM MAE: {mae}")

# Plot LSTM
plt.figure(figsize=(14, 5))
plt.plot(y_test_inv[:200], label="Actual")
plt.plot(lstm_preds_inv[:200], label="Predicted")
plt.title("Bi-LSTM CPU Usage Prediction (First 200 Test Points)")
plt.legend()
plt.savefig("lstm_results.png")
print("LSTM results plot saved to lstm_results.png")

# 9: AUTOENCODER FOR WORKLOAD ESTIMATION / ANOMALY DETECTION
print("\n--- TRAINING AUTOENCODER ---")

# Select features for AE
ae_cols = [CPU_COL, MEM_COL, 'Network received throughput [KB/s]']
valid_ae_cols = [c for c in ae_cols if c in df.columns]
print("Autoencoder features:", valid_ae_cols)

ae_data = df[valid_ae_cols].values
scaler_ae = MinMaxScaler()
ae_data_scaled = scaler_ae.fit_transform(ae_data)

input_dim = ae_data_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded) # Using sigmoid as data is 0-1

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

autoencoder.fit(ae_data_scaled, ae_data_scaled, epochs=10, batch_size=32, verbose=1)

# Reconstruction
reconstructed = autoencoder.predict(ae_data_scaled)
mse_ae = np.mean(np.power(ae_data_scaled - reconstructed, 2), axis=1)
print("Autoencoder Mean Reconstruction Error:", np.mean(mse_ae))

plt.figure(figsize=(14, 5))
plt.plot(mse_ae, label="Reconstruction Error")
plt.axhline(y=np.mean(mse_ae) + 2*np.std(mse_ae), color='r', linestyle='--', label='Threshold')
plt.title("Autoencoder Anomaly Detection (Reconstruction Error)")
plt.legend()
plt.savefig("autoencoder_results.png")
print("Autoencoder results plot saved to autoencoder_results.png")

# 10: CROW SEARCH ALGORITHM (CSA) FOR OFFLOADING OPTIMIZATION
print("\n--- RUNNING CSA FOR OFFLOADING OPTIMIZATION ---")

# Setup dummy problem for CSA
# Objective: Minimize Delay + Energy
# Variables: Offloading Decision (o), Bandwidth Allocation (b)

n_tasks = 50 # Optimization for 50 tasks for demo
if len(df) < n_tasks:
    n_tasks = len(df)

# Proxies from data
tau = df[MEM_COL].values[:n_tasks] # Task size proxy
mu = df['CPU capacity provisioned [MHZ]'].values[:n_tasks] # Processing speed proxy

# Normalize proxies
tau = (tau - tau.min()) / (tau.max() - tau.min() + 1e-9)
mu = (mu - mu.min()) / (mu.max() - mu.min() + 1e-9) + 0.5

def compute_cost(o, tau, mu, b):
    # o: vector of {-1, 0, 1} (Fog, Local, Cloud)
    # b: vector of bandwidths

    # Cost model: D + 0.5 * E
    # Local: D = tau / mu
    # Remote: D = tau / (b + epsilon) + some_latency

    D_local = tau / mu
    D_remote = tau / (b + 1e-6)

    D = np.where(o == 0, D_local, D_remote)
    E = 0.5 * D # Simplified energy

    return np.sum(D + E)

pop_size = 20
iters = 50
B_max = 5.0 # Total bandwidth budget

# Initialize Population
# Offloading: random integers -1, 0, 1
population_o = np.random.randint(-1, 2, size=(pop_size, n_tasks))
# Bandwidth: random floats, normalized to sum to B_max
population_b = np.random.rand(pop_size, n_tasks)
for i in range(pop_size):
    if population_b[i].sum() > 0:
        population_b[i] = population_b[i] / population_b[i].sum() * B_max

best_cost = float('inf')
cost_history = []
best_solution = None

# CSA Loop (Simplified)
for it in range(iters):
    for i in range(pop_size):
        # 1. Update/Mutate
        # Offloading mutation
        new_o = population_o[i].copy()
        # Randomly flip one decision
        idx = np.random.randint(0, n_tasks)
        new_o[idx] = np.random.choice([-1, 0, 1])

        # Bandwidth mutation
        new_b = population_b[i] + 0.1 * np.random.randn(n_tasks)
        new_b = np.abs(new_b) # Non-negative
        if new_b.sum() > 0:
            new_b = new_b / new_b.sum() * B_max

        # 2. Evaluate
        old_cost = compute_cost(population_o[i], tau, mu, population_b[i])
        new_cost = compute_cost(new_o, tau, mu, new_b)

        # 3. Selection (Greedy for this demo)
        if new_cost < old_cost:
            population_o[i] = new_o
            population_b[i] = new_b
            if new_cost < best_cost:
                best_cost = new_cost
                best_solution = (new_o, new_b)

    cost_history.append(best_cost)

print(f"CSA Finished. Best Cost: {best_cost:.4f}")

plt.figure(figsize=(10, 4))
plt.plot(cost_history)
plt.title("CSA Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.savefig("csa_convergence.png")
print("CSA convergence plot saved to csa_convergence.png")
