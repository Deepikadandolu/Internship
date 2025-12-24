# Converted from raproject.ipynb

# %% [markdown]
"""
<a href="https://colab.research.google.com/github/Deepikadandolu/Internship/blob/main/raproject.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# %% [code]
from google.colab import drive
drive.mount('/content/drive')

# %% [code]
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
# %matplotlib inline
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# %% [code]
excel_df = pd.read_excel('/content/drive/MyDrive/iot_dataset_normalized.xlsx')

excel_df.to_csv('iot_dataset_normalized.csv', index=False)

print("'/content/drive/MyDrive/iot_dataset_normalized.xlsx' has been converted and saved as 'iot_dataset_normalized.csv'")
print("First 5 rows of 'iot_dataset_normalized.csv':")
display(excel_df.head())

# %% [code]
df = pd.read_csv("iot_dataset_normalized.csv")

cpu = df['cpu_cycles'].values
tau = df['task_size'].values
b = df['bandwidth'].values
D = df['delay'].values
mu = df['processing_speed'].values
E = df['energy'].values

X = []  # Input features for the autoencoder
y = []  # Target workload for the autoencoder

for t in range(2, len(cpu) - 1):

    feature_vector = [
        cpu[t], cpu[t - 1], cpu[t - 2],   # Past CPU workloads
        tau[t],                           # Current task size
        b[t],                             # Current bandwidth
        D[t],                             # Current delay
        mu[t],                            # Current processing speed
        E[t]                              # Current energy consumption
    ]

    X.append(feature_vector)
    y.append(cpu[t + 1])   # Next-step CPU workload as the target

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

print("Feature vector example:", X[0])
print("X shape:", X.shape)
print("y shape:", y.shape)

# %% [code]
# Deep Autoencoder for Workload estimmation

# X must be the input features prepared in Step 2 (multi-feature X)
# X.shape should be (num_samples, num_features)
num_features = X.shape[1]   # Number of input features for the autoencoder

# 1. Define Autoencoder Structure
# Input layer: matches the number of features
input_layer = Input(shape=(num_features,))

# Encoder: compresses the input data into a lower-dimensional representation
e1 = Dense(32, activation='relu')(input_layer)
e2 = Dense(16, activation='relu')(e1)
bottleneck = Dense(8, activation='relu', name="latent_space")(e2) # Bottleneck layer (latent space)

# Decoder: reconstructs the original input from the latent space
d1 = Dense(16, activation='relu')(bottleneck)
d2 = Dense(32, activation='relu')(d1)
output_layer = Dense(num_features, activation='linear')(d2) # Output layer (reconstruction)

# Build the autoencoder model
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# 2 Compile Autoencoder
# Use Adam optimizer and Mean Squared Error (MSE) as loss function
autoencoder.compile(
    optimizer='adam',
    loss='mse'       # MSE is common for reconstruction tasks
)

# 3 Print Summary
autoencoder.summary()

# 4. Train the Autoencoder
print("Starting autoencoder training...")
history = autoencoder.fit(
    X, X,                       # Input and target for reconstruction are the same (X)
    epochs=50,                  # Number of training epochs
    batch_size=32,              # Batch size for training
    validation_split=0.1,       # 10% of data used for validation
    verbose=1                   # Show training progress
)

print("Training completed.")

# %% [code]
# STEP 4 – Workload Demand Estimation using Autoencoder

reconstructed_X = autoencoder.predict(X)

estimated_workload = reconstructed_X[:, 0]

print("Sample original workload c(t):")
print(X[:10, 0])

print("\nSample estimated workload demand (AE output):")
print(estimated_workload[:10])

recon_mse = np.mean((X[:, 0] - estimated_workload) ** 2)
print("\nAutoencoder reconstruction MSE (workload feature):", recon_mse)

# %% [code]
# STEP 5 – Workload Demand Analysis
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Build a DataFrame for inspection
demand_df = pd.DataFrame({
    "Original_Workload_c(t)": X[:, 0],
    "Estimated_Workload_Demand": estimated_workload
})

print("\nWorkload Demand Data Preview:")
display(demand_df.head(10))

# Plot 1: Original vs Estimated Workload Demand

plt.figure(figsize=(12,4))
plt.plot(demand_df["Original_Workload_c(t)"], label="Original Workload c(t)", linewidth=1.5)
plt.plot(demand_df["Estimated_Workload_Demand"], label="Estimated Demand (Autoencoder)", linestyle="--")
plt.title("Workload Demand Estimation using Deep Autoencoder")
plt.xlabel("Sample Index")
plt.ylabel("CPU Cycles (normalized)")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Demand Distribution

plt.figure(figsize=(10,4))
plt.hist(demand_df["Estimated_Workload_Demand"], bins=40, alpha=0.8)
plt.title("Distribution of Estimated Workload Demand")
plt.xlabel("Estimated CPU Cycles")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# Basic demand statistics ( for CSA interpretation)

print("\nEstimated Workload Demand Statistics:")
print(demand_df["Estimated_Workload_Demand"].describe())

# %% [code]
# STEP 6
# It has length L = 197, corresponding to c(t+1) for t from 2 to 198.
L = len(estimated_workload)
task_size = df['task_size'].values[3:3+L]
bandwidth = df['bandwidth'].values[3:3+L]
delay = df['delay'].values[3:3+L]
processing_speed = df['processing_speed'].values[3:3+L]
energy = df['energy'].values[3:3+L]

# 1.a unified optimization input structure

optimization_input = {
    "estimated_workload": estimated_workload, # c(t+1)
    "task_size":         task_size,          # τ_j
    "bandwidth":         bandwidth,          # b_j
    "delay":             delay,              # D_j
    "processing_speed":  processing_speed,   # μ_j
    "energy":            energy,             # E_j
    "num_tasks":         L                   # Number of tasks for optimization
}

# 2. Printing preview of the optimization input

print("------ Optimization Input Preview ------")
print("estimated_workload(first 10):", optimization_input["estimated_workload"][:10])
print("Task Sizes (first 10):        ", optimization_input["task_size"][:10])
print("Bandwidth (first 10):         ", optimization_input["bandwidth"][:10])
print("Delay (first 10):             ", optimization_input["delay"][:10])
print("Processing Speed (first 10):  ", optimization_input["processing_speed"][:10])
print("Energy (first 10):            ", optimization_input["energy"][:10])
print("Number of tasks:              ", optimization_input["num_tasks"])

# 3. Normalize values for optimization (if needed by optimizer)

scaler = MinMaxScaler()

# Stack all relevant features into a single matrix for normalization
opt_matrix = np.column_stack([
    optimization_input["estimated_workload"],
    optimization_input["task_size"],
    optimization_input["bandwidth"],
    optimization_input["delay"],
    optimization_input["processing_speed"],
    optimization_input["energy"]
])

# Apply Min-Max Scaling to the optimization matrix
opt_matrix_scaled = scaler.fit_transform(opt_matrix)
optimization_input["normalized_matrix"] = opt_matrix_scaled

print("\nOptimization matrix normalized and ready for CSA.")

# %% [code]
#step 7
import numpy as np

def quantize_o(o_cont):
    """Project continuous o values to discrete {-1,0,1}"""
    o_q = np.copy(o_cont)
    o_q = np.clip(o_q, -1, 1)
    o_q = np.where(o_q <= -0.5, -1, o_q)
    o_q = np.where((o_q > -0.5) & (o_q < 0.5), 0, o_q)
    o_q = np.where(o_q >= 0.5, 1, o_q)
    return o_q.astype(int)

def normalize_bandwidth(b, B_max):
    """Ensure b_j >= 0 and sum(b) <= B_max. If all zero, distribute evenly."""
    b = np.maximum(b, 0.0)
    total = b.sum()
    if total == 0:
        return np.full_like(b, B_max / len(b))
    if total <= B_max:
        return b
    return b * (B_max / total)

# Objective function U

def compute_cost_U(o_vec, b_vec, opt_in, params):
    """
    o_vec: array of int values {-1,0,1}
    b_vec: array of non-negative bandwidth allocations (sum <= B_max)
    opt_in: optimization_input dict (see usage)
    params: dict of system constants:
        p_ID, p_tx, p_FS, p_CD, mu_FS, mu_CD, lambda1, lambda2
    """
    # Extract arrays with fallbacks
    n = len(o_vec)
    tau = np.asarray(opt_in['task_size'], dtype=float)           # task sizes
    pred_work = np.asarray(opt_in['estimated_workload'], dtype=float)
    mu_ID = np.asarray(opt_in.get('processing_speed', np.ones(n)), dtype=float)
    prop_delay = np.asarray(opt_in.get('prop_delay', np.zeros(n)), dtype=float)

    # Params with defaults
    p_ID = params.get('p_ID', 0.5)
    p_tx = params.get('p_tx', 0.2)
    p_FS = params.get('p_FS', 1.0)
    p_CD = params.get('p_CD', 2.0)
    mu_FS = params.get('mu_FS', 1e6)
    mu_CD = params.get('mu_CD', 5e6)
    lambda1 = params.get('lambda1', 0.5)
    lambda2 = params.get('lambda2', 0.5)

    # Safety for division
    b_safe = np.maximum(b_vec, 1e-9)
    mu_ID_safe = np.maximum(mu_ID, 1e-9)
    mu_FS_safe = max(mu_FS, 1e-9)
    mu_CD_safe = max(mu_CD, 1e-9)

    # Local (ID)
    D_id_exe = tau / mu_ID_safe
    E_id = D_id_exe * p_ID
    u_id = lambda1 * D_id_exe + lambda2 * E_id

    # Fog
    D_id_fs_tx = tau / b_safe
    D_fs_exe = tau / mu_FS_safe
    E_tx_fs = p_tx * D_id_fs_tx
    E_fs = p_FS * D_fs_exe
    u_fs = lambda1 * (D_id_fs_tx + D_fs_exe) + lambda2 * (E_tx_fs + E_fs)

    # Cloud
    D_id_cd_tx = tau / b_safe
    D_cd_exe = tau / mu_CD_safe
    E_tx_cd = p_tx * D_id_cd_tx
    E_cd = p_CD * D_cd_exe
    u_cd = lambda1 * (D_id_cd_tx + D_cd_exe + prop_delay) + lambda2 * (E_tx_cd + E_cd)

    o = np.asarray(o_vec, dtype=int)
    part_local = (1 - o**2) * u_id                 # active when o==0
    part_fog   = o * (o - 1) / 2.0 * u_fs          # active when o==-1
    part_cloud = o * (1 + o) / 2.0 * u_cd          # active when o==1

    U_total = np.sum(part_local + part_fog + part_cloud)
    return float(U_total)

# CSA Optimizer

def csa_optimize(optimization_input, params, csa_params=None, random_seed=None):
    """
    optimization_input: dict (see top)
    params: system params (p_ID, p_tx, p_FS, p_CD, mu_FS, mu_CD, lambda1, lambda2)
    csa_params: hyperparams {
        pop_size, max_iter, AP (awareness probability), FL (flight length scalar),
        B_max (total bandwidth limit)
    }
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n = int(optimization_input['num_tasks'])

    # default CSA params
    if csa_params is None:
        csa_params = {}
    pop_size = int(csa_params.get('pop_size', 40))
    max_iter = int(csa_params.get('max_iter', 300))
    AP = float(csa_params.get('AP', 0.15))
    FL = float(csa_params.get('FL', 1.5))
    B_max = float(csa_params.get('B_max', np.sum(optimization_input.get('bandwidth', np.ones(n))))) # Ensure B_max for actual data

    # Initialize population (continuous representation)
    pop_o = np.random.uniform(-1, 1, size=(pop_size, n))   # continuous in [-1,1]
    pop_b = np.random.rand(pop_size, n)                    # random positive

    for i in range(pop_size):
        pop_b[i] = normalize_bandwidth(pop_b[i], B_max)

    # memory (each crow's personal best)
    mem_o = pop_o.copy()
    mem_b = pop_b.copy()
    mem_cost = np.full(pop_size, np.inf)

    # evaluate initial memory
    for i in range(pop_size):
        o_q = quantize_o(mem_o[i])
        b_norm = normalize_bandwidth(mem_b[i], B_max)
        mem_cost[i] = compute_cost_U(o_q, b_norm, optimization_input, params)

    best_idx = int(np.argmin(mem_cost))
    best_o = quantize_o(mem_o[best_idx])
    best_b = normalize_bandwidth(mem_b[best_idx], B_max)
    best_cost = mem_cost[best_idx]
    history = [best_cost]

    # Main CSA loop
    for t in range(max_iter):
        for i in range(pop_size):
            # choose a random crow j != i
            j = np.random.randint(pop_size)
            while j == i:
                j = np.random.randint(pop_size)

            r = np.random.rand()
            if r > AP:
                # follow j's memory
                rand_vec = np.random.rand(n)
                new_o = pop_o[i] + FL * rand_vec * (mem_o[j] - pop_o[i])
                new_b = pop_b[i] + FL * np.random.rand(n) * (mem_b[j] - pop_b[i])
            else:
                # random exploration
                new_o = np.random.uniform(-1, 1, size=n)
                new_b = np.random.rand(n)

            # repair / project
            new_b = normalize_bandwidth(new_b, B_max)
            new_o_q = quantize_o(new_o)

            # evaluate
            new_cost = compute_cost_U(new_o_q, new_b, optimization_input, params)

            # update memory if improved
            if new_cost < mem_cost[i]:
                mem_cost[i] = new_cost
                mem_o[i] = new_o
                mem_b[i] = new_b

                # update global best
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_o = quantize_o(new_o)
                    best_b = new_b

            # update current position
            pop_o[i] = new_o
            pop_b[i] = new_b

        history.append(best_cost)

        # optional tiny-convergence break
        if len(history) > 10 and abs(history[-2] - history[-1]) < 1e-9:
            break

    best_solution = {
        "o": np.array(best_o, dtype=int),
        "b": np.array(best_b, dtype=float),
        "U": float(best_cost),
        "history": history
    }
    return best_solution

# -----------------------
# Example usage (standalone) - COMMENTED OUT TO AVOID INTERFERENCE
# -----------------------
# if __name__ == "__main__":
#     # synthetic example — replace with your actual optimization_input from Step 6
#     n_tasks = 20
#     np.random.seed(0)
#     optimization_input = {
#         "predicted_workload": np.random.uniform(1000, 50000, size=n_tasks),
#         "task_size": np.random.uniform(500, 20000, size=n_tasks),
#         "bandwidth": np.full(n_tasks, 1.0),
#         "delay": np.random.uniform(1, 20, size=n_tasks),
#         "processing_speed": np.random.uniform(1e5, 5e5, size=n_tasks),
#         "energy": np.random.uniform(0.1, 5.0, size=n_tasks),
#         "prop_delay": np.zeros(n_tasks),
#         "num_tasks": n_tasks
#     }

#     params = {
#         "p_ID": 0.5,
#         "p_tx": 0.2,
#         "p_FS": 1.0,
#         "p_CD": 2.0,
#         "mu_FS": 1e6,
#         "mu_CD": 5e6,
#         "lambda1": 0.5,
#         "lambda2": 0.5
#     }

#     csa_params = {
#         "pop_size": 30,
#         "max_iter": 300,
#         "AP": 0.15,
#         "FL": 1.5,
#         "B_max": 10.0
#     }

#     best = csa_optimize(optimization_input, params, csa_params, random_seed=42)

#     print("\n=== CSA Result (example) ===")
#     print("Best cost U:", best['U'])
#     print("Best offloading decisions o_j:", best['o'])
#     print("Best bandwidths b_j:", np.round(best['b'], 4))
#     print("Convergence history length:", len(best['history']))

# %% [code]
# STEP 8 – Final Integration of CSA Output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# best_solution comes from Step 7
# optimization_input comes from Step 6

o_best = best_solution["o"]     # offloading decisions
b_best = best_solution["b"]     # optimized bandwidths
U_best = best_solution["U"]     # optimal cost
num_tasks = len(o_best)

# 1. Convert offloading values to readable form

def decode_offloading(o):
    if o == -1:
        return "Fog"
    elif o == 0:
        return "Local"
    elif o == 1:
        return "Cloud"

offloading_str = [decode_offloading(v) for v in o_best]

# 2. Build final decision DataFrame

result_df = pd.DataFrame({
    "Task_ID": np.arange(num_tasks),
    "Offloading_Decision": offloading_str,
    "o_j_value": o_best,
    "Allocated_Bandwidth": b_best,
    "estimated_workload": optimization_input["estimated_workload"][:num_tasks], # Explicitly slice
    "Task_Size": optimization_input["task_size"][:num_tasks]                 # Explicitly slice
})

print("\n================ FINAL DECISION TABLE ================\n")
display(result_df.head(15))

print("\nTotal Cost U* =", U_best)

# 3. Export results to CSV
csv_path = "final_offloading_and_bandwidth.csv"
result_df.to_csv(csv_path, index=False)
print(f"\n[SAVED] Final decisions saved to: {csv_path}")

# 4. Plot 1 — Bandwidth Allocation

plt.figure(figsize=(14,4))
plt.bar(result_df["Task_ID"], result_df["Allocated_Bandwidth"], color='skyblue')
plt.title("Optimized Bandwidth Allocation per Task")
plt.xlabel("Task ID")
plt.ylabel("Bandwidth")
plt.grid(True, alpha=0.3)
plt.show()

# 5. Plot 2 — Offloading Decisions

plt.figure(figsize=(14,4))
plt.bar(result_df["Task_ID"], result_df["o_j_value"], color='orange')
plt.title("Offloading Decisions (-1=Fog, 0=Local, 1=Cloud)")
plt.xlabel("Task ID")
plt.ylabel("o_j Value")
plt.grid(True, alpha=0.3)
plt.show()

print("\nPlots successfully generated!")

# %% [code]
best_solution = csa_optimize(optimization_input, params, csa_params, random_seed=42)

print("\n=== CSA Optimization Result ===")
print("Best cost U:", best_solution['U'])
print("Best offloading decisions o_j (first 10):", best_solution['o'][:10])
print("Best bandwidths b_j (first 10):", np.round(best_solution['b'][:10], 4))
print("Convergence history length:", len(best_solution['history']))

# %% [code]
# STEP 9 – PERFORMANCE ANALYSIS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Safety alignment

n = best_solution["o"].shape[0]

tau = optimization_input["task_size"][:n]
mu_ID = optimization_input["processing_speed"][:n]
b = best_solution["b"][:n]
o = best_solution["o"][:n]

# Numerical safety
mu_ID = np.maximum(mu_ID, 1e-9)
b = np.maximum(b, 1e-9)

# 2. Delay & Energy Models

# Local
D_local = tau / mu_ID
E_local = D_local * params["p_ID"]

# Fog
D_fog = (tau / params["mu_FS"]) + (tau / b)
E_fog = (tau / b) * params["p_tx"] + (tau / params["mu_FS"]) * params["p_FS"]

# Cloud
D_cloud = (tau / params["mu_CD"]) + (tau / b)
E_cloud = (tau / b) * params["p_tx"] + (tau / params["mu_CD"]) * params["p_CD"]

# 3. Select per-task values based on CSA decision

delay = np.where(o == 0, D_local,
         np.where(o == -1, D_fog, D_cloud))

energy = np.where(o == 0, E_local,
          np.where(o == -1, E_fog, E_cloud))

# 4. Aggregate Metrics

avg_delay = np.mean(delay)
avg_energy = np.mean(energy)
total_cost = best_solution["U"]

print("\n========== CSA PERFORMANCE ==========")
print(f"Number of Tasks      : {n}")
print(f"Average Delay        : {avg_delay:.6f}")
print(f"Average Energy       : {avg_energy:.6f}")
print(f"Total Cost (U*)      : {total_cost:.6f}")

# 5. Offloading Distribution

labels, counts = np.unique(o, return_counts=True)
label_map = {-1: "Fog", 0: "Local", 1: "Cloud"}
dist = {label_map[k]: v for k, v in zip(labels, counts)}

plt.figure(figsize=(6,4))
plt.bar(dist.keys(), dist.values())
plt.title("Offloading Decision Distribution (CSA)")
plt.ylabel("Number of Tasks")
plt.grid(True)
plt.show()

print("\nOffloading Distribution:", dist)

# 6. Bandwidth Utilization

plt.figure(figsize=(10,3))
plt.plot(b)
plt.title("Optimized Bandwidth Allocation")
plt.xlabel("Task ID")
plt.ylabel("Bandwidth")
plt.grid(True)
plt.show()

print("\nBandwidth Stats:")
print("Total Bandwidth Used :", np.sum(b))
print("Bandwidth Std Dev    :", np.std(b))
print("Bandwidth CV         :", np.std(b) / np.mean(b))

# 7. CSA Convergence

plt.figure(figsize=(7,4))
plt.plot(best_solution["history"])
plt.title("CSA Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Cost U")
plt.grid(True)
plt.show()

# 8. Save Summary

summary_df = pd.DataFrame({
    "Metric": ["Avg Delay", "Avg Energy", "Total Cost"],
    "Value": [avg_delay, avg_energy, total_cost]
})

display(summary_df)


# %% [code]
display(summary_df)
