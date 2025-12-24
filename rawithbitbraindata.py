# Converted from rawithbitbraindata.ipynb

# %% [markdown]
"""
<a href="https://colab.research.google.com/github/Deepikadandolu/Internship/blob/main/rawithbitbraindata.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# %% [code]
# 1: SETUP

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


# %% [code]
# 2: LOAD DATA

excel_df = pd.read_excel('/content/drive/MyDrive/837.xlsx')
excel_df.to_csv('837.csv', index=False)

print("Excel converted to CSV")
display(excel_df.head())


# %% [code]
# 3.

df = pd.read_csv('837.csv')

# Clean column names aggressively
df.columns = (
    df.columns
    .str.strip()
    .str.replace('\t', '', regex=True)
    .str.replace('\n', '', regex=True)
)

print("Available columns in dataset:\n")
for c in df.columns:
    print("-", c)

# ---- Auto-detect required columns ----
cpu_col = [c for c in df.columns if 'cpu usage' in c.lower()][0]
cap_col = [c for c in df.columns if 'capacity' in c.lower()][0]
net_rx  = [c for c in df.columns if 'received' in c.lower()][0]
net_tx  = [c for c in df.columns if 'transmitted' in c.lower()][0]

print("\nSelected columns:")
print("CPU usage        :", cpu_col)
print("CPU capacity     :", cap_col)
print("Network RX       :", net_rx)
print("Network TX       :", net_tx)

# Select and clean data
df = df[[cpu_col, cap_col, net_rx, net_tx]].dropna().reset_index(drop=True)

display(df.head())
print("Dataset shape:", df.shape)



# %% [code]
#4: CLEAN NUMERIC DATA + NORMALIZE

# Remove semicolons and convert everything to numeric
for col in df.columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(';', '', regex=False)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with non-numeric values
df = df.dropna().reset_index(drop=True)

print("After numeric cleaning:")
display(df.head())
print("Dataset shape:", df.shape)

# Normalize

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)

display(df_scaled.head())



# %% [code]
#5: SEQUENCE CREATION

# Re-detect CPU column from df_scaled
cpu_col = [c for c in df_scaled.columns if 'cpu' in c.lower() and 'usage' in c.lower()][0]

print("Using CPU column for workload:", cpu_col)

cpu = df_scaled[cpu_col].values

X, y = [], []
for i in range(len(cpu) - 1):
    X.append([cpu[i]])
    y.append(cpu[i + 1])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Sample X:", X[:5].flatten())
print("Sample y:", y[:5])



# %% [code]
# CELL 6: AUTOENCODER

inp = Input(shape=(1,))
e1 = Dense(16, activation='relu')(inp)
e2 = Dense(8, activation='relu')(e1)
latent = Dense(4, activation='relu')(e2)
d1 = Dense(8, activation='relu')(latent)
d2 = Dense(16, activation='relu')(d1)
out = Dense(1)(d2)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer=Adam(0.001), loss='mse')

autoencoder.summary()


# %% [code]
# CELL 7: TRAINING

history = autoencoder.fit(
    X, y,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Autoencoder Loss Curve")
plt.grid(True)
plt.show()


# %% [code]
# CELL 8: PREDICTION METRICS

y_pred = autoencoder.predict(X).flatten()

mse = mean_squared_error(y, y_pred)

threshold = np.median(y)
y_true_cls = (y >= threshold).astype(int)
y_pred_cls = (y_pred >= threshold).astype(int)

accuracy = accuracy_score(y_true_cls, y_pred_cls)
f1 = f1_score(y_true_cls, y_pred_cls)

print("MSE      :", mse)
print("Accuracy :", accuracy)
print("F1-score :", f1)


# %% [code]
print("Unique true labels:", np.unique(y_true_cls, return_counts=True))
print("Unique pred labels:", np.unique(y_pred_cls, return_counts=True))

print("Number of mismatches:",
      np.sum(y_true_cls != y_pred_cls))


# %% [code]
from sklearn.metrics import r2_score
print("R2:", r2_score(y, y_pred))

print("Correlation:", np.corrcoef(y, y_pred)[0,1])


# %% [code]
# 9. csa optimization

#Auto-detect columns
cpu_col = [c for c in df.columns if 'cpu' in c.lower() and 'usage' in c.lower()][0]
cap_col = [c for c in df.columns if 'capacity' in c.lower()][0]
rx_col  = [c for c in df.columns if 'received' in c.lower()][0]

print("Using columns:")
print("CPU usage    :", cpu_col)
print("CPU capacity :", cap_col)
print("Network RX   :", rx_col)

#Prepare variables
n = len(y_pred)

tau = df[cpu_col].values[:n]          # task size / workload
mu  = df[cap_col].values[:n]          # processing speed
b   = df[rx_col].values[:n] + 1e-6    # bandwidth (numerical safety)

#Simple cost function
def compute_cost(o, tau, mu, b):
    # count remote tasks
    remote_mask = (o != 0)
    N_remote = np.sum(remote_mask)
    N_remote = max(N_remote, 1)  # avoid divide by zero

    # local delay
    D_local = tau / mu

    # shared bandwidth delay
    D_remote = tau / (b / N_remote)

    # select delay
    D = np.where(o == 0, D_local, D_remote)

    # energy proxy
    E = 0.5 * D

    return np.sum(D + E)

# CSA with Bandwidth Optimization
pop_size = 20
iters = 50
B_max = 1.0          # total available bandwidth
eps = 1e-6           # numerical safety

# Offloading population
population = np.random.randint(-1, 2, size=(pop_size, n))

# Bandwidth population (random, normalized)
b_pop = np.random.rand(pop_size, n)
b_pop = b_pop / b_pop.sum(axis=1, keepdims=True) * B_max

best_cost = np.inf
best_o = None
best_b = None
cost_history = []

for it in range(iters):
    for i in range(pop_size):

        # -------- Offloading mutation --------
        new_o = population[i].copy()
        idx = np.random.randint(0, n)
        new_o[idx] = np.random.choice([-1, 0, 1])

        # -------- Bandwidth mutation --------
        new_b = b_pop[i] + 0.05 * np.random.randn(n)
        new_b = np.maximum(new_b, eps)
        new_b = new_b / new_b.sum() * B_max

        # -------- Cost evaluation --------
        old_cost = compute_cost(population[i], tau, mu, b_pop[i])
        new_cost = compute_cost(new_o, tau, mu, new_b)

        # -------- Accept improvement --------
        if new_cost < old_cost:
            population[i] = new_o
            b_pop[i] = new_b

        # -------- Update global best --------
        if new_cost < best_cost:
            best_cost = new_cost
            best_o = new_o.copy()
            best_b = new_b.copy()

    cost_history.append(best_cost)

print("CSA finished")
print("Best cost:", best_cost)




# %% [code]
#BEST_SOLUTION OBJECT

best_solution = {
    "o": best_o,              # offloading decisions
    "b": b,                   # bandwidth (not optimized)
    "U": best_cost,           # final cost
    "history": cost_history   # convergence
}

print("best_solution created with keys:", best_solution.keys())


# %% [code]
print("Unique offloading values:", np.unique(best_o, return_counts=True))


# %% [code]
# CELL 10: CONVERGENCE

plt.plot(cost_history)
plt.title("CSA Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.show()


# %% [code]
# 11: RESOURCE ALLOCATION DECISIONS

# Decode offloading decisions
def decode_offload(o):
    if o == -1:
        return "Fog"
    elif o == 0:
        return "Local"
    else:
        return "Cloud"

allocation = [decode_offload(x) for x in best_o]

# Build decision table
allocation_df = pd.DataFrame({
    "Task_ID": np.arange(len(best_o)),
    "Offloading_Decision": allocation
})

display(allocation_df.head(15))

print("\nAllocation counts:")
print(allocation_df["Offloading_Decision"].value_counts())


# %% [code]
#12: OFFLOADING DISTRIBUTION PLOT

counts = allocation_df["Offloading_Decision"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(counts.index, counts.values)
plt.title("Task Offloading Decisions")
plt.xlabel("Execution Location")
plt.ylabel("Number of Tasks")
plt.grid(True)
plt.show()


# %% [code]
# FIGURE 1: Bandwidth per Task

plt.figure(figsize=(14,4))
plt.bar(np.arange(200), best_b[:200])
plt.title("Optimized Bandwidth Allocation per Task")
plt.xlabel("Task ID")
plt.ylabel("Bandwidth")
plt.grid(True)
plt.show()



# %% [code]
# FIGURE 2: Offloading Decisions per Task

plt.figure(figsize=(14,4))
plt.stem(np.arange(200), best_o[:200], basefmt="k")
plt.yticks([-1,0,1], ["Fog", "Local", "Cloud"])
plt.title("Offloading Decisions per Task")
plt.xlabel("Task ID")
plt.ylabel("Decision")
plt.grid(True)
plt.show()


# %% [code]
#csa convergence
plt.figure(figsize=(6,4))
plt.plot(cost_history)
plt.title("CSA Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.show()


# %% [code]
#  SUMMARY

summary = pd.DataFrame({
    "Metric": ["MSE", "Accuracy", "F1-score", "Final Cost"],
    "Value": [mse, accuracy, f1, best_cost]
})

display(summary)
