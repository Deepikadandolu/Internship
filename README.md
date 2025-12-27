# Internship

Intelligent Workload Estimation and Resource Allocation in Fog–Cloud Computing

Project Title
Intelligent Workload Estimation and Resource Allocation using Deep Autoencoder and Crow Search Algorithm in Fog–Cloud Computing

Project Overview
With the rapid growth of IoT applications, cloud-centric architectures face challenges such as high latency, bandwidth congestion, and inefficient energy utilization. Fog computing mitigates these issues by enabling computation closer to data sources. However, efficient task offloading and resource allocation in fog–cloud environments require accurate workload awareness and intelligent optimization.
This project proposes an integrated Deep Autoencoder (DAE) and Crow Search Algorithm (CSA) framework to:
Estimate system workload demand
Optimally allocate tasks across Local, Fog, and Cloud layers
Allocate bandwidth efficiently under shared network constraints
Minimize overall delay and energy cost

Problem Statement
Traditional cloud-based processing suffers from:
High communication latency
Network congestion
Static and heuristic resource allocation
Inability to adapt to dynamic workloads
Existing models often ignore real-time workload estimation or treat bandwidth as a fixed resource, leading to inefficient utilization.

Proposed Solution
The proposed solution introduces:
Deep Autoencoder (DAE) for workload estimation
Crow Search Algorithm (CSA) for joint optimization of:
Task offloading decisions (Local / Fog / Cloud)
Bandwidth allocation per task
The estimated workload from DAE is directly used by CSA to make intelligent, workload-aware resource allocation decisions.

System Architecture
The system consists of three layers:
1. Local Layer
IoT / edge devices
Limited computational capability
Minimal communication delay
2. Fog Layer
Intermediate processing nodes
Moderate computational power
Reduced latency compared to cloud
3. Cloud Layer
High computational capacity
Higher communication delay
Tasks generated at the local layer can be processed locally or offloaded to fog or cloud based on estimated workload and system cost.

Dataset Description
The dataset contains system monitoring metrics:
Feature
Description
CPU usage [%]
Percentage CPU utilization
CPU capacity provisioned [MHz]
Available CPU capacity
Network received throughput [KB/s]
Incoming network traffic
Network transmitted throughput [KB/s]
Outgoing network traffic

CPU workload demand is computed as:
Workload (MHz) = CPU usage (%) × CPU capacity / 100


Project Pipeline
Load dataset from Excel and convert to CSV
Clean column names and remove malformed values
Convert CPU usage to actual workload demand
Normalize features using Min–Max scaling
Train Deep Autoencoder for workload estimation
Evaluate workload estimation using MSE and SMAPE
Formulate optimization problem (delay + energy cost)
Apply CSA for joint offloading and bandwidth allocation
Analyze cost convergence behavior
Visualize resource allocation and bandwidth distribution

Workload Estimation using Deep Autoencoder
DAE learns latent workload representations
Noise is smoothed through reconstruction
Reconstructed output is treated as estimated workload demand
MSE is used as the primary evaluation metric
Why DAE?
Captures nonlinear workload patterns
More robust than traditional regression
Suitable for unsupervised workload estimation

Resource Allocation using Crow Search Algorithm
CSA is a population-based metaheuristic inspired by the intelligent food-hiding behavior of crows.
Optimized Variables
Task offloading decision:
-1 → Fog
0 → Local
1 → Cloud
Bandwidth allocation per task
Objective Function
Minimize:
Total Cost = Delay Cost + Energy Cost

Key Features
Bandwidth sharing among offloaded tasks
Joint optimization of discrete and continuous variables
Convergence-based stopping criteria

Performance Analysis
Workload Estimation Metrics
Mean Squared Error (MSE)
SMAPE (evaluated on original scale)
Accuracy and F1-score reach 1.0 due to median-based binarization of smooth workload data and are treated as auxiliary metrics.
Optimization Metrics
Final system cost
Cost convergence curve
Task distribution across layers
Bandwidth allocation patterns

Results
Accurate workload estimation with low MSE
Stable CSA convergence after introducing bandwidth sharing
Balanced task distribution across local, fog, and cloud
Efficient bandwidth utilization under constraints

Challenges Faced and Solutions
Challenge
Solution
Dataset column mismatch
Dynamic column cleaning
Malformed numeric values (0.0;)
String cleaning and numeric conversion
Flat CSA convergence
Introduced bandwidth sharing
Constant bandwidth
Joint optimization of bandwidth
Accuracy misinterpretation
Emphasized MSE over classification metrics


Why This Model is Better
Predictive workload-aware allocation
Joint optimization instead of isolated decisions
Adaptive to dynamic workloads
More realistic network modeling

Tools and Technologies
Python
TensorFlow / Keras
NumPy, Pandas
Scikit-learn
Google Colab

Future Scope
Real-time deployment
Mobility-aware fog nodes
Reinforcement learning-based control
Multi-fog node coordination

Conclusion
This project demonstrates an intelligent and adaptive framework for fog–cloud resource allocation by integrating deep learning-based workload estimation with metaheuristic optimization. The proposed DAE–CSA approach effectively reduces system cost and improves scalability for IoT applications.


