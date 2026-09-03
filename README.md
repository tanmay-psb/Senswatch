# Senswatch

### Industrial Equipment Health Monitoring & Predictive Maintenance

<p align="center">
  <b>Machine Learning • Predictive Maintenance • Industrial Fault Diagnosis</b>
</p>

SensorWatch is a machine learning project for **industrial equipment health monitoring and predictive maintenance**.

The project combines two complementary machine learning tasks:

- **Remaining Useful Life (RUL) Prediction** using NASA C-MAPSS
- **Fault Detection & Diagnosis** using the Tennessee Eastman Process (TEP)

Together, these models answer two practical questions:

> **How much useful life does the equipment have remaining?**

and

> **Is the equipment operating normally, and if not, what fault is occurring?**

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Part I — RUL Prediction](#part-i--rul-prediction)
- [Part II — Fault Detection & Diagnosis](#part-ii--fault-detection--diagnosis)
- [Key Findings](#key-findings)
- [Results Summary](#results-summary)
- [Machine Learning Concepts](#machine-learning-concepts)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Datasets](#datasets)
- [Author](#author)
- [License](#license)

---

# Overview

Industrial equipment continuously generates sensor data during operation. Changes in these measurements can provide information about equipment degradation and abnormal operating conditions.

SensorWatch approaches this problem through two different datasets and machine learning objectives.

| Component | Dataset | ML Task | Output |
|---|---|---|---|
| RUL Prediction | NASA C-MAPSS FD001 | Regression | Remaining operational cycles |
| Fault Detection | Tennessee Eastman Process | Binary Classification | Normal / Fault |
| Fault Diagnosis | Tennessee Eastman Process | Multi-Class Classification | Specific fault type |

The datasets are treated as **separate modeling problems** because they represent different predictive-maintenance objectives.

---

# System Architecture

```text
                         SENSORWATCH
                              │
                    Industrial Sensor Data
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
          NASA C-MAPSS                  TEP Dataset
                │                           │
                ▼                           ▼
         RUL Regression             Fault Classification
                │                           │
                ▼                    ┌──────┴──────┐
       Remaining Useful Life         │             │
                │                    ▼             ▼
                │                 Binary       Multi-Class
                │                Detection      Diagnosis
                │                    │             │
                │                    ▼             ▼
                │                 Normal/       Fault
                │                  Fault          Type
                │
                └──────────────► Equipment Health
```

---

# Part I — RUL Prediction

## Dataset

### NASA C-MAPSS FD001

The NASA C-MAPSS Jet Engine Simulated Dataset contains multivariate sensor measurements collected over the operating lifetime of simulated turbofan engines.

Each row represents an **engine operating cycle**.

The FD001 subset contains:

- 100 training engines
- 100 official test engines
- Multiple operating parameters
- Multiple sensor measurements
- Progressive engine degradation

---

## Objective

The goal is to predict the number of operational cycles remaining before an engine reaches the end of its useful life.

This is a **regression problem** because the target is a numerical quantity.

```text
Sensor History
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Random Forest Regressor
      │
      ▼
Predicted RUL
```

For example:

```text
Predicted RUL = 32 cycles

Interpretation:
Approximately 32 operational cycles remain.
```

---

## Data Preprocessing

The C-MAPSS data was processed using the following workflow:

1. Load the raw FD001 training data.
2. Assign meaningful column names.
3. Identify near-constant sensors.
4. Remove sensors with negligible variation.
5. Calculate Remaining Useful Life for every engine cycle.
6. Split engines at the engine level.
7. Generate rolling sensor features.
8. Remove rows where rolling statistics are unavailable.
9. Cap the RUL target at 125 cycles.

---

## RUL Target Engineering

For each engine, RUL was calculated as:

```text
RUL = Maximum Engine Cycle - Current Cycle
```

Example:

```text
Engine final cycle = 200
Current cycle      = 173

RUL = 200 - 173
RUL = 27 cycles
```

Therefore:

```text
RUL = 0
```

represents the final recorded operating cycle of an engine.

---

## Engine-Level Train/Test Split

The data was split by **complete engine trajectories** rather than individual rows.

This is important because randomly splitting individual rows could allow measurements from the same engine to appear in both training and testing data.

### Incorrect Approach

```text
Engine 1

Cycle 1  → Training
Cycle 2  → Testing
Cycle 3  → Training
Cycle 4  → Testing
```

This can cause **data leakage** because the model sees part of the same engine trajectory during training.

### SensorWatch Approach

```text
Engine 1  → Training
Engine 2  → Training
Engine 3  → Testing
Engine 4  → Testing
```

This provides a more realistic evaluation of how the model performs on unseen engines.

---

## Feature Engineering

SensorWatch generates short-term temporal features using a **5-cycle rolling window**.

For each sensor:

### Rolling Mean

```text
Sensor values:

Cycle t-4
Cycle t-3
Cycle t-2
Cycle t-1
Cycle t

        │
        ▼

5-cycle rolling mean
```

### Rolling Standard Deviation

A 5-cycle rolling standard deviation was also calculated to capture recent sensor variability.

These features provide the model with information about recent sensor behavior rather than relying only on individual sensor readings.

---

## RUL Capping

The RUL target was capped at **125 cycles**.

Values above 125 are mapped to 125:

```text
Original RUL       Capped RUL

300       ─────►      125
250       ─────►      125
180       ─────►      125
150       ─────►      125
125       ─────►      125
100       ─────►      100
50        ─────►       50
10        ─────►       10
0         ─────►        0
```

The purpose is to focus the model on the degradation and failure-relevant region instead of requiring it to distinguish between many very large RUL values.

---

## Model

A **Random Forest Regressor** was used for RUL prediction.

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features=0.8,
    random_state=42
)
```

Random Forest was selected because it can model nonlinear relationships and interactions between sensor variables without requiring a strictly linear relationship between the inputs and RUL.

---

# RUL Results

## Internal Engine-Level Evaluation

The model achieved:

| Metric | Result |
|---|---:|
| MAE | **10.09 cycles** |
| R² | **0.869** |

### Interpretation

An MAE of approximately 10 cycles means that, on average, the model's predicted RUL differs from the actual RUL by about **10 operational cycles** on the internal held-out engine set.

---

## Official C-MAPSS FD001 Evaluation

The final model was also evaluated against the official unseen FD001 test trajectories.

| Metric | Result |
|---|---:|
| MAE | **13.94 cycles** |
| RMSE | **18.78 cycles** |
| Test Engines | **100** |

The official test provides a stronger generalization check because the test engines were not used during model training.

### Evaluation Note

The RUL model was trained using an upper target cap of 125 cycles.

The official evaluation compares predictions from this capped-target model against the provided raw RUL values.

Therefore, the official metrics should be interpreted in the context of this modeling decision rather than treated as a directly comparable standard C-MAPSS benchmark.

---

# Part II — Fault Detection & Diagnosis

## Dataset

### Tennessee Eastman Process (TEP)

The Tennessee Eastman Process dataset represents a simulated industrial chemical process operating under:

- Normal conditions
- Multiple simulated fault conditions

The faulty training data contains 20 fault conditions.

The dataset contains:

- 20 fault classes
- 500 simulation runs per fault
- 500 samples per simulation

The fault-free dataset contains:

- 500 normal simulation runs
- 500 samples per simulation

---

# TEP Dataset Structure

The TEP dataset contains **55 columns**.

These consist of:

```text
41 measured process variables
+
11 manipulated process variables
+
3 metadata/label columns
=
55 columns
```

### Measured Variables

```text
xmeas_1
xmeas_2
...
xmeas_41
```

### Manipulated Variables

```text
xmv_1
xmv_2
...
xmv_11
```

---

## Target Variable

The target variable is:

```text
faultNumber
```

It represents the operating condition/fault class.

The model **does not receive `faultNumber` as an input feature**.

Otherwise, the model would effectively be given the answer during training.

Therefore:

```text
X = xmeas + xmv
Y = faultNumber
```

---

## Excluded Columns

The following columns are excluded from the model features:

| Column | Reason |
|---|---|
| `faultNumber` | Target variable |
| `simulationRun` | Simulation identifier |
| `sample` | Time-step identifier |

The model therefore receives **52 process variables**:

```text
41 xmeas
+
11 xmv
=
52 features
```

---

# TEP Objective

The TEP component addresses two related classification problems.

## 1. Binary Fault Detection

The first task answers:

> Is the process operating normally or experiencing a fault?

```text
Normal
   │
   └── OR ──► Fault
```

---

## 2. Multi-Class Fault Diagnosis

The second task answers:

> Which specific fault condition is occurring?

```text
Normal
Fault 1
Fault 2
Fault 3
...
Fault 20
```

This produces a **21-class classification problem**.

---

# TEP Data Splitting

Because the TEP data represents complete simulated process trajectories, randomly splitting individual rows can cause data leakage.

For example, this approach would be problematic:

```text
Simulation 123

Sample 1 → Training
Sample 2 → Testing
Sample 3 → Training
Sample 4 → Testing
```

Instead, SensorWatch splits at the **simulation-run level**.

For each class:

```text
400 simulation runs → Training
100 simulation runs → Testing
```

Across 21 classes:

```text
21 × 400 = 8,400 training simulations

21 × 100 = 2,100 testing simulations
```

A complete simulation run is therefore kept entirely within either the training or testing set.

---

# TEP Downsampling

Each simulation contains 500 time samples.

To reduce computational cost while retaining temporal coverage, every 10th sample was selected.

```text
500 samples per simulation
          │
          ▼
     Every 10th sample
          │
          ▼
Approximately 50 samples
```

This reduces the number of samples while preserving observations across the complete simulation trajectory.

---

# TEP Model

A **Random Forest Classifier** was used as the baseline classification model.

```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)
```

Random Forest provides a strong baseline for nonlinear classification and can capture interactions between multiple process variables.

---

# Binary Fault Detection

For binary fault detection, the original TEP labels were converted into:

```text
0 → Normal
1 → Fault
```

The classifier was then evaluated on its ability to distinguish normal operation from any fault condition.

## Results

| Metric | Result |
|---|---:|
| Accuracy | **74.00%** |
| Normal Recall | **84%** |
| Fault Recall | **74%** |
| Fault F1-score | **0.84** |

The test distribution contains substantially more faulty samples than normal samples.

Therefore, accuracy alone should not be treated as the only performance indicator. Precision, recall, and F1-score provide additional information about the model's behavior.

---

# Multi-Class Fault Diagnosis

The second TEP task predicts the specific process condition:

```text
0  → Normal
1  → Fault 1
2  → Fault 2
3  → Fault 3
...
20 → Fault 20
```

## Results

| Metric | Result |
|---|---:|
| Accuracy | **65.28%** |
| Number of Classes | **21** |
| Input Features | **52** |

The model performs well on several fault types but struggles with others.

This demonstrates an important distinction:

```text
Fault Detection
       │
       ▼
"Is something wrong?"

        is easier than

Fault Diagnosis
       │
       ▼
"Exactly what went wrong?"
```

---

# Difficult Fault Conditions

Some fault classes were particularly difficult for the current model.

| Fault | F1-score |
|---:|---:|
| Fault 9 | **0.002** |
| Fault 15 | **0.040** |
| Fault 3 | **0.047** |
| Normal | **0.235** |
| Fault 16 | **0.453** |
| Fault 19 | **0.527** |
| Fault 10 | **0.548** |
| Fault 20 | **0.619** |

These results indicate that some process conditions have sensor patterns that overlap significantly with other conditions.

The confusion matrix provides a visual representation of these classification errors.

---

# Key Findings

## 1. RUL Prediction

The C-MAPSS model achieved:

```text
Internal Test MAE ≈ 10 cycles
Official Test MAE ≈ 14 cycles
```

The difference between the internal and official test results highlights the importance of evaluating predictive-maintenance models on genuinely unseen equipment.

---

## 2. Engine-Level Splitting Matters

Keeping complete engine trajectories within either training or testing prevents information from the same engine appearing in both sets.

This makes the evaluation more representative of real-world deployment.

---

## 3. Fault Detection Is Easier Than Fault Diagnosis

The TEP experiments demonstrate that identifying whether a fault exists is generally easier than identifying the exact fault type.

```text
Normal vs Fault
      │
      ▼
   74% Accuracy

21-Class Diagnosis
      │
      ▼
   65.28% Accuracy
```

---

## 4. Fault Difficulty Varies

Some TEP faults are highly distinguishable, while others have substantial overlap in their sensor patterns.

Faults such as **3, 9, and 15** were particularly challenging for the current baseline model.

This creates opportunities for future improvements using temporal modeling and more advanced feature engineering.

---

# Machine Learning Concepts Demonstrated

SensorWatch demonstrates practical machine learning concepts including:

- Regression
- Binary classification
- Multi-class classification
- Industrial sensor data
- Time-series data handling
- Feature engineering
- Rolling statistics
- Target engineering
- Remaining Useful Life prediction
- Train/test splitting
- Engine-level data splitting
- Simulation-level data splitting
- Data leakage prevention
- Random Forest regression
- Random Forest classification
- Hyperparameter selection
- Model evaluation
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R²
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Feature importance
- Error analysis
- Generalization analysis

---

# Results Summary

| Component | Task | Model | Result |
|---|---|---|---:|
| C-MAPSS | RUL Regression | Random Forest Regressor | **MAE: 10.09 cycles** |
| C-MAPSS Official Test | RUL Regression | Random Forest Regressor | **MAE: 13.94 cycles** |
| C-MAPSS Official Test | RUL Regression | Random Forest Regressor | **RMSE: 18.78 cycles** |
| TEP | Binary Fault Detection | Random Forest Classifier | **Accuracy: 74.00%** |
| TEP | Multi-Class Diagnosis | Random Forest Classifier | **Accuracy: 65.28%** |

---

# Project Structure

```text
SensorWatch/
│
├── README.md
├── .gitignore
│
└── notebooks/
    └── SensorWatch_RUL_Model.ipynb
```

The raw C-MAPSS and TEP datasets are intentionally excluded from the GitHub repository because of their size.

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computation |
| Pandas | Data manipulation |
| Scikit-learn | Machine learning |
| Matplotlib | Data visualization |
| Google Colab | Model development |
| Git | Version control |
| GitHub | Repository hosting |

---

# Limitations

SensorWatch is a **machine learning research and portfolio project** based on simulated industrial environments.

It should not be considered a production-ready safety-critical maintenance system.

Current limitations include:

- The datasets represent simulated industrial environments.
- The RUL model uses a Random Forest rather than an explicit sequence model.
- Long-term temporal dependencies are not directly modeled.
- Some TEP fault classes remain difficult to distinguish.
- The TEP classifier is a baseline model rather than a fully optimized diagnostic system.
- The current implementation does not provide real-time streaming inference.
- The RUL model uses a target cap of 125 cycles.
- The official C-MAPSS evaluation is therefore interpreted in the context of the capped target.

---

# Future Improvements

Potential future extensions include:

## Advanced Time-Series Modeling

- LSTM
- GRU
- Temporal Convolutional Networks
- Transformers
- Sequence-to-sequence architectures

## Advanced Fault Diagnosis

- Temporal fault classification
- Hierarchical fault diagnosis
- Anomaly detection
- One-class classification
- Feature selection
- Fault-specific models

## Production-Oriented Extensions

- Real-time sensor ingestion
- Streaming predictions
- Model monitoring
- Data drift detection
- Model drift detection
- Automated retraining
- Predictive-maintenance dashboards

---

# Datasets

## NASA C-MAPSS

NASA's C-MAPSS Jet Engine Simulated Dataset was used for Remaining Useful Life prediction.

The project uses the **FD001** subset for the RUL component.

## Tennessee Eastman Process

The Tennessee Eastman Process dataset was used for industrial fault detection and diagnosis.

The dataset contains normal operation and multiple simulated fault conditions.

### Dataset Availability

The raw datasets are **not included in this repository** because of their large size.

To reproduce the experiments, download the datasets from their respective public sources and place them locally according to the notebook's expected file paths.

---

# Why SensorWatch?

Traditional equipment monitoring can identify current measurements, but predictive maintenance aims to answer more useful questions:

```text
Traditional Monitoring
        │
        ▼
"What is happening right now?"
```

SensorWatch extends this idea:

```text
SensorWatch
     │
     ├──► "How much useful life remains?"
     │
     ├──► "Is there a fault?"
     │
     └──► "Which fault is occurring?"
```

This makes SensorWatch a practical demonstration of how machine learning can be applied to different stages of industrial equipment health monitoring.

---

# Conclusion

SensorWatch demonstrates a practical machine learning approach to industrial equipment health monitoring by combining **predictive maintenance** and **fault diagnosis**.

The project uses two complementary datasets:

```text
NASA C-MAPSS
      │
      ▼
Remaining Useful Life Prediction
      │
      ▼
"How much useful life remains?"
```

and:

```text
Tennessee Eastman Process
      │
      ▼
Fault Detection & Diagnosis
      │
      ├──► "Is there a fault?"
      │
      └──► "Which fault is occurring?"
```

The C-MAPSS component demonstrates how multivariate sensor histories can be used to estimate remaining operational life.

The TEP component demonstrates both binary fault detection and multi-class fault diagnosis, while also showing that identifying the exact fault condition is significantly more challenging than simply detecting abnormal operation.

Overall, SensorWatch provides a foundation for exploring machine learning techniques for industrial health monitoring, predictive maintenance, and automated fault diagnosis.

---

# Author

**Tanmay Pratap**

Machine Learning • AI/ML • Software Engineering

---

# License

This project is intended for educational, research, and portfolio purposes.
