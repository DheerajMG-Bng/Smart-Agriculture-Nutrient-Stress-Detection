<div align="center">

#  IoT-Based Real-Time Nutrient Stress Detection
### using Hybrid CNN-LSTM Model for Smart Agriculture

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Arduino](https://img.shields.io/badge/Arduino-IoT%20Hardware-00878F?style=for-the-badge&logo=arduino&logoColor=white)](https://www.arduino.cc/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue?style=for-the-badge)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

<br/>

> **End-to-end smart agriculture system** that detects soil nutrient stress in real time using IoT sensors and a parallel hybrid deep learning model — achieving **96–99% classification accuracy** on Government of India soil data.

<br/>

[![Kaggle Notebook](https://img.shields.io/badge/▶%20Run%20on-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/dheerajmg/cnn-lstm-nutrient-stress-detection-project)
[![Dataset](https://img.shields.io/badge/📊%20Dataset-Soil%20Data%20CSV-brightgreen?style=for-the-badge)](https://github.com/DheerajMG-Bng/Smart-Agriculture-Nutrient-Stress-Detection/blob/main/Soil%20data.csv)
[![Reference Paper](https://img.shields.io/badge/📄%20IEEE-Reference%20Paper-red?style=for-the-badge)](https://github.com/DheerajMG-Bng/Smart-Agriculture-Nutrient-Stress-Detection/blob/main/Reference%20Paper.pdf)

</div>

---


##  Overview

Agriculture is the backbone of India's economy, yet most farmers still rely on expensive, time-consuming laboratory soil tests. This project bridges that gap by building a **complete, deployable smart agriculture pipeline** — from raw sensor data to an intelligent classification decision — at a hardware cost of approximately **INR 800–1,200**.

The system combines two complementary technologies:

- **IoT Hardware Layer** — Arduino Uno with soil moisture, pH, temperature (DHT11), and light (LDR) sensors for real-time field data collection
- **Hybrid CNN-LSTM Model** — A parallel deep learning architecture that simultaneously captures spatial feature correlations (CNN branch) and temporal sequential dependencies (LSTM branch) in soil nutrient data

The model classifies each soil reading into one of three actionable categories:

| Class | Trigger Condition | Action |
|-------|------------------|--------|
|  **Normal** | Balanced N, P, K and pH 5.5–8.0 | No intervention needed |
|  **Deficient** | N < 50 OR P < 30 OR K < 30 | Apply targeted fertilizer |
|  **Stress** | pH < 5.5 OR pH > 8.0 | Soil amendment required |

---

##  Key Results

<div align="center">

| Metric | Value |
|--------|-------|
|  Test Accuracy | **96–99%** |
|  Weighted F1-Score | **0.94** |
|  Total Model Parameters | **9,251** (lightweight, edge-ready) |
|  Training Epochs (EarlyStopping) | < 50 (converges early) |
|  Hardware Cost | ~INR 800–1,200 |
|  Dataset Size | Government of India Soil Data Cards |

</div>

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.99 | 0.99 | 0.99 |
| Deficient | 0.83 | 0.83 | 0.83 |
| Stress | 0.90 | 0.88 | 0.89 |

---

##  System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SMART AGRICULTURE SYSTEM                  │
├──────────────────────────┬──────────────────────────────────┤
│     HARDWARE LAYER       │         SOFTWARE LAYER           │
│                          │                                  │
│  [Soil Moisture Sensor]  │   Soil Data (N, P, K, pH)       │
│  [DHT11 Temp/Humidity]  │          ↓                       │
│  [LDR Light Sensor]  ──►│   StandardScaler (Z-score)       │
│  [pH Sensor Module]      │          ↓                       │
│          ↓               │   ┌──────────────────┐          │
│  [Arduino Uno]           │   │  CNN Branch      │          │
│          ↓               │   │  Conv1D(32) + ReLU│         │
│  Threshold Logic         │   │  Dropout(0.3)    │          │
│  (Immediate Alert)       │   │  Flatten → R^32  │          │
│          ↓               │   └────────┬─────────┘          │
│  [LCD Display]           │            │  Concatenate        │
│  [LED Indicators]        │   ┌────────┘                    │
│                          │   │  LSTM Branch               │
│  🔗 Tinkercad Sim:       │   │  LSTM(32 units)            │
│  iZoJfhCpb4h            │   │  Dropout(0.3) → R^32       │
│                          │   └────────┬─────────┘          │
│                          │            ↓                     │
│                          │   Dense(64, ReLU) + Dropout(0.4)│
│                          │            ↓                     │
│                          │   Softmax(3) → Prediction       │
│                          │   Normal / Deficient / Stress   │
└──────────────────────────┴──────────────────────────────────┘
```

---

## 🔧 Hardware Layer — IoT Sensor Node

Built around the **Arduino Uno (ATmega328P, 16 MHz, 32 KB Flash)**:

| Sensor | Parameter | Range |
|--------|-----------|-------|
| Soil Moisture Sensor (Resistive) | Soil moisture % | 0–100% (ADC 0–1023) |
| DHT11 | Temperature & Humidity | 0–50°C, 20–90% RH |
| LDR (Light Dependent Resistor) | Light intensity | Voltage divider output |
| pH Sensor Module | Soil pH | 0–14 (analog voltage) |

**Embedded Threshold Logic** (hardware-level instant detection):
```
S = 1 (STRESS)  →  if Moisture < 30%  OR  Temp > 40°C  OR  pH < 5.5  OR  pH > 8.0
S = 0 (NORMAL)  →  otherwise
```

> 🔗 **Tinkercad Circuit Simulation:** [View Here](https://www.tinkercad.com/things/iZoJfhCpb4h-embedded-project/editel)

---

##  Model Architecture — Hybrid CNN-LSTM

The model uses a **parallel dual-branch architecture** built with the Keras Functional API:

```python
Input: (1, 4)  →  [N', P', K', pH']  (Z-score normalized)

 ┌──────────────────────┐     ┌──────────────────────┐
 │     CNN Branch       │     │     LSTM Branch       │
 │  Conv1D(32, k=1)     │     │  LSTM(units=32)       │
 │  + ReLU              │     │  + Dropout(0.3)       │
 │  + Dropout(0.3)      │     │  Output: R^32         │
 │  + Flatten → R^32    │     └──────────┬────────────┘
 └──────────┬───────────┘                │
            └──────────── Concat ────────┘
                              ↓
                       Dense(64, ReLU)
                       + Dropout(0.4)
                              ↓
                       Dense(3, Softmax)
                              ↓
               Normal  /  Deficient  /  Stress
```

**Training Configuration:**

```python
optimizer     = Adam(lr=0.001)
loss          = categorical_crossentropy
epochs        = 50 (EarlyStopping, patience=5)
batch_size    = 32
class_weight  = compute_class_weight('balanced')  # handles imbalance
dropout       = [0.3, 0.3, 0.4]                   # CNN, LSTM, Dense
```

---

##  Dataset

- **Source:** Government of India Soil Health Card Data and ([Kaggle: dheerajmg/soil-data](https://www.kaggle.com/datasets/dheerajmg/soil-data))
- **Features:** N (Nitrogen), P (Phosphorus), K (Potassium), pH
- **Labels:** Generated using agronomic threshold heuristics

**Label Generation Logic:**
```
Deficient  →  N < 50  OR  P < 30  OR  K < 30
Stress     →  pH < 5.5  OR  pH > 8.0
Normal     →  otherwise
```

**Class Distribution:**

| Class | Share |
|-------|-------|
| Normal | ~45% |
| Deficient | ~35% |
| Stress | ~20% |

**Preprocessing Pipeline:**
1. LabelEncoder → `to_categorical()` (one-hot)
2. StandardScaler (Z-score, fit on train split only)
3. 80/20 train-test split (`random_state=42`)
4. Reshape to `(n_samples, 1, 4)` for Conv1D + LSTM compatibility
5. `compute_class_weight('balanced')` for imbalance handling

---

##  Project Structure

```
Smart-Agriculture-Nutrient-Stress-Detection/
│
├──  cnn-lstm-nutrient-stress-detection-project_.ipynb   # Main model notebook
├──  Soil data.csv                                        # Government of India soil dataset
├──  Reference Paper.pdf                                  # IEEE reference paper
├──  Work update/                                         # Progress documentation
├──  CNN-LSTM_Smart_Agriculture                           # Model summary / architecture notes
├──  README.md
└──  LICENSE  (GPL-2.0)
```

---

## 📈 Results & Evaluation

### Model vs Baselines

| Model | Accuracy | Notes |
|-------|----------|-------|
| Decision Tree | 88% | Overfits on small datasets |
| SVM (RBF) | 90% | Limited non-linear handling |
| Random Forest | 92% | Poor feature interaction |
| MLP (3-layer) | 94% | No temporal learning |
| CNN (1D only) | 95% | No sequential modeling |
| **CNN-LSTM (Ours)** | **96–99%** |  Best performance |

### Why CNN-LSTM Outperforms
- **CNN branch** captures spatial correlations between N, P, K, pH simultaneously
- **LSTM branch** models sequential gradient patterns across sensor readings
- **Class weight balancing** prevents bias toward the majority Normal class
- **Dropout regularization** prevents overfitting on boundary cases

---

##  Tech Stack

<div align="center">

| Category | Technology |
|----------|-----------|
| Deep Learning | TensorFlow 2.x, Keras Functional API |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Hardware | Arduino Uno (ATmega328P), DHT11, pH Sensor, LDR, Soil Moisture Sensor |
| Simulation | Tinkercad |
| Notebook | Jupyter / Kaggle Kernels |
| Dataset | Government of India Soil Health Cards |

</div>

---

##  Team

**Indian Institute of Information Technology, Vadodara**
*Embedded System Project — Batch 2024–25*

| Name | Role |
|------|------|
| **Dheeraj MG** | Model Development, Dataset, Kaggle Pipeline,Hardware Design, Arduino Firmware,Preprocessing, Evaluation,System Integration, Documentation |



**Guide:** Dr. Kamal Kishor Jha

---
