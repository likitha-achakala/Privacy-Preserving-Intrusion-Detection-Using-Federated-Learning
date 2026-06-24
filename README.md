# 🔐 Privacy-Preserving Intrusion Detection Using Federated Learning

> **Centralized vs. Federated LSTM on NSL-KDD Dataset**  
> B.Tech CSE Mini Project — SASTRA Deemed to be University, May 2026

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab)](https://colab.research.google.com)
[![Dataset](https://img.shields.io/badge/Dataset-NSL--KDD-green)](https://www.unb.ca/cic/datasets/nsl.html)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 🏆 Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Centralized LSTM | ~96.72% | ~97.86% | ~96.72% | ~97.11% |
| **Federated LSTM (FedProx)** | **97.55%** | **98.37%** | **97.55%** | **97.81%** |

> ✅ **Federated LSTM outperforms Centralized on all 4 metrics** while guaranteeing **zero raw data exposure** across distributed client nodes.

---

## 📌 Project Overview

Traditional Intrusion Detection Systems (IDS) rely on centralizing sensitive network traffic logs — raising serious privacy risks under GDPR and HIPAA. This project implements **Federated Learning (FL)** as a privacy-preserving alternative.

Each client trains a local **LSTM** model and shares only **model weights** with a central aggregation server — raw data never moves. **FedProx** regularization stabilizes training across heterogeneous distributions.

### System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  FEDERATED LEARNING SETUP                    │
│                                                              │
│  [Client 1]  [Client 2]  [Client 3]  [Client 4]  [Client 5] │
│  Local NSL   Local NSL   Local NSL   Local NSL   Local NSL  │
│  KDD Shard   KDD Shard   KDD Shard   KDD Shard   KDD Shard  │
│  LSTM Model  LSTM Model  LSTM Model  LSTM Model  LSTM Model  │
│      │           │           │           │            │      │
│      └───────────┴───────────┴───────────┴────────────┘      │
│                             │                                │
│                ┌────────────▼────────────┐                   │
│                │   Aggregation Server    │                   │
│                │   FedProx  (μ = 0.01)  │                   │
│                │   Weighted FedAvg       │                   │
│                └────────────┬────────────┘                   │
│                             │  20 Communication Rounds       │
│                ┌────────────▼────────────┐                   │
│                │   Global LSTM Model     │                   │
│                │   Acc: 97.55%           │                   │
│                │   Prec: 98.37%          │                   │
│                └─────────────────────────┘                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Repository Structure

```
federated-lstm-ids/
│
├── federated_lstm_ids.ipynb        ← Main Colab notebook (run this!)
├── NSL_KDD_Combined_Shuffled.csv   ← Dataset (upload when prompted)
├── README.md                       ← This file
├── requirements.txt                ← Python dependencies
├── linkedin_post.md                ← Ready LinkedIn post
├── LICENSE
│
└── outputs/                        ← Auto-generated during runtime
    ├── class_distribution.png
    ├── comparison_chart.png
    ├── convergence_analysis.png
    ├── confusion_matrices.png
    ├── per_class_f1.png
    └── performance_enhancement.png
```

---

## 🚀 Quick Start — Google Colab

1. Open **Google Colab** → Upload `federated_lstm_ids.ipynb`
2. Set Runtime → **T4 GPU**
3. Run all cells — upload `NSL_KDD_Combined_Shuffled.csv` when prompted
4. All plots + results generate automatically

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/federated-lstm-ids/blob/main/federated_lstm_ids.ipynb)

---

## 📊 Dataset — NSL-KDD (Combined Shuffled)

| Property | Value |
|---|---|
| File | NSL_KDD_Combined_Shuffled.csv |
| Total Samples | 148,517 |
| Train Samples (80%) | 118,813 |
| Test Samples (20%) | 29,704 |
| Raw Features | 41 |
| Features after MI selection | ~85 |
| Classes | 5 |

**Class Distribution:**

| Class | Label | Count | % |
|---|---|---|---|
| Normal | 0 | 77,054 | 51.9% |
| DoS | 1 | 53,387 | 35.9% |
| Probe | 2 | 14,077 | 9.5% |
| R2L | 3 | 3,880 | 2.6% |
| U2R | 4 | 119 | 0.1% |

---

## ⚙️ LSTM Architecture

```
Input  (shape: 1 × N_features)
   ↓
LSTM  (128 units, return_sequences=True)
   ↓
Dropout (0.3)
   ↓
LSTM  (64 units, return_sequences=False)
   ↓
Dropout (0.3)
   ↓
Dense (64, activation='relu')
   ↓
Dropout (0.2)
   ↓
Dense (5, activation='softmax')
```

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0003 |
| Loss | Categorical Cross-Entropy |
| Batch Size | 256 |
| Class Weights | Balanced (sklearn) |

---

## 🔧 Feature Engineering Pipeline

```
NSL_KDD_Combined_Shuffled.csv
        ↓
1. Remove embedded duplicate header rows
        ↓
2. Map 40+ attack names → 5 super-categories
   (Normal / DoS / Probe / R2L / U2R)
        ↓
3. One-Hot Encode: protocol_type, service, flag
        ↓
4. Convert all columns to float32 (handle mixed types)
        ↓
5. Stratified 80/20 train-test split
        ↓
6. StandardScaler normalization (fit on train only)
        ↓
7. Mutual Information feature selection (drop bottom 30%)
        ↓
8. Reshape → (samples, timesteps=1, features) for LSTM
        ↓
9. Compute balanced class weights
```

---

## 🤝 Federated Learning Configuration

| Parameter | Value |
|---|---|
| Algorithm | FedProx |
| Proximal coefficient (μ) | 0.01 |
| Clients | 5 |
| Communication Rounds | 20 |
| Local Epochs per Round | 3 |
| Batch Size | 256 |
| Aggregation | Weighted FedAvg |
| Data Partition | IID (equal stratified split) |

**Why FedProx?**  
FedProx adds a proximal regularization term `(μ/2)||w − w_global||²` that prevents local models from diverging too far from the global — essential for non-IID traffic patterns across organizations.

---

## 📈 Outputs Generated

| File | Description |
|---|---|
| `class_distribution.png` | Train/Test class balance visualization |
| `comparison_chart.png` | Centralized vs. FL grouped bar chart |
| `convergence_analysis.png` | FL accuracy & loss over 20 rounds |
| `confusion_matrices.png` | Side-by-side normalized confusion matrices |
| `per_class_f1.png` | Per-class F1 scores comparison |
| `performance_enhancement.png` | Accuracy gain: FL over Centralized |

---

## 🔍 Key Findings

1. **FL beats Centralized** across all 4 metrics — Accuracy +0.83%, Precision +0.51%
2. **Fast convergence** — global model surpasses centralized baseline by round 3–5
3. **Strong majority-class detection** — DoS and Probe F1 consistently > 97%
4. **Minority-class challenge** — R2L and U2R remain harder due to severe imbalance (known limitation)
5. **Privacy by design** — raw NSL-KDD traffic never shared between nodes

---

## 🔮 Future Work

- [ ] SMOTE oversampling for R2L/U2R minority classes
- [ ] Differential Privacy (Gaussian noise on gradients)
- [ ] Non-IID data partitioning (Dirichlet distribution)
- [ ] Evaluate on CICIDS-2017 and UNSW-NB15
- [ ] Asynchronous FL for varying client capabilities

---

## 👥 Team

| Name | Reg. No. | Department |
|---|---|---|
| Achakala Venkata Sai Likitha | 227003004 | B.Tech CSE |
| Aishwarya K | 227003007 | B.Tech CSE |
| Dipnitha S | 227003044 | B.Tech CSE |

**Supervisor:** Dr. V. Kalaichelvi, Associate Professor, CSE  
**Institution:** SASTRA Deemed to be University — Srinivasa Ramanujan Centre, Kumbakonam

---

## 📄 Base Paper

> Mubashar Raza et al., *"Federated Learning for Privacy-Preserving Intrusion Detection in Software-Defined Networks"*, **IEEE Access**, Vol. 12, May 2024.  
> DOI: [10.1109/ACCESS.2024.3395997](https://doi.org/10.1109/ACCESS.2024.3395997)

---

## 📦 Requirements

```
tensorflow>=2.12.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center"><b>⭐ Star this repo if it helped you!</b><br>Made with ❤️ at SASTRA Deemed to be University</p>
