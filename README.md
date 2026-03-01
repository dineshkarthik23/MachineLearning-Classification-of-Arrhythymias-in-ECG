# Automated Detection and Classification of Arrhythmias in ECG Signals

> A machine learning pipeline for automated cardiac arrhythmia detection from 12-lead ECG signals.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Algorithms](#algorithms)
- [Results](#results)
- [Authors](#authors)
- [References](#references)

---

## Overview

Electrocardiograms (ECGs) are a fundamental diagnostic tool in cardiology, capturing the heart's electrical activity across characteristic waveforms — the P-wave, QRS complex, and T-wave. Manual interpretation of large ECG datasets is time-consuming and error-prone.

This project presents an automated system that:
- Preprocesses raw 12-lead ECG signals
- Extracts clinically relevant features (QRS duration, R-R intervals, HRV)
- Classifies cardiac arrhythmias using both supervised and unsupervised machine learning

### Arrhythmia Classes

| Label | Condition |
|---|---|
| AFib | Atrial Fibrillation |
| NSR | Normal Sinus Rhythm |
| SB | Sinus Bradycardia |
| STach | Sinus Tachycardia |
| SVT | Supraventricular Tachycardia |

---

## Dataset

- **Source:** Shaoxing People's Hospital & Ningbo First Hospital of Zhejiang University (2013–2018)
- **Size:** 40,258 12-lead ECG recordings
- **Patients:** 22,599 male, 17,659 female
- **Format:** `.mat` files, 10-second recordings sampled at 500 Hz
- **Distribution:** ~20% Normal Sinus Rhythm, ~80% abnormal readings
- **Demographics:** Majority of patients aged 51–80 years

---

## Pipeline

### 1. Preprocessing

Raw ECG signals go through a four-stage filtering process:

1. **Butterworth Low-Pass Filter** — Removes high-frequency noise above 50 Hz
2. **LOESS Smoothing** — Estimates and removes baseline wander via local regression
3. **Baseline Correction** — Centers the signal around zero
4. **Non-Local Means (NLM) Denoising** — Reduces random noise while preserving morphological features (P-waves, QRS complexes, T-waves) without manual threshold selection

### 2. Feature Extraction

A total of **11 distinct feature combinations** were analyzed, ranging from 11 to 39,830 features per sample.

Features extracted from Lead II ECG include:

- **Temporal:** RR intervals (mean, variance, count), PR intervals, QT intervals
- **Morphological:** QRS complex statistics, P-wave and T-wave prominence
- **Relational:** Peak/valley ratios (width, height, prominence differences over time)
- **Demographic:** Age and gender
- **Distribution:** Empirical frequency histograms (100 bins) for variable-length features → fixed-length vectors

### 3. Classification

Models are trained on extracted features and evaluated using **Accuracy, Precision, Recall, F1-score, and AUC-ROC**.

---

## Algorithms

### Supervised Learning

| Model | Description |
|---|---|
| **Logistic Regression** | Sigmoid-based binary/multi-class classifier with balanced class weights |
| **Decision Tree** | Hierarchical feature-split classifier; fast and interpretable |
| **K-Nearest Neighbors (KNN)** | Distance-based classifier with SMOTE oversampling for class imbalance |
| **Random Forest** | Ensemble of 150 decision trees (`max_depth=15`, `class_weight='balanced'`) |
| **SVM** | RBF kernel SVM with GridSearchCV hyperparameter tuning |

### Unsupervised Learning

| Model | Description |
|---|---|
| **K-Means Clustering** | Elbow Method + Silhouette Score used to determine optimal `k=3` |

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | **96.8%** | 98.1% | 96.1% | 97.1% |
| Random Forest | 94.13% | — | 94.13% | 93.08% |
| SVM | 91.4% | 91.6% | 88.2% | 88.5% |
| Decision Tree | 91.0% | — | — | — |
| KNN | 75.0% | — | — | — |

**Key observations:**
- Logistic Regression achieved the best overall accuracy (96.8%) and an AUC-ROC of 99.3%
- Random Forest showed the best balance of precision/recall for multi-class classification
- All supervised models performed well on common rhythms (NSR, SB) but struggled with minority classes (SVT, VFib) due to class imbalance
- K-Means (k=3) successfully separated ECG clusters, validating the natural structure of the data

---


## Authors

| Name | Email |
|---|---|
| Gowri Shankar A B | cb.en.u4cce23001@cb.students.amrita.edu |
| Mugilan S S | cb.en.u4cce23026@cb.students.amrita.edu |
| Dinesh Karthik V | cb.en.u4cce23008@cb.students.amrita.edu |
| Duvvuru Akshaya Saketh Reddy | cb.en.u4cce23011@cb.students.amrita.edu |

**Department of Electronics and Communication**  
Amrita Vishwa Vidhyapeetham, Coimbatore

---

## References

1. J. Zheng, H. Chu, D. Struppa, et al., "Optimal Multi-Stage Arrhythmia Classification Approach," *Scientific Reports*, vol. 10, no. 2898, February 2020.

2. H. K. Kim and M. H. Sunwoo, "An Automated Cardiac Arrhythmia Classification Network for 45 Arrhythmia Classes Using 12-Lead Electrocardiogram," *IEEE Access*, vol. 12, pp. 44527–44538, 2024.

3. N. S, J. M, I. T, S. R and K. G, "Pre-Cardiac Arrhythmia Detection using Machine Learning," *2023 International Conference on Intelligent Technologies for Sustainable Electric and Communications Systems (iTech SECOM)*, Coimbatore, India, 2023.

---

*© 20XX IEEE — Amrita Vishwa Vidhyapeetham*
