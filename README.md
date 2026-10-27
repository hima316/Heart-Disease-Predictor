
# ❤️ Heart Disease Prediction System

---

## 📌 Overview

The **Heart Disease Prediction System** is a Machine Learning based web application that predicts the likelihood of heart disease using patient medical information.

The project was developed using multiple classification algorithms and deployed using **Streamlit** to provide real-time predictions through an interactive user interface.

This project demonstrates:

* Data preprocessing
* Model training and evaluation
* Performance comparison of multiple ML algorithms
* Deployment of a Machine Learning model as a web application

---

## 🚀 Features

✅ Predicts heart disease risk using medical parameters

✅ Interactive Streamlit web application

✅ Multiple Machine Learning models implemented

✅ Real-time prediction system

✅ Model performance comparison

✅ Clean healthcare-focused UI

---

## 🧠 Machine Learning Models Used

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier
* AdaBoost Classifier

---

## 📊 Model Evaluation & Selection (Stratified 5-Fold Cross-Validation)

Evaluated on the 70% training split (212 samples) using identical stratified 5-fold splits with `StandardScaler` fitted strictly within each fold pipeline to prevent data leakage:

| Model | F1-Score (Mean ± Std) | Recall (Mean ± Std) | Accuracy (Mean ± Std) | Precision (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting** 🏆 | **0.8296 ± 0.0434** | **0.8783 ± 0.1079** | **0.8065 ± 0.0460** | **0.7990 ± 0.0749** |
| Random Forest | 0.8274 ± 0.0573 | 0.8522 ± 0.0758 | 0.8068 ± 0.0651 | 0.8104 ± 0.0822 |
| Logistic Regression | 0.8060 ± 0.0482 | 0.8174 ± 0.0797 | 0.7877 ± 0.0513 | 0.8010 ± 0.0603 |
| AdaBoost | 0.8010 ± 0.0518 | 0.8174 ± 0.1007 | 0.7828 ± 0.0492 | 0.7942 ± 0.0539 |

### 🏆 Selected Winning Model: Gradient Boosting
Selected dynamically based on the highest 5-fold cross-validation mean F1-Score (**0.8296**) and highest clinical sensitivity/recall (**87.83%**).

### 🎯 Final Evaluation on Untouched 30% Test Set (91 Patients)
The winning pipeline was fitted on the full 70% training split and evaluated once on the unseen 30% test set:
- **Test Accuracy**: **75.82%** (69 / 91)
- **Test Precision**: **79.17%** (38 / 48)
- **Test Recall**: **76.00%** (38 / 50)
- **Test F1-Score**: **77.55%**

---

## 🔍 Model Interpretability & Explainability
- **Global Interpretability**: Permutation feature importance on the 30% held-out test data (top predictors: `cp` chest pain type, `thal` thalassemia, `ca` major vessels, and `thalach` maximum heart rate).
- **Local Interpretability (SHAP)**: Patient-level risk breakdowns using `shap.TreeExplainer`, detailing each biomarker's direction and magnitude on prediction risk.
- **Influence vs. Causation**: Explicitly distinguishes statistical feature influence from physiological medical causation.


---

## ⚙️ Model Working Flow

```text
Data Collection
       ↓
Data Preprocessing
       ↓
Feature Selection
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Prediction System
       ↓
Deployment using Streamlit

```

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries & Frameworks

* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* SHAP (SHapley Additive exPlanations)
* Pickle

---

## 💻 Run Locally

### Clone Repository

```bash
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor

```

### Install Dependencies

```bash
pip install -r requirements.txt

```

### Run Streamlit App

```bash
streamlit run app.py

```