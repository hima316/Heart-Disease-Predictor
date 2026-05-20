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

## 📊 Model Performance

| Model | Training Accuracy | Testing Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 83.05% | 88.52% | 82.35% | 96.55% | 88.89% |
| Random Forest | 100.00% | **90.16%** | 82.86% | 100.00% | **90.63%** |
| Gradient Boosting | 99.59% | 88.52% | 80.56% | 100.00% | 89.23% |
| AdaBoost | 86.78% | 88.52% | 82.35% | 96.55% | 88.89% |

### 🏆 Best Performing Model

**Random Forest Classifier** achieved the highest testing accuracy of **90.16%**.

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
