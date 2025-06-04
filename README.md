# HypeXAI: A Real-Time Explainable AI Framework for Accurate Hypertension Prediction

![Framework Diagram](Assets/Hypertension_03.png)

## 📌 Overview

**HypeXAI** is an Explainable AI (XAI) framework designed for real-time and accurate **hypertension prediction**. It combines powerful ensemble machine learning techniques with SHAP-based interpretability, providing both **high performance** and **clinical transparency**.

This repository implements the methodology proposed in the paper:

> **HypeXAI: A Real-Time Explainable AI Framework for Accurate Hypertension Prediction**  
> _Pijush Kanti Roy Partho, Pankaj Bhowmik_  
> Department of Electronics and Communication Engineering & Department of Computer Science and Engineering,  
> Hajee Mohammad Danesh Science and Technology University, Dinajpur, Bangladesh  
> 📧 pijushkantiroy2040@gmail.com | pankaj.cshstu@gmail.com

---

## 🧠 Highlights

- ✅ **100% Accuracy** with ensemble models (Voting Classifier, Random Forest, Decision Tree)
- 📉 Comparisons with SVM (99.9%) and Gradient Boosting (97.6%)
- 🔍 Interpretability using **SHAP** visualizations
- ⚡ Real-time hypertension prediction service
- 🧬 Based on 26,083 samples & 14 clinical features from Kaggle dataset

---

## 💡 Methodology

- Dataset preprocessing & feature engineering
- Model training: Random Forest, Decision Tree, SVM, Gradient Boosting, and Voting Classifier
- SHAP-based explainability for feature importance
- Web-based real-time prediction interface

---

## 📊 Results

### ROC Curve

![ROC](Assets/ROC_Curves.png)

### Real-time Prediction Interface

![Server](Assets/server.png)

### SHAP Decision Plot

![SHAP Decision](Assets/shap_decision_plot.png)

### SHAP Summary Plot

![SHAP Summary](Assets/shap_summary_plot.png)

---

## ⚙️ Technologies Used

- Python, Pandas, NumPy
- Scikit-learn
- SHAP (SHapley Additive exPlanations)
- Flask (for real-time prediction API)
- Matplotlib, Seaborn

---

## 🚀 Run Locally

```bash
git clone https://github.com/yourusername/HypeXAI.git
cd HypeXAI
pip install -r requirements.txt
python app.py
```
