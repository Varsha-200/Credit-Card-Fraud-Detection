# 💳 Credit Card Fraud Detection Using Machine Learning

A machine learning project for detecting fraudulent credit card transactions by comparing multiple classification algorithms and addressing the challenges of highly imbalanced financial transaction data.

This project was developed as part of the **Intel Unnati Program**.

---

## 📌 Overview

Credit card fraud is a significant challenge for financial institutions because fraudulent transactions represent only a very small proportion of overall transactions.

This project explores the use of machine learning to distinguish fraudulent transactions from legitimate ones. Multiple classification algorithms were trained and evaluated, with particular attention given to handling class imbalance and comparing model performance using precision, recall, F1-score, accuracy, and RMSE.

Among the evaluated models, **Random Forest provided the strongest overall fraud-detection performance** in the reported experiments.

---

## 🎯 Project Objectives

The main objectives of this project were to:

- Analyse credit card transaction data
- Identify patterns associated with fraudulent transactions
- Preprocess and prepare transaction data for machine learning
- Address class imbalance using oversampling
- Train multiple classification models
- Compare model performance using appropriate evaluation metrics
- Identify the most effective model for fraud detection

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection dataset from Kaggle**.

### Dataset Characteristics

- **284,807 transactions**
- **31 features**
- Contains both legitimate and fraudulent transactions
- Includes transaction **Time** and **Amount**
- Other transaction features are anonymized
- Target variable: `Class`

### Target Classes

```text
Class 0 → Legitimate Transaction
Class 1 → Fraudulent Transaction
```

A major challenge with this dataset is the significant imbalance between legitimate and fraudulent transactions.

---

## 🔄 Machine Learning Workflow

```text
Credit Card Transaction Dataset
              ↓
      Exploratory Data Analysis
              ↓
       Data Preprocessing
              ↓
      Train / Test Split
              ↓
     Class Imbalance Handling
           (SMOTE)
              ↓
      Model Training
              ↓
 ┌────────────┼─────────────┐
 ↓            ↓             ↓
Logistic    Random       Decision
Regression  Forest         Tree
              ↓
             SVM
              ↓
       Model Evaluation
              ↓
 Accuracy | Precision | Recall
       F1-Score | RMSE
              ↓
      Model Comparison
              ↓
       Random Forest
      Selected as Best
```

---

## 🧹 Data Preprocessing

Several preprocessing steps were performed before model training.

### Missing Values

The dataset was inspected for missing values and prepared before training.

### Train-Test Split

The data was divided into training and testing datasets using an **80/20 split**.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Handling Class Imbalance

Fraudulent transactions are significantly less frequent than legitimate transactions.

To address this problem, the project used **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data.

```python
smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train,
    y_train
)
```

This creates additional synthetic samples of the minority fraud class to provide the models with a more balanced training dataset.

---

## 🤖 Machine Learning Models

Four classification algorithms were evaluated.

### 1. Logistic Regression

Used as a baseline binary classification model for predicting legitimate and fraudulent transactions.

### 2. Random Forest Classifier

An ensemble learning approach that combines multiple decision trees to improve classification performance.

### 3. Decision Tree

A tree-based classifier that recursively separates the feature space according to learned decision rules.

### 4. Support Vector Machine (SVM)

A Support Vector Machine with a **linear kernel** was used to identify a separating hyperplane between transaction classes.

---

## 📏 Evaluation Metrics

The models were compared using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **RMSE**
- **Processing Time**

For fraud detection, metrics such as **precision and recall are particularly important** because overall accuracy can be misleading when the dataset is highly imbalanced.

---

## 📈 Results

### Fraudulent Transaction Detection

| Model | RMSE | Precision | Accuracy | Recall | F1-Score | Processing Time |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest** | **0.021** | **0.87** | **1.00** | **0.85** | **0.86** | 390.33 sec |
| Logistic Regression | 0.100 | 0.14 | 0.99 | 0.92 | 0.24 | 37.18 sec |
| Decision Tree | 0.046 | 0.44 | 1.00 | 0.81 | 0.57 | 40.88 sec |
| SVM | 0.025 | 0.38 | 0.99 | 0.81 | 0.51 | 570.66 sec |

### Legitimate Transaction Detection

| Model | RMSE | Precision | Accuracy | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.021 | 1.00 | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 0.100 | 1.00 | 0.99 | 0.99 | 0.99 |
| Decision Tree | 0.046 | 1.00 | 1.00 | 1.00 | 1.00 |
| SVM | 0.025 | 1.00 | 0.99 | 1.00 | 1.00 |

---

## 🏆 Model Comparison

Although several models achieved very high overall accuracy, their ability to correctly identify the minority **fraud class** differed considerably.

**Random Forest was selected as the strongest model in the project**, providing:

- **87% precision** for fraudulent transactions
- **85% recall**
- **86% F1-score**
- Low RMSE of **0.021**

Logistic Regression achieved higher fraud recall (**92%**) but substantially lower precision (**14%**), demonstrating why accuracy alone should not be used to evaluate fraud-detection systems.

---

## 📊 Exploratory Data Analysis

The project explored several characteristics of the transaction dataset, including:

- Distribution of legitimate vs fraudulent transactions
- Transaction amount distribution
- Transaction time vs amount
- Feature correlations
- Transaction clusters

### Transaction Class Distribution

<!-- Add transaction-class-distribution.png here -->

![Transaction Class Distribution](transaction-class-distribution.png)

### Transaction Amount vs Time

<!-- Add amount-vs-time.png here -->

![Amount vs Time](amount-vs-time.png)

---

## 🛠️ Technologies & Libraries

### Language

- Python

### Machine Learning

- Scikit-learn
- Imbalanced-learn

### Data Processing

- Pandas
- NumPy
- SciPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning Techniques

- Logistic Regression
- Random Forest Classification
- Decision Tree Classification
- Support Vector Machine
- SMOTE
- Train/Test Splitting
- Model Evaluation

---

## 🧠 Key Challenges

### Imbalanced Dataset

Fraudulent transactions represent a very small proportion of credit card transactions.

SMOTE was therefore used to balance the training data before model development.

### Model Evaluation

High accuracy does not necessarily mean that a fraud-detection model performs well.

For this reason, the project compared **precision, recall and F1-score**, particularly for the fraudulent class.

### False Positives vs False Negatives

Fraud detection requires balancing two important errors:

- **False Positive:** A legitimate transaction is incorrectly classified as fraud.
- **False Negative:** A fraudulent transaction is incorrectly classified as legitimate.

The appropriate balance depends on the requirements of the financial system in which the model would be deployed.

---

## 💡 Key Learnings

Through this project, I gained practical experience with:

- Machine learning classification
- Exploratory data analysis
- Data preprocessing
- Handling imbalanced datasets
- SMOTE oversampling
- Training and testing ML models
- Comparing multiple classification algorithms
- Evaluating models using precision, recall and F1-score
- Understanding the limitations of accuracy for imbalanced datasets
- Data visualization using Matplotlib and Seaborn
- Applying machine learning to a real-world financial problem

---

## 🔮 Future Improvements

Potential extensions of this project include:

- Real-time fraud detection
- Hyperparameter optimization
- Cross-validation
- Additional ensemble-learning approaches
- Explainable AI techniques
- Improved feature engineering
- Cost-sensitive classification
- Automated model pipelines
- Deployment as a fraud-detection API
- Evaluation on additional transaction datasets

---

## 📄 Project Report

The complete project methodology, implementation, code excerpts, experiments, and results are available in the project report:

[View Project Report](Credit_Card_fraud_Detection_Using_Machine_Learning.pdf)

> **Note:** This repository currently contains the project documentation and report. The original implementation files are not currently included in the repository.

---

## 🎓 Intel Unnati Program

This project was completed as part of the **Intel Unnati Program**, providing practical exposure to machine learning and its application to real-world problems.

---

## 👥 Team

- Nandana A
- Lara Marium Jacob
- Varsha S Panicker
- Rizia Sara Prabin

**Project Mentor:** Dr. Starlet Ben Alex  
**Institution:** Saintgits College of Engineering and Technology

---

## 📌 Disclaimer

This project was developed for academic and educational purposes. It is not intended to be used as a production financial fraud-detection system.
