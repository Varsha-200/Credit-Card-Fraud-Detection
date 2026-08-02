# 💳 Credit Card Fraud Detection Using Machine Learning

A machine learning project for detecting fraudulent credit card transactions by comparing multiple classification algorithms and addressing the challenges of highly imbalanced transaction data.

This project was developed as part of the **Intel Unnati Program**.

---

## 📌 Overview

Credit card fraud presents a significant challenge to financial institutions and consumers as fraudulent transactions represent only a small proportion of overall financial transactions.

This project explores the application of machine learning to classify credit card transactions as **legitimate or fraudulent**.

Four machine learning algorithms were trained and compared:

- Logistic Regression
- Random Forest
- Decision Tree
- Support Vector Machine (SVM)

To address the highly imbalanced nature of the dataset, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

Based on the experimental results, **Random Forest demonstrated the strongest overall performance for detecting fraudulent transactions**.

---

## 🎯 Project Objectives

The main objectives of this project were to:

- Analyse credit card transaction data
- Identify patterns associated with fraudulent transactions
- Perform exploratory data analysis and preprocessing
- Address class imbalance using SMOTE
- Train multiple classification models
- Evaluate models using appropriate performance metrics
- Compare model performance
- Identify the most effective model for fraud detection

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection dataset from Kaggle**.

### Dataset Characteristics

- **284,807 transactions**
- **31 features**
- Contains legitimate and fraudulent transactions
- Includes transaction `Time` and `Amount`
- Other transaction features are anonymized
- Target variable: `Class`

### Target Classes

```text
Class 0 → Legitimate Transaction
Class 1 → Fraudulent Transaction
```

A major challenge with this dataset is the significant imbalance between legitimate and fraudulent transactions.

---

## 📊 Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the characteristics and distribution of the transaction data before model training.

The analysis included:

- Transaction class distribution
- Fraudulent vs legitimate transactions
- Transaction amount analysis
- Transaction time analysis
- Amount vs time relationships
- Feature correlation analysis

### Transaction Class Distribution

The class distribution illustrates the significant imbalance between legitimate and fraudulent transactions.

![Transaction Class Distribution](<Transaction Class Distribution.jpg>)

### Transaction Amount vs Time

Transaction amount and time were analysed to explore patterns within the transaction data.

![Amount Vs Time](<Amount Vs Time.jpg>)

---

## ⚙️ Machine Learning Workflow

```text
Credit Card Transaction Dataset
              │
              ▼
    Exploratory Data Analysis
              │
              ▼
       Data Preprocessing
              │
              ▼
       Train / Test Split
           (80 / 20)
              │
              ▼
      SMOTE Oversampling
              │
              ▼
        Model Training
              │
   ┌──────────┼──────────┬──────────┐
   ▼          ▼          ▼          ▼
Logistic    Random    Decision     SVM
Regression  Forest      Tree
   │          │          │          │
   └──────────┴──────────┴──────────┘
              │
              ▼
       Model Evaluation
              │
              ▼
 Accuracy | Precision | Recall
      F1-Score | RMSE
              │
              ▼
       Model Comparison
              │
              ▼
        Random Forest
```

---

## 🧹 Data Preprocessing

### Missing Values

The dataset was inspected for missing values and prepared before model training.

### Train-Test Split

The dataset was divided into **80% training data and 20% testing data**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Handling Class Imbalance

Fraudulent transactions occur much less frequently than legitimate transactions.

To address this imbalance, **SMOTE** was applied to the training dataset.

```python
smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train,
    y_train
)
```

This generates synthetic samples of the minority fraud class to create a more balanced training dataset.

---

## 🤖 Machine Learning Models

### 1. Logistic Regression

Used as a binary classification model for predicting whether transactions are legitimate or fraudulent.

### 2. Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees to improve classification performance.

### 3. Decision Tree

A tree-based classification algorithm that learns decision rules from transaction features.

### 4. Support Vector Machine (SVM)

A Support Vector Machine using a **linear kernel** was evaluated for transaction classification.

---

## 📏 Evaluation Metrics

The models were compared using:

- Accuracy
- Precision
- Recall
- F1-Score
- RMSE
- Processing Time

For highly imbalanced problems such as fraud detection, **precision, recall and F1-score are particularly important**, as overall accuracy alone may not accurately represent fraud-class performance.

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

Although several models achieved high overall accuracy, their ability to identify the minority **fraud class** differed considerably.

### Random Forest

Random Forest achieved:

```text
Fraud Precision : 0.87
Fraud Recall    : 0.85
Fraud F1-Score  : 0.86
RMSE            : 0.021
```

Based on the reported experiments, **Random Forest provided the strongest overall fraud-detection performance**.

### Why Accuracy Alone Is Not Enough

Logistic Regression achieved **99% overall accuracy** and **92% fraud recall**, but only **14% fraud precision**.

This demonstrates why accuracy alone can be misleading when evaluating models on highly imbalanced datasets.

---

## 🛠️ Technologies & Libraries

### Programming Language
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

### Techniques
- Exploratory Data Analysis
- Data Preprocessing
- SMOTE Oversampling
- Train/Test Splitting
- Logistic Regression
- Random Forest Classification
- Decision Tree Classification
- Support Vector Machine
- Model Evaluation

---

## 🧠 Key Challenges

### Imbalanced Dataset

Fraudulent transactions represent only a small proportion of the dataset.

SMOTE was therefore used to improve the representation of the minority fraud class during model training.

### False Positives vs False Negatives

Fraud detection requires consideration of two important errors:

**False Positive:**  
A legitimate transaction is incorrectly classified as fraudulent.

**False Negative:**  
A fraudulent transaction is incorrectly classified as legitimate.

The balance between precision and recall therefore plays an important role when evaluating fraud-detection models.

---

## 💡 Key Learnings

Through this project, I gained practical experience in:

- Machine learning classification
- Exploratory Data Analysis
- Data preprocessing
- Handling imbalanced datasets
- SMOTE oversampling
- Training and testing classification models
- Comparing multiple machine learning algorithms
- Evaluating models using accuracy, precision, recall and F1-score
- Understanding the limitations of accuracy on imbalanced datasets
- Data visualization using Matplotlib and Seaborn
- Applying machine learning to a real-world financial problem

---

## 🔮 Future Improvements

Potential extensions include:

- Hyperparameter tuning
- Cross-validation
- Additional ensemble-learning models
- Improved feature engineering
- Cost-sensitive classification
- Explainable AI techniques
- Real-time fraud detection
- Automated machine learning pipelines
- Deployment as a fraud-detection API
- Evaluation on additional transaction datasets

---

## 📄 Project Report

The complete project methodology, implementation details, code excerpts, experiments and results are available in the project report:

[View Project Report](Credit_Card_fraud_Detection_Using_Machine_Learning.pdf)

> **Repository Note:** This repository currently contains the project documentation and report. The original implementation files are not currently included in the repository.

---

## 🎓 Intel Unnati Program

This project was completed as part of the **Intel Unnati Program**, providing practical exposure to machine learning and its application to real-world problems.

---

## 👥 Team

- Nandana A
- Lara Marium Jacob
- **Varsha S Panicker**
- Rizia Sara Prabin

**Project Mentor:** Dr. Starlet Ben Alex  
**Institution:** Saintgits College of Engineering and Technology

---

## 📄 Disclaimer

This project was developed for academic and educational purposes and is not intended for use as a production financial fraud-detection system.
