# 📦 Customer Issue NLP Classifier

An end-to-end Natural Language Processing (NLP) and Machine Learning project that analyzes customer reviews, categorizes issue types, and trains a classification model using TF-IDF and Logistic Regression.

---

## 🚀 Project Overview

This project demonstrates a modular and production-style ML pipeline:

- Data exploration and visualization
- Complaint analysis
- Rule-based issue classification
- Text preprocessing
- Feature engineering using TF-IDF
- Supervised classification with Logistic Regression
- Model evaluation using confusion matrix and performance metrics

The system is structured using separate modules for better scalability and maintainability.

---

## 🏗 Project Architecture

```
customer-issue-nlp-classifier/
│
├── helperdata.py                  # Contains all processing & ML functions
├── system.py                  # Main execution file
├── Customer_Sentiment_csv.csv # Dataset
├── requirements.txt
└── README.md
```

---

## 🧠 Module Responsibilities

### 🔹 helperdata.py
Contains all reusable functions:
- Sentiment visualization
- Customer rating analysis
- Complaint analysis
- Issue type classification
- Text preprocessing
- Model training & evaluation

### 🔹 system.py
- Loads dataset
- Calls functions from `helperdata.py`
- Controls execution flow
- Displays all visualizations

---

## ⚙️ Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How To Run

From project root folder:

```bash
python system.py
```

---

## 📊 Output Graphs

The system generates 5 key visualizations:

1. Sentiment Distribution
2. Customer Rating Distribution
3. Review Length Distribution
4. Issue Type Distribution
5. Confusion Matrix

---

## 🔍 Workflow Explanation

### 1️⃣ Data Loading
- Reads customer review dataset
- Checks missing values

### 2️⃣ Exploratory Data Analysis
- Sentiment distribution
- Rating distribution
- Complaint percentage
- Review length histogram

### 3️⃣ Issue Classification
Rule-based categorization into:
- Delivery Delay
- Delivery Person Behavior
- Package Condition Issue
- Technical/Tracking Issue
- Positive Feedback
- Other/General

### 4️⃣ Text Preprocessing
- Lowercasing
- Removing special characters
- Stopword removal
- Whitespace normalization

### 5️⃣ Feature Engineering
- TF-IDF vectorization
- Unigrams + Bigrams
- English stopword filtering

### 6️⃣ Model Training
- Stratified 80/20 train-test split
- Logistic Regression classifier
- Class weight balancing

### 7️⃣ Model Evaluation
- Accuracy score
- Precision, Recall, F1-score
- Confusion matrix heatmap

---

## 📈 Model Summary
-----------------------------------------------------
| Component           | Technique Used              |
|---------------------|-----------------------------|
| Text Cleaning       | Regex + Stopword Removal    |
| Feature Extraction  | TF-IDF                      |
| Algorithm           | Logistic Regression         |
| Data Split          | Stratified 80/20            |
| Evaluation          | Accuracy + Confusion Matrix |
-----------------------------------------------------
---

## ⚠ Important Note

If the dataset contains repeated or limited unique reviews, accuracy may appear artificially high.  
This project focuses on demonstrating a complete machine learning pipeline structure.

---

This project is intended for educational and demonstration purposes.
