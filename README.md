# 💳 Credit Risk Assessment: Lending Club Default Prediction
### Handling Big Data & Class Imbalance in Financial Modeling

## 📌 Project Overview
This project focuses on building a robust machine learning pipeline to predict loan defaults using real-world data from **Lending Club**. The primary challenge addressed here is managing a massive dataset (**2.2 Million+ rows**) and overcoming extreme **Class Imbalance** to prioritize the detection of risky borrowers.

## 📊 The Dataset
The model utilizes the comprehensive Lending Club dataset (available on Kaggle), which includes historical records of over 2 million loans. 
* **Key Features:** Loan Amount, Interest Rate, Annual Income, Debt-to-Income (DTI) Ratio, and Employment Length.
* **Target Variable:** Binary classification (1 = Default/Charged-Off, 0 = Fully Paid).

## 🛠️ Data Engineering & Cleaning
Working with Big Data required a strategic preprocessing approach:
1.  **Dimensionality Reduction:** Removed columns with >30% missing values to optimize memory.
2.  **Feature Selection:** Focused on high-impact financial indicators used in credit scoring.
3.  **Handling Imbalance:** Initially, the dataset was heavily biased toward "Fully Paid" loans. I implemented **Random Under-Sampling** to balance the classes, which was crucial for improving the model's sensitivity to defaults.

## 🧠 Modeling & Optimization
* **Algorithm:** Random Forest Classifier.
* **Initial Challenge:** The baseline model achieved 80% accuracy but a near-zero **Recall (0.07)** for defaults, making it useless for risk management.
* **The Solution:** After balancing the dataset, the model's ability to identify defaulters improved significantly.

## 📈 Performance Results
After optimization, the model achieved a balanced and realistic performance suitable for banking standards:
* **Accuracy:** 64%
* **Recall (Default Detection):** 65% (A **9x improvement** over the baseline).
* **Precision (Reliability):** 64%
* **ROC-AUC Score:** Evaluated to ensure strong separability between risk classes.



## 🔍 Business Insights (Feature Importance)
The model identified that the **Interest Rate**, **Installment Amount**, and **Debt-to-Income (DTI)** are the top predictors of credit risk. This explainability is vital for financial compliance and transparent decision-making.



## 💻 Tech Stack
* **Language:** Python
* **Data Handling:** Pandas, NumPy, KaggleHub
* **Machine Learning:** Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

---
*Developed by Khalid Hamad Alsaab as part of Data Science coursework and financial risk exploration.*
