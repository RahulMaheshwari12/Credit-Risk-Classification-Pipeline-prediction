
# 📊 Loan Default Risk Classification 🏦

## Project Overview
This project implements an **end-to-end machine learning solution** to predict **loan default risk** using the **LendingClub dataset**.  
The goal is to identify **high-risk borrowers at the point of application**, ensuring the model reflects **real-world lending conditions**.

**Primary Challenge:**  
Handling and eliminating **data leakage** caused by “future information” present in the dataset.

---

## 🚀 Key Performance Metrics
- **ROC-AUC Score:** 0.71 (5-Fold Cross-Validation)
- **Default Recall:** 82%
- **Learning Strategy:** Cost-sensitive learning using **balanced class weights**

---

## 🛠️ Technical Implementation

### 1. Data Integrity & Leakage Prevention
A comprehensive data audit was performed to remove **post-origination features**, including:
- `recoveries`
- `total_rec_prncp`
- `out_prncp`
- and 25+ similar variables

**Why it matters:**  
Including these features can inflate accuracy to ~99%, producing a model that completely fails in real-world scenarios.  
This project strictly uses **pre-approval information only**, ensuring realistic performance.

---

### 2. Advanced Preprocessing Pipeline
Implemented a **Scikit-learn Pipeline** with **ColumnTransformer** to guarantee reproducible and leak-free preprocessing:

**Numerical Features**
- Median Imputation
- Standard Scaling

**Categorical Features**
- One-Hot Encoding  
  (e.g., `grade`, `home_ownership`, `purpose`)

---

### 3. Handling Class Imbalance
Loan defaults are significantly underrepresented in the dataset.  
To address this, **Cost-Sensitive Learning** was applied:

- `class_weight='balanced'`

This prioritizes catching defaults over avoiding false alarms.

---

### 4. Hyperparameter Optimization
Hyperparameters were tuned using **GridSearchCV**:

- **Regularization Strength (C):** 0.01 → 10
- **Penalty:** L1 (Lasso) vs L2 (Ridge)

**Result:**  
L1 Regularization was selected, enabling **automatic feature selection** by zeroing out weak predictors.

---

## 📈 Business Logic: Threshold Tuning
Instead of the default probability threshold (0.5), multiple thresholds were evaluated:

- 0.3
- 0.4
- 0.5

**Final Threshold:** **0.4**

**Reasoning:**  
In credit risk modeling:
- **False Negative (missed default)** = loss of principal
- **False Positive** = loss of potential interest

A 0.4 threshold maximizes financial safety while capturing **82% of defaulters**.

---

## 🧬 Top Predictive Insights
The most influential features identified by the model:

1. **Interest Rate** – Higher rates signal higher risk
2. **DTI (Debt-to-Income Ratio)** – Measures borrower’s financial burden
3. **Credit Inquiries (Last 6 Months)** – Indicates aggressive credit-seeking behavior

---

## 📂 Repository Structure
```
├── loan_pending_prediction.ipynb   # Complete ML pipeline & analysis
├── loan.csv                        # Raw LendingClub dataset
├── README.md                       # Project documentation
```

---

## 📌 Conclusion
This project demonstrates how **careful feature auditing**, **cost-aware modeling**, and **business-driven evaluation metrics** are essential for building **production-grade credit risk models**.

---

## 🧠 Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

### 👤 Author
**Rahul Maheshwari**

Feel free to fork, explore, or reach out with feedback!
