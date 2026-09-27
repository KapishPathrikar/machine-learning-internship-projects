# Customer Churn Prediction using Machine Learning

## 📌 Project Overview
This project develops an end-to-end Machine Learning classification pipeline to predict customer churn in the telecommunications industry using the **Telco Customer Churn** dataset. Customer churn represents a critical business challenge because customer retention is significantly more cost-effective than new customer acquisition.

By analyzing demographic characteristics, subscribed services, account profiles, and billing metrics, the predictive model identifies at-risk subscribers before they cancel, allowing businesses to execute targeted retention strategies.

---

## 📊 Dataset Information
- **Dataset**: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (Telco Customer Churn)
- **Total Records**: 7,043 customer accounts
- **Total Attributes**: 21 features
- **Target Variable**: `Churn` (`Yes` / `No` converted to binary `1` / `0`)

### Key Feature Categories
- **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Subscribed Services**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Account & Financial Information**: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

---

## 🛠️ Data Preprocessing & Feature Engineering
1. **Cleaning & Type Conversion**:
   - Converted `TotalCharges` from string to numeric; handled empty records with median imputation.
   - Dropped non-informative identifier (`customerID`).
2. **Categorical Encoding**:
   - Encoded all categorical features (`gender`, `Contract`, `PaymentMethod`, etc.) using `LabelEncoder`.
3. **Data Splitting**:
   - Split dataset into 80% training data and 20% test data with stratified sampling to preserve the target ratio.

---

## 🤖 Modeling & Implementation Pipeline
The project implements a complete modeling lifecycle:

1. **Baseline Model**: Trained a `RandomForestClassifier` on the encoded dataset.
2. **Hyperparameter Tuning (`GridSearchCV`)**:
   - Optimized `n_estimators`, `max_depth`, and `min_samples_split`.
   - Used multi-core parallelization (`n_jobs=-1`) and 3-fold cross validation for fast convergence.
3. **Model Selection & Comparison**:
   - **Logistic Regression** (with `StandardScaler` pipeline to ensure convergence): ~81.5% accuracy
   - **Tuned Random Forest**: ~80.7% accuracy
   - **Gradient Boosting Classifier**: ~80.9% accuracy
4. **Addressing Class Imbalance via Downsampling**:
   - Balanced majority non-churn instances with churn instances using `resample()`.
   - Boosted churn detection recall from ~59% to ~75%.
5. **Overfitting Mitigation**:
   - Applied structural tree regularization (`max_depth=6`, `min_samples_leaf=5`, `max_features='sqrt'`).
   - Narrowed train vs. test accuracy gap to < 1.5%.
6. **Stratified 5-Fold Cross Validation**:
   - Evaluated using `StratifiedKFold(n_splits=5)`, achieving a consistent **80.18% (±0.87%)** mean accuracy.
7. **End-to-End Predictive System**:
   - Implemented an inference function that accepts raw customer dictionaries and generates real-time churn predictions with probability scores.

---

## 📈 Key Results & Insights
| Model | Test Accuracy | Strengths |
|---|---|---|
| **Logistic Regression (Pipeline)** | **81.5%** | Fast, linear decision boundary with zero scaling warnings |
| **Gradient Boosting** | **80.9%** | Robust non-linear generalization on complex customer interactions |
| **Random Forest (Tuned)** | **80.7%** | High feature interpretability and resistance to outliers |

- **Top Risk Factors**: Month-to-Month contracts, short tenure (< 6 months), electronic check billing, and lack of Tech Support / Online Security.

---

## 🚀 How to Run the Project

### Prerequisites
Install the required Python packages:
```bash
pip install numpy pandas scikit-learn matplotlib
```

### Running the Notebook
Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab:
```bash
jupyter notebook Customer_Churn_Prediction_using_ML.ipynb
```
Run all cells sequentially (`Kernel -> Restart & Run All`). The complete run takes under 1 minute.

---

## 📁 Repository Structure
```
Customer Churn Prediction/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset file
├── Customer_Churn_Prediction_using_ML.ipynb # Complete Jupyter Notebook
└── README.md                             # Documentation file
```

---

## 👤 Author
- **Internship Project Submission**
- **Contact / Evaluation**: [info@naviotechsolution.com](mailto:info@naviotechsolution.com)
