# Heart Disease Prediction using Machine Learning

## 📌 Project Overview
Cardiovascular diseases (CVDs) are the leading cause of mortality globally, taking an estimated 17.9 million lives each year according to the World Health Organization. Early detection and risk assessment are critical for timely medical interventions.

This project implements a clinical decision-support classification pipeline that analyzes non-invasive medical diagnostics to predict the likelihood of heart disease in patients.

---

## 📊 Dataset Information
- **Dataset**: `heart_disease_data.csv` (Cleveland Heart Disease Database)
- **Total Records**: 303 patient records
- **Total Attributes**: 14 clinical features
- **Target Variable**: `target` (`1` = Heart Disease Detected, `0` = Healthy Patient)

### Clinical Diagnostic Features
| Feature | Description | Range / Values |
|---|---|---|
| `age` | Age in years | Continuous |
| `sex` | Biological sex | `1` = Male; `0` = Female |
| `cp` | Chest pain type | `0`: Typical angina, `1`: Atypical angina, `2`: Non-anginal, `3`: Asymptomatic |
| `trestbps` | Resting blood pressure | mm Hg on admission |
| `chol` | Serum cholesterol | mg/dl |
| `fbs` | Fasting blood sugar | `1` if > 120 mg/dl; `0` otherwise |
| `restecg` | Resting electrocardiographic results | `0`: Normal, `1`: ST-T wave abnormality, `2`: Left ventricular hypertrophy |
| `thalach` | Maximum heart rate achieved | Continuous |
| `exang` | Exercise-induced angina | `1` = Yes; `0` = No |
| `oldpeak` | ST depression induced by exercise | Continuous |
| `slope` | Slope of peak exercise ST segment | `0`: Upsloping, `1`: Flat, `2`: Downsloping |
| `ca` | Number of major vessels colored by fluoroscopy | `0` to `3` |
| `thal` | Thalassemia status | `1` = Normal, `2` = Fixed defect, `3` = Reversible defect |

---

## 🛠️ Data Preprocessing & Methodology
1. **Missing Data Inspection**: Verified that the dataset contains zero null or corrupted records.
2. **Feature Scaling & Pipeline Construction**:
   - Integrated `StandardScaler` inside a `make_pipeline` to scale continuous metrics (`trestbps`, `chol`, `thalach`, `oldpeak`).
   - Prevents data leakage and ensures clean optimizer convergence for gradient descent models.
3. **Train-Test Splitting**: Stratified 80/20 train-test split to preserve target class proportions across subsets.

---

## 🤖 Modeling & Implementation Pipeline
1. **Baseline Model**: Logistic Regression classifier providing an initial benchmark.
2. **Hyperparameter Tuning (`GridSearchCV`)**:
   - Optimized `RandomForestClassifier` parameters (`n_estimators`, `max_depth`, `min_samples_split`).
   - Used 3-fold cross validation with parallel multi-threading (`n_jobs=-1`).
3. **Model Selection & Comparison**:
   - **Logistic Regression (with Scaling Pipeline)**: ~83.6% accuracy
   - **Tuned Random Forest**: ~85.2% accuracy
   - **Gradient Boosting**: ~83.6% accuracy
4. **Handling Class Imbalance via Downsampling**:
   - Balanced healthy and diseased patient distributions using `sklearn.utils.resample` to ensure balanced clinical sensitivity.
5. **Mitigating Overfitting**:
   - Applied tree regularization (`max_depth=4`, `min_samples_leaf=4`) to tightly align training accuracy with test performance (< 3% gap).
6. **Stratified 5-Fold Cross Validation**:
   - Evaluated generalizability across 5 folds, achieving a stable **~82.4% (±2.1%)** mean accuracy.
7. **End-to-End Clinical Predictive System**:
   - Accepts real-time patient clinical vectors and outputs clear classification results (`Patient has Heart Disease` or `Patient is Healthy`).

---

## 📈 Key Results & Clinical Insights
- **Key Diagnostic Correlations**:
  - Higher ST depression (`oldpeak`) and exercise-induced angina (`exang=1`) are strong indicators of coronary disease.
  - Lower peak exercise heart rate (`thalach`) correlates strongly with cardiovascular impairment.
  - Fluoroscopy vessel count (`ca`) is a powerful predictor of arterial blockage severity.

---

## 🚀 How to Run the Project

### Prerequisites
Install required dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib
```

### Running the Notebook
Open in Jupyter Notebook, JupyterLab, or Google Colab:
```bash
jupyter notebook Heart_Disease_Prediction.ipynb
```
Select `Kernel -> Restart & Run All`. All cells run cleanly in under 15 seconds.

---

## 📁 Repository Structure
```
Heart Disease Prediction/
├── heart_disease_data.csv          # Clinical dataset
├── Heart_Disease_Prediction.ipynb  # Complete Jupyter Notebook
└── README.md                       # Comprehensive documentation
```

---

## 👤 Author
- **Internship Project Submission**
- **Contact / Evaluation**: [info@naviotechsolution.com](mailto:info@naviotechsolution.com)
