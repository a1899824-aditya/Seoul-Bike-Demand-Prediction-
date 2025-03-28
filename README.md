# 🚲 Seoul Bike Demand Prediction – Machine Learning Regression in Python

This project applies machine learning techniques to predict hourly bike rental demand in Seoul using a rich dataset with weather and temporal features. Developed as part of the *Machine Learning Tools (COMP SCI 7203)* coursework, this project explores data preprocessing, outlier handling, model selection, and hyperparameter tuning.

---

## 📁 Dataset

- **SeoulBikeData.csv**  
  Contains hourly records of bike rentals from 2017–2018 in Seoul, Korea  
  Features include:
  - Date, Hour, Rented Bike Count
  - Weather metrics: Temperature, Humidity, Wind speed, Solar radiation, Rainfall, Snowfall
  - Categorical: Seasons, Holiday, Functioning Day

---

## 🎯 Objective

- Clean and transform the dataset into a model-ready format  
- Predict **Rented Bike Count** using **regression models**  
- Use pipelines for preprocessing and model fitting  
- Compare **Linear Regression** and **Support Vector Regression (SVR)** with and without GridSearch optimization  
- Visualize performance of best model

---

## 🛠 Tools & Libraries

- Python, Jupyter Notebook  
- `pandas`, `numpy`, `matplotlib`, `seaborn`  
- `scikit-learn` (pipelines, regression, SVR, cross-validation, GridSearch)

---

## 🔄 Preprocessing Steps

- Removed non-functional days using the `Functioning Day` column  
- One-hot encoded `Seasons`, dropped unused features  
- Created a binary `Weekday` feature from `Date`  
- Converted all categorical columns to numerical  
- Handled outliers using the IQR method  
- Imputed missing values using `SimpleImputer(median)`  
- Scaled features using `StandardScaler`

---

## 🤖 Models Used

- Linear Regression (Baseline)
- Support Vector Regression (SVR)
- Pipeline with hyperparameter tuning via `GridSearchCV`

---

## 📊 Results

| Model               | Mean RMSE (CV) | Std Dev | Test RMSE |
|---------------------|----------------|----------|------------|
| Linear Regression   | ~425           | ~9.09    | **410.87** |
| Support Vector Regr | ~516           | ~12.6    | 632.47 (after tuning) |

✅ **Linear Regression** outperformed SVR in both cross-validation and test performance.

---

## 📈 Visualizations

- Feature distributions pre- and post-cleaning  
- Boxplots for outlier detection  
- Actual vs Predicted scatter plot for final model

![Predicted vs Actual](plot.png)

---

## 🧠 Key Learnings

- Pipelines simplify preprocessing and model management  
- Data cleaning (outlier handling, encoding) significantly boosts accuracy  
- Linear regression can outperform complex models when features are well-prepared  
- GridSearch for SVR did not improve performance due to model bias

---

## 📄 Report

See `Assignment_1_a1899824.pdf` for full analysis, discussion, and pipeline code with output.

---

## 👤 Author

Aditya Venugopalan Nediyirippil  
GitHub: [a1899824-aditya](https://github.com/a1899824-aditya)

---

## ✅ Future Improvements

- Experiment with ensemble methods (Random Forest, XGBoost)  
- Feature engineering using temporal cycles (e.g., sine/cosine for Hour)
- Apply time-series forecasting methods (e.g., ARIMA, Prophet)

