# House Price Prediction Project

## Objective
The objective of this project is to build a machine learning regression model to predict house prices using various house features.

---

## Dataset
House Price Prediction Dataset from Kaggle.

Target Variable:
- price

---

## Features Used
The dataset contains features related to:
- Area
- Bedrooms
- Bathrooms
- Stories
- Parking
- Furnishing Status
- Air Conditioning
- Preferred Area
- And more

---

## Algorithms Used
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

---

## Libraries Used
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## Project Workflow
1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Visualization
4. Log Transformation
5. Encoding Categorical Features
6. Feature Scaling
7. Train-Test Split
8. Model Training
9. Model Evaluation
10. Residual Analysis
11. Model Saving
12. Prediction Example

---

## Preprocessing Techniques
- Log Transformation using `np.log1p()`
- One-Hot Encoding using `pd.get_dummies()`
- Feature Scaling using `StandardScaler`

---

## Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## Best Model
Gradient Boosting Regressor achieved the best performance.

---

## Model Saving

```python
joblib.dump(gb, 'best_house_price_model.pkl')
```

---

## Example Prediction Code

```python
import pandas as pd
import numpy as np
import joblib

model = joblib.load('best_house_price_model.pkl')

sample_house = pd.DataFrame(
    [X.iloc[0]],
    columns=X.columns
)

sample_house_scaled = scaler.transform(sample_house)

prediction = model.predict(sample_house_scaled)

print("Predicted House Price:", np.exp(prediction[0]))
```

---

## Conclusion
This project demonstrates a complete machine learning regression workflow including preprocessing, feature engineering, model comparison, evaluation, and prediction using house price data.