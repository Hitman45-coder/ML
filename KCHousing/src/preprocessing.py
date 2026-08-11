# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: ML (3.14.4)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# %%
filepath = "./data/kc_house_data.csv"
data = pd.read_csv(filepath)

# %%
# 1. Parse dates to get sale year and month
data['date'] = pd.to_datetime(data['date'])
data['sale_year'] = data['date'].dt.year
data['sale_month'] = data['date'].dt.month

# 2. Calculate house age at the time of sale and flag renovations
data['house_age'] = data['sale_year'] - data['yr_built']
data['is_renovated'] = (data['yr_renovated'] > 0).astype(int)

# %%
# Drop 'id' (useless), raw dates/years (replaced by engineered features)
cols_to_drop = ['id', 'date', 'yr_built', 'yr_renovated']
X = data.drop(cols_to_drop + ['price'], axis=1)

# Log-transform the target variable to handle heavy skew in housing prices
y = np.log1p(data['price'])

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Tree-based models dominate real estate datasets because they handle non-linear spatial data (lat/long)
model = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_leaf_nodes=63,
    random_state=42
)

print("Training gradient boosting model...")
model.fit(X_train, y_train)

# %%
# Predict and reverse the log transformation (expm1) to get actual dollars
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

# Calculate metrics
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²:   {r2:.4f}")
print(data.head(5))
