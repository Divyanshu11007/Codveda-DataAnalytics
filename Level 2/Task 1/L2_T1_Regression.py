import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA = r'C:\Users\ASUS\Downloads\DataSet_Extracted\Data Set For Task'
OUT = r'C:\Users\ASUS\Downloads\CodeVeda_Internship_Projects\Level 2\Task 1'

# ---- Load & parse house data (space-separated, no header) ----
raw = pd.read_csv(f'{DATA}\\4) house Prediction Data Set.csv', header=None, sep=r'\s+')
print(f'Raw shape: {raw.shape}')
print('First 5 rows:')
print(raw.head())
print()

# Boston Housing column names
cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
        'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
raw.columns = cols

df = raw.copy()
print('=== Parsed Dataset ===')
print(df.info())
print()
print(df.describe())
print()

# ---- Check for missing values ----
print(f'Missing values total: {df.isnull().sum().sum()}')
print()

# ---- EDA before modeling ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Distribution of target
axes[0, 0].hist(df['MEDV'], bins=30, edgecolor='black', color='steelblue')
axes[0, 0].set_title('Distribution of MEDV (House Price)')
axes[0, 0].set_xlabel('MEDV ($1000s)')

# Correlation with target
corr = df.corr()['MEDV'].sort_values(ascending=False).drop('MEDV')
axes[0, 1].barh(corr.index, corr.values, color='coral')
axes[0, 1].set_title('Correlation with MEDV')

# RM vs MEDV
axes[1, 0].scatter(df['RM'], df['MEDV'], alpha=0.5, color='green')
axes[1, 0].set_xlabel('RM (avg rooms)')
axes[1, 0].set_ylabel('MEDV ($1000s)')
axes[1, 0].set_title('RM vs MEDV')

# LSTAT vs MEDV
axes[1, 1].scatter(df['LSTAT'], df['MEDV'], alpha=0.5, color='purple')
axes[1, 1].set_xlabel('LSTAT (% lower status)')
axes[1, 1].set_ylabel('MEDV ($1000s)')
axes[1, 1].set_title('LSTAT vs MEDV')

plt.tight_layout()
plt.savefig(f'{OUT}\\house_eda.png', dpi=150)
plt.show()
print('Saved: house_eda.png')
print()

# ---- Linear Regression ----
X = df.drop('MEDV', axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('=== Regression Results ===')
print(f'Intercept: {model.intercept_:.4f}')
print()
print('Coefficients:')
for name, coef in zip(cols[:-1], model.coef_):
    print(f'  {name:8s}: {coef:+.4f}')
print()

# ---- Evaluation ----
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f'R-squared  (R²): {r2:.4f}')
print(f'RMSE:           {rmse:.4f}')
print(f'MAE:            {mae:.4f}')
print()

# ---- Actual vs Predicted plot ----
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual MEDV ($1000s)')
plt.ylabel('Predicted MEDV ($1000s)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.tight_layout()
plt.savefig(f'{OUT}\\house_regression_results.png', dpi=150)
plt.show()
print('Saved: house_regression_results.png')

# ---- Residuals ----
residuals = y_test - y_pred
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.6, color='coral')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=20, edgecolor='black', color='steelblue')
plt.xlabel('Residual')
plt.title('Residual Distribution')
plt.tight_layout()
plt.savefig(f'{OUT}\\house_residuals.png', dpi=150)
plt.show()
print('Saved: house_residuals.png')



