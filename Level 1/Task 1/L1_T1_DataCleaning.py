import pandas as pd
import numpy as np

DATA = r'C:\Users\ASUS\Downloads\DataSet_Extracted\Data Set For Task'
OUT = r'C:\Users\ASUS\Downloads\CodeVeda_Internship_Projects\Level 1\Task 1'

# ---- Load dataset ----
df = pd.read_csv(f'{DATA}\\1) iris.csv')
print('=== Original Dataset ===')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(df.head())
print()

# ---- Check missing values ----
print('=== Missing Values ===')
print(df.isnull().sum())
print()

# ---- Check duplicates ----
print(f'Duplicate rows: {df.duplicated().sum()}')
print('Duplicates:')
print(df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist()))
print()

# ---- Remove duplicates ----
df_clean = df.drop_duplicates().reset_index(drop=True)
print(f'Shape after removing duplicates: {df_clean.shape}')
print()

# ---- Check data types ----
print('=== Data Types ===')
print(df_clean.dtypes)
print()

# ---- Standardize categorical values ----
df_clean['species'] = df_clean['species'].str.strip().str.lower()
print('Unique species after standardization:')
print(df_clean['species'].value_counts())
print()

# ---- Check for outliers (IQR method) ----
numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print('=== Outlier Detection (IQR) ===')
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
    print(f'{col}: {len(outliers)} outliers (bounds: [{lower:.2f}, {upper:.2f}])')
print()

# ---- Summary of cleaned data ----
print('=== Cleaned Dataset Summary ===')
print(f'Final shape: {df_clean.shape}')
print(df_clean.describe())
print()

# Save cleaned data
df_clean.to_csv(f'{OUT}\\1) iris_cleaned.csv', index=False)
print('Cleaned dataset saved as iris_cleaned.csv')

