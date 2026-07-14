import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Update DATA and OUT paths for Colab environment
DATA = r'/content'
OUT = r'/content'

df = pd.read_csv(f'{DATA}/1) iris_cleaned.csv')

print('=== Summary Statistics ===')
print(df.describe())
print()

print('=== Mode ===')
print(f"Mode of sepal_length: {df['sepal_length'].mode().values}")
print(f"Mode of sepal_width: {df['sepal_width'].mode().values}")
print(f"Mode of petal_length: {df['petal_length'].mode().values}")
print(f"Mode of petal_width: {df['petal_width'].mode().values}")
print()

# ---- Histograms ----
df.hist(figsize=(10, 8), bins=15, edgecolor='black')
plt.suptitle('Feature Distributions - Iris Dataset', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/iris_histograms.png', dpi=150)
plt.show()
print('Saved: iris_histograms.png')
print()

# ---- Boxplots ----
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for ax, col in zip(axes.ravel(), numeric_cols):
    sns.boxplot(data=df, x='species', y=col, ax=ax, palette='Set2')
    ax.set_title(f'{col} by Species')
plt.tight_layout()
plt.savefig(f'{OUT}/iris_boxplots.png', dpi=150)
plt.show()
print('Saved: iris_boxplots.png')
print()

# ---- Scatter plots ----
g = sns.pairplot(df, hue='species', palette='Set2', diag_kind='kde')
g.savefig(f'{OUT}/iris_pairplot.png', dpi=150)
plt.show()
plt.close('all')
print('Saved: iris_pairplot.png')
print()

# ---- Correlation ----
print('=== Correlation Matrix ===')
corr = df[numeric_cols].corr()
print(corr)
print()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap - Iris Features')
plt.tight_layout()
plt.savefig(f'{OUT}/iris_correlation_heatmap.png', dpi=150)
plt.show()
print('Saved: iris_correlation_heatmap.png')
