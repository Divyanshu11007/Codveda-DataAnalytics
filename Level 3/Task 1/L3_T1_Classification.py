import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

DATA = r'C:\Users\ASUS\Downloads\DataSet_Extracted\Data Set For Task\Churn Prdiction Data'
OUT = r'C:\Users\ASUS\Downloads\CodeVeda_Internship_Projects\Level 3\Task 1'

# ---- Load data ----
df = pd.read_csv(f'{DATA}/churn-bigml-80.csv')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print()
print('First 3 rows:')
print(df.head(3))
print()

# ---- Check missing ----
print(f'Missing values: {df.isnull().sum().sum()}')
print()

# ---- Preprocessing ----
# Encode binary categoricals
df['International plan'] = (df['International plan'] == 'Yes').astype(int)
df['Voice mail plan'] = (df['Voice mail plan'] == 'Yes').astype(int)

# Encode state
le = LabelEncoder()
df['State'] = le.fit_transform(df['State'])

# Target
df['Churn'] = df['Churn'].astype(int)

print('=== Churn Distribution ===')
print(df['Churn'].value_counts())
print(f"Churn rate: {df['Churn'].mean():.2%}")
print()

# Features & target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f'Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')
print()

# ---- Train multiple classifiers ----
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({'Model': name, 'Accuracy': acc, 'Precision': prec,
                    'Recall': rec, 'F1-Score': f1})

    print(f'=== {name} ===')
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    print()

results_df = pd.DataFrame(results)
print('=== Model Comparison ===')
print(results_df.to_string(index=False))
print()

# Plot comparison
results_df.set_index('Model').plot(kind='bar', figsize=(10, 6), rot=0)
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUT}/churn_model_comparison.png', dpi=150)
plt.show()
print('Saved: churn_model_comparison.png')
print()

# ---- Hyperparameter Tuning (Random Forest) ----
print('=== Grid Search: Random Forest ===')
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0
)
grid.fit(X_train_sc, y_train)

print(f'Best params: {grid.best_params_}')
print(f'Best CV F1: {grid.best_score_:.4f}')

best_model = grid.best_estimator_
y_pred_tuned = best_model.predict(X_test_sc)
print(f'Test F1 (tuned): {f1_score(y_test, y_pred_tuned):.4f}')
print(f'Test Accuracy (tuned): {accuracy_score(y_test, y_pred_tuned):.4f}')
print()

# ---- Confusion Matrix (best model) ----
cm = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Tuned Random Forest')
plt.tight_layout()
plt.savefig(f'{OUT}/churn_confusion_matrix.png', dpi=150)
plt.show()
print('Saved: churn_confusion_matrix.png')
print()

# ---- Feature Importance ----
importances = best_model.feature_importances_
feat_imp_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
feat_imp_df = feat_imp_df.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['feature'][:10], feat_imp_df['importance'][:10], color='steelblue')
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUT}/churn_feature_importance.png', dpi=150)
plt.show()
print('Saved: churn_feature_importance.png')