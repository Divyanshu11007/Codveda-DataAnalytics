# Codveda Technology — Data Analytics Internship

**Name:** Divyanshu Gupte

## Completed Tasks

### Level 1 (Basic)
- **Task 1:** Data Cleaning & Preprocessing — Iris dataset (handled duplicates, missing values, outliers, standardized data)
- **Task 2:** Exploratory Data Analysis (EDA) — Summary stats, correlation analysis, histograms, boxplots, pairplot

### Level 2 (Intermediate)
- **Task 1:** Regression Analysis — Linear regression on Boston Housing dataset (R² = 0.669)
- **Task 3:** K-Means Clustering — Iris dataset clustering with elbow method, PCA visualization

### Level 3 (Advanced)
- **Task 1:** Predictive Modeling (Classification) — Customer churn prediction using Logistic Regression, Decision Tree, Random Forest (95.3% accuracy)
- **Task 3:** NLP Sentiment Analysis — TextBlob-based sentiment classification with word clouds

## Colab Notebook

All task code is available as a single Colab notebook:

[`CodeVeda_DataAnalytics_Internship.ipynb`](CodeVeda_DataAnalytics_Internship.ipynb) — run all cells in order on Google Colab.

## Project Structure
```
Level 1/
├── Task 1/  (Data Cleaning)
├── Task 2/  (EDA)
Level 2/
├── Task 1/  (Regression)
├── Task 3/  (K-Means)
Level 3/
├── Task 1/  (Classification)
├── Task 3/  (Sentiment Analysis)
CodeVeda_DataAnalytics_Internship.ipynb  ← Colab notebook
```

## Running Locally

Each `.py` script has a **path toggle** at the top:

```python
# Toggle: comment/uncomment the DATA/OUT lines for Colab vs local run
DATA = r'/content'           # ← active (Colab default)
# DATA = r'C:\...'           # ← uncomment for local Windows run
OUT = r'/content'
# OUT = r'C:\...'
```

1. Uncomment the local lines and set paths to match your dataset location
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn nltk textblob wordcloud
   ```
3. Run any script: `python "Level 1/Task 1/L1_T1_DataCleaning.py"`

## Tools Used
Python, pandas, numpy, matplotlib, seaborn, scikit-learn, nltk, TextBlob, wordcloud

## Links
- [Codveda Technology](https://www.codveda.com)
