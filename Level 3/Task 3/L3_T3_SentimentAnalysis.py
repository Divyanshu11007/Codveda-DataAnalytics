import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter

# Download NLTK data if needed
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

DATA = r'C:\Users\ASUS\Downloads\DataSet_Extracted\Data Set For Task'
OUT = r'C:\Users\ASUS\Downloads\CodeVeda_Internship_Projects\Level 3\Task 3'

# ---- Load sentiment dataset ----
df = pd.read_csv(f'{DATA}/3) Sentiment dataset.csv')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print()
print('First 5 rows:')
print(df.head())
print()

# Check sentiment distribution
print('=== Sentiment Distribution ===')
print(df['Sentiment'].value_counts())
print()

# ---- Text Preprocessing ----
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)                 # remove mentions/hashtags
    text = re.sub(r'[^a-z\s]', '', text)                  # remove numbers/punctuation
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

print('Cleaning text...')
df['clean_text'] = df['Text'].apply(clean_text)
print('Sample clean text:')
for i in range(3):
    print(f'  Original: {repr(df["Text"].iloc[i][:80])}')
    print(f'  Cleaned:  {df["clean_text"].iloc[i][:80]}')
print()

# ---- Sentiment Analysis with TextBlob ----
from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1 to 1
    if polarity > 0.1:
        return 'positive', polarity
    elif polarity < -0.1:
        return 'negative', polarity
    else:
        return 'neutral', polarity

print('Running TextBlob sentiment analysis...')
sentiment_results = df['Text'].apply(get_sentiment)
df['predicted_sentiment'] = [r[0] for r in sentiment_results]
df['polarity'] = [r[1] for r in sentiment_results]

print()
print('=== Predicted Sentiment Distribution ===')
print(df['predicted_sentiment'].value_counts())
print()

# ---- Accuracy vs original labels ----
df['Sentiment'] = df['Sentiment'].str.lower()
accuracy = (df['predicted_sentiment'] == df['Sentiment']).mean()
print(f'Accuracy vs original labels: {accuracy:.2%}')
print()

# Confusion
print('=== Confusion Matrix (Original vs Predicted) ===')
print(pd.crosstab(df['Sentiment'], df['predicted_sentiment'], margins=True))
print()

# ---- Visualize sentiment distribution ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original
df['Sentiment'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red', 'gray'])
axes[0].set_title('Original Sentiment Labels')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)

# Predicted
df['predicted_sentiment'].value_counts().plot(kind='bar', ax=axes[1],
                                              color=['green', 'gray', 'red'])
axes[1].set_title('TextBlob Predicted Sentiment')
axes[1].set_xlabel('Sentiment')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(f'{OUT}/sentiment_distribution.png', dpi=150)
plt.show()
print('Saved: sentiment_distribution.png')
print()

# ---- Polarity distribution ----
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df['polarity'], bins=30, edgecolor='black', color='steelblue')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Polarity Score')
plt.ylabel('Frequency')
plt.title('Distribution of Polarity Scores')

plt.subplot(1, 2, 2)
colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
for sent, group in df.groupby('predicted_sentiment'):
    plt.hist(group['polarity'], bins=15, alpha=0.6, label=sent, color=colors[sent])
plt.xlabel('Polarity Score')
plt.ylabel('Frequency')
plt.title('Polarity by Sentiment Class')
plt.legend()

plt.tight_layout()
plt.savefig(f'{OUT}/sentiment_polarity.png', dpi=150)
plt.show()
print('Saved: sentiment_polarity.png')
print()

# ---- Word Cloud ----
try:
    from wordcloud import WordCloud
    print('Generating word clouds...')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sentiments = ['positive', 'negative', 'neutral']

    for ax, sent in zip(axes, sentiments):
        text = ' '.join(df[df['predicted_sentiment'] == sent]['clean_text'])
        if text.strip():
            wordcloud = WordCloud(width=400, height=300,
                                  background_color='white',
                                  colormap='viridis',
                                  max_words=50).generate(text)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'{sent.capitalize()} Words', fontsize=14)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUT}/sentiment_wordclouds.png', dpi=150)
    plt.show()
    print('Saved: sentiment_wordclouds.png')
except ImportError:
    print('wordcloud not installed. Skipping word cloud visualization.')
    print('Install: pip install wordcloud')
print()

# ---- Top words per sentiment ----
print('=== Top 10 Words per Sentiment ===')
for sent in ['positive', 'negative', 'neutral']:
    words = ' '.join(df[df['predicted_sentiment'] == sent]['clean_text']).split()
    top_words = Counter(words).most_common(10)
    print(f'\n{sent.capitalize()}:')
    for word, count in top_words:
        print(f'  {word}: {count}')