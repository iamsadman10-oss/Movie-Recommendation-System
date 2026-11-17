import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def load_dataset(file_path="dataset.csv"):
    print("=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)
    data = pd.read_csv(file_path)
    print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Columns: {list(data.columns)}")
    return data


def explore_dataset(data):
    print("\n" + "=" * 60)
    print("STEP 2: Dataset Exploration")
    print("=" * 60)
    print("\nDataset Info:")
    print(data.info())
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nMissing values:")
    missing = data.isnull().sum()
    print(missing)
    print("\nDuplicated rows:", data.duplicated().sum())
    return missing


def clean_dataset(data):
    print("\n" + "=" * 60)
    print("STEP 3: Data Cleaning")
    print("=" * 60)
    initial_rows = len(data)
    data = data.dropna(subset=['title', 'overview', 'genre_names'])
    print(f"Removed {initial_rows - len(data)} rows with missing critical values")
    duplicates = data.duplicated().sum()
    data = data.drop_duplicates()
    print(f"Removed {duplicates} duplicate rows")
    data = data[data['overview'].str.strip() != '']
    data = data[data['title'].str.strip() != '']
    print(f"Dataset after cleaning: {len(data)} rows")
    return data.reset_index(drop=True)


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)


def select_and_transform_features(data):
    print("\n" + "=" * 60)
    print("STEP 4: Feature Selection and Transformation")
    print("=" * 60)
    important_columns = ['title', 'overview', 'genre_names']
    data = data[important_columns].copy()
    print(f"Selected columns: {important_columns}")
    print("Preprocessing overview text (lowercase, remove special chars, stopwords)...")
    data['overview_cleaned'] = data['overview'].apply(preprocess_text)
    data['genre_names'] = data['genre_names'].str.strip()
    data['combined_features'] = data['overview_cleaned'] + ' ' + data['genre_names'].str.lower().str.replace(', ', ' ')
    print(f"Final dataset shape: {data.shape}")
    print("\nSample of cleaned data:")
    print(data[['title', 'overview_cleaned', 'genre_names']].head())
    return data


def save_cleaned_dataset(data, output_file="cleaned_movies.csv"):
    print("\n" + "=" * 60)
    print("STEP 5: Saving Cleaned Dataset")
    print("=" * 60)
    data.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}")
    print(f"Final dataset contains {len(data)} movies")
    print("\nGenre distribution (top 10):")
    genre_counts = data['genre_names'].value_counts().head(10)
    print(genre_counts)


def generate_eda_report(data, output_file="eda_report.png"):
    print("\n" + "=" * 60)
    print("STEP 6: Generating EDA Report")
    print("=" * 60)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    genre_counts = data['genre_names'].value_counts().head(10)
    axes[0, 0].barh(range(len(genre_counts)), genre_counts.values)
    axes[0, 0].set_yticks(range(len(genre_counts)))
    axes[0, 0].set_yticklabels([g[:30] for g in genre_counts.index], fontsize=8)
    axes[0, 0].set_xlabel('Count')
    axes[0, 0].set_title('Top 10 Genre Combinations')
    axes[0, 0].invert_yaxis()
    overview_lengths = data['overview_cleaned'].str.len()
    axes[0, 1].hist(overview_lengths, bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('Overview Length (characters)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Overview Lengths')
    word_counts = data['overview_cleaned'].str.split().str.len()
    axes[1, 0].hist(word_counts, bins=50, edgecolor='black', color='orange')
    axes[1, 0].set_xlabel('Word Count')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Word Counts in Overviews')
    genre_counts_per_movie = data['genre_names'].str.count(',') + 1
    single_genre = (genre_counts_per_movie == 1).sum()
    multi_genre = (genre_counts_per_movie > 1).sum()
    axes[1, 1].pie([single_genre, multi_genre],
                   labels=['Single Genre', 'Multi-Genre'],
                   autopct='%1.1f%%',
                   colors=['lightblue', 'lightgreen'])
    axes[1, 1].set_title('Single vs Multi-Genre Movies')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"EDA report saved to: {output_file}")
    plt.close()


def main():
    print("\n" + "=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM - DATA PREPROCESSING")
    print("25% Project Update - AI Lab Group Project")
    print("=" * 60)
    data = load_dataset()
    explore_dataset(data)
    data = clean_dataset(data)
    data = select_and_transform_features(data)
    save_cleaned_dataset(data)
    generate_eda_report(data)
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return data


if __name__ == "__main__":
    main()
