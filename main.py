import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def print_section(title):
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)


def data_collection_section():
    print_section("PART 1: DATA COLLECTION")

    print("\nDataset Source:")
    print("-" * 50)
    print("File: dataset.csv")
    print("Records: 10,000+ movies")
    print("Format: CSV with metadata and descriptions")

    print("\nDataset Suitability:")
    print("-" * 50)
    print("- Large enough for ML training (10K+ samples)")
    print("- Contains textual data (overviews) for content-based filtering")
    print("- Has genre labels for classification tasks")
    print("- Free and open-source (suitable for academic project)")
    print("- Diverse movie collection with multiple genres")

    print("\nSelected Columns for Recommendation:")
    print("-" * 50)
    print("- title: Movie title for identification")
    print("- overview: Plot summary for content analysis")
    print("- genre_names: Genre labels for classification")

    print("\nHow it Supports Recommendation:")
    print("-" * 50)
    print("- Overview text enables TF-IDF vectorization")
    print("- Genres allow content-based and classification approaches")
    print("- Titles provide user-friendly output")


def data_preprocessing_section():
    print_section("PART 2: DATA PREPROCESSING")

    print("\n1. Loading Dataset...")
    data = pd.read_csv("dataset.csv")
    print(f"   Initial shape: {data.shape}")
    print(f"   Columns: {list(data.columns)}")

    print("\n2. Checking Missing Values...")
    missing = data.isnull().sum()
    print(f"   Missing values:\n{missing[missing > 0]}")

    print("\n3. Handling Missing Values...")
    initial_count = len(data)
    data = data.dropna(subset=['title', 'overview', 'genre_names'])
    print(f"   Removed {initial_count - len(data)} rows with missing values")

    print("\n4. Removing Duplicates...")
    duplicates = data.duplicated().sum()
    data = data.drop_duplicates()
    print(f"   Removed {duplicates} duplicate rows")

    print("\n5. Selecting Important Columns...")
    data = data[['title', 'overview', 'genre_names']].copy()
    print(f"   Selected: {list(data.columns)}")

    print("\n6. Cleaning Text Data...")
    print("   - Converting to lowercase")
    print("   - Removing special characters and digits")
    print("   - Removing stopwords")
    print("   - Removing extra whitespace")

    data['overview_cleaned'] = data['overview'].apply(preprocess_text)

    data = data[data['overview_cleaned'].str.strip() != '']
    print(f"   Final shape after cleaning: {data.shape}")

    data['combined_features'] = (
        data['overview_cleaned'] + ' ' +
        data['genre_names'].str.lower().str.replace(',', ' ')
    )

    data = data.reset_index(drop=True)
    data.to_csv("cleaned_movies.csv", index=False)

    print("\n7. Cleaned Dataset Summary:")
    print(f"   - Saved to: cleaned_movies.csv")
    print(f"   - Final records: {len(data)}")
    print(f"   - Average overview length: {data['overview_cleaned'].str.len().mean():.0f} chars")
    print(f"   - Unique genres: {data['genre_names'].nunique()}")

    print("\n8. Sample Cleaned Data:")
    print(data[['title', 'genre_names']].head(3).to_string(index=False))

    return data


def model_selection_section():
    print_section("PART 3: MODEL SELECTION")

    print("\nCandidate Models for Implementation:")
    print("-" * 50)

    models = {
        "1. TF-IDF + Cosine Similarity (Content-Based)": {
            "type": "Similarity-based",
            "best_for": "Finding similar movies based on content",
            "pros": ["No training needed", "Fast and efficient", "Explainable"],
            "status": "PRIMARY RECOMMENDATION"
        },
        "2. K-Nearest Neighbors (KNN)": {
            "type": "Instance-based",
            "best_for": "Finding similar items in feature space",
            "pros": ["Simple to implement", "Effective for recommendations"],
            "status": "BASELINE MODEL"
        },
        "3. Random Forest Classifier": {
            "type": "Ensemble",
            "best_for": "Genre classification from plot text",
            "pros": ["Handles text features well", "Good accuracy", "Fast inference"],
            "status": "FINAL SELECTION"
        },
        "4. Naive Bayes Classifier": {
            "type": "Probabilistic",
            "best_for": "Text classification",
            "pros": ["Fast training", "Works well with TF-IDF"],
            "status": "ALTERNATIVE"
        },
        "5. Support Vector Machine (SVM)": {
            "type": "Discriminative",
            "best_for": "High-dimensional text classification",
            "pros": ["Effective for text", "Good generalization"],
            "status": "ALTERNATIVE"
        },
        "6. Decision Tree": {
            "type": "Tree-based",
            "best_for": "Interpretable classification",
            "pros": ["Easy to interpret", "No preprocessing needed"],
            "status": "BASELINE MODEL"
        }
    }

    for name, details in models.items():
        print(f"\n{name}")
        print(f"   Type: {details['type']}")
        print(f"   Best For: {details['best_for']}")
        print(f"   Status: {details['status']}")

    print("\n" + "-" * 50)
    print("FINAL MODEL SELECTION FOR NEXT PHASE:")
    print("-" * 50)
    print("1. PRIMARY:   TF-IDF + Cosine Similarity (Content-Based)")
    print("2. SECONDARY: Random Forest (for genre classification)")
    print("3. BASELINE:  KNN (for comparison)")

    print("\nEvaluation Metrics:")
    print("-" * 50)
    print("- Accuracy:  Overall prediction correctness")
    print("- Precision: Quality of recommendations")
    print("- Recall:    Coverage of relevant items")
    print("- F1-Score:  Balance between precision and recall")


def main():
    print("\n" + "=" * 70)
    print("MOVIE RECOMMENDATION SYSTEM".center(70))
    print("25% Project Update - AI Lab Group Project".center(70))
    print("=" * 70)

    data_collection_section()
    cleaned_data = data_preprocessing_section()
    model_selection_section()

    print("\n" + "=" * 70)
    print("25% UPDATE SUMMARY".center(70))
    print("=" * 70)
    print("\nCOMPLETED:")
    print("   [✓] Data Collection - Dataset loaded and analyzed")
    print("   [✓] Data Preprocessing - Clean data saved to cleaned_movies.csv")
    print("   [✓] Model Selection - Final models chosen for implementation")

    print("\nNEXT PHASE (50% Update):")
    print("   [ ] Model Training - Train selected models")
    print("   [ ] Evaluation - Compare model performance")
    print("   [ ] Recommendation Engine - Build recommendation pipeline")

    print("\n" + "=" * 70)
    print("25% PROJECT UPDATE COMPLETED SUCCESSFULLY".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()



