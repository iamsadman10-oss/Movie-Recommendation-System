from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np


class ModelSelection:
    def __init__(self, data_path="cleaned_movies.csv"):
        self.data = pd.read_csv(data_path)
        self.models = {}
        self.results = {}
        
    def prepare_data_for_classification(self):
        print("Preparing data for classification models...")
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        X = self.tfidf.fit_transform(self.data['combined_features'])
        y = self.data['genre_names'].apply(lambda x: x.split(',')[0])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Number of unique genres: {y.nunique()}")
        return X, y
    
    def initialize_models(self):
        print("\nInitializing candidate models...")
        
        self.models = {
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=20,
                min_samples_split=5,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                metric='cosine',
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='linear',
                C=1.0,
                random_state=42
            )
        }
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
        
        return self.models
    
    def evaluate_models(self):
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            self.results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            }
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
        return self.results
    
    def display_comparison(self):
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.round(4)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        print("\n" + comparison_df.to_string())
        best_model = comparison_df.index[0]
        print(f"\nBest performing model: {best_model}")
        print(f"F1-Score: {comparison_df.loc[best_model, 'F1-Score']:.4f}")
        return comparison_df
    
    def get_recommendation_engines(self):
        recommendation_engines = {
            'Content-Based Filtering': {
                'description': 'Recommends movies similar to what user liked based on content features',
                'technique': 'TF-IDF + Cosine Similarity',
                'features_used': ['overview', 'genre_names'],
                'pros': [
                    'No need for user history or ratings',
                    'Works for new movies (no cold start)',
                    'Explainable recommendations',
                    'CPU-friendly and fast'
                ],
                'cons': [
                    'Limited serendipity (only similar items)',
                    'Overspecialization possible'
                ]
            },
            'KNN-Based Recommendation': {
                'description': 'Finds similar movies using K-Nearest Neighbors in feature space',
                'technique': 'KNN with cosine similarity',
                'features_used': ['TF-IDF vectors of combined features'],
                'pros': [
                    'Simple and interpretable',
                    'Effective for finding similar items',
                    'Fast nearest neighbor search'
                ],
                'cons': [
                    'Performance degrades with very large datasets',
                    'Requires feature preprocessing'
                ]
            },
            'Genre Classification-Based': {
                'description': 'Classifies movies by genre and recommends within preferred genres',
                'technique': 'Multi-class classification (Random Forest/SVM)',
                'features_used': ['overview text'],
                'pros': [
                    'Can predict genre from plot summary',
                    'Useful for organizing recommendations',
                    'Good accuracy with text features'
                ],
                'cons': [
                    'Movies can have multiple genres',
                    'Requires labeled training data'
                ]
            }
        }
        
        return recommendation_engines
    
    def print_selection_report(self):
        print("\n" + "=" * 70)
        print("MOVIE RECOMMENDATION SYSTEM - MODEL SELECTION REPORT")
        print("25% Project Update - AI Lab Group Project")
        print("=" * 70)
        
        print("\n" + "-" * 70)
        print("1. SELECTED MODELS FOR IMPLEMENTATION")
        print("-" * 70)
        
        selected_models = {
            'Primary (Content-Based)': {
                'model': 'TF-IDF + Cosine Similarity',
                'reason': 'Most suitable for content-based recommendation; fast and effective'
            },
            'Secondary (Classification)': {
                'model': 'Random Forest Classifier',
                'reason': 'Best balance of accuracy and speed; handles text features well'
            },
            'Baseline': {
                'model': 'KNN with Cosine Similarity',
                'reason': 'Simple, interpretable, effective for similarity-based recommendation'
            }
        }
        
        for role, info in selected_models.items():
            print(f"\n{role}:")
            print(f"  Model: {info['model']}")
            print(f"  Reason: {info['reason']}")
        
        print("\n" + "-" * 70)
        print("2. MODEL COMPARISON PLAN")
        print("-" * 70)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        print("\nEvaluation Metrics:")
        for metric in metrics:
            print(f"  - {metric}: {self.get_metric_description(metric)}")
        
        print("\n" + "-" * 70)
        print("3. RECOMMENDATION ENGINES")
        print("-" * 70)
        
        engines = self.get_recommendation_engines()
        for name, details in engines.items():
            print(f"\n{name}:")
            print(f"  Description: {details['description']}")
            print(f"  Technique: {details['technique']}")
            print(f"  Pros: {', '.join(details['pros'][:2])}")
        
        print("\n" + "=" * 70)
        print("Model Selection Phase Completed")
        print("=" * 70)
    
    def get_metric_description(self, metric):
        descriptions = {
            'Accuracy': 'Overall correctness of predictions',
            'Precision': 'Ratio of correct positive predictions to total predicted positives',
            'Recall': 'Ratio of correct positive predictions to all actual positives',
            'F1-Score': 'Harmonic mean of precision and recall (balanced measure)'
        }
        return descriptions.get(metric, 'No description available')


def main():
    selector = ModelSelection()
    selector.prepare_data_for_classification()
    selector.initialize_models()
    selector.evaluate_models()
    selector.display_comparison()
    selector.print_selection_report()


if __name__ == "__main__":
    main()
