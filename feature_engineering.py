import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

class FeatureEngineer:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        print(f"✓ TF-IDF Vectorizer initialized with {max_features} max features")
    
    def create_features(self, df, text_column='cleaned_text', fit=True):
        """Convert text to TF-IDF features"""
        print("\n=== Creating TF-IDF Features ===")
        
        if fit:
            print("Fitting TF-IDF vectorizer on training data...")
            X = self.vectorizer.fit_transform(df[text_column])
            print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        else:
            print("Transforming text data using fitted vectorizer...")
            X = self.vectorizer.transform(df[text_column])
        
        print(f"Feature matrix shape: {X.shape}")
        return X
    
    def save_vectorizer(self, filepath='models/tfidf_vectorizer.pkl'):
        """Save the fitted vectorizer"""
        os.makedirs('models', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"✓ Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath='models/tfidf_vectorizer.pkl'):
        """Load a fitted vectorizer"""
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"✓ Vectorizer loaded from {filepath}")
        return self.vectorizer

def main():
    """Run feature engineering independently"""
    print("=" * 60)
    print("NEWS ARTICLE CLASSIFICATION - FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load preprocessed data
    print("\nLoading cleaned data...")
    df = pd.read_csv('data/processed/cleaned_news.csv')
    print(f"Loaded {len(df)} samples")
    
    # Create features
    engineer = FeatureEngineer(max_features=5000)
    X = engineer.create_features(df, fit=True)
    y = df['label_encoded']
    
    # Split data (stratified to maintain class distribution)
    print("\n=== Splitting Data ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Save vectorizer
    engineer.save_vectorizer()
    
    # Save features and splits
    os.makedirs('data/processed', exist_ok=True)
    
    # Save sparse matrices efficiently
    from scipy.sparse import save_npz
    save_npz('data/processed/X_train.npz', X_train)
    save_npz('data/processed/X_test.npz', X_test)
    
    # Save labels
    pd.Series(y_train).to_csv('data/processed/y_train.csv', index=False)
    pd.Series(y_test).to_csv('data/processed/y_test.csv', index=False)
    
    print("\n✓ Feature engineering completed!")
    print("✓ All features and splits saved successfully")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()
