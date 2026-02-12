import pandas as pd
import pickle
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import load_npz

class ModelTrainer:
    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            C=1.0,
            solver='lbfgs',
            multi_class='multinomial'
        )
        print("✓ Logistic Regression model initialized")
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("\n=== Training Logistic Regression Model ===")
        print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        self.model.fit(X_train, y_train)
        
        print("✓ Model training completed!")
        
        # Training accuracy
        y_pred = self.model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        print(f"Training Accuracy: {accuracy:.4f}")
        
        return self.model
    
    def save_model(self, filepath='models/classifier.pkl'):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='models/classifier.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Model loaded from {filepath}")
        return self.model

def main():
    """Run training independently"""
    print("=" * 60)
    print("NEWS ARTICLE CLASSIFICATION - MODEL TRAINING")
    print("=" * 60)
    
    # Load training data
    print("\nLoading training data...")
    X_train = load_npz('data/processed/X_train.npz')
    y_train = pd.read_csv('data/processed/y_train.csv')
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # Train model
    trainer = ModelTrainer()
    model = trainer.train(X_train, y_train.values.ravel())
    
    # Save model
    trainer.save_model()
    
    print("\n✓ Model training pipeline completed successfully!")
    
    return model

if __name__ == "__main__":
    main()
