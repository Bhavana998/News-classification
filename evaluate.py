import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.sparse import load_npz
import os
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self):
        self.model = None
        self.label_encoder = None
    
    def load_model_and_features(self):
        """Load trained model and test data"""
        print("\n=== Loading Model and Test Data ===")
        
        # Load model
        try:
            with open('models/classifier.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print("✓ Model loaded successfully")
        except FileNotFoundError:
            print("✗ Model not found. Please run train.py first.")
            raise
        
        # Load label encoder
        try:
            with open('models/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("✓ Label encoder loaded successfully")
        except FileNotFoundError:
            print("✗ Label encoder not found. Please run data_preprocessing.py first.")
            raise
        
        # Load test data
        try:
            X_test = load_npz('data/processed/X_test.npz')
            y_test = pd.read_csv('data/processed/y_test.csv')
            print(f"✓ Test data loaded: {X_test.shape[0]} samples")
        except FileNotFoundError:
            print("✗ Test data not found. Please run feature_engineering.py first.")
            raise
        
        return X_test, y_test
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model and save metrics"""
        print("\n=== Evaluating Model Performance ===")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Get class names
        class_names = self.label_encoder.classes_
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, 
                                           target_names=class_names)
        
        # Save results (using ASCII only for Windows compatibility)
        os.makedirs('results', exist_ok=True)
        
        with open('results/metrics.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("NEWS ARTICLE CLASSIFICATION - MODEL EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset: BBC News (Hugging Face)\n")
            f.write(f"Model: Logistic Regression (multinomial)\n")
            f.write(f"Features: TF-IDF (5000 features, unigrams + bigrams)\n")
            f.write(f"Test Set Size: {len(y_test)} samples\n\n")
            f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            f.write("-" * 60 + "\n")
            f.write("Confusion Matrix:\n")
            f.write("-" * 60 + "\n")
            
            # Format confusion matrix with labels (ASCII only)
            f.write(" " * 15)
            for name in class_names:
                f.write(f"{name[:8]:>8} ")
            f.write("\n")
            
            for i, row in enumerate(conf_matrix):
                f.write(f"{class_names[i][:15]:15}")
                for val in row:
                    f.write(f"{val:8} ")
                f.write("\n")
            
            f.write("\n" + "-" * 60 + "\n")
            f.write("Classification Report:\n")
            f.write("-" * 60 + "\n")
            f.write(class_report)
            
            # Add per-class accuracy
            f.write("\n" + "-" * 60 + "\n")
            f.write("Per-Class Accuracy:\n")
            f.write("-" * 60 + "\n")
            for i, class_name in enumerate(class_names):
                class_mask = (y_test == i).values.ravel()
                class_total = np.sum(class_mask)
                class_correct = np.sum((y_pred == i) & (y_test == i).values.ravel())
                class_acc = class_correct / class_total if class_total > 0 else 0
                f.write(f"{class_name:15}: {class_acc:.4f} ({class_correct}/{class_total})\n")
        
        # Plot confusion matrix
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix - News Classification', fontsize=16)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Confusion matrix plot saved to results/confusion_matrix.png")
        except Exception as e:
            print(f"⚠ Warning: Could not save confusion matrix plot: {e}")
        
        # Print results to console
        print(f"\n✓ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)
        print("\n✓ Results saved to results/metrics.txt")
        
        return accuracy, conf_matrix, class_report

def main():
    """Run evaluation independently"""
    print("=" * 60)
    print("NEWS ARTICLE CLASSIFICATION - MODEL EVALUATION")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    X_test, y_test = evaluator.load_model_and_features()
    accuracy, conf_matrix, class_report = evaluator.evaluate(X_test, y_test)
    
    return accuracy

if __name__ == "__main__":
    main()
