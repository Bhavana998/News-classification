import sys
import os
import subprocess
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def install_requirements():
    """Install required packages"""
    print("\n=== Installing Requirements ===")
    requirements = [
        'pandas',
        'numpy',
        'scikit-learn',
        'nltk',
        'datasets',
        'matplotlib',
        'seaborn',
        'kagglehub',
        'scipy'
    ]
    
    for package in requirements:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
                print(f"✓ {package} installed")
            except:
                print(f"⚠ Could not install {package}, continuing...")
    
    print("✓ All requirements satisfied")

def check_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data/raw', 'data/processed', 'models', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ Directory structure verified")

def run_pipeline():
    """Execute the complete ML pipeline"""
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("               NEWS ARTICLE CLASSIFICATION PIPELINE")
    print("=" * 70)
    print("\n📰 Dataset: BBC News (Hugging Face)")
    print("🤖 Model: Logistic Regression with TF-IDF")
    print("📊 Evaluation: Accuracy, Confusion Matrix, Classification Report")
    print("=" * 70)
    
    try:
        # Check directories
        check_directories()
        
        # Install requirements
        install_requirements()
        
        # Step 1: Data Preprocessing
        print("\n" + "🔷 [Step 1/4] Data Preprocessing")
        print("-" * 40)
        from src.data_preprocessing import main as preprocess_main
        preprocess_main()
        
        # Step 2: Feature Engineering
        print("\n" + "🔷 [Step 2/4] Feature Engineering")
        print("-" * 40)
        from src.feature_engineering import main as feature_main
        feature_main()
        
        # Step 3: Model Training
        print("\n" + "🔷 [Step 3/4] Model Training")
        print("-" * 40)
        from src.train import main as train_main
        train_main()
        
        # Step 4: Model Evaluation
        print("\n" + "🔷 [Step 4/4] Model Evaluation")
        print("-" * 40)
        from src.evaluate import main as evaluate_main
        accuracy = evaluate_main()
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("                  ✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n📊 Final Results:")
        print(f"   • Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   • Total Time: {elapsed_time:.2f} seconds")
        print(f"\n📁 Output Files:")
        print(f"   • Cleaned data: data/processed/cleaned_news.csv")
        print(f"   • TF-IDF vectorizer: models/tfidf_vectorizer.pkl")
        print(f"   • Trained model: models/classifier.pkl")
        print(f"   • Evaluation metrics: results/metrics.txt")
        print(f"   • Confusion matrix plot: results/confusion_matrix.png")
        print("=" * 70)
        
        return accuracy
        
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        return None
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Make sure you have internet connection to download the dataset")
        print("  2. Try running each script separately:")
        print("     python src/data_preprocessing.py")
        print("     python src/feature_engineering.py")  
        print("     python src/train.py")
        print("     python src/evaluate.py")
        return None

if __name__ == "__main__":
    run_pipeline()
