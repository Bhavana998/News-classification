import pandas as pd
import re
import nltk
import os
import sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

class DataPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def download_dataset(self):
        """Download BBC News dataset from Hugging Face"""
        try:
            from datasets import load_dataset
            print("Downloading BBC News dataset from Hugging Face...")
            
            # Load BBC News dataset
            dataset = load_dataset("SetFit/bbc-news")
            
            # Convert to pandas DataFrame
            df_train = pd.DataFrame(dataset['train'])
            df_test = pd.DataFrame(dataset['test'])
            
            # Combine train and test
            df = pd.concat([df_train, df_test], ignore_index=True)
            
            # Rename columns to match our expected format
            df = df.rename(columns={'text': 'Text', 'label_text': 'Category'})
            
            print(f"✓ Dataset downloaded successfully!")
            print(f"  • Total samples: {len(df)}")
            print(f"  • Categories: {df['Category'].unique()}")
            print(f"  • Samples per category:\n{df['Category'].value_counts()}")
            
            return df
            
        except ImportError:
            print("datasets library not found. Installing...")
            os.system('pip install datasets')
            return self.download_dataset()
        except Exception as e:
            print(f"⚠ Error downloading from Hugging Face: {e}")
            print("Trying alternative method...")
            return self.download_from_kaggle()
    
    def download_from_kaggle(self):
        """Fallback: Download from Kaggle"""
        try:
            import kagglehub
            print("Downloading BBC News dataset from Kaggle...")
            
            # Download BBC News dataset
            path = kagglehub.dataset_download("gpreda/bbc-news")
            
            # Find the CSV file
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(path, file))
                    break
            
            # Handle column names
            if 'description' in df.columns and 'category' in df.columns:
                df = df.rename(columns={'description': 'Text', 'category': 'Category'})
            elif 'text' in df.columns and 'category' in df.columns:
                df = df.rename(columns={'text': 'Text', 'category': 'Category'})
            
            print(f"✓ Dataset downloaded from Kaggle!")
            print(f"  • Total samples: {len(df)}")
            
            return df
            
        except Exception as e:
            print(f"⚠ Error downloading from Kaggle: {e}")
            print("Creating sample dataset as fallback...")
            return self.create_sample_dataset()
    
    def create_sample_dataset(self):
        """Create a sample dataset if downloads fail"""
        print("Creating sample BBC-style dataset...")
        
        news_data = {
            'Text': [
                "The stock market rallied as tech companies reported strong quarterly earnings",
                "Federal Reserve announces interest rate hike to combat inflation",
                "Oil prices surge amid supply concerns and global demand",
                "Prime minister announces new cabinet reshuffle",
                "Parliament passes landmark climate change legislation",
                "Champions league final ends in dramatic penalty shootout",
                "Olympic athlete breaks world record in 100m sprint",
                "AI breakthrough revolutionizes medical diagnosis",
                "New smartphone features advanced camera system",
                "Blockbuster movie breaks box office records",
                "Streaming service announces new original series",
                "Local team wins championship after 20-year drought",
                "Government unveils new education reform policy",
                "New trade deal boosts exports and manufacturing sector",
                "Cybersecurity experts warn of new malware threat"
            ],
            'Category': [
                'Business', 'Business', 'Business',
                'Politics', 'Politics',
                'Sports', 'Sports',
                'Tech', 'Tech',
                'Entertainment', 'Entertainment',
                'Sports', 'Politics', 'Business', 'Tech'
            ]
        }
        
        df = pd.DataFrame(news_data)
        print(f"✓ Sample dataset created with {len(df)} samples")
        return df
    
    def load_data(self, filepath=None):
        """Load the dataset from file or download"""
        if filepath and os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"✓ Dataset loaded from file with shape: {df.shape}")
        else:
            df = self.download_dataset()
            
            # Save the downloaded dataset
            os.makedirs('data/raw', exist_ok=True)
            df.to_csv('data/raw/bbc_news.csv', index=False, encoding='utf-8')
            print(f"✓ Dataset saved to data/raw/bbc_news.csv")
        
        return df
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization and stopword removal
        words = text.split()
        words = [self.stemmer.stem(word) for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def preprocess(self, df, text_column='Text', label_column='Category'):
        """Main preprocessing pipeline"""
        print("\n=== Starting Data Preprocessing ===")
        
        # Check columns and adapt if needed
        if text_column not in df.columns:
            text_candidates = ['text', 'article', 'content', 'description', 'news']
            for col in text_candidates:
                if col in df.columns:
                    text_column = col
                    break
        
        if label_column not in df.columns:
            label_candidates = ['category', 'label', 'topic', 'class']
            for col in label_candidates:
                if col in df.columns:
                    label_column = col
                    break
        
        print(f"  • Using text column: {text_column}")
        print(f"  • Using label column: {label_column}")
        
        # Handle missing values
        initial_shape = df.shape
        df = df.dropna(subset=[text_column, label_column])
        if initial_shape[0] > df.shape[0]:
            print(f"  • Dropped {initial_shape[0] - df.shape[0]} rows with missing values")
        
        # Clean text
        print("  • Cleaning text data...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Remove empty texts after cleaning
        before_empty = df.shape[0]
        df = df[df['cleaned_text'].str.len() > 0]
        if before_empty > df.shape[0]:
            print(f"  • Removed {before_empty - df.shape[0]} empty texts after cleaning")
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        df['label_encoded'] = self.label_encoder.fit_transform(df[label_column])
        
        # Save label encoder
        import pickle
        os.makedirs('models', exist_ok=True)
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"\n✓ Preprocessing completed!")
        print(f"  • Final dataset shape: {df.shape}")
        print(f"  • Categories: {list(df[label_column].unique())}")
        print(f"  • Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        return df

def main():
    """Run preprocessing independently"""
    print("=" * 60)
    print("NEWS ARTICLE CLASSIFICATION - DATA PREPROCESSING")
    print("=" * 60)
    
    preprocessor = DataPreprocessor()
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load/download dataset
    df = preprocessor.load_data()
    
    # Preprocess the data
    processed_df = preprocessor.preprocess(df)
    
    # Save processed data
    processed_df.to_csv('data/processed/cleaned_news.csv', index=False, encoding='utf-8')
    print(f"\n✓ Cleaned data saved to data/processed/cleaned_news.csv")
    
    return processed_df

if __name__ == "__main__":
    main()
