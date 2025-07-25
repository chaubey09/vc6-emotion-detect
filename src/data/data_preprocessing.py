import os
import re
import numpy as np
import pandas as pd
import nltk
import string
import logging
from typing import Any
from pandas import DataFrame

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Download required NLTK resources
try:
    nltk.download('wordnet')
    nltk.download('stopwords')
except Exception as e:
    logging.error(f"Error downloading NLTK resources: {e}")

def lemmatization(text: str) -> str:
    """Lemmatize each word in the text."""
    lemmatizer = WordNetLemmatizer()
    try:
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)
    except Exception as e:
        logging.error(f"Lemmatization failed: {e}")
        return text

def remove_stop_words(text: str) -> str:
    """Remove stop words from the text."""
    try:
        stop_words = set(stopwords.words("english"))
        words = [word for word in str(text).split() if word not in stop_words]
        return " ".join(words)
    except Exception as e:
        logging.error(f"Removing stop words failed: {e}")
        return text

def removing_numbers(text: str) -> str:
    """Remove all digits from the text."""
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logging.error(f"Removing numbers failed: {e}")
        return text

def lower_case(text: str) -> str:
    """Convert all words in the text to lowercase."""
    try:
        words = text.split()
        words = [word.lower() for word in words]
        return " ".join(words)
    except Exception as e:
        logging.error(f"Lowercasing failed: {e}")
        return text

def removing_punctuations(text: str) -> str:
    """Remove punctuations and extra whitespace from the text."""
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logging.error(f"Removing punctuations failed: {e}")
        return text

def removing_urls(text: str) -> str:
    """Remove URLs from the text."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logging.error(f"Removing URLs failed: {e}")
        return text

def remove_small_sentences(df: DataFrame) -> None:
    """Set text to NaN if sentence has fewer than 3 words."""
    try:
        for i in range(len(df)):
            if len(str(df.text.iloc[i]).split()) < 3:
                df.text.iloc[i] = np.nan
    except Exception as e:
        logging.error(f"Removing small sentences failed: {e}")

def normalize_text(df: DataFrame) -> DataFrame:
    """Apply all preprocessing steps to the 'content' column of the DataFrame."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        logging.error(f"Normalizing text failed: {e}")
        return df

def normalized_sentence(sentence: str) -> str:
    """Apply all preprocessing steps to a single sentence."""
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error(f"Normalizing sentence failed: {e}")
        return sentence

def load_data(file_path: str) -> DataFrame:
    """Load CSV data from a file path with error handling."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {file_path}: {e}")
        raise

def save_data(df: DataFrame, file_path: str) -> None:
    """Save DataFrame to CSV with error handling."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info(f"Saved data to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save data to {file_path}: {e}")
        raise

def main() -> None:
    """Main function to preprocess train and test data."""
    try:
        # Load raw train and test data
        train_data = load_data("data/raw/train.csv")
        test_data = load_data("data/raw/test.csv")

        # Normalize train and test data
        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)

        # Save processed data to CSV files
        save_data(train_data, "data/processed/train.csv")
        save_data(test_data, "data/processed/test.csv")
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")

if __name__ == "__main__":
    main()
