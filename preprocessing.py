# preprocessing.py
import re
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# --- 1. Safely Download NLTK Resources ---
# We use try-except to ensure the app doesn't crash if it's run multiple times
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet')
    nltk.download('stopwords')

# Load the official English stop words
STOP_WORDS = set(stopwords.words('english'))

# Instantiate stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def run_preprocessing_pipeline(text, apply_stop_words=True, norm_method="Stemming", handle_hyphens=True):
    """
    Executes the preprocessing pipeline on a given text string.
    Returns a dictionary containing the output at each stage so we can display it in Streamlit.
    """
    stages = {}
    
    # Step 1: Lowercasing
    text_lower = text.lower()
    stages['Lowercasing'] = text_lower
    
    # Step 2: Hyphen Handling
    # If a word is hyphenated (e.g., "state-of-the-art"), we can either split it or keep it.
    if handle_hyphens:
        # Replaces hyphens with a space to treat them as separate words
        text_hyphen = text_lower.replace('-', ' ')
    else:
        # Keeps the hyphenated word as a single token
        text_hyphen = text_lower
    stages['Hyphen Handling'] = text_hyphen
    
    # Step 3: Tokenization
    # We use regex \b\w+\b to extract words, removing punctuation
    tokens = re.findall(r'\b\w+\b', text_hyphen)
    stages['Tokenization'] = tokens
    
    # Step 4: Stop word removal
    if apply_stop_words:
        filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    else:
        filtered_tokens = tokens
    stages['Stop Word Removal'] = filtered_tokens
    
    # Step 5: Stemming or Lemmatization
    if norm_method == "Stemming":
        normalized_tokens = [stemmer.stem(word) for word in filtered_tokens]
    else:
        normalized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        
    stages['Normalization'] = normalized_tokens
    
    return stages


def build_inverted_index(documents_dict, apply_stop_words, norm_method, handle_hyphens):
    """
    Takes a dictionary of {doc_id: raw_text} and builds an inverted index.
    Returns the inverted index dictionary.
    """
    inverted_index = defaultdict(list)
    
    for doc_id, text in documents_dict.items():
        # Get the final normalized tokens for this document
        stages = run_preprocessing_pipeline(text, apply_stop_words, norm_method, handle_hyphens)
        final_tokens = stages['Normalization']
        
        # Add doc_id to the inverted index for each unique term
        for term in set(final_tokens): # Use set to avoid adding the same doc_id multiple times per term
            inverted_index[term].append(doc_id)
            
    return dict(inverted_index)


def compare_stem_lem(documents_dict, apply_stop_words, handle_hyphens):
    """
    Evaluates Stemming vs Lemmatization across the entire uploaded dataset.
    Returns vocabulary sizes to prove which technique compresses data more.
    """
    stem_vocab = set()
    lem_vocab = set()
    total_words = 0
    
    for text in documents_dict.values():
        # Process with Stemming
        stem_stages = run_preprocessing_pipeline(text, apply_stop_words, "Stemming", handle_hyphens)
        # Process with Lemmatization
        lem_stages = run_preprocessing_pipeline(text, apply_stop_words, "Lemmatization", handle_hyphens)
        
        stem_vocab.update(stem_stages['Normalization'])
        lem_vocab.update(lem_stages['Normalization'])
        total_words += len(stem_stages['Tokenization'])
        
    return {
        "Total Raw Tokens": total_words,
        "Stemming Vocabulary Size": len(stem_vocab),
        "Lemmatization Vocabulary Size": len(lem_vocab)
    }