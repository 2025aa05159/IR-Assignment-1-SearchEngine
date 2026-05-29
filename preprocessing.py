# preprocessing.py
import re
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Make sure we have the required NLTK files downloaded so the code doesn't break
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet')
    nltk.download('stopwords')

# Grab our list of common English filler words
english_stopwords = set(stopwords.words('english'))

# Set up our tools for chopping words down to their roots
word_stemmer = PorterStemmer()
word_lemmatizer = WordNetLemmatizer()


def process_text_pipeline(raw_text, drop_stopwords=True, cleanup_type="Stemming", split_hyphens=True):
    """
    Cleans up a raw string of text step-by-step.
    Returns a dictionary of every stage so we can show the user the progress in Streamlit.
    """
    pipeline_steps = {}
    
    # Step 1: Make everything lowercase so capitalization doesn't mess up matches
    lowercase_text = raw_text.lower()
    pipeline_steps['Lowercasing'] = lowercase_text
    
    # Step 2: Handle words with dashes (like "built-in")
    if split_hyphens:
        # Swap the dash for a space to turn it into two separate words
        processed_text = lowercase_text.replace('-', ' ')
    else:
        # Leave it alone and treat it as a single word
        processed_text = lowercase_text
    pipeline_steps['Hyphen Handling'] = processed_text
    
    # Step 3: Break the text into individual words
    # The regex '\b\w+\b' looks for word boundaries and completely ignores punctuation marks
    word_list = re.findall(r'\b\w+\b', processed_text)
    pipeline_steps['Tokenization'] = word_list
    
    # Step 4: Throw out common filler words if requested
    if drop_stopwords:
        meaningful_words = [word for word in word_list if word not in english_stopwords]
    else:
        meaningful_words = word_list
    pipeline_steps['Stop Word Removal'] = meaningful_words
    
    # Step 5: Reduce words to their base form (e.g., "running" becomes "run")
    if cleanup_type == "Stemming":
        # Stemming is fast but aggressive (cuts off endings brutally)
        cleaned_words = [word_stemmer.stem(word) for word in meaningful_words]
    else:
        # Lemmatization is smarter and looks up actual dictionary words
        cleaned_words = [word_lemmatizer.lemmatize(word) for word in meaningful_words]
        
    pipeline_steps['Normalization'] = cleaned_words
    
    return pipeline_steps


def create_word_index(documents, drop_stopwords, cleanup_type, split_hyphens):
    """
    Maps out which words appear in which documents.
    Takes a dictionary of {doc_name: raw_text} and returns a search index.
    """
    word_index = defaultdict(list)
    
    for doc_name, raw_text in documents.items():
        # Get the final list of clean words for the current document
        current_steps = process_text_pipeline(raw_text, drop_stopwords, cleanup_type, split_hyphens)
        final_words = current_steps['Normalization']
        
        # We turn the list into a 'set' here so we don't accidentally list 
        # the same document multiple times if a word appears in it twice.
        for word in set(final_words): 
            word_index[word].append(doc_name)
            
    return dict(word_index)


def compare_cleanup_methods(uploaded_documents):
    """
    Compares Stemming vs Lemmatization across the entire dataset 
    by looking at unique vocabulary size and word compression rates.
    """
    # Temporary lists to track all tokens generated
    all_stemmed_tokens = []
    all_lemmatized_tokens = []
    
    # Process every document using both methods
    for doc_id, text in uploaded_documents.items():
        # Run stemming pipeline 
        # FIXED: Changed argument names to match process_text_pipeline signature
        stem_res = process_text_pipeline(text, drop_stopwords=True, cleanup_type="Stemming", split_hyphens=True)
        all_stemmed_tokens.extend(stem_res['Normalization'])
        
        # Run lemmatization pipeline
        # FIXED: Changed argument names to match process_text_pipeline signature
        lem_res = process_text_pipeline(text, drop_stopwords=True, cleanup_type="Lemmatization", split_hyphens=True)
        all_lemmatized_tokens.extend(lem_res['Normalization'])
        
    # Calculate unique vocabulary sizes
    unique_stems = set(all_stemmed_tokens)
    unique_lems = set(all_lemmatized_tokens)
    
    # Pack everything into a clean dictionary for the UI to read
    # FIXED: Added logic to fallback gracefully if documents are empty to avoid a division-by-zero error
    total_processed = len(all_stemmed_tokens)
    if total_processed == 0:
        return {
            "Total Words Processed": 0,
            "Unique Stems (Vocabulary)": 0,
            "Unique Lemmas (Vocabulary)": 0,
            "Stemming Compression Rate": "0.00%",
            "Lemmatization Compression Rate": "0.00%"
        }

    comparison_results = {
        "Total Words Processed": total_processed,
        "Unique Stems (Vocabulary)": len(unique_stems),
        "Unique Lemmas (Vocabulary)": len(unique_lems),
        "Stemming Compression Rate": f"{((total_processed - len(unique_stems)) / total_processed) * 100:.2f}%",
        "Lemmatization Compression Rate": f"{((total_processed - len(unique_lems)) / total_processed) * 100:.2f}%"
    }
    
    return comparison_results