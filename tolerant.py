# tolerant.py
"""
BITS Pilani - Information Retrieval Assignment 1
Module: Tolerant Retrieval Engine - Edit Distance & K-Gram Indexing (Task E)
Implements algorithms from scratch to correct typos and parse wildcard structures.
"""
from collections import defaultdict

# ==========================================
# 1. EDIT DISTANCE (LEVENSHTEIN DISTANCE)
# ==========================================
def compute_levenshtein_distance(str1, str2):
    """
    Computes the minimum edit distance between two strings from scratch.
    Supports character insertion, deletion, and substitution.
    """
    matrix = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
    
    for i in range(len(str1) + 1):
        matrix[i][0] = i
    for j in range(len(str2) + 1):
        matrix[0][j] = j
        
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,       # Deletion
                matrix[i][j - 1] + 1,       # Insertion
                matrix[i - 1][j - 1] + cost  # Substitution
            )
            
    return matrix[len(str1)][len(str2)]

def get_spelling_corrections(misspelled_token, vocabulary_list, max_distance=2):
    """
    Scans the vocabulary list to find closest matching terms within the edit distance limit.
    """
    matches = []
    for term in vocabulary_list:
        dist = compute_levenshtein_distance(misspelled_token, term)
        if dist <= max_distance:
            matches.append((term, dist))
    # Sort by closest match first
    matches.sort(key=lambda x: x[1])
    return matches


# ==========================================
# 2. K-GRAM INDEX ENGINE (Using Bigrams k=2)
# ==========================================
def build_kgram_index(vocabulary_list):
    """
    Constructs a k-gram index mapping where keys are bigrams 
    and values are lists of vocabulary terms containing that bigram.
    """
    kgram_index = defaultdict(list)
    
    for term in vocabulary_list:
        # Pad the term with boundary markers
        padded_term = f"${term}$"
        for i in range(len(padded_term) - 1):
            bi_gram = padded_term[i:i+2]
            if term not in kgram_index[bi_gram]:
                kgram_index[bi_gram].append(term)
                
    return dict(kgram_index)

def resolve_wildcard_query(wildcard_pattern, kgram_index, vocabulary_list):
    """
    Resolves a wildcard string (e.g., 'comput*' or '*trieval') using the k-gram index.
    Filters out false matches via post-filtering constraints.
    """
    wildcard_pattern = wildcard_pattern.lower().strip()
    if '*' not in wildcard_pattern:
        return [wildcard_pattern] if wildcard_pattern in vocabulary_list else []
        
    # Split pattern by the wildcard operator
    parts = wildcard_pattern.split('*')
    
    # Generate required bigrams based on wildcard positions
    required_bigrams = []
    
    # Case 1: Prefix query (e.g., 'comput*') -> starts with '$'
    if wildcard_pattern.startswith('*'):
        padded_pattern = parts[1] + '$'
    # Case 2: Suffix query (e.g., '*trieval') -> ends with '$'
    elif wildcard_pattern.endswith('*'):
        padded_pattern = '$' + parts[0]
    # Case 3: Standard split query (e.g., 'comp*t')
    else:
        padded_pattern = '$' + parts[0] + parts[1] + '$'
        
    # Extract structural bigrams from explicit text blocks
    for part in parts:
        if part:
            for i in range(len(part) - 1):
                required_bigrams.append(part[i:i+2])
                
    # Add boundary bigrams explicitly
    if wildcard_pattern.startswith('*') and parts[1]:
        required_bigrams.append(parts[1][-1] + '$')
    elif wildcard_pattern.endswith('*') and parts[0]:
        required_bigrams.append('$' + parts[0][0])
        
    if not required_bigrams:
        # Fallback to absolute manual lookup if query is too brief
        return [term for term in vocabulary_list if wildcard_pattern.replace('*', '') in term]

    # Intersect postings array maps from our K-Gram index
    candidate_terms = None
    for bg in required_bigrams:
        postings = kgram_index.get(bg, [])
        if candidate_terms is None:
            candidate_terms = set(postings)
        else:
            candidate_terms = candidate_terms.intersection(postings)
            
    if not candidate_terms:
        return []
        
    # Post-filtering phase: Eliminate false tracking entries using exact regex check logic
    filtered_terms = []
    prefix = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""
    
    for term in candidate_terms:
        if term.startswith(prefix) and term.endswith(suffix):
            filtered_terms.append(term)
            
    return sorted(filtered_terms)