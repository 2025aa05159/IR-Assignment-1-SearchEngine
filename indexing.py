# indexing.py
"""
Information Retrieval Assignment 1
Module: Phrase Indexing and Core Structural Architecture (Task C)
Contains clean, human-readable logic for Biword and Positional Index generation.
"""

from collections import defaultdict

def build_biword_index(documents_dict, preprocess_func, apply_stop_words, norm_method, handle_hyphens):
    """
    Constructs a Biword Index framework.
    Combines consecutive pairs of overlapping normalized tokens into an index key string.
    Example: 'information retrieval' -> 'information_retrieval'
    """
    biword_index = defaultdict(list)
    
    for doc_id, text in documents_dict.items():
        # Invoke the imported preprocessor pipeline function passed down from app.py
        stages = preprocess_func(text, apply_stop_words, norm_method, handle_hyphens)
        tokens = stages['Normalization']
        
        # Parse adjacent word structures sequentially
        for i in range(len(tokens) - 1):
            biword_key = f"{tokens[i]}_{tokens[i+1]}"
            # Prevent populating duplicate Document IDs for the same biword entry
            if doc_id not in biword_index[biword_key]:
                biword_index[biword_key].append(doc_id)
                
    return dict(biword_index)


def build_positional_index(documents_dict, preprocess_func, apply_stop_words, norm_method, handle_hyphens):
    """
    Constructs a detailed Positional Index mapping structure.
    Mapping Scheme: term -> { doc_id -> [integer_position_1, integer_position_2, ...] }
    Allows complete verification of sequential structural limits.
    """
    positional_index = defaultdict(lambda: defaultdict(list))
    
    for doc_id, text in documents_dict.items():
        stages = preprocess_func(text, apply_stop_words, norm_method, handle_hyphens)
        tokens = stages['Normalization']
        
        # Enumerate each token position coordinate index
        for current_position, term_token in enumerate(tokens):
            positional_index[term_token][doc_id].append(current_position)
            
    # Cast nested defaultdicts back to base dict for compatibility with Streamlit json viewer
    clean_serializable_index = {}
    for term, doc_positions_map in positional_index.items():
        clean_serializable_index[term] = dict(doc_positions_map)
        
    return clean_serializable_index


def search_biword(query_phrase, biword_index, preprocess_func, apply_stop_words, norm_method, handle_hyphens):
    """
    Performs search using Biword Intersections.
    Can trigger false positives if phrase length matches > 2 elements.
    """
    stages = preprocess_func(query_phrase, apply_stop_words, norm_method, handle_hyphens)
    query_tokens = stages['Normalization']
    
    if not query_tokens:
        return []
    
    # Fallback to structural postings verification if only 1 query term is supplied
    if len(query_tokens) == 1:
        single_term = query_tokens[0]
        # Iterate over biwords to look for entries starting/ending with that word
        matched_docs = set()
        for bw_key, postings in biword_index.items():
            if bw_key.startswith(f"{single_term}_") or bw_key.endswith(f"_{single_term}"):
                matched_docs.update(postings)
        return list(matched_docs)
        
    # Generate biwords from query terms
    query_biword_list = [f"{query_tokens[i]}_{query_tokens[i+1]}" for i in range(len(query_tokens) - 1)]
    
    # Intersect the postings list across all query bigrams
    accumulated_results = None
    for biword in query_biword_list:
        postings_list = biword_index.get(biword, [])
        if accumulated_results is None:
            accumulated_results = set(postings_list)
        else:
            accumulated_results = accumulated_results.intersection(postings_list)
            
    return list(accumulated_results) if accumulated_results else []


def search_positional(query_phrase, positional_index, preprocess_func, apply_stop_words, norm_method, handle_hyphens):
    """
    Performs precise sequential phrase parsing using Positional index coordinates.
    Guarantees zero false positives by checking consecutive positioning adjustments (pos, pos+1, ...).
    """
    stages = preprocess_func(query_phrase, apply_stop_words, norm_method, handle_hyphens)
    query_tokens = stages['Normalization']
    
    if not query_tokens:
        return []
        
    # Standard fallback if phrase query consists of a single term element
    if len(query_tokens) == 1:
        target_term = query_tokens[0]
        return list(positional_index.get(target_term, {}).keys())

    first_term = query_tokens[0]
    if first_term not in positional_index:
        return []
        
    candidate_documents = positional_index[first_term]
    valid_matching_docs = []

    # Iterate through each document that features the initial term anchor
    for doc_id, starting_positions_list in candidate_documents.items():
        # Check every matching position coordinate of the first term inside this document
        for initial_pos in starting_positions_list:
            is_valid_phrase_sequence = True
            
            # Verify if all subsequent terms appear consecutively
            for structural_offset in range(1, len(query_tokens)):
                target_next_term = query_tokens[structural_offset]
                expected_next_position = initial_pos + structural_offset
                
                # Check structural parameters: Does next term exist, does it cover this doc, is it at the exact pos?
                if (target_next_term not in positional_index or 
                    doc_id not in positional_index[target_next_term] or 
                    expected_next_position not in positional_index[target_next_term][doc_id]):
                    is_valid_phrase_sequence = False
                    break
            
            # If the complete sequential sequence checks out, register the matched document
            if is_valid_phrase_sequence:
                if doc_id not in valid_matching_docs:
                    valid_matching_docs.append(doc_id)
                break # Conclude processing for this document and skip to next
                
    return valid_matching_docs