# indexing.py
"""
This file handles how we store phrases and search for them later.
We use two main approaches here: pairing words up (Biword Index) 
or remembering exactly where each word lives (Positional Index).
"""

from collections import defaultdict

def build_biword_index(all_documents, clean_text_function, remove_stopwords, word_format, split_hyphens):
    """
    Builds a search index that combines words into pairs (two consecutive words).
    For example: 'information retrieval' becomes 'information_retrieval'.
    This makes searching for exact two-word phrases super fast.
    """
    paired_words_index = defaultdict(list)
    
    for doc_name, raw_text in all_documents.items():
        # Clean up the raw text using the pipeline from our preprocessing file
        cleanup_steps = clean_text_function(raw_text, remove_stopwords, word_format, split_hyphens)
        final_words = cleanup_steps['Normalization']
        
        # Look at words side-by-side to create our pairs
        for i in range(len(final_words) - 1):
            word_pair = f"{final_words[i]}_{final_words[i+1]}"
            
            # If we haven't already linked this document to this word pair, add it
            if doc_name not in paired_words_index[word_pair]:
                paired_words_index[word_pair].append(doc_name)
                
    return dict(paired_words_index)


def build_positional_index(all_documents, clean_text_function, remove_stopwords, word_format, split_hyphens):
    """
    Builds a highly detailed index that remembers the exact position of every single word.
    Format: word -> { document_name -> [position_1, position_4, ...] }
    This lets us search for phrases of any length by checking if words appear right next to each other.
    """
    # Create a dictionary inside a dictionary to hold lists of positions
    exact_positions_index = defaultdict(lambda: defaultdict(list))
    
    for doc_name, raw_text in all_documents.items():
        cleanup_steps = clean_text_function(raw_text, remove_stopwords, word_format, split_hyphens)
        final_words = cleanup_steps['Normalization']
        
        # Loop through the list of words, keeping track of the current index (position)
        for position, word in enumerate(final_words):
            exact_positions_index[word][doc_name].append(position)
            
    # Streamlit sometimes complains about displaying 'defaultdicts', 
    # so we convert it back into standard Python dictionaries here.
    clean_index = {}
    for word, doc_positions in exact_positions_index.items():
        clean_index[word] = dict(doc_positions)
        
    return clean_index


def search_biword(search_query, paired_words_index, clean_text_function, remove_stopwords, word_format, split_hyphens):
    """
    Searches our database using word pairs.
    Note: If someone searches for a 3+ word phrase, we break it down into multiple pairs 
    and find documents that contain ALL of those pairs.
    """
    # Clean the user's search query exactly like we cleaned the documents
    cleanup_steps = clean_text_function(search_query, remove_stopwords, word_format, split_hyphens)
    search_words = cleanup_steps['Normalization']
    
    if not search_words:
        return []
    
    # EDGE CASE: The user only typed one word.
    if len(search_words) == 1:
        one_word = search_words[0]
        matching_docs = set()
        
        # Since our index stores PAIRS, we have to find any pair that starts or ends with this one word
        for pair_key, doc_list in paired_words_index.items():
            if pair_key.startswith(f"{one_word}_") or pair_key.endswith(f"_{one_word}"):
                matching_docs.update(doc_list)
        return list(matching_docs)
        
    # Standard case: Turn the search query into pairs (e.g., A B C -> A_B, B_C)
    search_pairs = [f"{search_words[i]}_{search_words[i+1]}" for i in range(len(search_words) - 1)]
    
    # We want to find documents that have EVERY pair in the query
    final_matching_docs = None
    for pair in search_pairs:
        docs_for_this_pair = paired_words_index.get(pair, [])
        
        # If this is the first pair we are checking, grab its documents
        if final_matching_docs is None:
            final_matching_docs = set(docs_for_this_pair)
        # For every pair after that, keep ONLY the documents that also have this new pair (intersection)
        else:
            final_matching_docs = final_matching_docs.intersection(docs_for_this_pair)
            
    return list(final_matching_docs) if final_matching_docs else []


def search_positional(search_query, exact_positions_index, clean_text_function, remove_stopwords, word_format, split_hyphens):
    """
    Searches for an exact phrase by checking the positional coordinates of every word.
    This is very accurate because we ensure word 2 is exactly one spot after word 1, etc.
    """
    cleanup_steps = clean_text_function(search_query, remove_stopwords, word_format, split_hyphens)
    search_words = cleanup_steps['Normalization']
    
    if not search_words:
        return []
        
    # EDGE CASE: Only one word typed. Just return every document that has it.
    if len(search_words) == 1:
        single_word = search_words[0]
        # Return just the document names (keys), not the positions
        return list(exact_positions_index.get(single_word, {}).keys())

    # Start the hunt by looking for the very first word in the phrase
    first_word = search_words[0]
    if first_word not in exact_positions_index:
        return [] # If the first word isn't anywhere, the whole phrase definitely isn't.
        
    docs_with_first_word = exact_positions_index[first_word]
    final_matched_docs = []

    # Let's check each document that actually contains our starting word
    for doc_name, starting_positions in docs_with_first_word.items():
        
        # A document might have the first word multiple times. We need to check every instance.
        for start_position in starting_positions:
            is_exact_phrase = True
            
            # Now, check the rest of the words in the user's search query
            for word_distance in range(1, len(search_words)):
                next_word = search_words[word_distance]
                expected_position = start_position + word_distance
                
                # Ask three questions:
                # 1. Is this next word in our whole database?
                # 2. Is it in THIS specific document?
                # 3. Is it sitting at the exact position we expect it to be?
                if (next_word not in exact_positions_index or 
                    doc_name not in exact_positions_index[next_word] or 
                    expected_position not in exact_positions_index[next_word][doc_name]):
                    
                    is_exact_phrase = False # The chain is broken.
                    break
            
            # If we made it through the whole loop and the chain didn't break, we found a match!
            if is_exact_phrase:
                if doc_name not in final_matched_docs:
                    final_matched_docs.append(doc_name)
                break # We found it in this doc, no need to check the other starting positions here.
                
    return final_matched_docs