# tolerant.py
"""
This file handles the "forgiving" parts of our search engine.
If a user makes a typo, we fix it (Edit Distance).
If a user searches with a wildcard like "compu*", we find the matches (K-Gram Index).
"""
from collections import defaultdict

# ==========================================
# 1. FIXING TYPOS (Levenshtein Edit Distance)
# ==========================================
def calculate_edit_distance(word_a, word_b):
    """
    Calculates exactly how many single-letter changes (insertions, deletions, or swaps) 
    it takes to turn word_a into word_b.
    """
    # Create a grid (matrix) to track the cost of changing letters
    # The grid size is (length of word_a + 1) by (length of word_b + 1)
    grid = [[0] * (len(word_b) + 1) for _ in range(len(word_a) + 1)]
    
    # Fill in the base costs for the first row and column (e.g., matching against an empty string)
    for i in range(len(word_a) + 1):
        grid[i][0] = i
    for j in range(len(word_b) + 1):
        grid[0][j] = j
        
    # Go through the grid and calculate the cost at each step
    for i in range(1, len(word_a) + 1):
        for j in range(1, len(word_b) + 1):
            
            # If the current letters match, it costs nothing to transition!
            if word_a[i - 1] == word_b[j - 1]:
                substitution_cost = 0
            else:
                # If they don't match, we have to pay '1' to swap them
                substitution_cost = 1
                
            # The value for this cell is the cheapest of the three possible moves:
            grid[i][j] = min(
                grid[i - 1][j] + 1,                    # Cost to delete a letter
                grid[i][j - 1] + 1,                    # Cost to insert a letter
                grid[i - 1][j - 1] + substitution_cost # Cost to swap (or do nothing if they match)
            )
            
    # The bottom-right corner of the grid holds the final total cost
    return grid[len(word_a)][len(word_b)]


def find_spelling_suggestions(bad_word, dictionary_words, max_typos_allowed=2):
    """
    Looks through our entire dictionary to find real words that are very close 
    (in edit distance) to the misspelled word.
    """
    suggestions = []
    
    for real_word in dictionary_words:
        distance = calculate_edit_distance(bad_word, real_word)
        
        # If the word is close enough, add it to our list of suggestions
        if distance <= max_typos_allowed:
            suggestions.append((real_word, distance))
            
    # Sort the list so the closest matches (lowest distance) are at the very top
    suggestions.sort(key=lambda item: item[1])
    return suggestions


# ==========================================
# 2. WILDCARD SEARCHES (K-Gram Index)
# ==========================================
def create_bigram_index(dictionary_words):
    """
    Breaks every word in our dictionary down into two-letter chunks (bigrams).
    Example: "apple" becomes "$a", "ap", "pp", "pl", "le", "e$"
    We use the '$' to mark the start and end of a word.
    """
    bigram_index = defaultdict(list)
    
    for word in dictionary_words:
        # Add the special dollar signs so we know where the word begins and ends
        word_with_markers = f"${word}$"
        
        # Slide through the word two letters at a time
        for i in range(len(word_with_markers) - 1):
            two_letter_chunk = word_with_markers[i:i+2]
            
            # Link this chunk back to the original word
            if word not in bigram_index[two_letter_chunk]:
                bigram_index[two_letter_chunk].append(word)
                
    return dict(bigram_index)


def process_wildcard_search(search_term, bigram_index, dictionary_words):
    """
    Handles searches that have a '*' in them (like 'comput*' or '*ing').
    We use our bigrams to find potential matches quickly, then filter out the bad ones.
    """
    clean_search = search_term.lower().strip()
    
    # If there's no asterisk at all, just do a normal, exact word check
    if '*' not in clean_search:
        return [clean_search] if clean_search in dictionary_words else []
        
    # Split the search term at the asterisk (e.g., "comp*t" becomes ["comp", "t"])
    pieces = clean_search.split('*')
    
    # We need to figure out which two-letter chunks we absolutely MUST find
    chunks_to_look_for = []
    
    # Extract the normal, middle-of-the-word chunks from whatever text the user provided
    for piece in pieces:
        if piece: # Make sure it's not empty
            for i in range(len(piece) - 1):
                chunks_to_look_for.append(piece[i:i+2])
                
    # Now, add the special boundary markers based on where the asterisk is
    
    # If it's a prefix search like "*ing", the word MUST end with the last letter + '$'
    if clean_search.startswith('*') and pieces[1]:
        chunks_to_look_for.append(pieces[1][-1] + '$')
        
    # If it's a suffix search like "comp*", the word MUST start with '$' + the first letter
    elif clean_search.endswith('*') and pieces[0]:
        chunks_to_look_for.append('$' + pieces[0][0])
        
    # If the search is too short to make any bigrams (like just "*"), fallback to a manual scan
    if not chunks_to_look_for:
        text_without_star = clean_search.replace('*', '')
        return [word for word in dictionary_words if text_without_star in word]

    # Step 1: Find all words that contain ALL of the necessary two-letter chunks
    possible_matches = None
    for chunk in chunks_to_look_for:
        words_with_this_chunk = bigram_index.get(chunk, [])
        
        if possible_matches is None:
            # First pass: grab the list of words for the first chunk
            possible_matches = set(words_with_this_chunk)
        else:
            # Subsequent passes: keep only the words that also have this new chunk
            possible_matches = possible_matches.intersection(words_with_this_chunk)
            
    # If nothing matched our chunks, we're done.
    if not possible_matches:
        return []
        
    # Step 2: The Final Filter
    # Sometimes bigrams give us "false positives" (words that have the chunks, but in the wrong order).
    # We loop through our candidates and make absolutely sure they start and end correctly.
    verified_matches = []
    start_text = pieces[0]
    end_text = pieces[1] if len(pieces) > 1 else ""
    
    for word in possible_matches:
        if word.startswith(start_text) and word.endswith(end_text):
            verified_matches.append(word)
            
    return sorted(verified_matches)