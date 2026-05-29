# trees.py
"""
This file handles how we store our dictionary of words so we can search it quickly.
We are building and comparing two classic ways to organize data: 
a simple Binary Search Tree (BST) and a bulkier, more balanced B-Tree.
"""
import time

# ==========================================
# 1. BINARY SEARCH TREE (BST)
# A simple tree where smaller words go to the left, and bigger words go to the right.
# ==========================================
class BSTNode:
    def __init__(self, word, document_list):
        self.word = word
        self.document_list = document_list  # The list of documents containing this word
        self.left_child = None              # Pointer to words alphabetically smaller
        self.right_child = None             # Pointer to words alphabetically larger

class BinarySearchTree:
    def __init__(self):
        self.root = None # The very top of our tree

    def add_word(self, word, document_list):
        # If the tree is completely empty, this word becomes the root
        if not self.root:
            self.root = BSTNode(word, document_list)
        else:
            # Otherwise, we travel down the tree to find the right spot
            self._add_word_recursive(self.root, word, document_list)

    def _add_word_recursive(self, current_node, word, document_list):
        # If the word comes before the current node's word alphabetically, go left
        if word < current_node.word:
            if current_node.left_child is None:
                current_node.left_child = BSTNode(word, document_list)
            else:
                self._add_word_recursive(current_node.left_child, word, document_list)
                
        # If the word comes after, go right
        elif word > current_node.word:
            if current_node.right_child is None:
                current_node.right_child = BSTNode(word, document_list)
            else:
                self._add_word_recursive(current_node.right_child, word, document_list)
                
        # If the word is exactly the same, it already exists! Just update the documents.
        else:
            current_node.document_list = document_list

    def find_word(self, word):
        # Start a stopwatch to see exactly how long the search takes
        start_time = time.perf_counter()
        found_node = self._find_word_recursive(self.root, word)
        end_time = time.perf_counter()
        
        search_duration = end_time - start_time
        
        # If we found the word, return its documents and the time it took. Otherwise, return empty.
        if found_node:
            return found_node.document_list, search_duration
        return [], search_duration

    def _find_word_recursive(self, current_node, word):
        # Base case: we hit a dead end (None) or we actually found the word we want!
        if current_node is None or current_node.word == word:
            return current_node
            
        # Keep searching left or right depending on alphabetical order
        if word < current_node.word:
            return self._find_word_recursive(current_node.left_child, word)
        return self._find_word_recursive(current_node.right_child, word)


# ==========================================
# 2. B-TREE
# A chunkier tree where each node holds multiple words. 
# It stays perfectly balanced, making it great for huge amounts of data.
# ==========================================
class BTreeNode:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.words = []           # A list of words stored in this specific node
        self.document_lists = []  # The matching documents for each of those words
        self.children = []        # Pointers to nodes further down the tree

class BTree:
    def __init__(self, min_degree=3):
        self.root = BTreeNode(is_leaf=True)
        # 'min_degree' controls how fat the nodes can get. 
        # A node can hold a maximum of (2 * min_degree) - 1 words.
        self.min_degree = min_degree 

    def add_word(self, word, document_list):
        old_root = self.root
        max_words_allowed = (2 * self.min_degree) - 1
        
        # If the root is completely full, we have to split it before we can add anything new
        if len(old_root.words) == max_words_allowed:
            new_root = BTreeNode()
            self.root = new_root
            new_root.children.insert(0, old_root)
            
            # Split the old full root in half
            self._split_full_child(parent_node=new_root, child_index=0, full_child=old_root)
            
            # Now that there is space, insert the new word normally
            self._add_to_node_with_space(new_root, word, document_list)
        else:
            # If the root isn't full, just insert it normally
            self._add_to_node_with_space(old_root, word, document_list)

    def _add_to_node_with_space(self, node, word, document_list):
        current_index = len(node.words) - 1
        
        if node.is_leaf:
            # If we are at the bottom of the tree, just find the right alphabetical spot and push it in
            node.words.append(None)
            node.document_lists.append(None)
            
            # Shift larger words to the right to make room
            while current_index >= 0 and word < node.words[current_index]:
                node.words[current_index + 1] = node.words[current_index]
                node.document_lists[current_index + 1] = node.document_lists[current_index]
                current_index -= 1
                
            # Drop the new word into the space we just created
            node.words[current_index + 1] = word
            node.document_lists[current_index + 1] = document_list
        else:
            # We are not at the bottom yet. Figure out which child node we need to travel down into.
            while current_index >= 0 and word < node.words[current_index]:
                current_index -= 1
            current_index += 1
            
            # Before going down, check if the child we are about to visit is full
            target_child = node.children[current_index]
            max_words_allowed = (2 * self.min_degree) - 1
            
            if len(target_child.words) == max_words_allowed:
                # If it's full, split it!
                self._split_full_child(node, current_index, target_child)
                # After splitting, the middle word moved up. Decide if we need to go to the left or right of it.
                if word > node.words[current_index]:
                    current_index += 1
                    
            # Safely travel down to the (now guaranteed to have space) child
            self._add_to_node_with_space(node.children[current_index], word, document_list)

    def _split_full_child(self, parent_node, child_index, full_child):
        """
        Takes a child node that has reached maximum capacity and splits it perfectly in half.
        The middle word is pushed up into the parent node to act as a divider.
        """
        degree = self.min_degree
        new_sibling_node = BTreeNode(is_leaf=full_child.is_leaf)
        
        # Make room in the parent for the middle word that's about to be pushed up
        parent_node.children.insert(child_index + 1, new_sibling_node)
        parent_node.words.insert(child_index, full_child.words[degree - 1])
        parent_node.document_lists.insert(child_index, full_child.document_lists[degree - 1])
        
        # Give the right half of the words to the brand new sibling node
        new_sibling_node.words = full_child.words[degree : (2 * degree) - 1]
        new_sibling_node.document_lists = full_child.document_lists[degree : (2 * degree) - 1]
        
        # Keep only the left half of the words in the original child node
        full_child.words = full_child.words[0 : degree - 1]
        full_child.document_lists = full_child.document_lists[0 : degree - 1]

        # If these weren't leaf nodes, we also have to split up their children pointers
        if not full_child.is_leaf:
            new_sibling_node.children = full_child.children[degree : 2 * degree]
            full_child.children = full_child.children[0 : degree]

    def find_word(self, word):
        start_time = time.perf_counter()
        found_documents = self._find_word_recursive(self.root, word)
        end_time = time.perf_counter()
        
        search_duration = end_time - start_time
        
        if found_documents:
            return found_documents, search_duration
        return [], search_duration

    def _find_word_recursive(self, node, word):
        current_index = 0
        
        # Scan through the words in this specific node to find where our target word fits
        while current_index < len(node.words) and word > node.words[current_index]:
            current_index += 1
            
        # Did we find an exact match right here?
        if current_index < len(node.words) and word == node.words[current_index]:
            return node.document_lists[current_index]
            
        # If we didn't find it and we're at the very bottom of the tree, it doesn't exist
        if node.is_leaf:
            return None
            
        # Otherwise, follow the appropriate child pointer further down the tree
        return self._find_word_recursive(node.children[current_index], word)


# ==========================================
# 3. SPEED TEST & SETUP FUNCTIONS
# ==========================================
def build_both_trees(word_dictionary):
    """
    Takes our fully compiled dictionary of unique words and loads them 
    into both tree structures at the same time.
    """
    bst = BinarySearchTree()
    b_tree = BTree(min_degree=3)
    
    # We sort the words first so we can evaluate performance fairly across both structures
    for word, document_list in sorted(word_dictionary.items()):
        bst.add_word(word, document_list)
        b_tree.add_word(word, document_list)
        
    return bst, b_tree

def run_speed_test(search_query, bst, b_tree, clean_text_function, drop_stopwords, format_type, split_hyphens):
    """
    Cleans up a user's test query, searches for every word in both trees, 
    and records exactly how fast each tree was able to find the results.
    """
    # Clean the input text using our pipeline so it matches the format of the words stored in our trees
    cleanup_steps = clean_text_function(search_query, drop_stopwords, format_type, split_hyphens)
    search_words = cleanup_steps['Normalization']
    
    test_results = []
    
    # Search both trees for each individual word and log the times
    for word in search_words:
        bst_docs, bst_time = bst.find_word(word)
        btree_docs, btree_time = b_tree.find_word(word)
        
        test_results.append({
            "Word Searched": word,
            "BST Search Time (sec)": f"{bst_time:.8f}",
            "B-Tree Search Time (sec)": f"{btree_time:.8f}",
            "BST Matches Found": len(bst_docs),
            "B-Tree Matches Found": len(btree_docs)
        })
        
    return test_results