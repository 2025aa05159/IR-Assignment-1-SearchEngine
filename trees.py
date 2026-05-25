# trees.py
"""
Information Retrieval Assignment 1
Module: Dictionary Search Trees - BST vs B-Tree (Task D)
Implements native search tree indexes from scratch to benchmark word dictionary lookups.
"""
import time

# ==========================================
# BINARY SEARCH TREE (BST) IMPLEMENTATION
# ==========================================
class BSTNode:
    def __init__(self, key, postings):
        self.key = key
        self.postings = postings  # Inverted index postings/metadata
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key, postings):
        if not self.root:
            self.root = BSTNode(key, postings)
        else:
            self._insert_recursive(self.root, key, postings)

    def _insert_recursive(self, node, key, postings):
        if key < node.key:
            if node.left is None:
                node.left = BSTNode(key, postings)
            else:
                self._insert_recursive(node.left, key, postings)
        elif key > node.key:
            if node.right is None:
                node.right = BSTNode(key, postings)
            else:
                self._insert_recursive(node.right, key, postings)
        else:
            # Key exists, update its posting parameters
            node.postings = postings

    def search(self, key):
        start_time = time.perf_counter()
        node = self._search_recursive(self.root, key)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        if node:
            return node.postings, elapsed_time
        return [], elapsed_time

    def _search_recursive(self, node, key):
        if node is None or node.key == key:
            return node
        if key < node.key:
            return self._search_recursive(node.left, key)
        return self._search_recursive(node.right, key)


# ==========================================
# B-TREE IMPLEMENTATION (Order t=3)
# ==========================================
class BTreeNode:
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []        # Array of terms (keys)
        self.postings = []    # Array of corresponding inverted indexes
        self.child = []       # Child pointers array

class BTree:
    def __init__(self, t=3):
        self.root = BTreeNode(True)
        self.t = t  # Minimum degree (defines node capacities)

    def insert(self, key, postings):
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            temp = BTreeNode()
            self.root = temp
            temp.child.insert(0, root)
            self._split_child(temp, 0, root)
            self._insert_non_full(temp, key, postings)
        else:
            self._insert_non_full(root, key, postings)

    def _insert_non_full(self, node, key, postings):
        i = len(node.keys) - 1
        if node.leaf:
            # Find the correct slot for the new element key
            node.keys.append(None)
            node.postings.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.postings[i + 1] = node.postings[i]
                i -= 1
            # Insert the item
            node.keys[i + 1] = key
            node.postings[i + 1] = postings
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.child[i].keys) == (2 * self.t) - 1:
                self._split_child(node, i, node.child[i])
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.child[i], key, postings)

    def _split_child(self, parent, i, child):
        t = self.t
        new_node = BTreeNode(child.leaf)
        parent.child.insert(i + 1, new_node)
        parent.keys.insert(i, child.keys[t - 1])
        parent.postings.insert(i, child.postings[t - 1])
        
        # Split keys and postings maps
        new_node.keys = child.keys[t:(2 * t) - 1]
        new_node.postings = child.postings[t:(2 * t) - 1]
        child.keys = child.keys[0:t - 1]
        child.postings = child.postings[0:t - 1]

        if not child.leaf:
            new_node.child = child.child[t:2 * t]
            child.child = child.child[0:t]

    def search(self, key):
        start_time = time.perf_counter()
        result = self._search_recursive(self.root, key)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        if result:
            return result, elapsed_time
        return [], elapsed_time

    def _search_recursive(self, node, key):
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        if i < len(node.keys) and key == node.keys[i]:
            return node.postings[i]
        if node.leaf:
            return None
        return self._search_recursive(node.child[i], key)


# ==========================================
# TREE LOADER AND PERFORMANCE BENCHMARKING ENGINE
# ==========================================
def populate_trees(inverted_index):
    """
    Ingests the standard inverted index vocabulary dictionary 
    and constructs both tree representations simultaneously.
    """
    bst = BinarySearchTree()
    btree = BTree(t=3)
    
    # Sort keys to evaluate performance traits fairly
    for term, postings in sorted(inverted_index.items()):
        bst.insert(term, postings)
        btree.insert(term, postings)
        
    return bst, btree

def execute_tree_benchmarks(query_string, bst, btree, preprocess_pipeline_func, apply_stop, norm, hyphen):
    """
    Tokenizes a text query string, runs structural searches across 
    both indexes, and aggregates real performance timing lookups.
    """
    # Extract clean dictionary terms from the query using the text preprocessor
    stages = preprocess_pipeline_func(query_string, apply_stop, norm, hyphen)
    query_tokens = stages['Normalization']
    
    results = []
    
    for token in query_tokens:
        bst_postings, bst_time = bst.search(token)
        btree_postings, btree_time = btree.search(token)
        
        results.append({
            "Token Key": token,
            "BST Search Time (s)": f"{bst_time:.8f}",
            "B-Tree Search Time (s)": f"{btree_time:.8f}",
            "BST Matches Count": len(bst_postings),
            "B-Tree Matches Count": len(btree_postings)
        })
        
    return results