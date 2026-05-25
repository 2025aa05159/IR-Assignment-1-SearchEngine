# IR Assignment 1 - End-to-End Information Retrieval System

**Submitted by:** Suresh Kumar
**ID:** 2026aa05159 
**EMAIL:** 2026aa05159@wilp.bits-pilani.ac.in

## 1. Problem Statement
The goal of this assignment is to design and implement a from-scratch, end-to-end Information Retrieval (IR) pipeline for processing structural document collections. This involves:
1. Building a robust **Text Preprocessing Pipeline** running tokenization, lowercasing, aggressive stop word removal, and variable morphological normalization.
2. Developing core indexing data structures to execute phrase queries using both **Biword Index** and **Positional Index** frameworks.
3. Implementing persistent internal dictionary representations utilizing an un-balanced **Binary Search Tree (BST)** and a multi-way **B-Tree** block layout.
4. Engineering a comprehensive **Tolerant Retrieval Engine** leveraging **Levenshtein Edit Distance** for typo handling and a **K-Gram Index** for wildcard expansion logic.
5. Deploying the fully integrated solution as an interactive, zero-latency web interface via **Streamlit**.

## 2. Dataset Description
* **Dataset Name:** Custom Technical Information Retrieval Corpus (`IR_Corpus.txt`)
* **Source:** Academic Text Compilation (Simulating real-world software engineering documentation logs)
* **Description:**
    * The dataset consists of a text collection split explicitly using discrete document boundary headers (`=== DOC_1 ===` through `=== DOC_10 ===`). The documents contain dense technical passages covering modern computing fields including Machine Learning, Data Science, Autonomous Systems, and Information Retrieval indexing protocols.
    * **Collection Scale:** 10 Distinct Documents
    * **Target Entities:** Raw Source Text Content extracted via specialized regex segmentation parsing layers.

## 3. Preprocessing Mappings & Structural Comparison
The system processes text streams across modular normalization pipelines. The table below represents empirical data extracted from the application logs comparing structural token transformations under independent backend modes:

| Raw Sample Document Token | Stemming Pipeline Output (Porter) | Lemmatization Pipeline Output (WordNet) |
| :--- | :--- | :--- |
| `artificial` | `"artifici"` | `"artificial"` |
| `intelligence` | `"intellig"` | `"intelligence"` |
| `rapidly` | `"rapidli"` | `"rapidly"` |
| `transforming` | `"transform"` | `"transforming"` |
| `computing` | `"comput"` | `"computing"` |

## 4. Observations & Architectural Inferences

### **Why we chose Lemmatization over Stemming for this Dataset**
When handling complex, domain-specific computer science literature, traditional rule-based suffix truncation presents significant drawbacks. Thus, **Lemmatization** was chosen as the superior text normalization framework for retrieval precision.

**Justification:** Porter-style stemming relies on crude truncation rules that clip suffixes blindly, leaving damaged strings (e.g., stripping `artificial` down to `"artifici"`). This creates high **over-stemming risks** where distinct concepts collapse into a single broken token root, injecting false-positives into query results. Lemmatization references a dictionary dataset, ensuring tokens remain grammatically valid words that preserve the exact semantic definitions needed for precise lookups.

### **How the Structural Core Components Performed**
Observations regarding how each implemented component handled processing across the 10-document evaluation corpus:

| IR Component Layer | Analysis & Key System Takeaways |
| :--- | :--- |
| **Biword Index** | **Prone to False Positives.** This structural layout links terms into binary phrase pairs ($w_1, w_2$). If a user inputs a multi-term phrase like `"information retrieval system"`, it independently looks up the bigram entries. If those bigrams exist split apart across entirely separate sentences in a single document, the biword index incorrectly returns it as a match. |
| **Positional Index** | **The Precision Champion.** By explicitly logging precise integer coordinate positions for every term across all documents, this model tests exact positional adjacency constraints mathematically ($pos(w_2) = pos(w_1) + 1$). It eliminates false positives entirely, ensuring absolute structural accuracy. |
| **Binary Search Tree (BST)** | **Struggled on Ordered Ingestion.** Because unique dictionary terms are sorted alphabetically before being populated, standard sequential insertion causes the BST to decay into an un-balanced linear chain. Traversal times degrade from an optimal $O(\log N)$ toward a slow linear scan of $O(N)$. |
| **Multi-Way B-Tree** | **Reliable & High-Speed.** By utilizing multi-way horizontal branching blocks, the B-Tree limits tree height and balances itself dynamically. This guarantees uniform, fast dictionary lookup times regardless of alphabetical input order. |
| **Levenshtein Distance Engine** | **Excellent Typo Recovery.** Computes minimum character edits (insertions, deletions, substitutions) within a strict threshold of $\le 2$, instantly offering correct terms when users mistype queries. |
| **K-Gram Index ($k=2$)** | **Optimized Wildcard Processor.** Dissects wildcard strings (e.g., `comput*`) into explicit bigram slices, intersects matching postings arrays, and identifies candidate terms without performing a full vocabulary table scan. |

## 5. Project Structure
This repository maintains a fully decoupled, modular architecture to prevent dependency conflicts within virtual environment instances:

```text
IR_Assignment-1/
│
├── app.py                  # Streamlit central user interface & custom CSS injection block
├── preprocessing.py        # Text pipeline (Lowercasing, Tokenization, Stop Word Filters)
├── indexing.py             # Phase engine (Biword index and Positional coordinate posting arrays)
├── trees.py                # Dictionary lookups (Binary Search Tree and Multi-way B-Tree)
├── tolerant.py             # Fault tolerance (Levenshtein edit distance & Bigram K-Gram index mappings)
├── requirements.txt        # Virtual lab environment external package declarations
├── README.md               # Code deployment and evaluation documentation
└── IR_Corpus.txt           # Structured collection containing the 10 testing source documents
```
## 6. How to Run Locally
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/2026aa05159/IR-Assignment-1-SearchEngine.git](https://github.com/2026aa05159/IR-Assignment-1-SearchEngine.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
	
## 7. 🔍 End-to-End IR System Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://bits-ir-assignment-1-suresh-kumar.streamlit.app/](https://ir-assignment-1-searchengine-suresh-kumar.streamlit.app/))

**Live Demo:** [Click here to launch the App 🚀]([https://bits-ir-assignment-1-suresh-kumar.streamlit.app/](https://ir-assignment-1-searchengine-suresh-kumar.streamlit.app/))

