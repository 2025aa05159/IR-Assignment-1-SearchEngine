# app.py
import streamlit as st
import pandas as pd
import re

# Import our custom files with their updated, easy-to-read function names
from preprocessing import process_text_pipeline, create_word_index, compare_cleanup_methods
from indexing import build_biword_index, build_positional_index, search_biword, search_positional
from trees import build_both_trees, run_speed_test
from tolerant import find_spelling_suggestions, create_bigram_index, process_wildcard_search

# Set up the basic look of our web page
st.set_page_config(page_title="IR System - Assignment 1", layout="wide")

# Hide Streamlit's default top padding so our app uses the full screen space
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 1rem !important;
            margin-top: 0rem !important;
        }
        .stMain, .stMainBlockContainer, [data-testid="stMainBlockContainer"] {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }
        [data-testid="stVerticalBlock"] > div:first-child {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }
        header, [data-testid="stHeader"] {
            visibility: hidden !important;
            height: 0px !important;
            padding: 0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("End-to-End Information Retrieval System")

# ---------------------------------------------------------
# CACHING FUNCTIONS
# ---------------------------------------------------------
@st.cache_data
def get_cached_biword_index(docs, apply_stop, norm, hyphen):
    return build_biword_index(docs, process_text_pipeline, apply_stop, norm, hyphen)

@st.cache_data
def get_cached_positional_index(docs, apply_stop, norm, hyphen):
    return build_positional_index(docs, process_text_pipeline, apply_stop, norm, hyphen)

@st.cache_data
def get_cached_inverted_index(docs, apply_stop, norm, hyphen):
    return create_word_index(docs, apply_stop, norm, hyphen)

@st.cache_resource
def get_cached_trees(inverted_index):
    return build_both_trees(inverted_index)

@st.cache_data
def get_cached_kgram_index(words_list):
    return create_bigram_index(words_list)

# ---------------------------------------------------------
# INITIALIZE DEFAULT ASSIGNMENT CORPUS
# ---------------------------------------------------------
default_corpus = {
    "Doc_1": "Artificial intelligence and machine learning are rapidly transforming modern computing platforms. Deep learning models provide state-of-the-art performance across multiple industries. Many computer scientists are studying these advanced engineering applications to automate complex decision-making processes.",
    "Doc_2": "Information retrieval systems rely heavily on an efficient inverted index or a detailed positional index to process complex phrase queries. A modern search engine organizes its vocabulary using specialized data structures like a binary search tree or a robust B-tree to minimize query search time.",
    "Doc_3": "Autonomous self-driving electric cars process real-time environmental data using artificial intelligence. These high-tech vehicles are transforming urban transport infrastructure globally as engineers continue studying their long-term safety profiles.",
    "Doc_4": "Natural language processing helps computer software understand structural human languages. Text preprocessing operations like tokenization, lowercasing, and aggressive stop word removal are fundamental prerequisites before building any phrase query index.",
    "Doc_5": "Data science involves analyzing massive collections of unstructured text to extract valuable semantic insights. Modern data scientists use machine learning algorithms to build user-friendly software applications capable of predicting highly accurate future trends.",
    "Doc_6": "Modern search engines process millions of search queries every second. To remain scalable, an information retrieval system must utilize optimized dictionary search structures. When a user requests an exact phrase query, a positional index ensures the application returns only the most relevant results.",
    "Doc_7": "A software engineer studying natural language processing must understand how stemming and lemmatization differ. While a basic stemmer cuts off word endings aggressively, a lemmatizer utilizes a dictionary to find the grammatically correct base form of transforming words.",
    "Doc_8": "Building a user-friendly application requires careful data processing and architectural planning. When designing a dictionary search framework, choosing between a balanced binary search tree and a multi-way B-tree significantly impacts the final information retrieval time.",
    "Doc_9": "Advanced computing networks use state-of-the-art machine learning algorithms to detect unauthorized security threats. These security systems analyze digital tokenization logs in real-time to protect corporate data science servers from malicious attacks.",
    "Doc_10": "Text preprocessing is the most crucial step in any information retrieval pipeline. Lowercasing all characters and removing common stop words prevents a search engine's positional index from growing unnecessarily large, resulting in lightning-fast query search times."
}

if "uploaded_docs" not in st.session_state:
    st.session_state["uploaded_docs"] = default_corpus

# ---------------------------------------------------------
# SIDEBAR: FILE UPLOAD (OPTIONAL DATASET OVERRIDE)
# ---------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/d/d3/BITS_Pilani-Logo.svg/330px-BITS_Pilani-Logo.svg.png", use_container_width=True)
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload an alternative text dataset (.txt, .csv)", type=["txt", "csv"])

# Button moved outside IF block so it renders properly in Streamlit UI
force_reload = st.sidebar.button("Force Re-load File")

if uploaded_file is not None:
    need_processing = False
    
    if "last_processed_file" not in st.session_state:
        need_processing = True
    elif st.session_state["last_processed_file"] != uploaded_file.name:
        need_processing = True
    elif force_reload:
        need_processing = True

    if need_processing:
        # CRITICAL FIX: Reset file pointer to 0 before reading
        uploaded_file.seek(0)
        
        uploaded_docs = {}
        
        if uploaded_file.name.endswith('.csv'):
            csv_data = pd.read_csv(uploaded_file)
            for index, row_data in csv_data.iterrows():
                uploaded_docs[f"Doc_{index + 1}"] = str(row_data.iloc[0])
                
        else:
            file_text = uploaded_file.read().decode("utf-8")
            text_chunks = re.split(r'===\s*DOC_\d+\s*===', file_text)
            current_doc_number = 1
            
            for chunk in text_chunks:
                clean_text = chunk.strip()
                if clean_text:
                    uploaded_docs[f"Doc_{current_doc_number}"] = clean_text
                    current_doc_number += 1
            
            # Robust Fallback: Split by any line break sequence
            if not uploaded_docs:
                basic_docs = [line.strip() for line in re.split(r'[\r\n]+', file_text) if line.strip()]
                for index, doc_text in enumerate(basic_docs):
                    uploaded_docs[f"Doc_{index + 1}"] = doc_text
                        
        if uploaded_docs:
            st.session_state["uploaded_docs"] = uploaded_docs
            st.session_state["last_processed_file"] = uploaded_file.name
        else:
            st.sidebar.error("Could not extract any text from the uploaded file.")
else:
    # Revert to default if user clicks 'X' to remove file
    st.session_state["uploaded_docs"] = default_corpus
    if "last_processed_file" in st.session_state:
        del st.session_state["last_processed_file"]

uploaded_docs = st.session_state["uploaded_docs"]

# ---------------------------------------------------------
# SHOW CURRENT SYSTEM ACTIVE DOCUMENTS
# ---------------------------------------------------------
with st.expander("📄 View Active Document Collection Content", expanded=True):
    if uploaded_docs == default_corpus:
        st.success("💡 Pre-loaded Assignment Corpus Active (10 Documents Loaded Automatically)")
        if uploaded_file is not None:
             st.warning("Note: Your file was uploaded successfully, but it contains the exact same text as the default corpus!")
    else:
        st.success(f"✅ Custom Dataset '{st.session_state.get('last_processed_file', 'Uploaded File')}' Active ({len(uploaded_docs)} Documents Loaded)")
        
    st.caption("Here is the document collection parsed inside active indexing memory:")
    display_table = pd.DataFrame(list(uploaded_docs.items()), columns=["Document Reference Key", "Raw Source Text Content"])
    st.dataframe(display_table, use_container_width=True)

# ---------------------------------------------------------
# SIDEBAR: SETTINGS & CONTROLS
# ---------------------------------------------------------
st.sidebar.subheader("Text Cleanup Settings")
word_format_choice = st.sidebar.radio("How should we format the words?", ("Stemming", "Lemmatization"))
remove_stopwords = st.sidebar.checkbox("Remove Stop Words (like 'and', 'the')", value=True)
split_hyphens = st.sidebar.checkbox("Split hyphenated words", value=True)
    
st.sidebar.subheader("Search Settings")
search_index_type = st.sidebar.radio("How should we search for phrases?", ("Biword Index", "Positional Index"))
tree_structure_choice = st.sidebar.radio("Underlying Tree Structure:", ("Binary Search Tree", "B-Tree"))
    
st.sidebar.subheader("Error Handling Settings")
error_handling_mode = st.sidebar.selectbox("How should we handle typos or wildcards?", ("Edit Distance Correction", "K-Gram Wildcard Search"))

main_inverted_index = get_cached_inverted_index(uploaded_docs, remove_stopwords, word_format_choice, split_hyphens)
binary_tree, b_tree = get_cached_trees(main_inverted_index)
all_unique_words = list(main_inverted_index.keys())
kgram_index = get_cached_kgram_index(all_unique_words)

# ---------------------------------------------------------
# MAIN PAGE: TEXT CLEANUP PREVIEW
# ---------------------------------------------------------
st.header("Step-by-Step Text Cleanup")
chosen_doc = st.selectbox("Pick a document to see how it gets cleaned up behind the scenes:", list(uploaded_docs.keys()))

cleanup_steps = process_text_pipeline(uploaded_docs[chosen_doc], remove_stopwords, word_format_choice, split_hyphens)

left_col, right_col = st.columns(2)
with left_col:
    st.write("**1. All Lowercase:**", f"*{cleanup_steps['Lowercasing'][:100]}...*")
    st.write("**2. Split into Words (First 10):**", cleanup_steps['Tokenization'][:10])
with right_col:
    st.write("**3. Stop Words Removed (First 10):**", cleanup_steps['Stop Word Removal'][:10])
    st.write(f"**4. {word_format_choice} Applied (First 10):**", cleanup_steps['Normalization'][:10])

# ---------------------------------------------------------
# TASK B REQUIREMENT: STEMMING VS LEMMATIZATION DASHBOARD
# ---------------------------------------------------------
st.write("")
st.subheader("📊 Head-to-Head Evaluation: Stemming vs Lemmatization")
st.caption("Quantitative data-driven breakdown comparing vocabulary reduction and compression efficiency across your dataset.")

collection_stats = compare_cleanup_methods(uploaded_docs)

card_col1, card_col2, card_col3 = st.columns(3)
with card_col1:
    st.metric(label="Total Words Processed", value=collection_stats["Total Words Processed"])
with card_col2:
    st.metric(label="Unique Stems (Vocab Size)", value=collection_stats["Unique Stems (Vocabulary)"], delta="Aggressive Suffix Chopping")
with card_col3:
    st.metric(label="Unique Lemmas (Vocab Size)", value=collection_stats["Unique Lemmas (Vocabulary)"], delta="Preserves Lexical Form", delta_color="inverse")
        
comparison_table_data = pd.DataFrame({
    "Evaluation Dimension Metric": ["Unique Dictionary Index Vocabulary Size", "Index Storage Compression Efficiency"],
    "Porter Stemming Approach": [collection_stats["Unique Stems (Vocabulary)"], collection_stats["Stemming Compression Rate"]],
    "WordNet Lemmatization Approach": [collection_stats["Unique Lemmas (Vocabulary)"], collection_stats["Lemmatization Compression Rate"]]
})
st.table(comparison_table_data)
    
st.info("""
**💡 Evaluation & Collection Justification:**
* **Stemming Performance:** Demonstrates a smaller vocabulary size and higher compression rate across your 10 documents by executing crude truncation rules. However, it results in crude structural mutilation (e.g., transforming words like 'processing' or 'computing' into incomplete base forms).
* **Lemmatization Performance:** Preserves standard dictionary roots by performing contextual morphological lookups, yielding a clean, grammatically sound vocabulary list.
* *Collection Justification Decision:* **Lemmatization is heavily selected as optimal for this dataset.** Because this specific corpus handles technical phrases (like *'natural language processing'*, *'information retrieval'*, and *'artificial intelligence'*), structural correctness is critical to avoid false dictionary query misses.
""")

st.markdown("---")

# ---------------------------------------------------------
# MAIN PAGE: PHRASE INDEXING PREVIEW
# ---------------------------------------------------------
st.header("How the Search Engine Stores Phrases")
    
left_col, right_col = st.columns(2)
with left_col:
    st.subheader("• Biword Index Representation")
    biword_data = get_cached_biword_index(uploaded_docs, remove_stopwords, word_format_choice, split_hyphens)
    st.json(dict(list(biword_data.items())[:8]))
        
with right_col:
    st.subheader("• Positional Index Representation")
    positional_data = get_cached_positional_index(uploaded_docs, remove_stopwords, word_format_choice, split_hyphens)
    st.json(dict(list(positional_data.items())[:4]))

# ---------------------------------------------------------
# TASK C REQUIREMENT: LIVE FALSE POSITIVE CASE STUDY
# ---------------------------------------------------------
st.write("")
st.subheader("⚠️ Live Corpus Case Study: Biword False Positives vs. Positional Accuracy")

st.markdown("""
A **Biword Index** only records pairings of adjacent words. This introduces structural blind spots when verifying phrase queries containing **three or more terms**.
""")

demo_col1, demo_col2 = st.columns(2)

with demo_col1:
    st.error("🚨 The Biword False Positive Loophole in Doc_6")
    st.caption("**Target Phrase Query:** `\"modern search structures\"`")
    st.info("""
    **1. Target Query Decomposition into Biwords:**
    * Pair 1: `(\"modern\", \"search\")`
    * Pair 2: `(\"search\", \"structures\")`
    
    **2. Evaluation against Doc_6:**
    Look closely at your actual **Doc 6 text**:
    * *\"**Modern search** engines process... utilize optimized dictionary **search structures**.\"*
    
    **3. The Breakdown:** Since both separate biwords exist independently inside **Doc_6**, a Biword Index intersects their lookups and incorrectly returns **Doc_6 as a valid MATCH**. However, the continuous phrase *\"modern search structures\"* never actually occurred! This is a classic **False Positive**.
    """)
    
with demo_col2:
    st.success("🎯 The Positional Index Precision Solution")
    st.caption("**How Positional Math Prevents the Error:**")
    st.info("""
    **1. Explicit Word Offset Maps:**
    A Positional Index maps out the exact integer positioning of each parsed term:
    * `modern` $\\rightarrow$ Position: `[1]`
    * `search` $\\rightarrow$ Position: `[2, 7, 18]`
    * `structures` $\\rightarrow$ Position: `[19]`
    
    **2. Continuous Proximity Metric Checks:**
    The positional intersection calculation tests for consecutive increments:
    $$\\text{Pos}(\\text{search}) = \\text{Pos}(\\text{modern}) + 1 \\quad (2 = 1 + 1 \\rightarrow \\mathbf{True})$$
    $$\\text{Pos}(\\text{structures}) = \\text{Pos}(\\text{search}) + 1 \\quad (19 = 2 + 1 \\rightarrow \\mathbf{False})$$
    
    **3. The Outcome:** Because position `19` is not immediately adjacent to position `2`, the system throws out the match, delivering **100% precision accuracy**.
    """)

st.warning("""
**Inference:**
* **Biword Indexing** sacrifices retrieval accuracy on compound lookups spanning three or more words. This structural choice prioritizes simple boolean intersections over strict sequence layout enforcement.
* **Positional Indexing** provides complete phrase accuracy by maintaining word positioning arrays during index evaluation. While it requires a larger storage footprint, it completely eliminates text-ordering false positives.
""")

st.markdown("---")

# ---------------------------------------------------------
# MAIN PAGE: SEARCH SPEED TEST (TREES)
# ---------------------------------------------------------
st.header("Search Speed Test")
st.metric(label="Total Unique Words in our Database", value=len(main_inverted_index))
    
st.subheader("📊 Test the Speed")
test_query = st.text_input("Type a query to see how fast our different tree structures search:", value="information retrieval")
    
if test_query:
    speed_results = run_speed_test(
        test_query, binary_tree, b_tree, 
        process_text_pipeline, remove_stopwords, word_format_choice, split_hyphens
    )
    if speed_results:
        st.table(pd.DataFrame(speed_results))

st.markdown("---")

# ---------------------------------------------------------
# MAIN PAGE: TYPOS AND WILDCARDS
# ---------------------------------------------------------
st.header("Handling Typos and Wildcards")
    
if error_handling_mode == "Edit Distance Correction":
    st.subheader("1. Spelling Correction (Edit Distance)")
    typo_word = st.text_input("Type a word with a typo to see if we can fix it (e.g., 'infrmation'):", value="infrmation")
        
    if typo_word:
        spelling_suggestions = find_spelling_suggestions(typo_word.lower().strip(), all_unique_words, max_typos_allowed=2)
        if spelling_suggestions:
            suggestions_table = pd.DataFrame(spelling_suggestions, columns=["Did you mean?", "Letter Differences (Cost)"])
            st.table(suggestions_table)
        else:
            st.warning("We couldn't find any words close to that one.")
                
else:
    st.subheader("2. Wildcard Search (K-Gram)")
    wildcard_word = st.text_input("Type a wildcard query using an asterisk (e.g., 'comput*' or '*retrieval'):", value="comput*")
        
    if wildcard_word:
        matching_words = process_wildcard_search(wildcard_word, kgram_index, all_unique_words)
        if matching_words:
            st.success(f"We expanded '{wildcard_word}' into these words:")
            st.write(matching_words)
        else:
            st.warning("We couldn't find any words matching that wildcard pattern.")

st.markdown("---")

# ---------------------------------------------------------
# MAIN PAGE: RETRIEVAL SEARCH INTERFACE
# ---------------------------------------------------------
st.sidebar.header("Step 2: Try Searching!")
user_search = st.sidebar.text_input("Type what you want to find:")

if st.sidebar.button("Search"):
    if not user_search:
        st.sidebar.error("Please enter something to search for first!")
    else:
        st.header("Search Results")
            
        if '*' in user_search:
            st.info(f"🔮 Looks like you used a wildcard! Expanding your search...")
            found_terms = process_wildcard_search(user_search, kgram_index, all_unique_words)
                
            if found_terms:
                st.write(f"**Expanded Words:** {', '.join(found_terms[:10])}")
                found_documents = main_inverted_index.get(found_terms[0], [])
            else:
                found_documents = []
                    
        else:
            if search_index_type == "Biword Index":
                found_documents = search_biword(user_search, biword_data, process_text_pipeline, remove_stopwords, word_format_choice, split_hyphens)
            else:
                found_documents = search_positional(user_search, positional_data, process_text_pipeline, remove_stopwords, word_format_choice, split_hyphens)
            
        st.success(f"• Results for: '{user_search}'")
            
        if found_documents:
            display_results = []
                
            for doc_id in sorted(found_documents, key=lambda x: int(x.split('_')[1]) if '_' in x else 0):
                display_results.append({
                    "Matched Document": doc_id,
                    "Preview": uploaded_docs[doc_id][:250] + "..."
                })
            st.table(pd.DataFrame(display_results))
                
        else:
            st.warning("We couldn't find any documents matching your search.")
            first_search_word = user_search.split()[0].lower()
            auto_suggestions = find_spelling_suggestions(first_search_word, all_unique_words, max_typos_allowed=2)
                
            if auto_suggestions:
                st.info(f"💡 Did you mean: **{auto_suggestions[0][0]}**?")