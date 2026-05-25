# app.py
import streamlit as st
import pandas as pd
import re

# --- Import  custom modular backends ---
from preprocessing import run_preprocessing_pipeline, build_inverted_index, compare_stem_lem
from indexing import build_biword_index, build_positional_index, search_biword, search_positional
from trees import populate_trees, execute_tree_benchmarks
from tolerant import get_spelling_corrections, build_kgram_index, resolve_wildcard_query

# -- Page Configuration --
st.set_page_config(page_title="IR System - Assignment 1", layout="wide")
# ==========================================
# ADVANCED REMOVAL OF STREAMLIT TOP PADDING
# ==========================================
st.markdown(
    """
    <style>
        /* 1. Eliminate default main layout block-container paddings */
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 1rem !important;
            margin-top: 0rem !important;
        }
        
        /* 2. Target the primary structural multi-div layout wrappers */
        .stMain, .stMainBlockContainer, [data-testid="stMainBlockContainer"] {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }

        /* 3. Strip padding from the root vertical element stacks inside main view */
        [data-testid="stVerticalBlock"] > div:first-child {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }

        /* 4. Kill blank top application header zone completely */
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

# ==========================================
# PERFORMANCE CACHING LAYER
# Prevents the application from freezing or hanging on re-renders
# ==========================================
@st.cache_data
def cached_biword_index(docs, apply_stop, norm, hyphen):
    return build_biword_index(docs, run_preprocessing_pipeline, apply_stop, norm, hyphen)

@st.cache_data
def cached_positional_index(docs, apply_stop, norm, hyphen):
    return build_positional_index(docs, run_preprocessing_pipeline, apply_stop, norm, hyphen)

@st.cache_data
def cached_inverted_index(docs, apply_stop, norm, hyphen):
    return build_inverted_index(docs, apply_stop, norm, hyphen)

@st.cache_resource
def cached_compiled_trees(inverted_index):
    return populate_trees(inverted_index)

@st.cache_data
def cached_kgram_index(vocab_list):
    return build_kgram_index(vocab_list)

# ==========================================
# 1. DATA INGESTION (Sidebar Layout)
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/d/d3/BITS_Pilani-Logo.svg/330px-BITS_Pilani-Logo.svg.png", use_container_width=True)
st.sidebar.header("Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload a text dataset or document collection (.txt, .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    
    # Secure Session State Initialization to manage dictionary objects cleanly
    if "documents_dict" not in st.session_state or st.sidebar.button("Force Re-load File"):
        documents_dict = {}
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            for idx, row in df.iterrows():
                documents_dict[f"Doc_{idx+1}"] = str(row.iloc[0])
        else:
            raw_text = uploaded_file.read().decode("utf-8")

            # Finds matches for headers like === DOC_1 === and splits content accordingly for uploaded text corpus
            segments = re.split(r'===\s*DOC_\d+\s*===', raw_text)
            headers = re.findall(r'===\s*(DOC_\d+)\s*===', raw_text)
            
            # Clean up out-of-bounds segments
            doc_index = 1
            for seg in segments:
                cleaned_content = seg.strip()
                if cleaned_content:
                    documents_dict[f"Doc_{doc_index}"] = cleaned_content
                    doc_index += 1
            
            # Fallback block if document headers aren't parsed by regex patterns
            if not documents_dict:
                docs = raw_text.split('\n\n')
                for idx, doc in enumerate(docs):
                    if doc.strip():
                        documents_dict[f"Doc_{idx+1}"] = doc.strip()
                        
        st.session_state["documents_dict"] = documents_dict
    else:
        documents_dict = st.session_state["documents_dict"]

    # ==========================================
    # RAW DOCUMENT VIEWER
    # ==========================================
    with st.expander("📄 View Uploaded Document Collection Content (Task A Workflow Requirement)", expanded=True):
        st.caption("Inspect the raw textual entries split across discrete index map identifiers:")
        display_df = pd.DataFrame(list(documents_dict.items()), columns=["Document Reference Key", "Raw Source Text Content"])
        st.dataframe(display_df, use_container_width=True)

    # ==========================================
    # 2. SELECT INTERACTIVE CONTROLS (Sidebar)
    # ==========================================
    st.sidebar.subheader("Preprocessing Options")
    prep_option = st.sidebar.radio("Text Normalization:", ("Stemming", "Lemmatization"))
    stop_option = st.sidebar.checkbox("Remove Stop Words", value=True)
    hyphen_option = st.sidebar.checkbox("Handle Hyphens (Split)", value=True)
    
    st.sidebar.subheader("Indexing & Data Structures")
    index_option = st.sidebar.radio("Phrase Query Index Framework:", ("Biword Index", "Positional Index"))
    tree_option = st.sidebar.radio("Dictionary Tree Type:", ("Binary Search Tree", "B-Tree"))
    
    st.sidebar.subheader("Imperfect Queries")
    tolerant_option = st.sidebar.selectbox("Tolerant Retrieval Mode:", ("Edit Distance Correction", "K-Gram Wildcard Search"))

    # Compile the Indexes and Dictionary structures in memory
    inv_index = cached_inverted_index(documents_dict, stop_option, prep_option, hyphen_option)
    bst_index, btree_index = cached_compiled_trees(inv_index)
    
    vocab_list = list(inv_index.keys())
    kgram_idx = cached_kgram_index(vocab_list)

    # ==========================================
    # TASK B: TEXT PREPROCESSING PREVIEW
    # ==========================================
    st.header("Text Preprocessing Pipeline")
    sample_doc_id = st.selectbox("Select a Document to View Pipeline Stages:", list(documents_dict.keys()))
    
    pipeline_stages = run_preprocessing_pipeline(documents_dict[sample_doc_id], stop_option, prep_option, hyphen_option)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**1. Raw Lowercased:**", f"*{pipeline_stages['Lowercasing'][:100]}...*")
        st.write("**2. Tokenization (First 10):**", pipeline_stages['Tokenization'][:10])
    with col2:
        st.write("**3. Stop Word Removal (First 10):**", pipeline_stages['Stop Word Removal'][:10])
        st.write(f"**4. {prep_option} (First 10):**", pipeline_stages['Normalization'][:10])

    st.markdown("---")

    # ==========================================
    # TASK C: PHRASE QUERY PROCESSING
    # ==========================================
    st.header("Phrase Query Index Representations")
    
    col_bw, col_pos = st.columns(2)
    with col_bw:
        st.subheader("• Biword Index Representation")
        biword_idx = cached_biword_index(documents_dict, stop_option, prep_option, hyphen_option)
        st.json(dict(list(biword_idx.items())[:8]))
        
    with col_pos:
        st.subheader("• Positional Index Representation")
        positional_idx = cached_positional_index(documents_dict, stop_option, prep_option, hyphen_option)
        st.json(dict(list(positional_idx.items())[:4]))

    st.markdown("---")

    # ==========================================
    # TASK D: DICTIONARY SEARCH USING BST AND B-TREE
    # ==========================================
    st.header("Dictionary Search Performance Evaluation")
    st.metric(label="Total Unique Terms Count in Dictionary", value=len(inv_index))
    
    st.subheader("📊 Live Experimental Results Table")
    sample_bench_query = st.text_input("Enter a test query to generate tree performance benchmarks:", value="information retrieval")
    
    if sample_bench_query:
        bench_metrics = execute_tree_benchmarks(
            sample_bench_query, bst_index, btree_index, 
            run_preprocessing_pipeline, stop_option, prep_option, hyphen_option
        )
        if bench_metrics:
            st.table(pd.DataFrame(bench_metrics))

    st.markdown("---")

    # ==========================================
    # TASK E: TOLERANT RETRIEVAL BENCHMARKS
    # ==========================================
    st.header("Tolerant Retrieval Testing Engine")
    
    if tolerant_option == "Edit Distance Correction":
        st.subheader("1. Edit Distance Spelling Correction Panel")
        typo_input = st.text_input("Type a word with a typo to check index spelling corrections (e.g., 'infrmation'):", value="infrmation")
        
        if typo_input:
            corrections = get_spelling_corrections(typo_input.lower().strip(), vocab_list, max_distance=2)
            if corrections:
                corr_df = pd.DataFrame(corrections, columns=["Suggested Vocabulary Term", "Levenshtein Distance Cost"])
                st.table(corr_df)
            else:
                st.warning("No related vocabulary entries matched within an edit distance of 2.")
                
    else:
        st.subheader("2. K-Gram Wildcard Processing Panel")
        wildcard_input = st.text_input("Enter a wildcard text query to execute K-gram index lookups (e.g., 'comput*' or '*retrieval'):", value="comput*")
        
        if wildcard_input:
            matched_vocab_terms = resolve_wildcard_query(wildcard_input, kgram_idx, vocab_list)
            if matched_vocab_terms:
                st.success(f"K-Gram expansion resolved '{wildcard_input}' into the following terms:")
                st.write(matched_vocab_terms)
            else:
                st.warning("No matching vocabulary words found for this specific wildcard pattern.")

    st.markdown("---")

    # ==========================================
    # RETRIEVAL CONSOLE (Executes Search Logic)
    # ==========================================
    st.sidebar.header("Step 2: Search Engine Console")
    search_query = st.sidebar.text_input("Enter your phrase search query:")

    if st.sidebar.button("Run Search Pipeline"):
        if not search_query:
            st.sidebar.error("Please enter a query first!")
        else:
            st.header("Search Results & Live Metrics")
            
            # Auto-handle Wildcard patterns directly inside the search bar console
            if '*' in search_query:
                st.info(f"🔮 Wildcard symbol detected. Query expanded via K-gram Indexing...")
                terms_found = resolve_wildcard_query(search_query, kgram_idx, vocab_list)
                if terms_found:
                    st.write(f"**Expanded Wildcard Terms:** {', '.join(terms_found[:10])}")
                    matched_docs = list(inv_index.get(terms_found[0], {}).keys())
                else:
                    matched_docs = []
            else:
                if index_option == "Biword Index":
                    matched_docs = search_biword(search_query, biword_idx, run_preprocessing_pipeline, stop_option, prep_option, hyphen_option)
                else:
                    matched_docs = search_positional(search_query, positional_idx, run_preprocessing_pipeline, stop_option, prep_option, hyphen_option)
            
            st.success(f"• Query Results for: '{search_query}'")
            
            if matched_docs:
                results_data = []
                for doc_id in sorted(matched_docs, key=lambda x: int(x.split('_')[1]) if '_' in x else 0):
                    results_data.append({
                        "Matched Doc ID": doc_id,
                        "Document Snippet": documents_dict[doc_id][:250] + "..."
                    })
                st.table(pd.DataFrame(results_data))
            else:
                st.warning("No direct document matches found.")
                first_word = search_query.split()[0].lower()
                suggestions = get_spelling_corrections(first_word, vocab_list, max_distance=2)
                if suggestions:
                    st.info(f"💡 Did you mean: **{suggestions[0][0]}**? (Detected via Levenshtein Edit Distance)")

else:
    st.info("👈 Please upload a document collection in the sidebar to begin processing.")