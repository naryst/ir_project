import sys
import os

# Set environment variable to disable PyTorch's custom class path examination
# These must be set before importing any PyTorch-related modules
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCH_DISABLE_CUSTOM_CLASS_LOOKUP"] = "1"
os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import glob
import pandas as pd
from PIL import Image
import io
import base64

# Add the parent directory to the path so we can import the KGramIndex
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kgram_index.build_index import KGramIndex, load_dataset

# Set page configuration
st.set_page_config(
    page_title="KGram Index Search Demo",
    page_icon="🔍",
    layout="wide"
)

def get_available_indices():
    """Get a list of available index files in the data directory."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kgram_index", "data")
    index_files = glob.glob(os.path.join(data_dir, "*.pkl"))
    
    # Extract just the filenames without the path
    index_names = [os.path.basename(f) for f in index_files]
    
    # If no indices found, return an empty list
    if not index_names:
        return []
    
    return index_names

@st.cache_resource
def load_kgram_index(index_filename):
    """Load the KGram index with caching to avoid reloading on each rerun."""
    try:
        # Construct the full path to the index file
        index_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "kgram_index", "data", index_filename
        )
        
        # Extract k value from filename if it follows the pattern K=X_index.pkl
        k = 3  # Default value
        if index_filename.startswith("K="):
            try:
                k = int(index_filename.split("=")[1].split("_")[0])
            except (IndexError, ValueError):
                pass
                
        index = KGramIndex.load(path=index_path)
        return index
    except FileNotFoundError:
        st.error(f"Index file {index_filename} not found. Please make sure you've built the index first.")
        return None
    except Exception as e:
        st.error(f"Error loading index: {str(e)}")
        return None

@st.cache_data
def get_dataset(_start_index=0, _end_index=7599):
    """Load the dataset with caching to avoid reloading on each rerun."""
    try:
        return load_dataset(_start_index, _end_index)
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return []

@st.cache_data
def get_documents_by_ids(doc_ids):
    """Load only the specific documents by their IDs."""
    try:
        documents = {}
        for doc_id in doc_ids:
            # Load one document at a time
            doc_data = load_dataset(doc_id, doc_id + 1)
            if doc_data and len(doc_data) > 0:
                documents[doc_id] = doc_data[0]
        return documents
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return {}

def main():
    st.title("🔍 KGram Index Search Demo")
    
    # Get available indices
    available_indices = get_available_indices()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    if not available_indices:
        st.sidebar.error("No index files found in the data directory. Please build an index first.")
        st.stop()
    
    # Let user select which index to load
    selected_index = st.sidebar.selectbox(
        "Select Index File", 
        available_indices,
        index=0
    )
    
    top_k = st.sidebar.slider("Number of results to show", 1, 20, 5)
    
    # Add checkbox for wildcard search
    use_wildcards = st.sidebar.checkbox("Enable wildcard search (use * in queries)", value=False)
    
    # Load the selected index
    with st.spinner(f"Loading index {selected_index}..."):
        index = load_kgram_index(selected_index)
    
    if index is None:
        st.stop()
    
    st.sidebar.success(f"Index loaded with {index.total_docs} documents")
    
    # Query input
    st.subheader("Enter your search query")
    query = st.text_input("", placeholder="Type your search query here...")
    
    if query:
        with st.spinner("Searching..."):
            # Get the top-k results
            results = index.compute_tf_idf_query(query, top_k=top_k, use_wildcards=use_wildcards)
            
            if not results:
                st.warning("No results found. Try a different query.")
            else:
                st.subheader(f"Top {len(results)} Results")
                
                # Extract document IDs from results
                doc_ids = [doc_id for doc_id, _ in results]
                
                # Load only the documents that match the query
                documents = get_documents_by_ids(doc_ids)
                
                if not documents:
                    st.error("Could not load documents to display results.")
                    st.stop()
                
                # Display results
                for doc_id, score in results:
                    if doc_id in documents:
                        try:
                            entry = documents[doc_id]
                            caption = entry[0]["caption"]
                            
                            # Create a result card using Streamlit components
                            with st.container():
                                st.subheader(f"Document ID: {doc_id}")
                                st.write(caption)
                                st.write(f"**Score:** {score:.4f}")
                                st.divider()
                            
                            # Try to display the image if available
                            try:
                                if "image" in entry[0]:
                                    image = entry[0]["image"]
                                    st.image(image, caption=f"Image for Document {doc_id}", use_container_width=True)
                            except Exception as e:
                                st.error(f"Error displaying image: {e}")
                        except Exception as e:
                            st.error(f"Error displaying result {doc_id}: {str(e)}")
                    else:
                        st.warning(f"Document ID {doc_id} is out of range or could not be loaded.")

if __name__ == "__main__":
    main()
