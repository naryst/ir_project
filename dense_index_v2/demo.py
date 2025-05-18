import os

# Set environment variables for PyTorch
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCH_DISABLE_CUSTOM_CLASS_LOOKUP"] = "1"

import streamlit as st
import torch
from PIL import Image
import faiss
import numpy as np
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers import AutoModel
import glob
import pickle
from densely_captioned_images.dataset.impl import get_complete_dataset_with_settings
from densely_captioned_images.dataset.dense_image import DenseCaptionedImage

# Set page configuration
st.set_page_config(
    page_title="Vector Search Demo",
    page_icon="🔍",
    layout="wide"
)

COMPLETE_DATASET_SIZE = 7599
SPLIT = "train"

MODE = "jina"
# MODE = "colqwen"

@st.cache_data
def load_dataset(start_index=0, end_index=COMPLETE_DATASET_SIZE):
    print(f"Loading dataset from {start_index} to {end_index}")
    return get_complete_dataset_with_settings(
        split=SPLIT, start_index=start_index, end_index=end_index
    )

@st.cache_data
def get_documents_by_ids(doc_ids):
    """Load only the specific documents by their IDs."""
    documents = {}
    for doc_id in doc_ids:
        documents[doc_id] = DenseCaptionedImage(img_id=doc_id)
    return documents

# Initialize model and processor
@st.cache_resource
def load_model(mode="jina"):
    if mode == "jina":
        model = AutoModel.from_pretrained(
            'jinaai/jina-clip-v2', 
            trust_remote_code=True,
            device_map="cpu",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        return model, None
    else:
        model_name = "nomic-ai/colnomic-embed-multimodal-7b"
        model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        processor = ColQwen2_5_Processor.from_pretrained(model_name, use_fast=True)
        return model, processor

def get_available_indices(mode="jina"):
    """Get a list of available index files in the results directory."""
    base_dir = '.'

    faiss_dir = os.path.join(base_dir, "faiss_index")
    ball_tree_dir = os.path.join(base_dir, "ball_tree_index")
    
    # Get FAISS index files
    faiss_files = glob.glob(os.path.join(faiss_dir, "*.bin"))
    # Get Ball Tree index files
    ball_tree_files = glob.glob(os.path.join(ball_tree_dir, "*.pkl"))
    
    # Combine and format the results
    index_files = []
    for f in faiss_files + ball_tree_files:
        index_type = "FAISS" if "faiss" in f else "Ball Tree"
        index_files.append({
            "path": f,
            "name": os.path.basename(f),
            "type": index_type
        })
    
    return index_files

@st.cache_resource
def load_index(index_info):
    """Load the index with caching to avoid reloading on each rerun."""
    try:
        with open(index_info["path"], 'rb') as f:
            if index_info["type"] == "FAISS":
                # For FAISS, we need to load both the index and the ID lists
                index = faiss.read_index(index_info["path"])
                # base_path = os.path.dirname(index_info["path"])
                # with open(os.path.join(base_path, "query_id_list.pkl"), 'rb') as id_f:
                #     query_ids = pickle.load(id_f)
                # with open(os.path.join(base_path, "image_id_list.pkl"), 'rb') as id_f:
                #     image_ids = pickle.load(id_f)
                return {
                    "index": index,
                    # "query_ids": query_ids,
                    # "image_ids": image_ids,
                    "type": "FAISS"
                }
            else:
                # For Ball Tree, just load the pickle file
                return {
                    "index": pickle.load(f),
                    "type": "Ball Tree"
                }
    except Exception as e:
        st.error(f"Error loading index: {str(e)}")
        return None

def search_faiss(query_embedding, index, top_k=5):
    """Search using FAISS index."""
    distances, indices = index.search(query_embedding.reshape(1, -1).astype('float32'), top_k)
    vectors = np.zeros((len(indices[0]), index.d), dtype='float32')
    for i, idx in enumerate(indices[0]):
        vectors[i] = index.reconstruct(int(idx))
    return indices[0], distances[0], vectors

def search_ball_tree(query_embedding, index, top_k=5):
    """Search using Ball Tree index."""
    indices, distances = index.query(query_embedding, k=top_k)
    if distances.ndim == 1:
        distances = distances.reshape(1, -1)
    if indices.ndim == 1:
        indices = indices.reshape(1, -1)
    return indices[0], distances[0]

def main():
    st.title("🔍 Vector Search Demo")
    
    # Load model and processor
    model, processor = load_model(MODE)
    
    # Get available indices
    available_indices = get_available_indices(MODE)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    if not available_indices:
        st.sidebar.error("No index files found in the index directories. Please build an index first.")
        st.stop()
    
    # Let user select which index to load
    index_options = [f"{idx['name']} ({idx['type']})" for idx in available_indices]
    selected_index_name = st.sidebar.selectbox(
        "Select Index File", 
        index_options,
        index=0,
        label_visibility="visible"
    )
    
    # Get the selected index info
    selected_index = available_indices[index_options.index(selected_index_name)]
    
    # Set search method based on index type
    search_method = selected_index["type"]
    
    top_k = st.sidebar.slider(
        "Number of results to show", 
        1, 20, 5,
        label_visibility="visible"
    )
    
    # Load the selected index
    with st.spinner(f"Loading index {selected_index['name']}..."):
        index_data = load_index(selected_index)
    
    if index_data is None:
        st.stop()
    
    # Get the number of vectors based on index type
    if index_data["type"] == "FAISS":
        num_vectors = index_data["index"].ntotal
    else:  # Ball Tree
        num_vectors = len(index_data["index"].data)
    
    st.sidebar.success(f"Index loaded with {num_vectors} vectors")
    
    # Query input
    st.subheader("Enter your search query")
    query = st.text_input(
        "Search Query",
        placeholder="Type your search query here...",
        label_visibility="visible"
    )
    
    if query:
        with st.spinner("Processing query..."):
            if MODE == "colqwen":
                # Process query
                query_input = processor.process_queries([query])
                with torch.no_grad():
                    query_embedding = model(**query_input)
                # Convert BFloat16 to float32 before converting to numpy
                query_embedding = query_embedding.mean(dim=1).to(torch.float32).cpu().numpy()
            elif MODE == "jina":
                query_embedding = model.encode_text([query], task='retrieval.query', truncate_dim=512)
            # normalize the query embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search
            if search_method == "FAISS":
                indices, distances, vectors = search_faiss(query_embedding, index_data['index'], top_k)
            else:
                indices, distances = search_ball_tree(query_embedding, index_data['index'], top_k)
            print(indices)
            print([np.linalg.norm(vector) for vector in vectors])
            for i, vector in enumerate(vectors):
                assert vector.shape == (512,)
                assert np.linalg.norm(vector) > 1e-6

            indices = [x // 104 for x in indices] # get id of the image by id of mask
            
            st.subheader(f"Top {len(indices)} Results")
            
            # Get document IDs from indices
            doc_ids = [int(idx) for idx in indices]
            
            # Load only the documents that match the query
            documents = get_documents_by_ids(doc_ids)
            
            if not documents:
                st.error("Could not load documents to display results.")
                st.stop()
            
            # Display results
            for i, (idx, dist) in enumerate(zip(indices, distances)):
                try:
                    doc_id = int(idx)
                    if doc_id in documents:
                        entry = documents[doc_id]
                        caption = entry.get_formatted_complete_description()[0]['caption']
                        
                        # Create a result card
                        with st.container():
                            st.subheader(f"Result {i+1}")
                            st.write(f"Document ID: {doc_id}")
                            st.write(f"**Distance:** {dist:.4f}")
                            st.write(f"**Caption:** {caption}")
                            
                            # Display image if available
                            try:
                                if entry.get_image() is not None:
                                    image = entry.get_image()
                                    st.image(image, caption=f"Image for Result {i+1}", use_container_width=True)
                            except Exception as e:
                                st.error(f"Error displaying image: {e}")
                                
                            st.divider()
                    else:
                        st.warning(f"Document ID {doc_id} is out of range or could not be loaded.")
                        
                except Exception as e:
                    st.error(f"Error displaying result {i+1}: {str(e)}")

if __name__ == "__main__":
    main()
