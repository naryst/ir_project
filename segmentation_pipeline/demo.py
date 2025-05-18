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
from ollama_request import refine_query_for_clip

# Set page configuration
st.set_page_config(
    page_title="Vector Search Demo",
    page_icon="🔍",
    layout="wide"
)

def refine_prompt(query):
    return refine_query_for_clip(query)

COMPLETE_DATASET_SIZE = 100  # Adjust based on number of images you have
SPLIT = "train"

MODE = "jina"
# MODE = "colqwen"

# Define paths relative to the script location to ensure consistency
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masks")

@st.cache_data
def load_dataset(start_index=0, end_index=COMPLETE_DATASET_SIZE):
    """Load dataset directly from masks folder."""
    print(f"Loading dataset from {start_index} to {end_index}")
    documents = []
    
    # Ensure indices are within bounds
    start_index = max(0, start_index)
    end_index = min(COMPLETE_DATASET_SIZE, end_index)
    
    for i in range(start_index, end_index):
        image_folder = os.path.join(MASKS_DIR, f"image_{i}")
        if os.path.exists(image_folder):
            # Load the whole image and its description
            whole_image_path = os.path.join(image_folder, f"image_{i}_whole.png")
            whole_text_path = os.path.join(image_folder, f"image_{i}_whole.txt")
            
            if os.path.exists(whole_image_path) and os.path.exists(whole_text_path):
                with open(whole_text_path, 'r') as f:
                    caption = f.read().strip()
                
                # Create entry similar to DCI format
                entry = {
                    "image": Image.open(whole_image_path),
                    "caption": caption,
                    "id": i
                }
                documents.append([entry])
    
    return documents

@st.cache_data
def get_documents_by_ids(doc_ids):
    """Load only the specific documents by their IDs from masks folder."""
    try:
        documents = {}
        
        for doc_id in doc_ids:
            image_folder = os.path.join(MASKS_DIR, f"image_{doc_id}")
            
            if os.path.exists(image_folder):
                # Load the whole image and its description
                whole_image_path = os.path.join(image_folder, f"image_{doc_id}_whole.png")
                whole_text_path = os.path.join(image_folder, f"image_{doc_id}_whole.txt")
                
                if os.path.exists(whole_image_path) and os.path.exists(whole_text_path):
                    with open(whole_text_path, 'r') as f:
                        caption = f.read().strip()
                    
                    # Create entry similar to DCI format
                    entry = {
                        "image": Image.open(whole_image_path),
                        "caption": caption,
                        "id": doc_id
                    }
                    documents[doc_id] = [entry]
                else:
                    st.write(f"Required files don't exist for document {doc_id}")
            else:
                st.write(f"Folder doesn't exist for document {doc_id}")
        
        if not documents:
            # List all available image folders as a fallback
            available_folders = glob.glob(os.path.join(MASKS_DIR, "image_*"))
            
            # Try to load the first few available images as a fallback
            if available_folders:
                for folder in available_folders[:5]:  # Try first 5 folders
                    folder_name = os.path.basename(folder)
                    try:
                        doc_id = int(folder_name.replace("image_", ""))
                        
                        whole_image_path = os.path.join(folder, f"{folder_name}_whole.png")
                        whole_text_path = os.path.join(folder, f"{folder_name}_whole.txt")
                        
                        if os.path.exists(whole_image_path) and os.path.exists(whole_text_path):
                            with open(whole_text_path, 'r') as f:
                                caption = f.read().strip()
                            
                            entry = {
                                "image": Image.open(whole_image_path),
                                "caption": caption,
                                "id": doc_id
                            }
                            documents[doc_id] = [entry]
                    except Exception as e:
                        st.write(f"Error loading fallback document: {str(e)}")
        
        return documents
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        import traceback
        st.error(f"Stack trace: {traceback.format_exc()}")
        return {}

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
                return {
                    "index": index,
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

def scan_masks_directory():
    """Scan the masks directory to understand the available structure."""
    base_dir = MASKS_DIR
    
    if not os.path.exists(base_dir):
        return {
            "exists": False,
            "message": f"Directory {base_dir} does not exist"
        }
    
    image_dirs = glob.glob(os.path.join(base_dir, "image_*"))
    
    result = {
        "exists": True,
        "dir_count": len(image_dirs),
        "sample_dirs": image_dirs[:5],
        "samples": []
    }
    
    # Check some example directories in detail
    for image_dir in image_dirs[:3]:
        dir_name = os.path.basename(image_dir)
        image_id = dir_name.replace("image_", "")
        
        whole_png = os.path.join(image_dir, f"{dir_name}_whole.png")
        whole_txt = os.path.join(image_dir, f"{dir_name}_whole.txt")
        
        dir_info = {
            "dir_name": dir_name,
            "image_id": image_id,
            "whole_png_exists": os.path.exists(whole_png),
            "whole_txt_exists": os.path.exists(whole_txt),
            "all_files": glob.glob(os.path.join(image_dir, f"{dir_name}_*.png"))[:5],
            "all_txts": glob.glob(os.path.join(image_dir, f"{dir_name}_*.txt"))[:5]
        }
        
        result["samples"].append(dir_info)
    
    return result

def load_fallback_documents(count=5):
    """Load any available documents as a fallback."""
    try:
        documents = {}
        available_folders = glob.glob(os.path.join(MASKS_DIR, "image_*"))
        
        for i, folder in enumerate(available_folders[:count]):
            folder_name = os.path.basename(folder)
            try:
                doc_id = int(folder_name.replace("image_", ""))
                
                whole_image_path = os.path.join(folder, f"{folder_name}_whole.png")
                whole_text_path = os.path.join(folder, f"{folder_name}_whole.txt")
                
                st.write(f"Debug: Trying fallback image: {whole_image_path}")
                st.write(f"Debug: Trying fallback text: {whole_text_path}")
                
                if os.path.exists(whole_image_path) and os.path.exists(whole_text_path):
                    with open(whole_text_path, 'r') as f:
                        caption = f.read().strip()
                    
                    entry = {
                        "image": Image.open(whole_image_path),
                        "caption": caption,
                        "id": doc_id
                    }
                    documents[doc_id] = [entry]
                    st.write(f"Debug: Fallback loaded document {doc_id}")
                else:
                    st.write(f"Debug: Fallback files don't exist for {folder_name}")
            except Exception as e:
                st.write(f"Debug: Error loading fallback document {folder_name}: {str(e)}")
        
        return documents
    except Exception as e:
        st.error(f"Error loading fallback documents: {str(e)}")
        import traceback
        st.error(f"Stack trace: {traceback.format_exc()}")
        return {}

def main():
    st.title("🔍 Vector Search Demo")
    
    # Load model and processor
    model, processor = load_model(MODE)
    
    # Get available indices
    available_indices = get_available_indices(MODE)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Add diagnostics section to sidebar
    if st.sidebar.checkbox("Show Diagnostics"):
        st.sidebar.subheader("Data Directory Diagnostics")
        masks_info = scan_masks_directory()
        
        if masks_info["exists"]:
            st.sidebar.success(f"Found {masks_info['dir_count']} image directories")
            st.sidebar.write("Sample directories:")
            for sample_dir in masks_info["sample_dirs"]:
                st.sidebar.write(f"- {os.path.basename(sample_dir)}")
            
            st.sidebar.write("Directory contents:")
            for dir_info in masks_info["samples"]:
                st.sidebar.write(f"Directory: {dir_info['dir_name']}")
                st.sidebar.write(f"Whole image: {'✅' if dir_info['whole_png_exists'] else '❌'}")
                st.sidebar.write(f"Whole text: {'✅' if dir_info['whole_txt_exists'] else '❌'}")
        else:
            st.sidebar.error(masks_info["message"])
    
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
    
    # Add button right after query input
    refine_button = st.button("Run with refined prompt", help="Use AI to optimize your query for better results")
    
    if query:
        # Process query
        processed_query = query
        
        # Check if refinement button was clicked
        if refine_button:
            with st.spinner("Refining query..."):
                processed_query = refine_prompt(query)
                st.info(f"Original query: '{query}'\nRefined query: '{processed_query}'")
        
        with st.spinner("Processing query..."):
            if MODE == "colqwen":
                # Process query
                query_input = processor.process_queries([processed_query])
                with torch.no_grad():
                    query_embedding = model(**query_input)
                # Convert BFloat16 to float32 before converting to numpy
                query_embedding = query_embedding.mean(dim=1).to(torch.float32).cpu().numpy()
            elif MODE == "jina":
                query_embedding = model.encode_text([processed_query], task='retrieval.query', truncate_dim=512)
            
            # Search
            if search_method == "FAISS":
                indices, distances, vectors = search_faiss(query_embedding, index_data['index'], top_k)
                print(indices)
                print([np.linalg.norm(vector) for vector in vectors])
                for i, vector in enumerate(vectors):
                    assert vector.shape == (512,)
                    assert np.linalg.norm(vector) > 1e-6, f"Vector {i} has norm {np.linalg.norm(vector)}"
            else:
                indices, distances = search_ball_tree(query_embedding, index_data['index'], top_k)
                print(indices)
                        
            # Convert mask indices to image indices (36 masks per image)
            original_indices = indices.copy()
            indices = [int(x // 36) for x in indices]            
            st.subheader(f"Top {len(indices)} Results")
            
            # Get document IDs from indices
            doc_ids = [int(idx) for idx in indices]
            
            # Check for existence of folders before loading
            available_ids = []
            for doc_id in doc_ids:
                folder_path = os.path.join(MASKS_DIR, f"image_{doc_id}")
                if os.path.exists(folder_path):
                    available_ids.append(doc_id)
            
            # Load only the documents that match the query
            documents = get_documents_by_ids(doc_ids)
            
            if not documents:
                st.error("Could not load documents to display results. Trying alternative loading...")
                
                # Try loading with the original indices
                alt_doc_ids = [int(idx) for idx in original_indices]
                documents = get_documents_by_ids(alt_doc_ids)
                
                if not documents:
                    st.error("Still could not load documents. Trying fallback loading...")
                    documents = load_fallback_documents(count=5)
                    
                    if not documents:
                        st.error("All loading methods failed. No results to display.")
                        st.stop()
                    else:
                        st.warning("Showing fallback documents instead of search results")
                        indices = list(documents.keys())
                        distances = [0.0] * len(indices)  # Placeholder distances
            
            # Display results
            for i, (idx, dist) in enumerate(zip(indices, distances)):
                try:
                    doc_id = int(idx)
                    if doc_id in documents:
                        entry = documents[doc_id][0]  # Get the first (and only) entry
                        caption = entry["caption"]
                        
                        # Create a result card
                        with st.container():
                            st.subheader(f"Result {i+1}")
                            st.write(f"Document ID: {doc_id}")
                            st.write(f"**Distance:** {dist:.4f}")
                            st.write(f"**Caption:** {caption}")
                            
                            # Display image if available
                            try:
                                if "image" in entry:
                                    image = entry["image"]
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
