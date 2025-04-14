import sys
import os
import faiss
import pickle
import numpy as np
import time

# Set environment variables for PyTorch
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCH_DISABLE_CUSTOM_CLASS_LOOKUP"] = "1"
os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import glob
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Add parent directory to path for KGramIndex import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kgram_index.build_index import KGramIndex, load_dataset

# Set page configuration
st.set_page_config(
    page_title="Search Demo - KGram, FAISS & BallTree",
    page_icon="🔍",
    layout="wide"
)

# VP‑Tree реализация (Vantage‑Point Tree)
import heapq

class VPTree:
    class Node:
        def __init__(self, index, threshold, left, right):
            self.index = index          # индекс опорной точки в массиве эмбеддингов
            self.threshold = threshold  # пороговое расстояние (медиана)
            self.left = left            # поддерево для точек внутри порога
            self.right = right          # поддерево для точек за пределами порога

    def __init__(self, points):
        """
        points: numpy массив размерности (N, D), где N — число точек, D — размерность эмбеддингов.
        """
        self.points = points
        indices = np.arange(points.shape[0])
        self.root = self._build_tree(indices)

    def _build_tree(self, indices):
        if len(indices) == 0:
            return None
        if len(indices) == 1:
            return self.Node(indices[0], None, None, None)

        # Выбираем опорную точку; для простоты – первую точку
        vp = indices[0]
        vp_point = self.points[vp]
        others = indices[1:]
        # Вычисляем расстояния от опорной точки до остальных (евклидово расстояние)
        distances = np.linalg.norm(self.points[others] - vp_point, axis=1)
        # Вычисляем медиану расстояний
        median = np.median(distances)

        # Разбиваем остальные точки по медиане
        inner_mask = distances < median
        left_indices = others[inner_mask]
        right_indices = others[~inner_mask]

        left_tree = self._build_tree(left_indices)
        right_tree = self._build_tree(right_indices)

        return self.Node(vp, median, left_tree, right_tree)

    def _search(self, node, target, k, heap):
        if node is None:
            return

        vp_index = node.index
        vp_point = self.points[vp_index]
        dist = np.linalg.norm(target - vp_point)

        # Добавляем в max-heap (используем отрицательные расстояния)
        if len(heap) < k:
            heapq.heappush(heap, (-dist, vp_index))
        else:
            if dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, vp_index))

        if node.threshold is None:
            return

        # Решаем, в какую сторону двигаться
        if dist < node.threshold:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left

        self._search(first, target, k, heap)
        # Если расстояние до разделяющей границы позволяет получить лучшего кандидата, ищем и во втором поддереве
        if len(heap) < k or abs(dist - node.threshold) < -heap[0][0]:
            self._search(second, target, k, heap)

    def query(self, target, k=1):
        """
        target: numpy массив размерности (D,)
        k: число ближайших соседей
        Возвращает список кортежей (расстояние, индекс), отсортированный по расстоянию.
        """
        heap = []
        self._search(self.root, target, k, heap)
        results = sorted([(-d, idx) for d, idx in heap])
        return results

# Initialize CLIP model globally for BallTree text encoding
def init_clip_model():
    try:
        clip_model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(clip_model_name)
        processor = CLIPProcessor.from_pretrained(clip_model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        return model, processor
    except Exception as e:
        st.error(f"Error loading CLIP model: {str(e)}")
        return None, None


clip_model, clip_processor = init_clip_model()


def get_available_indices():
    """Get list of available KGram index files."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kgram_index", "data")
    index_files = glob.glob(os.path.join(data_dir, "*.pkl"))
    return [os.path.basename(f) for f in index_files] if index_files else []


@st.cache_resource
def load_kgram_index(index_filename):
    """Load KGram index with caching."""
    try:
        index_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "kgram_index", "data", index_filename
        )
        return KGramIndex.load(path=index_path)
    except Exception as e:
        st.error(f"Error loading KGram index: {str(e)}")
        return None


@st.cache_resource
def load_faiss_index(index_path="faiss_ivf_index.bin"):
    """Load FAISS index with caching."""
    try:
        return faiss.read_index(index_path)
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        return None


@st.cache_resource
def load_balltree_index():
    """Load embeddings and build VP-Tree index with caching."""
    try:
        import sys
        import numpy as np
        sys.modules['numpy._core'] = np.core

        with open("embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)

        # Filter only image embeddings
        image_entries = [e for e in embeddings if e.get("type") == "image"]
        image_ids = [e["id"] for e in image_entries]
        image_vectors = np.array([e["embedding"] for e in image_entries], dtype=np.float32)

        # Normalize vectors
        norms = np.linalg.norm(image_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        image_vectors = image_vectors / norms

        # Build VP‑Tree вместо BallTree
        tree = VPTree(image_vectors)
        return tree, image_vectors, image_ids
    except Exception as e:
        st.error(f"Error loading VP-Tree index: {str(e)}")
        return None, None, None


@st.cache_data
def get_dataset(_start_index=0, _end_index=7599):
    """Load dataset with caching."""
    try:
        return load_dataset(_start_index, _end_index)
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return []


@st.cache_data
def get_documents_by_ids(doc_ids):
    """Load specific documents by IDs with caching."""
    try:
        documents = {}
        for doc_id in doc_ids:
            doc_data = load_dataset(doc_id, doc_id + 1)
            if doc_data and len(doc_data) > 0:
                documents[doc_id] = doc_data[0]
        return documents
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return {}


def encode_text_for_balltree(text):
    """Encode text query using CLIP model for VP‑Tree search."""
    if clip_model is None or clip_processor is None:
        raise RuntimeError("CLIP model not loaded")

    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(clip_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)

    text_vector = text_features.cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(text_vector)
    return (text_vector / norm).flatten() if norm > 0 else text_vector.flatten()


def search_balltree(query_text, tree, vectors, ids, top_k=5):
    """Search VP‑Tree index for similar images."""
    query_vector = encode_text_for_balltree(query_text)
    # Получаем список кортежей (distance, index) от VP‑Tree
    neighbors = tree.query(query_vector, k=top_k)
    results = []
    for d, i in neighbors:
        image_id = ids[i]
        cosine_sim = 1 - (d ** 2) / 2.0
        results.append((image_id, cosine_sim))
    return results


def main():
    st.title("🔍 Search Demo - KGram, FAISS & VP-Tree")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    available_indices = get_available_indices()

    if not available_indices:
        st.sidebar.error("No KGram index files found. Please build an index first.")
        st.stop()

    selected_index = st.sidebar.selectbox(
        "Select KGram Index File",
        available_indices,
        index=0
    )

    top_k = st.sidebar.slider("Number of results", 1, 20, 5)
    use_wildcards = st.sidebar.checkbox("Enable wildcard search (*)", value=False)

    search_type = st.sidebar.selectbox(
        "Search Method",
        ["KGram (TF-IDF)", "FAISS (Embeddings)", "BallTree (CLIP)"]
    )

    # Load appropriate index based on selection
    index = None
    faiss_index = None
    balltree, balltree_vectors, balltree_ids = None, None, None

    if search_type == "KGram (TF-IDF)":
        with st.spinner("Loading KGram index..."):
            index = load_kgram_index(selected_index)
        if index is None:
            st.stop()
        st.sidebar.success(f"KGram Index loaded ({index.total_docs} docs)")

    elif search_type == "FAISS (Embeddings)":
        with st.spinner("Loading FAISS index..."):
            faiss_index = load_faiss_index()
        if faiss_index is None:
            st.stop()
        st.sidebar.success("FAISS Index loaded")

    elif search_type == "BallTree (CLIP)":
        with st.spinner("Loading VP‑Tree index..."):
            balltree, balltree_vectors, balltree_ids = load_balltree_index()
        if balltree is None:
            st.stop()
        st.sidebar.success(f"VP‑Tree Index loaded ({len(balltree_ids)} images)")

    # Query input
    st.subheader("Enter your search query")
    query = st.text_input("", placeholder="Type your search query here...", key="query_input")

    if query:
        with st.spinner("Searching..."):
            results = []

            if search_type == "KGram (TF-IDF)":
                results = index.compute_tf_idf_query(query, top_k=top_k, use_wildcards=use_wildcards)

            elif search_type == "FAISS (Embeddings)":
                # Using dummy encoding for FAISS as in original code
                query_embedding = np.random.rand(512).astype("float32")
                faiss_results = faiss_index.search(np.array([query_embedding]), top_k)
                results = list(zip(faiss_results[1][0], faiss_results[0][0]))

            elif search_type == "BallTree (CLIP)":
                results = search_balltree(query, balltree, balltree_vectors, balltree_ids, top_k)

            if not results:
                st.warning("No results found. Try a different query.")
            else:
                st.subheader(f"Top {len(results)} Results")

                # Extract document IDs (handle different result formats)
                doc_ids = []
                if search_type == "BallTree (CLIP)":
                    # VP‑Tree returns (image_id, score) as original BallTree did
                    doc_ids = [int(id_.split('_')[1]) for id_, _ in results]
                else:
                    # KGram and FAISS return (doc_id, score)
                    doc_ids = [doc_id for doc_id, _ in results]

                documents = get_documents_by_ids(doc_ids)

                if not documents:
                    st.error("Could not load documents to display results.")
                    return

                # Display results
                for i, (doc_id, score) in enumerate(results, 1):
                    if search_type == "BallTree (CLIP)":
                        # For VP‑Tree, doc_id is actually the image_id string
                        display_id = int(doc_id.split('_')[1])
                    else:
                        display_id = doc_id

                    if display_id in documents:
                        entry = documents[display_id]
                        try:
                            with st.container():
                                st.markdown(f"**Result {i}**")
                                st.write(f"**ID:** {display_id}")
                                st.write(f"**Score:** {score:.4f}")

                                if "caption" in entry[0]:
                                    st.write(f"**Caption:** {entry[0]['caption']}")

                                if "image" in entry[0]:
                                    st.image(
                                        entry[0]["image"],
                                        caption=f"Image for ID {display_id}",
                                        use_container_width=True
                                    )

                                st.divider()
                        except Exception as e:
                            st.error(f"Error displaying result {display_id}: {str(e)}")
                    else:
                        st.warning(f"Document ID {display_id} not found in dataset.")


if __name__ == "__main__":
    main()