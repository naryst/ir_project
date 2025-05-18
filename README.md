# Information Retrieval Project - Multimodal Search Engine

This project implements a multimodal search engine for images and text, developed as part of the Information Retrieval course at Innopolis University. The system allows users to search for images using text queries through different search methodologies.

## Project Overview

The project implements three main search approaches:

1. **K-gram Index with TF-IDF**: A text-based search engine that breaks down text into k-grams and uses TF-IDF scoring to match queries to captions.

2. **Dense Vector Search**: Uses neural embeddings to encode both text and images into the same vector space, allowing for semantic search.

3. **Image Segmentation Pipeline**: Segments images and generates descriptions for specific parts of images, enabling more precise and localized search.

## Project Structure

The project is organized into several key components:

### 1. K-gram Index
- Located in the `kgram_index/` directory
- Implements a k-gram based index with TF-IDF scoring
- Supports flexible k values and wildcard searches
- Main files:
  - `build_index.py`: Core implementation of the k-gram index
  - `test_index.ipynb`: Notebook for testing the index

### 2. Dense Vector Search
- Located in two directories:
  - `dense_index/`: Initial implementation
  - `dense_index_v2/`: Improved version with optimizations
- Uses embeddings from neural models (JINA-CLIP and ColQwen) to encode queries and images
- Supports different index types:
  - FAISS index for fast vector search
  - Ball Tree index for nearest neighbor search
- Main files:
  - `demo.py`: Interactive demo application
  - `faiss_index.py`: FAISS index implementation
  - `ball_tree.py`: Ball Tree index implementation
  - `colqwen_emb.py`/`siglig_embeddings.py`: Embedding generation

### 3. Segmentation Pipeline
- Located in the `segmentation_pipeline/` directory
- Segments images and generates descriptions for specific regions
- Creates a search index for these localized descriptions
- Main files:
  - `demo.py`: Interactive demo application
  - `mask_images.py`: Image segmentation implementation
  - `generate_descriptions.py`: Description generation for segments
  - `embed_data.py`: Embedding generation for segments

### 4. Demo Applications
- Located in the `demo/` directory
- Streamlit-based web interfaces for the search engines
- Allows interactive querying and result visualization

## Dataset

The project uses the DCI (Densely Captioned Images) dataset from Meta, which provides images with detailed captions. The dataset usage and processing code is referenced from Meta's implementation.

## Getting Started

### Prerequisites
- Python 3.10+
- Required packages (see `REPRODUCE.md` for detailed setup)

### Setup
Follow the instructions in `REPRODUCE.md` to set up the environment and download the dataset“

Important remark:
for prompt refinement you need to have ollama with gemma3:4b installed and running.
```bash
ollama serve gemma3:4b
```

## Technologies Used

- **FAISS**: For efficient similarity search
- **PyTorch**: For neural network models
- **Streamlit**: For interactive demo interfaces
- **Transformer models**: JINA-CLIP and ColQwen for text/image embeddings
- **Ollama**: For AI-assisted query refinement

## Research Contributions

This project explores different approaches to multimodal search and compares their effectiveness:

1. Traditional text search with k-grams and TF-IDF
2. Neural embedding-based search with different models
3. Segmentation-based search for more localized results

The implementation demonstrates how these approaches can be combined to create a comprehensive search engine for images and text.