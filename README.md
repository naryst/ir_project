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

The project uses the DCI (Densely Captioned Images) dataset from Meta, which provides images with detailed captions. The dataset usage and processing code is referenced from Meta's implementation. Detailed instructions for downloading and setting up the DCI dataset are provided in the 'Prerequisites and Initial Setup' section below.

## Getting Started

This section guides you through setting up the project environment and downloading the necessary data.

### Prerequisites and Initial Setup

Follow these steps to prepare your environment:

1.  **Create Conda Environment:**
    Open your terminal and run the following command to create a conda environment named `ir_project` with Python 3.10:
    ```bash
    conda create -n ir_project python=3.10
    ```
    Activate the environment:
    ```bash
    conda activate ir_project
    ```

2.  **Install Core DCI Dependencies:**
    Navigate to the `DCI/dataset` directory from the project root and install the DCI dataset dependencies:
    ```bash
    cd DCI/dataset
    pip install -e .
    cd ../..
    ```
    *(Note: `cd ../..` returns you to the project root from `DCI/dataset/`.)*

3.  **Install Additional Project Dependencies:**
    Install further requirements using the `environment.yml` file located in the project root:
    ```bash
    conda env update -f environment.yml
    ```

4.  **Configure Data Installation Path:**
    The DCI dataset scripts (and by extension, this project) rely on a configuration file to know where to find and store dataset files.
    *   **File Location:** This configuration file is `config.yaml`, located at `DCI/dataset/config.yaml` relative to the project root.
    *   **Action Required:** You **must** edit this `config.yaml` file. Specifically, you need to set the `data_dir` key to the absolute path of the directory on your system where you want to store the DCI dataset and related files. For example:
        ```yaml
        data_dir: /mnt/large_storage/my_datasets
        ```
        Replace `/mnt/large_storage/my_datasets` with your actual desired path.
    *   **Expected Structure:** The DCI scripts will then create a subdirectory structure within your specified `data_dir`. This project expects the DCI images and annotations to be ultimately accessible at `<data_dir>/densely_captioned_images/`. For instance, if `data_dir` is `/mnt/large_storage/my_datasets`, the images should eventually reside in `/mnt/large_storage/my_datasets/densely_captioned_images/photos/`.
    *   **Initial Creation:** If `DCI/dataset/config.yaml` does not exist, running a DCI script (like the download script in the next step) might prompt you to define data saving locations, which will then create this file. However, it's recommended to check and edit it manually to ensure correctness.

5.  **Download the DCI Dataset:**
    The DCI dataset consists of text/annotations and images.

    *   **Text and Annotations:**
        From the project root, run the following script to download the DCI metadata (captions, etc.):
        ```bash
        python DCI/dataset/densely_captioned_images/dataset/scripts/download.py
        ```
        This script uses the `data_dir` from `DCI/dataset/config.yaml` to determine where to download and place files like `dci.tar.gz`.

    *   **Images:**
        The images for the DCI dataset are sourced from the SA-1B dataset.
        1.  Go to the [SA-1B dataset page](https://ai.meta.com/datasets/segment-anything-downloads/) and accept their license agreement.
        2.  From the downloads provided, download **only the `sa_000138.tar` archive**. This specific archive contains the images required for the DCI dataset.
        3.  Extract the `sa_000138.tar` archive. This will produce a collection of individual image files (e.g., `.jpg` files).
        4.  Place all these extracted image files **directly** into the `photos` subdirectory within your DCI data path. The target directory should be: `<your_data_dir>/densely_captioned_images/photos/`. Create the `photos` subdirectory if it doesn't exist. For example, if your `data_dir` in `config.yaml` is `/mnt/large_storage/my_datasets`, then the images should be placed in `/mnt/large_storage/my_datasets/densely_captioned_images/photos/`.
        For further details on DCI data handling, you can consult the "Setup" section in `DCI/README.md`.

6.  **Ollama for Prompt Refinement (Optional but Recommended):**
    For the prompt refinement feature used in some parts of this project, you need to have Ollama installed and running with the `gemma3:4b` model.
    *   Install Ollama from [ollama.ai](https://ollama.ai/).
    *   Pull the `gemma3:4b` model:
        ```bash
        ollama pull gemma3:4b
        ```
    *   Start the Ollama server (this command may need to be run in a separate terminal session):
        ```bash
        ollama serve
        ```
        Ensure the `gemma3:4b` model is available through Ollama. Once the server is running, it can serve any models you have pulled.

With these steps completed, you should have the necessary environment, dependencies, and data to run the various components of this project.

### Building and Running the K-gram Index

Once your environment is set up and the data is downloaded, you can build the K-gram index. This index is used for text-based search using k-grams and TF-IDF scoring.

1.  **Navigate to the K-gram Index Directory:**
    From the project root, change to the `kgram_index` directory:
    ```bash
    cd kgram_index
    ```

2.  **Build the Index:**
    Run the `build_index.py` script. You need to specify the size of the k-grams using the `--k` parameter. For example, to use 3-grams (trigrams):
    ```bash
    python build_index.py --k=3
    ```
    Adjust the value of `k` as needed for your experiments.

3.  **Index Output:**
    The script will process the dataset and save the generated index as a `.pkl` file (e.g., `kgram_index_k3.pkl`) in the `kgram_index/data/` directory. This file will be used by the search components that rely on the K-gram index.

4.  **Testing and Demonstration:**
    To test the index or see an example of its usage, you can explore the Jupyter notebook located at `kgram_index/test_index.ipynb`. This notebook provides code examples for loading the index and performing searches.

When you are finished, navigate back to the project root directory by running:
```bash
cd ..
```

### Building and Running the Dense Vector Search

This component uses neural embeddings (e.g., from JINA-CLIP) to represent both text and images in a shared vector space, enabling semantic search. The improved version (`dense_index_v2`) utilizes a FAISS index for efficient similarity search.

1.  **Navigate to the Dense Vector Search Directory:**
    From the project root, change to the `dense_index_v2` directory:
    ```bash
    cd dense_index_v2
    ```

2.  **Generate Embeddings:**
    Run the `get_jina_embeddings.py` script to process the dataset and generate embeddings for images and captions. These embeddings will be saved as torch tensors.
    ```bash
    python get_jina_embeddings.py
    ```
    *(Ensure your DCI dataset is correctly configured and accessible as per the "Prerequisites and Initial Setup" section, as this script will need to read the data.)*

3.  **Build the FAISS Index:**
    Once the embeddings are generated, build the FAISS index by running:
    ```bash
    python faiss_index.py
    ```
    This script will create an index file (e.g., `faiss_jina.index`) that the demo application will use for searching.

4.  **Run the Interactive Demo:**
    To explore the Dense Vector Search capabilities, run the Streamlit demo application:
    ```bash
    streamlit run demo.py
    ```
    This will typically open the demo in your web browser.

When you are finished, you can stop the Streamlit application (usually Ctrl+C in the terminal) and navigate back to the project root directory by running:
```bash
cd ..
```

### Building and Running the Image Segmentation Pipeline

The Image Segmentation Pipeline allows for a more granular search by segmenting images, generating descriptions for these segments, and then searching over these localized descriptions. This pipeline can use custom images.

1.  **Navigate to the Segmentation Pipeline Directory:**
    From the project root, change to the `segmentation_pipeline` directory:
    ```bash
    cd segmentation_pipeline
    ```

2.  **Load Images:**
    Run the `retriever.py` script to load your custom images into the pipeline. Ensure your images are accessible by the script (e.g., by placing them in an expected directory or configuring the script).
    ```bash
    python retriever.py
    ```

3.  **Generate Segmentation Masks:**
    Execute `mask_images.py` to generate segmentation masks for the loaded images.
    ```bash
    python mask_images.py
    ```

4.  **Generate Descriptions for Segments:**
    Run `generate_descriptions.py` to create textual descriptions for each image segment.
    ```bash
    python generate_descriptions.py
    ```
    *(Note: This step might require an AI model for description generation. If using a model served by Ollama, ensure Ollama with `gemma3:4b` (or the relevant model) is running, as detailed in the "Prerequisites and Initial Setup" section.)*

5.  **Embed Segment Descriptions:**
    Create embeddings for the generated textual descriptions of segments by running:
    ```bash
    python embed_data.py
    ```

6.  **Build FAISS Index for Segments:**
    Build a FAISS index from the segment description embeddings:
    ```bash
    python build_faiss.py
    ```
    This index will be used for searching over the specific image regions.

7.  **Run the Interactive Demo:**
    Launch the Streamlit demo for the segmentation pipeline:
    ```bash
    streamlit run demo.py
    ```
    This will allow you to perform searches that leverage the segmented image data.

When you are finished, you can stop the Streamlit application (usually Ctrl+C in the terminal) and navigate back to the project root directory by running:
```bash
cd ..
```

### Running the Main Streamlit Demo

This project includes a main Streamlit demonstration application located in the `demo/` directory. This demo provides a unified interface to interact with the different search methodologies implemented. Ensure that the respective indexes (K-gram, Dense Vector) are built before trying to use them in the main demo.

1.  **Navigate to the Main Demo Directory:**
    From the project root, change to the `demo` directory:
    ```bash
    cd demo
    ```

2.  **Install Demo-Specific Dependencies:**
    If you haven't installed dependencies for this specific demo yet, or to ensure all are up to date, run:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure your conda environment `ir_project` is active.)*

3.  **Run the Main Demo Application:**
    Launch the Streamlit application:
    ```bash
    streamlit run streamlit_demo.py
    ```
    This will open the main demo interface in your web browser, allowing you to select and test different search backends.

After you are done with the demo, you can stop the Streamlit application (usually Ctrl+C in the terminal) and navigate back to the project root:
```bash
cd ..
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