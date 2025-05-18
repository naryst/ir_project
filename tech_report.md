# Technical Report: Implementation of a Multimodal Image Search System

## Executive Summary

This report details the development of a comprehensive multimodal image search system capable of retrieving images based on text queries. The system implements three distinct search methodologies: k-gram indexing with TF-IDF, dense vector embedding search, and segmentation-based search. The project successfully demonstrates how different information retrieval techniques can be combined to create an effective image search engine.

## System Architecture

The system is built as a modular framework with three main components:

1. **K-gram Text Search Engine**
2. **Dense Vector Search Engine**
3. **Image Segmentation and Description Pipeline**

Each component implements a different approach to the image search problem and can be used independently or in combination.

## Implementation Details

### 1. K-gram Index with TF-IDF

The k-gram index implementation breaks down text into character-level n-grams and uses TF-IDF scoring to match queries to image captions.

**Key features:**
- Custom implementation of a k-gram index that supports variable k values
- TF-IDF scoring with log normalization for term frequency and smoothed IDF
- Support for wildcard searches (using * character)
- Efficient index serialization and loading mechanisms
- Optimized for processing large datasets in chunks

The implementation processes captions from the DCI dataset and builds an inverted index mapping k-grams to document IDs. During query time, the system computes TF-IDF scores between the query and indexed documents to retrieve the most relevant results.

### 2. Dense Vector Search

The dense vector search component in version 2 leverages neural models to encode image segments and their descriptions into a common embedding space, enabling fine-grained semantic search across modalities.
**Key implementations:**
- Segment-level processing that embeds individual regions of images with their associated descriptions
- Integration with state-of-the-art embedding models (JINA-CLIP v2)
- FAISS index for efficient similarity search in high-dimensional spaces
- Processing of densely captioned image segments with detailed spatial information
- Padding mechanisms to handle variable numbers of segments per image

The system generates embeddings by first segmenting images into meaningful regions using mask data. For each segment, it extracts both the visual content and its textual description, then encodes both into a common embedding space using JINA-CLIP. These segment-level embeddings are stored in FAISS indices to enable fast similarity search. During query time, the user's text query is embedded into the same space, and the system retrieves the most semantically similar image segments, allowing for precise localization of search results within images.

### 3. Segmentation Pipeline

The segmentation pipeline extends the system's capabilities by enabling more precise and localized search within images.

**Components:**
1. **Image Segmentation** - Using Meta's Segment Anything Model (SAM) to automatically generate masks for distinct objects and regions within images
2. **Description Generation** - Leveraging Google's Gemma 3-4B-it model to generate detailed descriptions for each segmented region
3. **Region-Based Embedding** - Creating embeddings for each segment and its description
4. **Localized Search** - Enabling search that can match queries to specific parts of images

The pipeline first segments images into meaningful regions, then generates descriptions for each segment. These segment-description pairs are embedded into vector space and indexed. During search, the system can retrieve not just relevant images but also highlight the specific regions within those images that match the query.

## Technologies Used

The implementation leverages several key technologies:

1. **PyTorch** - For neural network model operations and tensor processing
2. **FAISS** - For efficient similarity search in high-dimensional embedding spaces
3. **Transformer Models**:
   - JINA-CLIP - For multimodal embedding generation
   - ColQwen - For alternative embedding generation
   - Gemma 3-4B-it - For generating detailed image segment descriptions
4. **Segment Anything Model (SAM)** - For automatic image segmentation
5. **Streamlit** - For creating interactive web interfaces for the search engines
6. **Ollama** - For local AI-powered query refinement

## Performance and Evaluation

The system demonstrates different trade-offs across the three approaches:

1. **K-gram Index**:
   - Pros: Fast indexing and retrieval, works well for exact term matching
   - Cons: Limited semantic understanding, dependent on caption quality

2. **Dense Vector Search**:
   - Pros: Strong semantic understanding, can match concepts even with different terminology
   - Cons: Computationally intensive for large datasets, requires GPU for optimal performance

3. **Segmentation Pipeline**:
   - Pros: Enables precise localization within images, more detailed search capability
   - Cons: Most resource-intensive, dependent on quality of segmentation and description generation

## Innovations and Research Contributions

The project makes several notable contributions:

1. **Integration of Multiple Search Paradigms** - Successfully combining traditional information retrieval techniques with modern neural approaches
2. **Segmentation-Based Search** - Implementing a novel approach that enables search at the sub-image level
3. **Modular Architecture** - Designing a system where components can be used independently or combined
4. **Query Refinement** - Using AI (via Ollama) to help users refine their search queries

## Deployment and Usage

The system is implemented with user-friendly interfaces through Streamlit, allowing for interactive querying and result visualization. Each component has its own demo application, and the system includes comprehensive documentation for setup and deployment.

## Conclusion

The implemented image search system successfully demonstrates the application of various information retrieval techniques to the multimodal search domain. By combining traditional indexing methods with modern neural approaches and innovative segmentation techniques, the system provides a powerful and flexible framework for image search that can be extended and refined for specific use cases.

The exclusion of the DCI (Data of images with descriptions) folder from the implementation scope is noted, as this data is separately provided and referenced from Meta's implementation. 