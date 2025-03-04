# KGram Index Streamlit Demo

This is a Streamlit demo application that showcases the KGram index search functionality. It allows users to search through a dataset of captions using a k-gram based index with TF-IDF scoring.

## Prerequisites

- Python 3.7+
- Streamlit
- The KGram index must be built and available in the `kgram_index/data` directory

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Make sure you have built the KGram index. If not, run:
```bash
cd kgram_index
python build_index.py --k 3
```

## Running the Demo

To run the Streamlit demo:

```bash
cd demo
streamlit run streamlit_demo.py
```

This will start a local web server and open the demo in your default web browser.

## Features

- Search through captions using a text query
- Adjust the number of results to display
- View the relevance score for each result
- See the original caption for each result
- View images (if available) associated with the captions

## How It Works

1. The demo loads a pre-built KGram index (default k=3)
2. When a user enters a query, it's processed to extract k-grams
3. The index is searched to find documents containing those k-grams
4. Results are ranked using TF-IDF scoring
5. The top-k results are displayed with their captions and scores