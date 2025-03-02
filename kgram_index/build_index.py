from densely_captioned_images.dataset.impl import get_complete_dataset_with_settings
import numpy as np
from collections import defaultdict
import re
import os
import pickle
import argparse
from tqdm import tqdm

COMPLETE_DATASET_SIZE = 7599
SPLIT = "train"
DATA_FOLDER = "data"  # Folder to save index files


class KGramIndex:
    """
    Builds a k-gram index from a dataset of captions using TF-IDF.
    """

    def __init__(self, k):
        self.k = k
        self.kgram_index = defaultdict(set)  # k-gram -> {doc_ids}
        self.documents = defaultdict(
            lambda: defaultdict(int)
        )  # doc_id -> {kgram -> frequency}
        self.doc_count = defaultdict(int)  # k-gram -> number of documents containing it
        self.total_docs = 0  # Total number of documents

    def generate_kgrams(self, word):
        """Generate k-grams for a word with padding."""
        word = word.lower()
        padded_word = f"${word}$"
        return {
            padded_word[i : i + self.k] for i in range(len(padded_word) - self.k + 1)
        }

    def add_document(self, document_id, text):
        """Add a document (caption) to the TF-IDF model and update the k-gram index."""
        words = re.findall(r"\w+", text.lower())  # Tokenize words
        kgram_set = set()

        for word in words:
            kgrams = self.generate_kgrams(word)
            kgram_set.update(kgrams)

            for kgram in kgrams:
                self.kgram_index[kgram].add(document_id)
                self.documents[document_id][kgram] += 1  # Store frequency

        # Count unique k-grams per document
        for kgram in kgram_set:
            self.doc_count[kgram] += 1

        self.total_docs += 1

    def compute_tf_idf(self, kgram, doc_id):
        """Compute TF-IDF for a k-gram in a specific document."""
        tf = self.documents[doc_id].get(kgram, 0)
        if tf == 0:
            return 0
        tf = 1 + np.log(tf)  # Log-normalized TF
        idf = (
            np.log((self.total_docs + 1) / (1 + self.doc_count[kgram])) + 1
        )  # Smoothed IDF
        return tf * idf

    def compute_tf_idf_query(self, query, top_k=10):
        """Compute TF-IDF scores for a query."""

        if top_k < 1:
            top_k = COMPLETE_DATASET_SIZE
        words = re.findall(r"\w+", query.lower())
        query_kgrams = set()

        for word in words:
            query_kgrams.update(self.generate_kgrams(word))

        tf_idf_scores = defaultdict(float)

        # Compute scores for each document
        for kgram in query_kgrams:
            for doc_id in self.kgram_index.get(kgram, []):
                tf_idf_scores[doc_id] += self.compute_tf_idf(kgram, doc_id)

        return sorted(tf_idf_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]  # Rank results

    def save(self, filename=None):
        """
        Save the index to a file in the data folder.
        If filename is not provided, a default name with k-value will be used.
        """
        if filename is None:
            filename = f"K={self.k}_index.pkl"

        # Ensure data directory exists
        os.makedirs(DATA_FOLDER, exist_ok=True)
        path = os.path.join(DATA_FOLDER, filename)

        # Convert defaultdict(set) to dict of lists for serialization
        serializable_kgram_index = {k: list(v) for k, v in self.kgram_index.items()}

        # Convert defaultdict(lambda: defaultdict(int)) to dict of dicts
        serializable_documents = {k: dict(v) for k, v in self.documents.items()}

        # Create a serializable representation of the index
        index_data = {
            "k": self.k,
            "kgram_index": serializable_kgram_index,
            "documents": serializable_documents,
            "doc_count": dict(self.doc_count),
            "total_docs": self.total_docs,
        }

        with open(path, "wb") as f:
            pickle.dump(index_data, f)

        print(f"Index saved to {path}")
        return path

    @classmethod
    def load(cls, path=None, k=None):
        """
        Load an index from a file.
        If filename is not provided but k is, it will look for the default filename with that k-value.
        """

        if path is None:
            raise ValueError("Path must be provided")


        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")

        with open(path, "rb") as f:
            index_data = pickle.load(f)

        # Create a new instance
        index = cls(index_data["k"])

        # Restore the index data
        index.total_docs = index_data["total_docs"]
        index.doc_count = defaultdict(int, index_data["doc_count"])

        # Convert dict of lists back to defaultdict(set)
        for k, v in index_data["kgram_index"].items():
            index.kgram_index[k] = set(v)

        # Convert dict of dicts back to defaultdict(lambda: defaultdict(int))
        for doc_id, kgram_freqs in index_data["documents"].items():
            for kgram, freq in kgram_freqs.items():
                index.documents[doc_id][kgram] = freq

        print(f"Index loaded from {path}")
        return index


def load_dataset(start_index=0, end_index=COMPLETE_DATASET_SIZE):
    print(f"Loading dataset from {start_index} to {end_index}")
    return get_complete_dataset_with_settings(
        split=SPLIT, start_index=start_index, end_index=end_index
    )


def load_chunked_dataset(chunk_size=500, documents_to_load=None):
    if documents_to_load is None:
        documents_to_load = COMPLETE_DATASET_SIZE
    for i in range(0, documents_to_load, chunk_size):
        print(f"Loading chunk {i} of {documents_to_load}")
        if i + chunk_size > documents_to_load:
            chunk_size = documents_to_load - i
        data_chunk = load_dataset(i, i + chunk_size)
        yield data_chunk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--documents_to_load", type=int, default=COMPLETE_DATASET_SIZE)
    args = parser.parse_args()

    kgram_index = KGramIndex(k=args.k)
    chunk_size = args.chunk_size
    cur_document_id = 0
    for data_chunk in load_chunked_dataset(chunk_size, args.documents_to_load):
        print("Adding documents to index")
        for entry in tqdm(data_chunk):
            # use only caption for the whole image, not segments
            total_entry = entry[0]["caption"]
            kgram_index.add_document(cur_document_id, total_entry)
            cur_document_id += 1
    print("Saving index")
    kgram_index.save(f"K={args.k}_index.pkl")
