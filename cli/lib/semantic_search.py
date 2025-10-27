from sentence_transformers import SentenceTransformer
import numpy as np, pathlib, os
from search_utils import CACHE_DIR
from data_handling import load_movies
from tqdm import tqdm
from time import sleep

class SemanticSearch:

  def __init__(self) -> None:
    print("Loading model...")
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model ready!")
    self.embeddings = None
    self.embeddings_filepath = os.path.join(CACHE_DIR, "movie_embeddings.npy")
    self.documents = None
    self.document_map = {}

  def generate_embedding(self, text: str):
    if text.strip() == "":
      raise ValueError("SemanticSearch - generate embedding: text input is empty")
    embeddings = self.model.encode([text])
    return embeddings[0]
  
  def build_embeddings(self, documents: list[dict]):
    self.documents = documents
    string_docs = []
    for doc in documents:
      self.document_map[doc["id"]] = doc
      string_docs.append(f"{doc['title']}: {doc['description']}")
    print("Encoding embeddings...")
    self.embeddings = self.model.encode(string_docs, show_progress_bar=True)
    self.save_embeddings()
    return self.embeddings
  
  def save_embeddings(self):
    if self.embeddings is None:
      return print("No embeddings to save")
    pathlib.Path(CACHE_DIR).mkdir(exist_ok=True)
    with open(self.embeddings_filepath, "wb") as file:
      print(f"Saving embeddings to {self.embeddings_filepath}")
      np.save(file, self.embeddings)
  
  def load_or_create_embeddings(self, documents: list[dict]):
    self.documents = documents
    print("Building doc map...")
    for doc in documents:
      self.document_map[doc["id"]] = doc
    if pathlib.Path(self.embeddings_filepath).exists():
      print(f"Loading embeddings from {self.embeddings_filepath}...")
      self.embeddings = np.load(self.embeddings_filepath)
      if len(self.embeddings) == len(self.documents):
        return self.embeddings
      print("Cache mismatch. Rebuilding cache...")
    return self.build_embeddings(documents)

def verify_model():
  semantic_search = SemanticSearch()
  print(f"Model loaded: {semantic_search.model}")
  print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text: str):
  semantic_search = SemanticSearch()
  embedding = semantic_search.generate_embedding(text)
  print(f"Text: {text}")
  print(f"First 3 dimensions: {embedding[:3]}")
  print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
  semantic_search = SemanticSearch()
  movies_list = load_movies()["movies"]
  embeddings = semantic_search.load_or_create_embeddings(movies_list)
  print(f"Number of docs:   {len(movies_list)}")
  print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")