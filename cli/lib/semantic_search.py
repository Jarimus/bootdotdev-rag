from sentence_transformers import SentenceTransformer
import numpy as np, pathlib, os
from search_utils import *
from data_handling import load_movies
import regex as re

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
    text = text.strip()
    if text == "":
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
  
  def search(self, query: str, limit: int = 5):
    if self.embeddings is None or self.documents is None:
      raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
    query_embedding = self.generate_embedding(query)
    search_result: list[tuple[float, dict]] = []
    for doc_embedding, doc in zip(self.embeddings, self.documents):
      similarity_score = cosine_similarity(query_embedding, doc_embedding)
      search_result.append((similarity_score, doc))
    search_result.sort(key=lambda item: item[0], reverse=True)
    return search_result[:limit]


def verify_model():
  semantic_search = SemanticSearch()
  print(f"Model loaded: {semantic_search.model}")
  print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def verify_embeddings():
  semantic_search = SemanticSearch()
  movies_list = load_movies()["movies"]
  embeddings = semantic_search.load_or_create_embeddings(movies_list)
  print(f"Number of docs:   {len(movies_list)}")
  print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(text: str):
  semantic_search = SemanticSearch()
  embedding = semantic_search.generate_embedding(text)
  print(f"Text: {text}")
  print(f"First 3 dimensions: {embedding[:3]}")
  print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query: str):
  semantic_search = SemanticSearch()
  embedding = semantic_search.generate_embedding(query)
  print(f"Query: {query}")
  print(f"First 5 dimensions: {embedding[:5]}")
  print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search_command(query: str, limit: int = 5):
  semantic_search = SemanticSearch()
  movies = load_movies()["movies"]
  semantic_search.load_or_create_embeddings(movies)
  search_result = semantic_search.search(query, limit)
  for i, movie in enumerate(search_result):
    abbreviated_description = " ".join(movie[1]['description'].split()[:20])
    print(f"{i+1}: {movie[1]['title']} (score: {movie[0]})\n{abbreviated_description}...")

def fixed_size_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = 0) -> list[str]:
  words: list[str] = text.split()
  chunks: list[str] = []
  i = 0
  while i < len(words):
    if i == 0:
      chunks.append(" ".join(words[ i : i+chunk_size ]))
    else:
      chunks.append(" ".join(words[ i-overlap : i+chunk_size ]))
    i += chunk_size
  return chunks

def semantic_chunking(text: str, max_chunk_size: int = MAX_SEMANTIC_CHUNK_SIZE, overlap: int = 0) -> list[str]:
  sentences = re.split(r"(?<=[.!?])\s+", text)
  chunks: list[str] = []
  i = 0
  while i < len(sentences):
    if i == 0:
      chunks.append(" ".join(sentences[ i : i+max_chunk_size ]))
    else:
      chunks.append(" ".join(sentences[ i-overlap : i+max_chunk_size ]))
    i += max_chunk_size
  return chunks