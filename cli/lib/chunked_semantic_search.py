from lib.semantic_search import SemanticSearch
from data_handling import *
from search_utils import *
import numpy as np
import regex as re
from pathlib import Path

class ChunkedSemanticSearch(SemanticSearch):
  def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
    super().__init__(model_name)
    self.chunk_embeddings = None
    self.chunk_metadata = None

  def build_chunk_embeddings(self, documents: list[dict]):
    self.documents = documents
    chunk_list: list[str] = []
    chunk_metadata: list[dict] = []
    for doc in documents:
      self.document_map[doc["id"]] = doc
      if not doc['description'] == "":
        chunks = semantic_chunking(doc['description'], 4, 1)
        chunk_list.extend(chunks)
        for i in range(len(chunks)):
          chunk_metadata.append({
            "movie_idx": doc['id'],
            "chunk_idx": i,
            "total_chunks": len(chunks)
          })
    self.chunk_embeddings = self.model.encode(chunk_list, show_progress_bar=True)
    self.chunk_metadata = chunk_metadata
    with open(Path(CACHE_DIR, CHUNK_EMBEDDINGS_FILE), "wb") as file:      
      np.save(file, self.chunk_embeddings)
      print("Chunk embeddings written to cache.")
    with open(Path(CACHE_DIR, CHUNK_METADATA_FILE), "w") as file:
      json.dump({"chunks": chunk_metadata, "total_chunks": len(chunk_list)}, file, indent=2)
      print("Chunk metadata written to cache.")
    return self.chunk_embeddings
  
  def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
    self.documents = documents
    for doc in documents:
      self.document_map[doc["id"]] = doc
    if Path(CACHE_DIR, CHUNK_EMBEDDINGS_FILE).exists() and Path(CACHE_DIR, CHUNK_METADATA_FILE).exists():
      print("Chunk embeddings and metadata in cache. Loading...")
      with open(Path(CACHE_DIR, CHUNK_EMBEDDINGS_FILE), "rb") as file:
        self.chunk_embeddings = np.load(file)
      with open(Path(CACHE_DIR, CHUNK_METADATA_FILE), "r") as file:
        self.chunk_metadata = json.load(file)
      return self.chunk_embeddings
    else:
      print("Chunk embeddings or metadata not found in cache. Building...")
      return self.build_chunk_embeddings(documents)
    
  def search_chunks(self, query: str, limit: int = 10):
    pass


def semantic_chunking(text: str, max_chunk_size: int = MAX_SEMANTIC_CHUNK_SIZE, overlap: int = 0) -> list[str]:
  sentences = re.split(r"(?<=[.!?])\s+", text)
  chunks: list[str] = []
  i = 0
  while i < len(sentences) - overlap:
    chunks.append(" ".join(sentences[ i : i+max_chunk_size ]))
    i += max_chunk_size - overlap
  return chunks

def embed_chunks_command() -> np.ndarray:
  movies = load_movies()["movies"]
  CSS = ChunkedSemanticSearch()
  return CSS.load_or_create_chunk_embeddings(movies)