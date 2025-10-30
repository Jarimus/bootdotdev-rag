from lib.semantic_search import SemanticSearch, cosine_similarity
from data_handling import *
from search_utils import *
import numpy as np
import regex as re
from pathlib import Path

class ChunkedSemanticSearch(SemanticSearch):
  def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
    print("--- Initialize chunked semantic search ---")
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
    Path(CACHE_DIR).mkdir(exist_ok=True)
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
        self.chunk_metadata = json.load(file)["chunks"]
      return self.chunk_embeddings
    else:
      print("Chunk embeddings or metadata not found in cache. Building...")
      return self.build_chunk_embeddings(documents)
    
  def search_chunks(self, query: str, limit: int = 10) -> list:
    if self.chunk_embeddings is None or self.chunk_metadata is None:
      print("Chunk embeddings or metadata not found. Exiting...")
      return []
    query_embedding = self.generate_embedding(query)
    chunk_scores: list[dict] = []
    for chunk_emb, chunk_meta in zip(self.chunk_embeddings, self.chunk_metadata):
      score = cosine_similarity(query_embedding, chunk_emb)
      chunk_scores.append({
        "chunk_idx": chunk_meta['chunk_idx'],
        "movie_idx": chunk_meta['movie_idx'],
        "score": score
      })
    movie_scores: dict[str, float] = {}
    for c_score in chunk_scores:
      if c_score['movie_idx'] not in movie_scores or c_score['score'] > movie_scores[c_score['movie_idx']]:
        movie_scores[c_score['movie_idx']] = c_score['score']
    movie_scores_sorted: list[tuple[str, float]] = sorted(movie_scores.items(), key=lambda item: item[1], reverse=True)
    movie_scores_sorted = movie_scores_sorted[:limit]
    final_result: list[dict] = []
    for score in movie_scores_sorted:
      id = score[0]
      score = score[1]
      doc = self.document_map[id]
      final_result.append({
      "id": id,
      "title": doc["title"],
      "document": doc["description"][:100],
      "score": round(score, SCORE_PRECISION),
      "metadata": {}
    })
    return final_result


def semantic_chunking(text: str, max_chunk_size: int = MAX_SEMANTIC_CHUNK_SIZE, overlap: int = 0) -> list[str]:
  text = text.strip()
  if text == "":
    return []
  sentences = re.split(r"(?<=[.!?])\s+", text)
  if len(sentences) == 1 and not sentences[0].endswith(("!", ".", "?")):
    return [text]
  stripped_sentences: list[str] = []
  for s in sentences:
    s = s.strip()
    if s != "":
      stripped_sentences.append(s.strip())
  chunks: list[str] = []
  i = 0
  while i < len(stripped_sentences) - overlap:
    chunks.append(" ".join(stripped_sentences[ i : i+max_chunk_size ]))
    i += max_chunk_size - overlap
  return chunks

def embed_chunks_command() -> np.ndarray:
  movies = load_movies()["movies"]
  CSS = ChunkedSemanticSearch()
  return CSS.load_or_create_chunk_embeddings(movies)

def search_chunked_command(query: str, limit: int = 10) -> list[dict]:
  movies = load_movies()["movies"]
  CSS = ChunkedSemanticSearch()
  CSS.load_or_create_chunk_embeddings(movies)
  return CSS.search_chunks(query, limit)