from text_handling import process_string
from data_handling import load_movies
from collections import Counter
from search_utils import BM25_K1, BM25_B, CACHE_DIR
import pickle, pathlib, math, shutil, os

class InvertedIndex:

  def __init__(self) -> None:
    self.index_filepath = os.path.join(CACHE_DIR, "index.pkl")
    self.index: dict[str, set[int]] =  {}
    self.docmap_filepath = os.path.join(CACHE_DIR, "docmap.pkl")
    self.docmap: dict[int, dict] = {}
    self.doc_lengths: dict = {}
    self.doc_lengths_filepath = os.path.join(CACHE_DIR, "doc_lengths.pkl")
    self.tf_filepath = os.path.join(CACHE_DIR, "term_frequencies.pkl")
    self.term_frequencies: dict[int, Counter[str]] = {}
  
  def __single_term_to_token(self, term: str) -> str:
    token = process_string(term)
    if len(token) > 1:
      raise ValueError("command supports single tokens only")
    return token[0]
  
  def __get_avg_doc_length(self) -> float:
    if len(self.doc_lengths) == 0:
      return 0.0
    return (sum(self.doc_lengths.values()))/len(self.doc_lengths)

  def __add_document(self, doc_id: int, text: str):
    tokens = process_string(text)
    # Add tokens to index and increment term frequencies
    for token in tokens:
      # index
      if token not in self.index:
        self.index[token] = set()
      self.index[token].add(doc_id)
      # term frequencies
      if doc_id not in self.term_frequencies:
        self.term_frequencies[doc_id] = Counter()
      self.term_frequencies[doc_id].update([token])
    # document lengths
    self.doc_lengths[doc_id] = len(tokens)

  def get_documents(self, token: str) -> list[int]:
    return sorted(self.index.get(token, []))

  def get_tf(self, doc_id: int, term: str) -> int:
    token = self.__single_term_to_token(term)
    return self.term_frequencies[doc_id][token]
  
  def get_df(self, term: str) -> int:
    token = self.__single_term_to_token(term)
    return len(self.get_documents(token))
  
  def get_idf(self, term: str) -> float:
    token = self.__single_term_to_token(term)
    return math.log((len(self.docmap) + 1) / (len(self.index.get(token, [])) + 1))
  
  def get_bm25_idf(self, term: str) -> float:
    df = self.get_df(term)
    N = len(self.docmap)
    return math.log((N - df + 0.5) / (df + 0.5) + 1)
  
  def get_tfidf(self, doc_id: int, term: str) -> float:
    return self.get_tf(doc_id, term) * self.get_idf(term)
  
  def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    tf = self.get_tf(doc_id, term)
    length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
    return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    # return (tf * (k1 + 1)) / (tf + k1)

  def get_bm25score(self, doc_id: int, term: str):
    return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)
  
  def bm25_search(self, query: str, limit: int = 5):
    # tokenize the query
    search_tokens = process_string(query)
    # Calculate bm25 score for each document
    bm25_scores: dict[int, float] = {}
    for doc_id in self.docmap.keys():
      score = 0
      for token in search_tokens:
        score += self.get_bm25score(doc_id, token)
      bm25_scores[doc_id] = score
    # Sort the scores, descending
    sorted_scores: list[tuple[int, float]] = sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)
    # Pick the top results by limit
    top_scores = sorted_scores[:limit]
    return top_scores

  def build(self):
    # Only build if cache dir does not exist
    if pathlib.Path(CACHE_DIR).exists():
      print("Cache directory already exists, loading cache...")
      try:
        self.load()
        return
      except FileNotFoundError:
        print("Cache file missing. Building cache.")
        return
    # First load data into memory
    movies = load_movies()
    for i, m in enumerate(movies["movies"]):
      self.__add_document(int(m["id"]), f"{m['title']} {m['description']}")
      self.docmap[int(m["id"])] = m
      print(f"Building... ({i}/{len(movies["movies"])})")
    # Save to file
    self.save()

  def save(self):
    # Ensure directory exists
    pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    # Write index cache
    with open(self.index_filepath, "wb") as file:
      pickle.dump(self.index, file)
    # Write docmap cache
    with open(self.docmap_filepath, "wb") as file:
      pickle.dump(self.docmap, file)
    # Write tf cache
    with open(self.tf_filepath, "wb") as file:
      pickle.dump(self.term_frequencies, file)
    # Write doc_lengths cache
    with open(self.doc_lengths_filepath, "wb") as file:
      pickle.dump(self.doc_lengths, file)
  
  def load(self):
    try:
      with open(self.index_filepath, "rb") as file:
        self.index = pickle.load(file)
      with open(self.docmap_filepath, "rb") as file:
        self.docmap = pickle.load(file)
      with open(self.tf_filepath, "rb") as file:
        self.term_frequencies = pickle.load(file)
      with open(self.doc_lengths_filepath, "rb") as file:
        self.doc_lengths = pickle.load(file)
    except FileNotFoundError:
      print("Cache file missing.")
      while True:
        want_cache = input("Build a new cache? (Y/N)\n")
        if want_cache.lower() == "n":
          break
        if want_cache.lower() == "y":
          if pathlib.Path(CACHE_DIR).exists(): shutil.rmtree(CACHE_DIR)
          self.build()
          break