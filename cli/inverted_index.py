from text_handling import process_string
from data_handling import load_movies
from collections import Counter
import pickle, pathlib, math, shutil

class InvertedIndex:

  def __init__(self) -> None:
    self.index_filepath = "cache/index.pkl"
    self.index: dict[str, set[int]] =  {}
    self.docmap_filepath = "cache/docmap.pkl"
    self.docmap: dict[int, dict] = {}
    self.tf_filepath = "cache/term_frequencies.pkl"
    self.term_frequencies: dict[int, Counter[str]] = {}
  
  def __single_term_to_token(self, term: str) -> str:
    token = process_string(term)
    if len(token) > 1:
      raise ValueError("command supports single tokens only")
    return token[0]

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

  def build(self):
    # Only build if cache dir does not exist
    if pathlib.Path("cache").exists():
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
    pathlib.Path("cache").mkdir(parents=True, exist_ok=True)
    # Write index cache
    with open(self.index_filepath, "wb") as file:
      pickle.dump(self.index, file)
    # Write docmap cache
    with open(self.docmap_filepath, "wb") as file:
      pickle.dump(self.docmap, file)
    # Write tf cache
    with open(self.tf_filepath, "wb") as file:
      pickle.dump(self.term_frequencies, file)
  
  def load(self):
    try:
      with open(self.index_filepath, "rb") as file:
        self.index = pickle.load(file)
      with open(self.docmap_filepath, "rb") as file:
        self.docmap = pickle.load(file)
      with open(self.tf_filepath, "rb") as file:
        self.term_frequencies = pickle.load(file)
    except FileNotFoundError:
      print("Cache file missing.")
      while True:
        want_cache = input("Build a new cache? (Y/N)\n")
        if want_cache.lower() == "n":
          break
        if want_cache.lower() == "y":
          shutil.rmtree("cache")
          self.build()
          break