from text_handling import process_string
from data_handling import load_movies
import pickle
import pathlib

class InvertedIndex:

  def __init__(self) -> None:
    self.index_filepath = "cache/index.pkl"
    self.index: dict[str, set[int]] =  {}
    self.docmap_filepath = "cache/docmap.pkl"
    self.docmap: dict[int, dict] = {}
  

  def __add_document(self, doc_id: int, text: str):
    tokens = process_string(text)
    for token in tokens:
      if token not in self.index:
        self.index[token] = set()
      self.index[token].add(doc_id)

  def get_documents(self, term: str) -> list[int]:
    return sorted(self.index.get(term, []))

  def build(self):
    movies = load_movies()
    for i, m in enumerate(movies["movies"]):
      self.__add_document(int(m["id"]), f"{m['title']} {m['description']}")
      self.docmap[int(m["id"])] = m
      print(f"Building... ({i}/{len(movies["movies"])})")

  def save(self):
    # Ensure directory exists
    pathlib.Path("cache").mkdir(parents=True, exist_ok=True)
    # Write index cache
    with open(self.index_filepath, "wb") as file:
      pickle.dump(self.index, file)
    # Write docmap cache
    with open(self.docmap_filepath, "wb") as file:
      pickle.dump(self.docmap, file)
  
  def load(self):
    try:
      with open(self.index_filepath, "rb") as file:
        self.index = pickle.load(file)
      with open(self.docmap_filepath, "rb") as file:
        self.docmap = pickle.load(file)
    except FileNotFoundError:
      print("Cache does not exist")
      while True:
        want_cache = input("Build a new cache? (Y/N)\n")
        if want_cache.lower() == "n":
          break
        if want_cache.lower() == "y":
          self.build()
          self.save()
          break