from text_handling import process_string
from data_handling import load_movies

class InvertedIndex:

  def __init__(self) -> None:
    self.index: dict[str, set[int]] =  {}
    self.docmap: dict[int, dict] = {}
  

  def __add_document(self, doc_id: int, text: str):
    tokens = process_string(text)
    for token in tokens:
      if token not in self.index:
        self.index[token] = set()
      self.index[token].add(doc_id)

  def get_documents(self, term: str) -> list[int]:
    return sorted(self.index[term])

  def build(self):
    movies = load_movies()
    for m in movies:
      self.__add_document(int(movies["id"]), f"{m['title']} {m['description']}")
      self.docmap[int(m["id"])] = m

  def save(self):
    pass