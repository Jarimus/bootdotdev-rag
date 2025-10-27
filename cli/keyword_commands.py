from inverted_index import InvertedIndex
from text_handling import process_string
from search_utils import BM25_K1, BM25_B

def search_command(query: str):
  print(f"Searching for: {query}")
  InvertedIndexer = InvertedIndex()
  InvertedIndexer.load()
  search_tokens = process_string(query)
  search_result = []
  done = False
  for token in search_tokens:
    if done: break
    doc_ids = InvertedIndexer.get_documents(token)
    for id in doc_ids:
      search_result.append(InvertedIndexer.docmap[id])
      if len(search_result) >= 5:
        done = True
        break
  print("Search results:")
  for i, movie in enumerate(search_result):
    print(f"{i+1}: {movie['title']}")

def tf_command(doc_id: int, term: str):
  InvertedIndexer = InvertedIndex()
  InvertedIndexer.load()
  tf = InvertedIndexer.get_tf(doc_id, term)
  print(f"Term frequency of '{term}' in doc {doc_id}: {tf}")

def idf_command(term: str):
  InvertedIndexer = InvertedIndex()
  InvertedIndexer.load()
  idf = InvertedIndexer.get_idf(term)
  print(f"idf for '{term}': {idf:.2f}")

def tfidf_command(doc_id: int, term: str):
  InvertedIndexer = InvertedIndex()
  InvertedIndexer.load()
  tf_idf = InvertedIndexer.get_tfidf(doc_id, term)
  print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")

def bm25idf_command(term: str):
  InvertedIndexer = InvertedIndex()
  InvertedIndexer.load()
  bm25idf = InvertedIndexer.get_bm25_idf(term)
  print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")

def bm25tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B):
  InvertedIndexer = InvertedIndex()
  InvertedIndexer.load()
  bm25tf = InvertedIndexer.get_bm25_tf(doc_id, term, k1)
  print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")

def bm25search_command(query: str, limit: int):
  # Load the data into cache
  InvertedIndexer = InvertedIndex()
  InvertedIndexer.load()
  # initiate the search
  bm25search_result = InvertedIndexer.bm25_search(query, limit)
  for doc_id, score in bm25search_result:
    movie = InvertedIndexer.docmap[doc_id]
    print(f"({doc_id}) {movie['title']} - Score: {score:.2f}")