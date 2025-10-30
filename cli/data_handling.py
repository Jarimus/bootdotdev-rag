import json

MOVIE_FILEPATH = "data/movies.json"
STOPWORDS_FILEPATH = "data/stopwords.txt"
GOLDEN_DATASET_FILEPATH = "data/golden_dataset.json"

CACHE_DIR = "cache"
DOC_LENGTHS_FILE = "doc_lengths.pkl"
DOCMAP_FILE = "docmap.pkl"
INDEX_FILE = "index.pkl"
MOVIE_EMBEDDINGS_FILE = "movie_embeddings.npy"
TERM_FREQ_FILE = "term_frequencies.pkl"
CHUNK_EMBEDDINGS_FILE = "chunk_embeddings.npy"
CHUNK_METADATA_FILE = "chunk_metadata.json"

def load_movies() -> dict[str, list[dict]]:
  with open(MOVIE_FILEPATH) as file:
    movie_data = json.load(file)
  return movie_data

def load_stopwords() -> list[str]:
  result = []
  with open(STOPWORDS_FILEPATH) as file:
    for word in file:
      result.append(word.strip())
  return result