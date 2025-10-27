from sentence_transformers import SentenceTransformer

class SemanticSearch:

  def __init__(self) -> None:
    self.model = SentenceTransformer('all-MiniLM-L6-v2')

  def generate_embedding(self, text: str):
    if text.strip() == "":
      raise ValueError("SemanticSearch - generate embedding: text input is empty")
    embeddings = self.model.encode([text])
    return embeddings[0]
    

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