from lib.logging import rrf_results_log
from lib.inverted_index import InvertedIndex
from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.gemini import LLM_Evaluate_results, rerank_batch, rerank_individual
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm
from time import sleep
import json


class HybridSearch:
  def __init__(self, documents):
    self.documents = documents
    self.semantic_search = ChunkedSemanticSearch()
    self.semantic_search.load_or_create_chunk_embeddings(documents)
    self.idx = InvertedIndex()
    self.idx.build()

  def _bm25_search(self, query, limit):
    self.idx.load()
    return self.idx.bm25_search(query, limit)

  def weighted_search(self, query, alpha, limit=5) -> list[dict]:
    bm25_results = self._bm25_search(query, limit * 500)
    semantic_results = self.semantic_search.search_chunks(query, limit * 500)
    # normalize bm25 scores
    normalized_bm25 = list(zip( [int(x[0]) for x in bm25_results], normalize_values([x[1] for x in bm25_results ])))
    # normalize semantic scores
    normalized_semantic = list(zip( [ int(x["id"]) for x in semantic_results ], normalize_values([x["score"] for x in semantic_results]) ))
    # Combine results
    results: dict[int, dict] = {}
    for doc_id, bm25_score in normalized_bm25:
      if doc_id not in results:
        results[doc_id] = {}
      results[doc_id]["bm25"] = bm25_score
      results[doc_id]["doc"] = self.idx.docmap[doc_id]
      semantic_score = results[doc_id].get("semantic", 0)
      results[doc_id]["hybrid"] = compute_hybrid_score(bm25_score, semantic_score, alpha)
    for doc_id, semantic_score in normalized_semantic:
      if doc_id not in results:
        results[doc_id] = {}
      results[doc_id]["semantic"] = semantic_score
      bm25_score = results[doc_id].get("bm25", 0)
      results[doc_id]["hybrid"] = compute_hybrid_score(bm25_score, semantic_score, alpha)

    return sorted(results.values(), key=lambda item: item["hybrid"], reverse=True)[:limit]

    
  def rrf_search(self, query, k=60, limit=10, alpha=0.5):
    bm25_ranks = [ x[0] for x in self._bm25_search(query, limit * 500) ]
    semantic_ranks = [ x["id"] for x in self.semantic_search.search_chunks(query, limit * 500) ]
    # Combine results
    results: dict[int, dict] = {}
    for i, doc_id in enumerate(bm25_ranks):
      if doc_id not in results:
        results[doc_id] = {}
      bm25_rrf = compute_rrf_score(i+1, k)
      results[doc_id]["bm25"] = i+1
      results[doc_id]["bm25_score"] = bm25_rrf
      results[doc_id]["doc"] = self.idx.docmap[doc_id]
      semantic_rrf = results[doc_id].get("semantic_score", 0)
      results[doc_id]["hybrid"] = bm25_rrf + semantic_rrf
    for i, doc_id in enumerate(semantic_ranks):
      if doc_id not in results:
        results[doc_id] = {}
      semantic_rrf = compute_rrf_score(i+1, k)
      results[doc_id]["semantic"] = i+1
      results[doc_id]["semantic_score"] = semantic_rrf
      results[doc_id]["doc"] = self.idx.docmap[doc_id]
      bm25_rrf = results[doc_id].get("bm25_score", 0)
      results[doc_id]["hybrid"] = bm25_rrf + semantic_rrf

    return sorted(results.values(), key=lambda item: item["hybrid"], reverse=True)[:limit]
  

def normalize_values(values: list[float]) -> list[float]:
  if values == []:
    return []
  result = []
  max_v = max(values)
  min_v = min(values)
  if min_v == max_v:
    return [1.0] * len(values)
  for v in values:
   result.append( (v - min_v) / (max_v - min_v) )
  return result

def compute_hybrid_score(bm25score: float ,semantic_score: float, alpha: float = 0.5) -> float:
  return alpha*bm25score + (1-alpha)*semantic_score

def compute_rrf_score(rank, k=60):
    return 1 / (k + rank)

def rrf_search_individual(hybrid_search: HybridSearch, query: str, k: int = 50, limit: int = 5):
  results = hybrid_search.rrf_search(query, k, limit * 5)
  rrf_results_log(results)
  for r in tqdm(results, "LLM Reranking", len(results)):
    doc = r["doc"]
    try:
      LLM_score = float(rerank_individual(query, doc))
      r["LLM_score"] = LLM_score
    except ValueError:
      print(f"LLM did not provide an appropriate ranking: {LLM_score}")
      r["LLM_score"] = 0
    sleep(3)
  results.sort(key=lambda item: item["LLM_score"], reverse=True)
  return results

def rrf_search_batch(hybrid_search: HybridSearch, query: str, k: int = 50, limit: int = 5):
  results = hybrid_search.rrf_search(query, k, limit * 5)
  rrf_results_log(results)
  doc_list_str = ""
  for i, r in enumerate(results):
    doc_list_str += f"""Title: {r['doc']['title']}
ID: {i}
Description: {r['doc']['description']}
-------------------------------------------------------"""
  print(f"Reranking the top {limit} results using batch method...\n")
  LLM_json = rerank_batch(query, doc_list_str)
  LLM_IDs = json.loads(LLM_json)
  print("LLM ranking:", LLM_IDs)
  final_results = []
  for i in LLM_IDs:
    final_results.append(results[i])
  return final_results

def rrf_search_cross_encoder(hybrid_search: HybridSearch, query: str, k: int = 50, limit: int = 5):
  print("Reranking top 25 results using cross_encoder method...")
  results = hybrid_search.rrf_search(query, k, limit * 5)
  rrf_results_log(results)
  pairs: list[tuple[str, str]] = []
  for r in results:
    pairs.append( (query, f"{r['doc'].get('title', '')} - {r.get('doc', '')}"))
  encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
  scores = encoder.predict(pairs)
  for i, score in enumerate(scores):
    results[i]['encoder_score'] = score
  return sorted(results, key=lambda item: item['encoder_score'], reverse=True)