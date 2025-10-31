import argparse
from lib.hybrid_search import normalize_values, HybridSearch, rrf_search_individual, rrf_search_batch, rrf_search_cross_encoder
from data_handling import load_movies
from sentence_transformers.cross_encoder import CrossEncoder
from lib.gemini import enhance_rewrite_query, enhance_spell_query, enhance_expand_query
from lib.logging import rrf_results_log
from time import sleep
from tqdm import tqdm


def main() -> None:
  parser = argparse.ArgumentParser(description="Hybrid Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  # Normalize command
  normalize_parser = subparsers.add_parser("normalize", help="normalize the scale to 0.0-1.0")
  normalize_parser.add_argument("values", type=float, nargs="+", help="list of values (separated by space) to normalize")

  # weighted_search command
  weighted_search_parser = subparsers.add_parser("weighted-search", help="perform a weighted search using bm25 and semantic scores")
  weighted_search_parser.add_argument("query", type=str, help="text to initiate the search with")
  weighted_search_parser.add_argument("--alpha", type=float, nargs="?", default=0.5, help="Adjust the weight of bm25 and semantic scores. 0.0: Full weight on bm25 - 1.0: Full weight on semantic")
  weighted_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Limits the number of results. Defaults to 5.")

  # rrf_search command
  rrf_search_parser = subparsers.add_parser("rrf-search", help="perform an rrf-search")
  rrf_search_parser.add_argument("query", type=str, help="text to initiate the search with")
  rrf_search_parser.add_argument("--k", type=int, nargs="?", default=50, help="The k parameter (a constant) controls the weight between higher-ranked results and lower-ranked ones. Defaults to 50. Suggested range: 20-100")
  rrf_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Limits the number of results. Defaults to 5.")
  rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
  rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="perform reranking after search.")
  rrf_search_parser.add_argument("--evaluate", type=str, help="Use LLM to evaluate the relevance of results")

  args = parser.parse_args()

  match args.command:
    case "normalize":
      normalized_values = normalize_values(args.values)
      for v in normalized_values:
        print(f"* {v:.4f}")

    case "weighted-search":
      movies = load_movies()["movies"]
      hybrid_search = HybridSearch(movies)
      results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
      for i, r in enumerate(results):
        print(f"""{i+1}. {r['doc']['title']}
   Hybrid score: {r['hybrid']:.3f}
   BM25: {r['bm25']:.3f}, Semantic: {r['semantic']:.3f}
   {r['doc']['description'][:100]}
""")
         
    case "rrf-search":
      print("Original query:", args.query)
      match args.enhance:
        case "spell":
          new_query = enhance_spell_query(args.query)
          if new_query != args.query:
            print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{new_query.strip()}'\n")
            args.query = new_query
        case "rewrite":
          new_query = enhance_rewrite_query(args.query)
          if new_query != args.query:
            print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{new_query.strip()}'\n")
            args.query = new_query
        case "expand":
          new_query = enhance_expand_query(args.query)
          if new_query != args.query:
            print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{new_query.strip()}'\n")
            args.query = new_query

      movies = load_movies()["movies"]
      hybrid_search = HybridSearch(movies)
      match args.rerank_method:
        case "individual":
          results = rrf_search_individual(hybrid_search, args.query, args.k, args.limit)
          for i, r in enumerate(results[:args.limit]):
            print(f"""{i+1}. {r['doc']['title']}
   Rerank score: {r["LLM_score"]:.1f}/10
   RRF score: {r['hybrid']:.3f}
   BM25 Rank: {r['bm25']:.0f}, Semantic Rank: {r['semantic']:.0f}
   {r['doc']['description'][:100]}...
   """)
            
        case "batch":
          final_results = rrf_search_batch(hybrid_search, args.query, args.k, args.limit)
          for i, r in enumerate(final_results[:args.limit]):
            print(f"""{i+1}. {r['doc']['title']}
   Rerank rank: {i+1}
   RRF score: {r['hybrid']:.3f}
   BM25 Rank: {r['bm25']:.0f}, Semantic Rank: {r['semantic']:.0f}
   {r['doc']['description'][:100]}...
   """)
            
        case "cross_encoder":
          print("Reranking top 25 results using cross_encoder method...")
          results = rrf_search_cross_encoder(hybrid_search, args.query, args.k, args.limit)
          for i, r in enumerate(results[:args.limit]):
            print(f"""{i+1}. {r['doc']['title']}
   Cross Encoder Score: {r['encoder_score']:.3f}
   RRF score: {r['hybrid']:.3f}
   BM25 Rank: {r['bm25']:.0f}, Semantic Rank: {r['semantic']:.0f}
   {r['doc']['description'][:100]}...
   """)

        case _:
          results = hybrid_search.rrf_search(args.query, args.k, args.limit)
          for i, r in enumerate(results):
            print(f"""{i+1}. {r['doc']['title']}
   RRF score: {r['hybrid']:.3f}
   BM25 Rank: {r['bm25']:.0f}, Semantic Rank: {r['semantic']:.0f}
   {r['doc']['description'][:100]}...
   """)

    case _:
      parser.print_help()


if __name__ == "__main__":
  main()