import argparse
from lib.hybrid_search import normalize_values, HybridSearch
from data_handling import load_movies
from lib.gemini import enhance_query


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
  rrf_search_parser.add_argument("--enhance", type=str, choices=["spell"], help="Query enhancement method")

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
{r['doc']['description'][:100]}""")
         
    case "rrf-search":
      match args.enhance:
        case "spell":
          new_query = enhance_query(args.query)
          if new_query != args.query:
            print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{new_query.strip()}'\n")
            args.query = new_query

      movies = load_movies()["movies"]
      hybrid_search = HybridSearch(movies)
      results = hybrid_search.rrf_search(args.query, args.k, args.limit)
      for i, r in enumerate(results):
        print(f"""{i+1}. {r['doc']['title']}
   RRF score: {r['hybrid']:.3f}
   BM25 Rank: {r['bm25']:.0f}, Semantic Rank: {r['semantic']:.0f}
{r['doc']['description'][:100]}...""")

    case _:
      parser.print_help()


if __name__ == "__main__":
  main()