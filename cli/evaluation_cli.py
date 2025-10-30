import argparse, json
from data_handling import GOLDEN_DATASET_FILEPATH, load_movies
from lib.hybrid_search import HybridSearch

def main():
  parser = argparse.ArgumentParser(description="Search Evaluation CLI")
  parser.add_argument(
    "--limit",
    type=int,
    default=5,
    help="Number of results to evaluate (k for precision@k, recall@k)",
  )

  args = parser.parse_args()
  limit = args.limit

  # Evaluation logic
  with open(GOLDEN_DATASET_FILEPATH) as file:
    test_cases: list[dict] = json.load(file)["test_cases"]
  
  test_precisions = []
  movies = load_movies()["movies"]
  hybrid_search = HybridSearch(movies)
  for tc in test_cases:
    print(f"Next test query: {tc['query']}")
    results = hybrid_search.rrf_search(tc['query'], 60, limit)
    relevant_found = 0
    for r in results:
      if r['doc']['title'] in tc['relevant_docs']:
        relevant_found += 1
    test_precisions.append({
      "query": tc['query'],
      "precision": relevant_found / len(results),
      "recall": relevant_found / len(tc['relevant_docs']),
      "retrieved": ", ".join([ r['doc']['title'] for r in results]),
      "relevant": tc["relevant_docs"]
    })
    print("-"*60)

  print(f"k={limit}")
  for t in test_precisions:
    print(f"""
- Query: {t['query']}
\tPrecision@{limit}: {t['precision']:.4f}
\tRecall@{limit}: {t['recall']:.4f}
\tRetrieved: {t['retrieved']}
\tRelevant: {t['relevant']}""")
    

if __name__ == "__main__":
    main()