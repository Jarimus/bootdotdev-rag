def rrf_results_log(results: list[dict]):
  print("-"*60)
  print(f"results after initial search:")
  for i, r in enumerate(results):
    print(f"""{i+1}. {r['doc']['title']} ({r['doc']['id']})
Ranks (bm25, semantic): {r['bm25']} --- {r['semantic']}
RRF score: {r['hybrid']:.4f}
{"-"*60}""")