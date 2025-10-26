#!/usr/bin/env python3
import argparse, math
from data_handling import load_movies
from text_handling import process_string, normalize_string
from inverted_index import InvertedIndex

def main() -> None:
  parser = argparse.ArgumentParser(description="Keyword Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  # Search command
  search_parser = subparsers.add_parser("search", help="Search movies using BM25")
  search_parser.add_argument("query", type=str, help="Search query")

  # Build command
  subparsers.add_parser("build", help="Build and save inverted index to disk")

  # tf command
  tf_parser = subparsers.add_parser("tf", help="Get the term frequency for a document")
  tf_parser.add_argument("doc_id", type=int, help="Document id")
  tf_parser.add_argument("term", type=str, help="target token for frequnecy")

  # idf command
  idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency (idf) of a term")
  idf_parser.add_argument("term", type=str, help="term to idf")

  args = parser.parse_args()
  
  match args.command:
    case "search":
      print(f"Searching for: {args.query}")
      InvertedIndexer = InvertedIndex()
      InvertedIndexer.load()
      search_tokens = process_string(args.query)
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

    case "build":
      InvertedIndexer = InvertedIndex()
      InvertedIndexer.build()
      InvertedIndexer.save()

      test_search: list[int] = InvertedIndexer.get_documents("merida")
      print("docID of first:", test_search[0])

    case "tf":
      InvertedIndexer = InvertedIndex()
      InvertedIndexer.load()
      tf_result = InvertedIndexer.get_tf(args.doc_id, args.term)
      print(f"Term frequency of '{args.term}' in doc {args.doc_id}: {tf_result}")

    case "idf":
      InvertedIndexer = InvertedIndex()
      InvertedIndexer.load()
      idf = InvertedIndexer.get_idf(args.term)
      print(f"idf for '{args.term}': {idf:.2f}")

    case _:
      parser.print_help()


if __name__ == "__main__":
    main()