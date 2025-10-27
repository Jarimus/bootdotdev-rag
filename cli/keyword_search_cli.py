#!/usr/bin/env python3
import argparse
from keyword_commands import *
from inverted_index import InvertedIndex
from search_utils import BM25_K1, BM25_B

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
  tf_parser.add_argument("term", type=str, help="target token for frequency")

  # idf command
  idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency (idf) of a term")
  idf_parser.add_argument("term", type=str, help="term to idf")

  # tf-idf command
  tf_idf_parser = subparsers.add_parser("tfidf", help="Get the TF-IDF score of a term")
  tf_idf_parser.add_argument("doc_id", type=int, help="Target document id")
  tf_idf_parser.add_argument("term", type=str, help="target token for TF-IDF score")

  # bm25idf command
  bm25idf_parser = subparsers.add_parser("bm25idf", help="Get the bm25idf score of a term")
  bm25idf_parser.add_argument("term", type=str, help="target token for bm25idf score")

  # bm25tf command
  bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
  bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
  bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
  bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
  bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

  # bm25search command
  bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
  bm25search_parser.add_argument("query", type=str, help="Search query")
  bm25search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Tunable limit for results (default 5)")

  args = parser.parse_args()
  
  match args.command:
    case "search":
      search_command(args.query)

    case "build":
      InvertedIndexer = InvertedIndex()
      InvertedIndexer.build()

    case "tf":
      tf_command(args.doc_id, args.term)

    case "idf":
      idf_command(args.term)

    case "tfidf":
      tfidf_command(args.doc_id, args.term)

    case "bm25idf":
      bm25idf_command(args.term)

    case "bm25tf":
      bm25tf_command(args.doc_id, args.term, args.k1, args.b)

    case "bm25search":
      bm25search_command(args.query, args.limit)

    case _:
      parser.print_help()


if __name__ == "__main__":
    main()