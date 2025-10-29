#!/usr/bin/env python3

from lib.semantic_search import *
from search_utils import DEFAULT_SEMANTIC_SEARCH_LIMIT, DEFAULT_CHUNK_SIZE

import argparse

def main():
  parser = argparse.ArgumentParser(description="Semantic Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  # Verify model command
  subparsers.add_parser("verify", help="Verify the semantic search model")

  # embed_text command
  embed_text_subparser = subparsers.add_parser("embed_text", help="Embed text with the semantic model")
  embed_text_subparser.add_argument("text", help="text to embed")

  # Verify embeddings command
  subparsers.add_parser("verify_embeddings", help="Verifies the embeddings")

  # embedquery command
  embedquery_subparser = subparsers.add_parser("embedquery", help="Embed query with the semantic model")
  embedquery_subparser.add_argument("query", help="query to be embedded")

  # search command
  search_subparser = subparsers.add_parser("search", help="perform a semantic search")
  search_subparser.add_argument("query", help="query for the search")
  search_subparser.add_argument("--limit", type=int, nargs="?", default=DEFAULT_SEMANTIC_SEARCH_LIMIT, help="tunable limit for top results to display")

  # chunk command
  chunk_subparser = subparsers.add_parser("chunk", help="Splits a string into chunks. Default 200 words.")
  chunk_subparser.add_argument("text", help="text to split into chunks")
  chunk_subparser.add_argument("--chunk-size", type=int, nargs="?", default=DEFAULT_CHUNK_SIZE, help="the size of chunks in words")

  # Parse arguments
  args = parser.parse_args()

  match args.command:
    case "verify":
      verify_model()

    case "embed_text":
      embed_text(args.text)

    case "verify_embeddings":
      verify_embeddings()

    case "embedquery":
      embed_query_text(args.query)

    case "search":
      semantic_search_command(args.query, args.limit)

    case "chunk":
      chunks = fixed_size_chunking(args.text, args.chunk_size)
      print(f"Chunking {len(args.text)} characters:")
      for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")


    case _:
      parser.print_help()

if __name__ == "__main__":
  main()