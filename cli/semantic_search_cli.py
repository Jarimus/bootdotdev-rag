#!/usr/bin/env python3

from lib.semantic_search import *

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
  search_subparser.add_argument("--limit", type=int, nargs="?", default=5, help="tunable limit for top results to display")

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

    case _:
      parser.print_help()

if __name__ == "__main__":
  main()