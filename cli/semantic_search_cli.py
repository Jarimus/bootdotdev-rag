#!/usr/bin/env python3

from lib.semantic_search import verify_model, embed_text, verify_embeddings

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

  # Parse arguments
  args = parser.parse_args()

  match args.command:
    case "verify":
      verify_model()

    case "embed_text":
      embed_text(args.text)

    case "verify_embeddings":
      verify_embeddings()

    case _:
      parser.print_help()

if __name__ == "__main__":
  main()