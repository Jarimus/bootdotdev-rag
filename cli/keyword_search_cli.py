#!/usr/bin/env python3
import argparse
from data_handling import load_movies
from text_handling import process_string, normalize_string

def main() -> None:
  parser = argparse.ArgumentParser(description="Keyword Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  search_parser = subparsers.add_parser("search", help="Search movies using BM25")
  search_parser.add_argument("query", type=str, help="Search query")

  args = parser.parse_args()
  
  match args.command:
    case "search":
      # Load the movies data
      print(f"Searching for: {args.query}")
      movies_data = load_movies()
      result: list[dict[str, str]] = []

      # Search the movies database
      search_terms = process_string(args.query)
      done = False
      for term in search_terms:
        if done: break
        for movie in movies_data["movies"]:
          if done: break
          movie_title = normalize_string(movie["title"])
          if term in movie_title and movie not in result:
            result.append(movie)
            if len(result) >= 5:
                done = True
      
      # Sort by id
      result.sort(key=lambda x: x["id"])
              
      # Show results
      for i, movie in enumerate(result):
        print(f"{i+1}: {movie['title']} ({movie['id']})")

    case _:
      parser.print_help()


if __name__ == "__main__":
    main()