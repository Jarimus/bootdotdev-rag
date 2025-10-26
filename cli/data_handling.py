import json

movie_filepath = "data/movies.json"
stopwords_filepath = "data/stopwords.txt"

def load_movies() -> dict[str, list[dict]]:
  with open(movie_filepath) as file:
    movie_data = json.load(file)
  return movie_data

def load_stopwords() -> list[str]:
  result = []
  with open(stopwords_filepath) as file:
    for word in file:
      result.append(word.strip())
  return result