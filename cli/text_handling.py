from string import punctuation
from data_handling import load_stopwords
from nltk.stem import PorterStemmer

def normalize_string(keywords: str) -> str:
  processed = keywords.lower()
  translations = str.maketrans("", "", punctuation)
  processed = processed.translate(translations)
  return processed

def tokenize_string(keywords: str) -> list[str]:
  return keywords.strip().split()

def remove_stopwords(keywords: list[str]) -> list[str]:
  stopwords = load_stopwords()
  result = []
  for keyword in keywords:
    if keyword not in stopwords:
      result.append(keyword)
  return result

def stem_words(words: list[str]) -> list[str]:
  stemmer = PorterStemmer()
  result = []
  for word in words:
    result.append(stemmer.stem(word))
  return result

def process_string(text: str) -> list[str]:
  return stem_words(remove_stopwords(tokenize_string(normalize_string(text))))