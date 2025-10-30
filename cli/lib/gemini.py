import os
from dotenv import load_dotenv
from google import genai

def enhance_spell_query(query: str):
  load_dotenv()
  api_key = os.environ.get("GEMINI_API_KEY")
  client = genai.Client(api_key=api_key)

  res = client.models.generate_content(model="gemini-2.0-flash-001", contents=f"""Fix any spelling errors in this movie search query.

  Only correct obvious typos. Don't change correctly spelled words.

  Query: "{query}"

  If no errors, return the original query. Return only the enhanced or original query and nothing else.
  Corrected:""")

  if res.text:
    return res.text
  else:
    return query
  
def enhance_rewrite_query(query: str):
  load_dotenv()
  api_key = os.environ.get("GEMINI_API_KEY")
  client = genai.Client(api_key=api_key)

  res = client.models.generate_content(model="gemini-2.0-flash-001", contents=f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:""")

  if res.text:
    return res.text
  else:
    return query
  
def enhance_expand_query(query: str):
  load_dotenv()
  api_key = os.environ.get("GEMINI_API_KEY")
  client = genai.Client(api_key=api_key)

  res = client.models.generate_content(model="gemini-2.0-flash-001", contents=f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
""")

  if res.text:
    return res.text
  else:
    return query
  
def rerank_individual(query: str, doc: dict):
  load_dotenv()
  api_key = os.environ.get("GEMINI_API_KEY")
  client = genai.Client(api_key=api_key)

  res = client.models.generate_content(model="gemini-2.0-flash-001", contents=f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match). Accuracy in tenths.
Give me ONLY the number in your response, no other text or explanation.

Score:""")

  if res.text:
    return res.text
  else:
    return query
  
def rerank_batch(query: str, doc_list_str: str):
  load_dotenv()
  api_key = os.environ.get("GEMINI_API_KEY")
  client = genai.Client(api_key=api_key)

  res = client.models.generate_content(model="gemini-2.0-flash-001", contents=f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[1, 12, 34, 2, 75]
""")

  if res.text:
    return res.text
  else:
    return query