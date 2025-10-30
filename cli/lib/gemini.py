import os
from dotenv import load_dotenv
from google import genai

def enhance_query(query: str):
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