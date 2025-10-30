import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

#Test api key
if api_key is None:
  print("api key could not be loaded")
  exit(1)
print(f"Using key {api_key[:3]}...")
print("""Prompt Tokens: 19
Response Tokens: Lots""")

client = genai.Client(api_key=api_key)

res = client.models.generate_content(model="gemini-2.0-flash-001", contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.")

print(res.text)

if res.usage_metadata is None:
  print("no metadata to print")
else:
  print(f"Prompt tokens: {res.usage_metadata.prompt_token_count}")
  print(f"Response tokens: {res.usage_metadata.candidates_token_count}")