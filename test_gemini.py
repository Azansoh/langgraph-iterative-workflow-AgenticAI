import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print("API KEY FOUND:", bool(api_key))

client = genai.Client(api_key=api_key)

print("\nTesting Gemini...\n")

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Say hello in one sentence."
)

print(response.text)