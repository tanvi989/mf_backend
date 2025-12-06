import os
from google import genai

def get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)
