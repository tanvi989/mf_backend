import base64
import requests
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

def remove_eyeglasses(image_bytes: bytes):
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Remove eyeglasses from the person and restore natural eyes. Keep face identical."},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
    }

    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    data = response.json()

    # Extract edited image
    edited_base64 = data["candidates"][0]["content"]["parts"][0]["inline_data"]["data"]
    edited_bytes = base64.b64decode(edited_base64)

    return edited_bytes
