import os
import base64
from google import genai

def get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)


def png_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to Base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def remove_glasses_service(image_bytes: bytes) -> str:
    """
    Sends image to Gemini to remove glasses.
    Returns Base64 string directly for frontend display.
    """
    client = get_gemini_client()
    base64_image = png_to_base64(image_bytes)

    prompt = (
        "Remove the eyeglasses from this face and naturally restore the eyes and eyebrows. "
        "Preserve identity, skin texture, lighting, and facial structure."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/png", "data": base64_image}}
            ]
        }]
    )

    # Extract the edited image Base64
    edited_base64 = None
    for part in response.candidates[0].content.parts:
        if getattr(part, "inlineData", None):
            edited_base64 = part.inlineData.data
            break
        elif getattr(part, "inline_data", None):
            edited_base64 = part.inline_data.data
            break

    if not edited_base64:
        raise Exception("Gemini did not return an edited image")

    # Return Base64 string directly — do not use Pillow
    return edited_base64
