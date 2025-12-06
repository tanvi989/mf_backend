import os
import base64
from io import BytesIO
from google import genai


def get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)


def png_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def base64_to_png(base64_str: str) -> bytes:
    return base64.b64decode(base64_str)


def remove_glasses_service(image_bytes: bytes) -> bytes:
    """
    Sends image to Gemini to remove glasses.
    Returns the edited PNG bytes.
    """
    client = get_gemini_client()

    base64_image = png_to_base64(image_bytes)

    prompt = (
        "Remove the eyeglasses from this face and naturally restore the eyes and eyebrows. "
        "Preserve identity, skin texture, lighting, and facial structure."
    )

    # *** CORRECT CALL FOR NEW GOOGLE SDK ***
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/png", "data": base64_image}}
                ]
            }
        ]
    )

    print("Gemini response:", response)

    # Extract image part
    edited_base64 = None

    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            edited_base64 = part.inline_data.data
            break

    if not edited_base64:
        raise Exception("Gemini did not return an edited image")

    return base64_to_png(edited_base64)
