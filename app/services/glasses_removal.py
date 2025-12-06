import base64
from app.utils.gemini_client import get_gemini_client
from google import genai

def remove_glasses_service(image_bytes: bytes) -> str:
    client = get_gemini_client()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            base64_image,
            "Remove the eyeglasses from this face and naturally restore the eyes and eyebrows. Preserve identity, skin texture, lighting, and facial structure."
        ],
        config=genai.GenerateContentConfig(
            response_modalities=[genai.Modality.IMAGE],
            candidate_count=1
        )
    )

    edited_base64 = None

    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data"):
            edited_base64 = part.inline_data.data

    if not edited_base64:
        raise Exception("Gemini did not return an edited image")

    return edited_base64
