from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

def get_authorized_session(service_account_file: str):
    """
    Creates an authorized session for Google Vertex AI API.
    """
    creds = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return AuthorizedSession(creds)
