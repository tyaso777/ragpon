import os

from openai import AzureOpenAI, OpenAI


def create_openai_client():
    """
    Reads environment variables, decides whether to use Azure or regular OpenAI,
    and returns the client plus associated model/deployment info.

    Returns:
        tuple: (client, MODEL_NAME, DEPLOYMENT_ID, OPENAI_TYPE)
    Raises:
        ValueError: If required environment variables are missing.
    """
    # Load environment variables
    OPENAI_TYPE = os.getenv("OPENAI_TYPE", "openai").lower()  # Default is "openai"

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    # Check required environment variables
    if OPENAI_TYPE == "azure":
        required_vars = {
            "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
            "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
            "AZURE_OPENAI_DEPLOYMENT": AZURE_OPENAI_DEPLOYMENT,
        }
    else:
        required_vars = {"OPENAI_API_KEY": OPENAI_API_KEY}

    missing_vars = [key for key, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    # Initialize the client
    if OPENAI_TYPE == "azure":
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        MODEL_NAME = AZURE_OPENAI_MODEL
        DEPLOYMENT_ID = AZURE_OPENAI_DEPLOYMENT
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)
        MODEL_NAME = OPENAI_MODEL
        DEPLOYMENT_ID = None

    return client, MODEL_NAME, DEPLOYMENT_ID, OPENAI_TYPE
