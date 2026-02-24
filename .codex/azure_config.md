Create the file `src/utils/azure_config.py` with Azure configuration boilerplate for the RiskSentinel project.

Requirements:
- Load environment variables from .env file using python-dotenv
- Define a Settings class (pydantic BaseSettings) with these fields:
  - AZURE_OPENAI_ENDPOINT: str
  - AZURE_OPENAI_API_KEY: str
  - AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o"
  - AZURE_OPENAI_API_VERSION: str = "2024-12-01-preview"
  - AZURE_AI_PROJECT_ENDPOINT: str (for Foundry Agent Service)
  - AZURE_SUBSCRIPTION_ID: str = ""
  - AZURE_RESOURCE_GROUP: str = ""
- Add a get_settings() function that returns a cached Settings instance
- Add a get_openai_client() function that returns an AzureOpenAI client (from openai import AzureOpenAI)
- Add a get_azure_credential() function that returns DefaultAzureCredential (from azure.identity)
- All imports should be inside functions to avoid import errors when packages aren't installed
- Add docstrings explaining each function
- Create also the `.env.example` file at project root with all variables as placeholders

Keep it simple — no over-engineering. This is hackathon code.
