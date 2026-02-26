"""Azure configuration helpers for RiskSentinel."""

from typing import Any

Settings = None
_SETTINGS_CACHE = None


def _get_settings_class():
    """Build and return the pydantic Settings class lazily.

    Third-party imports are kept inside this function so importing this
    module does not fail when optional Azure dependencies are missing.
    """
    global Settings
    if Settings is not None:
        return Settings

    try:
        from pydantic_settings import BaseSettings, SettingsConfigDict
    except ImportError:
        from pydantic import BaseSettings

        class _Settings(BaseSettings):
            AZURE_OPENAI_ENDPOINT: str
            AZURE_OPENAI_API_KEY: str
            AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o"
            AZURE_OPENAI_FALLBACK_DEPLOYMENT: str = "gpt-4o-mini"
            AZURE_OPENAI_API_VERSION: str = "2025-03-01-preview"
            AZURE_AI_PROJECT_ENDPOINT: str = ""
            AZURE_SUBSCRIPTION_ID: str = ""
            AZURE_RESOURCE_GROUP: str = ""

            class Config:
                env_file = ".env"
                env_file_encoding = "utf-8"

        Settings = _Settings
        return Settings

    class _Settings(BaseSettings):
        AZURE_OPENAI_ENDPOINT: str
        AZURE_OPENAI_API_KEY: str
        AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o"
        AZURE_OPENAI_FALLBACK_DEPLOYMENT: str = "gpt-4o-mini"
        AZURE_OPENAI_API_VERSION: str = "2025-03-01-preview"
        AZURE_AI_PROJECT_ENDPOINT: str = ""
        AZURE_SUBSCRIPTION_ID: str = ""
        AZURE_RESOURCE_GROUP: str = ""

        model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    Settings = _Settings
    return Settings


def get_settings():
    """Return a cached Settings instance loaded from environment variables.

    Environment values are read from process environment and the local `.env`
    file when present.
    """
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is not None:
        return _SETTINGS_CACHE

    from dotenv import load_dotenv

    load_dotenv()
    settings_cls = _get_settings_class()
    _SETTINGS_CACHE = settings_cls()
    return _SETTINGS_CACHE


def get_openai_client():
    """Create and return an AzureOpenAI client configured from settings."""
    from openai import AzureOpenAI

    settings = get_settings()
    return AzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    )


def get_agent_framework_chat_client(deployment_name: str | None = None) -> Any:
    """Create an Agent Framework Azure OpenAI client from settings.

    Uses the Responses client because the underlying Azure endpoint in this
    project is validated with the Responses API.
    """
    from agent_framework.azure import AzureOpenAIResponsesClient

    settings = get_settings()
    deployment = deployment_name or settings.AZURE_OPENAI_DEPLOYMENT
    return AzureOpenAIResponsesClient(
        api_key=settings.AZURE_OPENAI_API_KEY,
        deployment_name=deployment,
        endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        function_invocation_configuration={
            "enabled": True,
            "max_iterations": 6,
            "max_consecutive_errors_per_request": 2,
            "include_detailed_errors": True,
        },
    )


def get_azure_credential():
    """Create and return a DefaultAzureCredential for Azure SDK clients."""
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()
