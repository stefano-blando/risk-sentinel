import os

import pytest

from src.utils.azure_config import get_openai_client, get_settings


def _missing_required_vars() -> list[str]:
    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    return [name for name in required if not os.getenv(name, "").strip()]


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_AZURE_INTEGRATION_TESTS", "").strip().lower() not in {"1", "true", "yes", "on"},
    reason="Set RUN_AZURE_INTEGRATION_TESTS=1 to enable live Azure round-trip test.",
)
def test_azure_openai_live_roundtrip() -> None:
    missing = _missing_required_vars()
    if missing:
        pytest.skip(f"Missing Azure env vars: {', '.join(missing)}")

    settings = get_settings()
    client = get_openai_client()
    resp = client.responses.create(
        model=settings.AZURE_OPENAI_FALLBACK_DEPLOYMENT or settings.AZURE_OPENAI_DEPLOYMENT,
        input="Reply with OK only.",
        timeout=25,
    )
    text = (resp.output_text or "").strip().upper()
    assert "OK" in text, f"Unexpected response text: {text!r}"
