"""LLM runtime helpers for agent execution and diagnostics."""

from __future__ import annotations

import asyncio
import importlib.metadata
import time

from src.agents.critic import create_critic_agent
from src.agents.orchestrator import (
    create_orchestrator,
    create_simple_agent,
    run_parallel_workflow,
    run_query,
)
from src.utils.azure_config import (
    get_agent_framework_chat_client,
    get_openai_client,
    get_settings,
)


def run_async(coro):
    """Run coroutine from Streamlit sync context."""
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "asyncio.run() cannot be called from a running event loop" not in str(exc):
            raise
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


async def run_orchestrator_query_async(
    query: str,
    timeout_sec: int,
    deployment_name: str | None = None,
) -> str:
    client = get_agent_framework_chat_client(deployment_name=deployment_name)
    orchestrator = create_orchestrator(client)
    return await asyncio.wait_for(run_query(orchestrator, query), timeout=timeout_sec)


async def run_simple_query_async(
    query: str,
    timeout_sec: int,
    deployment_name: str | None = None,
) -> str:
    client = get_agent_framework_chat_client(deployment_name=deployment_name)
    simple_agent = create_simple_agent(client)
    return await asyncio.wait_for(run_query(simple_agent, query), timeout=timeout_sec)


async def run_parallel_workflow_async(
    query: str,
    timeout_sec: int,
    deployment_name: str | None = None,
) -> str:
    client = get_agent_framework_chat_client(deployment_name=deployment_name)
    return await asyncio.wait_for(
        run_parallel_workflow(client, query, timeout_sec=timeout_sec),
        timeout=timeout_sec,
    )


def run_direct_commentary_query(
    query: str,
    timeout_sec: int,
    deployment_name: str | None = None,
) -> str:
    settings = get_settings()
    model_name = deployment_name or settings.AZURE_OPENAI_DEPLOYMENT
    client = get_openai_client()
    response = client.responses.create(
        model=model_name,
        input=query,
        timeout=timeout_sec,
    )
    return (response.output_text or "").strip()


async def run_critic_validation_async(
    *,
    query: str,
    facts_plain: str,
    candidate_json_text: str,
    timeout_sec: int = 12,
    deployment_name: str | None = None,
    extract_json_payload_fn,
) -> dict:
    client = get_agent_framework_chat_client(deployment_name=deployment_name)
    critic = create_critic_agent(client)
    prompt = (
        "Validate candidate JSON against deterministic evidence. Return strict JSON only.\n\n"
        f"User query:\n{query}\n\n"
        f"Deterministic evidence:\n{facts_plain}\n\n"
        f"Candidate JSON:\n{candidate_json_text}"
    )
    output = await asyncio.wait_for(run_query(critic, prompt), timeout=timeout_sec)
    parsed = extract_json_payload_fn(output)
    if not isinstance(parsed, dict):
        return {
            "approved": False,
            "issues": ["Critic output was not valid JSON."],
            "required_fixes": ["Return strict JSON only."],
            "uncertainty_score": 0.8,
            "confidence_reason": "Critic parsing failed.",
        }
    return parsed


def run_critic_validation(
    *,
    query: str,
    facts_plain: str,
    candidate_json_text: str,
    timeout_sec: int = 12,
    deployment_name: str | None = None,
    extract_json_payload_fn,
) -> dict:
    return run_async(
        run_critic_validation_async(
            query=query,
            facts_plain=facts_plain,
            candidate_json_text=candidate_json_text,
            timeout_sec=timeout_sec,
            deployment_name=deployment_name,
            extract_json_payload_fn=extract_json_payload_fn,
        )
    )


def run_agent_query(
    query: str,
    timeout_sec: int = 35,
    strategy: str = "simple",
    deployment_name: str | None = None,
) -> str:
    if strategy == "commentary_direct":
        return run_direct_commentary_query(query, timeout_sec, deployment_name=deployment_name)
    if strategy == "orchestrator":
        return run_async(run_orchestrator_query_async(query, timeout_sec, deployment_name=deployment_name))
    if strategy == "workflow_parallel":
        return run_async(run_parallel_workflow_async(query, timeout_sec, deployment_name=deployment_name))
    return run_async(run_simple_query_async(query, timeout_sec, deployment_name=deployment_name))


def is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def is_timeout_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "timeout" in text or "timed out" in text


def is_retryable_gpt_error(exc: Exception) -> bool:
    if is_rate_limit_error(exc) or is_timeout_error(exc):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(sig in text for sig in ["connection", "temporarily unavailable", "service unavailable"])


def run_agent_query_with_backoff(
    *,
    query: str,
    timeout_sec: int,
    strategy: str,
    deployment_name: str | None = None,
    max_retries: int = 2,
    base_delay_sec: float = 1.5,
    max_total_wait_sec: float = 24.0,
    on_backoff=None,
) -> str:
    """Retry agent query on transient failures with bounded total wait."""
    attempt = 0
    started = time.perf_counter()
    while True:
        try:
            return run_agent_query(
                query=query,
                timeout_sec=timeout_sec,
                strategy=strategy,
                deployment_name=deployment_name,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            remaining = max_total_wait_sec - elapsed
            if (not is_retryable_gpt_error(exc)) or attempt >= max_retries or remaining <= 0.8:
                raise
            wait_sec = min(base_delay_sec * (2 ** attempt), max(0.5, remaining - 0.4))
            if on_backoff:
                on_backoff(wait_sec, attempt + 1, max_retries + 1)
            time.sleep(wait_sec)
            attempt += 1


def get_deployment_routing(
    *,
    high_quality_mode: bool,
    get_runtime_value_fn,
) -> tuple[str, str]:
    """Return (primary, fallback) deployment names for the current run."""
    default_primary = get_runtime_value_fn("AZURE_OPENAI_DEPLOYMENT", "gpt-4o") or "gpt-4o"
    default_fallback = (
        get_runtime_value_fn("AZURE_OPENAI_FALLBACK_DEPLOYMENT", "gpt-4o-mini") or "gpt-4o-mini"
    )
    try:
        settings = get_settings()
        primary = str(getattr(settings, "AZURE_OPENAI_DEPLOYMENT", "")).strip() or default_primary
        fallback = (
            str(getattr(settings, "AZURE_OPENAI_FALLBACK_DEPLOYMENT", "")).strip() or default_fallback
        )
    except Exception:
        primary = default_primary
        fallback = default_fallback

    if high_quality_mode:
        return primary, fallback
    return fallback, primary


def run_gpt_diagnostic(run_agent_query_fn) -> str:
    """Run direct and agent-level GPT checks and return a text report."""
    lines = []
    settings = get_settings()
    lines.append(f"endpoint={settings.AZURE_OPENAI_ENDPOINT}")
    lines.append(f"deployment={settings.AZURE_OPENAI_DEPLOYMENT}")
    lines.append(f"fallback_deployment={settings.AZURE_OPENAI_FALLBACK_DEPLOYMENT}")
    lines.append(f"api_version={settings.AZURE_OPENAI_API_VERSION}")
    try:
        lines.append(f"openai_pkg={importlib.metadata.version('openai')}")
    except Exception:
        lines.append("openai_pkg=unknown")
    try:
        lines.append(f"agent_framework_pkg={importlib.metadata.version('agent-framework')}")
    except Exception:
        lines.append("agent_framework_pkg=unknown")

    t0 = time.perf_counter()
    try:
        client = get_openai_client()
        response = client.responses.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT,
            input="Reply with OK only.",
            timeout=20,
        )
        elapsed = time.perf_counter() - t0
        lines.append(f"direct_openai=OK ({elapsed:.2f}s) text={(response.output_text or '').strip()[:80]}")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        lines.append(f"direct_openai=ERR ({elapsed:.2f}s) {type(exc).__name__}: {exc}")

    t1 = time.perf_counter()
    try:
        text = run_agent_query_fn("Reply with OK only.", timeout_sec=30, strategy="simple")
        elapsed = time.perf_counter() - t1
        lines.append(f"agent_simple=OK ({elapsed:.2f}s) text={text.strip()[:80]}")
    except Exception as exc:
        elapsed = time.perf_counter() - t1
        lines.append(f"agent_simple=ERR ({elapsed:.2f}s) {type(exc).__name__}: {exc}")

    return "\n".join(lines)
