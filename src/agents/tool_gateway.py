"""RiskSentinel Tool Gateway.

Single entrypoint for deterministic tool invocation with:
- timeout + retry
- contract validation (MCP-ready envelope)
- normalized error taxonomy
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

TOOL_CONTRACT_VERSION = "mcp.tool.result.v1"


@dataclass(frozen=True)
class GatewayResult:
    tool: str
    status: str
    data: dict[str, Any]
    error_code: str | None
    retryable: bool
    message: str
    attempts: int
    duration_ms: int


class ToolGatewayError(RuntimeError):
    """Raised when tool execution fails and no retry budget remains."""


class ToolGateway:
    """Deterministic gateway for tool invocations."""

    def __init__(
        self,
        tools: dict[str, Callable[..., str]],
        timeout_sec: float = 8.0,
        max_retries: int = 1,
    ) -> None:
        self._tools = dict(tools)
        self._timeout_sec = float(timeout_sec)
        self._max_retries = int(max_retries)

    @property
    def allowed_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    def invoke(self, tool_name: str, **kwargs: Any) -> GatewayResult:
        if tool_name not in self._tools:
            raise ToolGatewayError(f"TOOL_NOT_ALLOWED:{tool_name}")

        attempts = 0
        t0 = time.perf_counter()
        last_error: GatewayResult | None = None

        while attempts <= self._max_retries:
            attempts += 1
            raw: str | None = None
            try:
                raw = self._run_with_timeout(self._tools[tool_name], kwargs)
                parsed = self._parse_and_validate_contract(tool_name, raw)
                result = GatewayResult(
                    tool=tool_name,
                    status=str(parsed.get("status", "error")),
                    data=parsed.get("data", {}) if isinstance(parsed.get("data"), dict) else {},
                    error_code=parsed.get("error_code"),
                    retryable=bool(parsed.get("retryable", False)),
                    message=str(parsed.get("message", "")).strip(),
                    attempts=attempts,
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                )
                if result.status == "ok":
                    return result
                last_error = result
                if not result.retryable or attempts > self._max_retries:
                    return result
            except TimeoutError:
                last_error = GatewayResult(
                    tool=tool_name,
                    status="error",
                    data={},
                    error_code="TIMEOUT",
                    retryable=True,
                    message=f"Tool timed out after {self._timeout_sec:.1f}s",
                    attempts=attempts,
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                )
                if attempts > self._max_retries:
                    return last_error
            except Exception as exc:
                code, retryable = self._classify_error(exc)
                last_error = GatewayResult(
                    tool=tool_name,
                    status="error",
                    data={},
                    error_code=code,
                    retryable=retryable,
                    message=str(exc),
                    attempts=attempts,
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                )
                if not retryable or attempts > self._max_retries:
                    return last_error

        if last_error is not None:
            return last_error
        raise ToolGatewayError(f"Unknown gateway failure for tool={tool_name}")

    def _run_with_timeout(self, fn: Callable[..., str], kwargs: dict[str, Any]) -> str:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(fn, **kwargs)
            try:
                out = fut.result(timeout=self._timeout_sec)
            except concurrent.futures.TimeoutError as exc:
                fut.cancel()
                raise TimeoutError(str(exc)) from exc
        if not isinstance(out, str):
            raise TypeError("Tool returned non-string payload")
        return out

    def _parse_and_validate_contract(self, tool_name: str, raw: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw)
        except Exception as exc:
            raise ValueError(f"SCHEMA_INVALID: invalid JSON from {tool_name}") from exc

        if not isinstance(payload, dict):
            raise ValueError("SCHEMA_INVALID: payload is not an object")

        if payload.get("contract_version") != TOOL_CONTRACT_VERSION:
            raise ValueError("SCHEMA_INVALID: unsupported contract_version")

        if payload.get("tool") != tool_name:
            raise ValueError("SCHEMA_INVALID: tool mismatch")

        status = payload.get("status")
        if status not in {"ok", "error"}:
            raise ValueError("SCHEMA_INVALID: status must be ok|error")

        if "data" not in payload:
            raise ValueError("SCHEMA_INVALID: missing data field")

        return payload

    @staticmethod
    def _classify_error(exc: Exception) -> tuple[str, bool]:
        text = f"{type(exc).__name__}: {exc}".lower()
        if "timeout" in text:
            return "TIMEOUT", True
        if "rate limit" in text or "429" in text:
            return "RATE_LIMIT", True
        if "invalid" in text or isinstance(exc, (ValueError, TypeError)):
            return "INVALID_INPUT", False
        if "not found" in text or "missing" in text:
            return "NOT_FOUND", False
        if "schema_invalid" in text:
            return "SCHEMA_INVALID", False
        return "INTERNAL_ERROR", False
