import json
import time

from src.agents.tool_gateway import TOOL_CONTRACT_VERSION, ToolGateway


def _ok_tool(**kwargs):
    return json.dumps(
        {
            "contract_version": TOOL_CONTRACT_VERSION,
            "tool": "ok_tool",
            "status": "ok",
            "error_code": None,
            "retryable": False,
            "data": {"echo": kwargs},
        }
    )


def _invalid_schema_tool(**kwargs):
    _ = kwargs
    return "not-json"


def _slow_tool(**kwargs):
    _ = kwargs
    time.sleep(0.2)
    return json.dumps(
        {
            "contract_version": TOOL_CONTRACT_VERSION,
            "tool": "slow_tool",
            "status": "ok",
            "error_code": None,
            "retryable": False,
            "data": {"done": True},
        }
    )


def test_tool_gateway_ok_path():
    gateway = ToolGateway({"ok_tool": _ok_tool}, timeout_sec=1.0, max_retries=0)
    result = gateway.invoke("ok_tool", x=1)
    assert result.status == "ok"
    assert result.data["echo"]["x"] == 1
    assert result.attempts == 1


def test_tool_gateway_schema_error_is_normalized():
    gateway = ToolGateway({"bad_tool": _invalid_schema_tool}, timeout_sec=1.0, max_retries=0)
    result = gateway.invoke("bad_tool")
    assert result.status == "error"
    assert result.error_code in {"INVALID_INPUT", "SCHEMA_INVALID"}
    assert result.retryable is False


def test_tool_gateway_timeout():
    gateway = ToolGateway({"slow_tool": _slow_tool}, timeout_sec=0.05, max_retries=0)
    result = gateway.invoke("slow_tool")
    assert result.status == "error"
    assert result.error_code == "TIMEOUT"
    assert result.retryable is True
