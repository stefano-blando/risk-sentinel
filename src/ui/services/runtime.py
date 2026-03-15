"""Runtime, access policy, and rate-limit helpers."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import time


def get_runtime_value(st_module, name: str, default: str = "") -> str:
    """Read config from Streamlit secrets first, then env vars."""
    try:
        if hasattr(st_module, "secrets") and name in st_module.secrets:
            return str(st_module.secrets[name]).strip()
    except Exception:
        pass
    return str(os.getenv(name, default)).strip()


def get_runtime_int(st_module, name: str, default: int) -> int:
    raw = get_runtime_value(st_module, name, str(default))
    try:
        return int(raw)
    except Exception:
        return default


def prune_events(events: list[float], now_ts: float, window_sec: int = 60) -> list[float]:
    return [ts for ts in events if (now_ts - ts) <= window_sec]


def agentic_cache_key(op_name: str, **kwargs) -> str:
    raw = json.dumps({"op": op_name, **kwargs}, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def run_agentic_operation(
    *,
    session_state,
    op_name: str,
    cache_key: str,
    fn,
    timeout_sec: int,
    ttl_sec: int,
) -> tuple[dict, bool]:
    """Run deterministic agentic op with timeout + session cache."""
    now = time.time()
    cache = session_state.agentic_ops_cache
    row = cache.get(cache_key)
    if isinstance(row, dict) and (now - float(row.get("ts", 0.0))) <= ttl_sec:
        return dict(row.get("data", {})), True

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            result = future.result(timeout=max(3, int(timeout_sec)))
        except concurrent.futures.TimeoutError:
            future.cancel()
            return {
                "ok": False,
                "error": f"{op_name} timed out after {timeout_sec}s",
                "timeout": True,
            }, False
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }, False

    if not isinstance(result, dict):
        result = {"ok": False, "error": f"{op_name} returned non-dict payload"}
    if "ok" not in result:
        result["ok"] = True
    cache[cache_key] = {"ts": now, "data": result}
    session_state.agentic_ops_cache = cache
    return result, False


def get_gpt_access_policy(
    *,
    st_module,
    session_state,
) -> dict:
    """GPT access is open by default (no judge gate)."""
    del st_module, session_state
    return {"gate_enabled": False, "allowed": True, "reason": "open"}


def unlock_judge_access(
    *,
    st_module,
    session_state,
    user_code: str,
) -> bool:
    del st_module, user_code
    session_state.judge_unlocked = True
    return True


def check_gpt_rate_limit(
    *,
    session_state,
    get_runtime_int_fn,
    get_global_bucket_fn,
    prune_events_fn,
) -> tuple[bool, str]:
    """Soft limiter to keep demo stable and avoid quota spikes."""
    max_session = get_runtime_int_fn("GPT_MAX_CALLS_PER_SESSION", 120)
    max_per_min_session = get_runtime_int_fn("GPT_MAX_CALLS_PER_MINUTE_SESSION", 8)
    max_per_min_global = get_runtime_int_fn("GPT_MAX_CALLS_PER_MINUTE_GLOBAL", 20)
    max_per_day_global = get_runtime_int_fn("GPT_MAX_CALLS_PER_DAY_GLOBAL", 600)

    now = time.time()
    session_state.gpt_rate_events = prune_events_fn(session_state.gpt_rate_events, now, 60)
    bucket = get_global_bucket_fn()
    bucket["events"] = prune_events_fn(bucket.get("events", []), now, 60)
    day_key = time.strftime("%Y-%m-%d", time.gmtime(now))
    if bucket.get("day_key") != day_key:
        bucket["day_key"] = day_key
        bucket["day_calls"] = 0

    if session_state.gpt_calls_total_session >= max_session:
        return False, f"session_cap_reached:{max_session}"
    if len(session_state.gpt_rate_events) >= max_per_min_session:
        return False, f"session_rate_limit:{max_per_min_session}/min"
    if len(bucket["events"]) >= max_per_min_global:
        return False, f"global_rate_limit:{max_per_min_global}/min"
    if int(bucket.get("day_calls", 0)) >= max_per_day_global:
        return False, f"global_daily_cap:{max_per_day_global}/day"
    return True, "ok"


def register_gpt_call(
    *,
    session_state,
    get_global_bucket_fn,
    prune_events_fn,
) -> None:
    now = time.time()
    session_state.gpt_rate_events = prune_events_fn(session_state.gpt_rate_events, now, 60)
    session_state.gpt_rate_events.append(now)
    session_state.gpt_calls_total_session += 1
    bucket = get_global_bucket_fn()
    bucket["events"] = prune_events_fn(bucket.get("events", []), now, 60)
    bucket["events"].append(now)
    day_key = time.strftime("%Y-%m-%d", time.gmtime(now))
    if bucket.get("day_key") != day_key:
        bucket["day_key"] = day_key
        bucket["day_calls"] = 0
    bucket["day_calls"] = int(bucket.get("day_calls", 0)) + 1


def is_gpt_circuit_open(*, session_state) -> tuple[bool, float]:
    now = time.time()
    open_until = float(session_state.get("gpt_circuit_open_until", 0.0) or 0.0)
    if open_until > now:
        return True, open_until - now
    return False, 0.0


def register_gpt_success(*, session_state) -> None:
    session_state.gpt_fail_streak = 0
    session_state.gpt_circuit_open_until = 0.0


def register_gpt_failure(*, session_state, reason: str, cooldown_sec: int) -> None:
    streak = int(session_state.get("gpt_fail_streak", 0)) + 1
    session_state.gpt_fail_streak = streak
    if streak >= 3 or reason in {"rate_limit", "timeout"}:
        session_state.gpt_circuit_open_until = max(
            float(session_state.get("gpt_circuit_open_until", 0.0) or 0.0),
            time.time() + cooldown_sec,
        )


def estimate_eta_seconds(history: list[dict], strategy: str, np_module, fallback: float = 18.0) -> float:
    vals = []
    for row in history[-20:]:
        if row.get("policy", {}).get("router", {}).get("effective_strategy") != strategy:
            continue
        gpt_sec = row.get("timings", {}).get("gpt_sec")
        if isinstance(gpt_sec, (int, float)) and gpt_sec > 0:
            vals.append(float(gpt_sec))
    if not vals:
        return fallback
    return float(np_module.percentile(vals, 75))
