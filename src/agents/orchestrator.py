"""
RiskSentinel — Orchestrator
Two modes:
1. Workflow mode: sequential pipeline Architect → Quant → Advisor (for full scenarios)
2. Single-agent mode: route to one agent based on query type (for quick questions)
3. Parallel workflow mode: Architect + Quant in parallel, then Advisor + Critic

Uses Microsoft Agent Framework's agent-as-tool pattern for composition.
"""

import asyncio
import json
import re
from typing import Optional

from .architect import create_architect_agent, ARCHITECT_INSTRUCTIONS
from .advisor import create_advisor_agent, ADVISOR_INSTRUCTIONS
from .critic import create_critic_agent, CRITIC_INSTRUCTIONS
from .simulator import create_quant_agent, QUANT_INSTRUCTIONS
from .tools import ALL_TOOLS

# ---------------------------------------------------------------------------
# ORCHESTRATOR INSTRUCTIONS
# ---------------------------------------------------------------------------
ORCHESTRATOR_INSTRUCTIONS = """You are the RiskSentinel Orchestrator, a systemic risk analysis system.

You coordinate specialist agents to analyze financial contagion risk:
- **TheArchitect**: Builds and analyzes the S&P 500 correlation network topology
- **TheQuant**: Runs shock propagation simulations (DebtRank, cascades)
- **TheAdvisor**: Interprets results and provides risk mitigation recommendations
- **TheCritic**: Validates consistency with deterministic evidence

## How to handle user queries

For a "what-if" scenario (e.g., "What happens if JPM crashes 40%?"):
1. Call TheArchitect to analyze the current network and the target stock's connections
2. Call TheQuant to simulate the shock propagation
3. Call TheAdvisor to interpret results and recommend mitigation
4. Call TheCritic to validate evidence consistency

For simple questions:
- Network structure questions → call TheArchitect
- Simulation requests → call TheQuant
- Risk assessment/advice → call TheAdvisor

## Output Format
After gathering all agent inputs, synthesize a final response that includes:
1. **Network Context** (from Architect): market regime, node importance
2. **Simulation Results** (from Quant): cascade impact, affected nodes
3. **Strategic Advice** (from Advisor): recommendations, risk rating
4. **Validation** (from Critic): consistency status and uncertainty

Always be clear about which agent provided which insight.

## Scope Guardrail
Only answer topics related to:
- financial network topology
- crisis regime context
- contagion/shock propagation
- risk mitigation tied to simulation outputs

If the user asks for generic macro commentary outside these topics, refuse briefly
and ask them to provide a network/crisis/contagion scenario.
"""


# ---------------------------------------------------------------------------
# FULL PIPELINE (Architect → Quant → Advisor via agent-as-tool)
# ---------------------------------------------------------------------------
def create_orchestrator(client):
    """Create the orchestrator agent that uses the three specialists as tools.

    Uses the agent-as-tool pattern: each specialist agent is converted to a
    callable tool that the orchestrator can invoke.

    Args:
        client: An Agent Framework client (AzureOpenAIChatClient or similar).
    """
    architect = create_architect_agent(client)
    quant = create_quant_agent(client)
    advisor = create_advisor_agent(client)
    critic = create_critic_agent(client)

    # Convert specialist agents to tools for the orchestrator
    architect_tool = architect.as_tool(
        name="analyze_network",
        description="Analyze the S&P 500 correlation network: build topology, find systemic nodes, map connections, check market regime.",
        arg_name="query",
        arg_description="What to analyze about the network (e.g., 'Build network for 2025-12-01 and find JPM connections')",
    )
    quant_tool = quant.as_tool(
        name="simulate_shock",
        description="Run a contagion shock simulation on the financial network. Propagate stress from a shocked node and measure cascade impact.",
        arg_name="query",
        arg_description="What shock to simulate (e.g., 'Simulate JPM defaulting with 50% shock using DebtRank')",
    )
    advisor_tool = advisor.as_tool(
        name="get_risk_advice",
        description="Get strategic risk assessment and mitigation recommendations based on network analysis and simulation results.",
        arg_name="query",
        arg_description="What risk advice is needed (e.g., 'Assess systemic risk from JPM shock and recommend hedging strategies')",
    )
    critic_tool = critic.as_tool(
        name="validate_analysis",
        description="Validate that candidate analysis is consistent with deterministic evidence and highlight required fixes.",
        arg_name="query",
        arg_description="Analysis and evidence payload to validate.",
    )

    orchestrator = client.as_agent(
        name="RiskSentinel",
        description="Systemic risk analysis orchestrator. Coordinates network analysis, shock simulation, and risk advisory for financial contagion scenarios.",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        tools=[architect_tool, quant_tool, advisor_tool, critic_tool],
    )

    return orchestrator


# ---------------------------------------------------------------------------
# SIMPLE SINGLE-AGENT MODE (all tools, one agent — for MVP/testing)
# ---------------------------------------------------------------------------
def create_simple_agent(client):
    """Create a single agent with all tools (no multi-agent orchestration).

    Useful for MVP testing when you want to skip the agent-as-tool overhead.
    One agent handles everything directly.

    Args:
        client: An Agent Framework client.
    """
    return client.as_agent(
        name="RiskSentinel",
        description="Systemic risk analysis system for S&P 500 financial contagion.",
        instructions=f"""You are RiskSentinel, an AI system that analyzes systemic financial risk
using network science and contagion simulation.

You have tools to:
1. Build and analyze S&P 500 correlation networks (210 stocks, 11 sectors)
2. Run shock propagation simulations (DebtRank, Linear Threshold, Cascade Removal)
3. Assess market regimes and provide risk recommendations

Scope guardrail:
- Stay strictly within network/crisis/contagion analysis.
- If the user asks for generic macro or unrelated investment advice, refuse briefly and
  redirect to a concrete scenario (ticker/date/shock/model).

When a user asks a "what-if" question (e.g., "What happens if JPM crashes 40%?"):
1. First check the market regime and build the network
2. Run the shock simulation (default: DebtRank)
3. Analyze the results and provide recommendations

For consistency with the UI, use correlation threshold 0.5 unless the user
explicitly asks for a different threshold.

Return ONLY valid JSON (no markdown) with this schema:
{{
  "schema_version":"v1",
  "situation":["..."],
  "quant_results":["..."],
  "risk_rating":"LOW|ELEVATED|HIGH|CRITICAL",
  "actions":["..."],
  "monitoring_triggers":["..."],
  "evidence_used":["..."],
  "notes":"...",
  "insufficient_data":false,
  "uncertainty_score":0.2,
  "confidence_reason":"..."
}}

Rules:
- Prefer numbers from tool outputs; do not invent values.
- Keep each list concise (max 4 items).
- uncertainty_score must be between 0.0 and 1.0.

Data range: 2013-09-06 to 2025-12-04 (3081 daily network snapshots).
""",
        tools=ALL_TOOLS,
    )


# ---------------------------------------------------------------------------
# RUNNER UTILITIES
# ---------------------------------------------------------------------------
async def run_query(agent, query: str) -> str:
    """Run a query against any RiskSentinel agent and return the text response."""
    result = await agent.run(query)
    return result.text


def _extract_json_dict(text: str) -> dict:
    text = text.strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())
    obj = re.search(r"(\{[\s\S]*\})", text)
    if obj:
        candidates.append(obj.group(1).strip())
    for raw in candidates:
        try:
            val = json.loads(raw)
            if isinstance(val, dict):
                return val
        except Exception:
            continue
    return {}


def _clip(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + " ..."


async def run_parallel_workflow(client, query: str, timeout_sec: int = 45) -> str:
    """Run explicit multi-agent workflow with parallel Architect/Quant + Critic validation."""
    architect = create_architect_agent(client)
    quant = create_quant_agent(client)
    advisor = create_advisor_agent(client)
    critic = create_critic_agent(client)

    architect_prompt = (
        "Analyze network context for this request. Focus on regime, topology, connections, and cross-sector channels.\n"
        f"Request: {query}"
    )
    quant_prompt = (
        "Run contagion simulation relevant to this request. Focus on cascade depth, affected nodes, stress tiers, sectors.\n"
        f"Request: {query}"
    )

    architect_task = asyncio.create_task(run_query(architect, architect_prompt))
    quant_task = asyncio.create_task(run_query(quant, quant_prompt))
    architect_out, quant_out = await asyncio.wait_for(
        asyncio.gather(architect_task, quant_task), timeout=timeout_sec
    )

    advisor_prompt = (
        "Synthesize this analysis into strict JSON schema v1 only.\n"
        "Use only values from provided evidence and keep uncertainty explicit.\n\n"
        f"User request:\n{query}\n\n"
        f"Architect evidence:\n{_clip(architect_out)}\n\n"
        f"Quant evidence:\n{_clip(quant_out)}"
    )
    advisor_out = await asyncio.wait_for(run_query(advisor, advisor_prompt), timeout=timeout_sec)

    critic_prompt = (
        "Audit candidate output against evidence and return strict JSON validation payload.\n\n"
        f"User request:\n{query}\n\n"
        f"Evidence (Architect):\n{_clip(architect_out)}\n\n"
        f"Evidence (Quant):\n{_clip(quant_out)}\n\n"
        f"Candidate output:\n{_clip(advisor_out)}"
    )
    critic_out = await asyncio.wait_for(run_query(critic, critic_prompt), timeout=min(timeout_sec, 30))
    critic_json = _extract_json_dict(critic_out)
    advisor_json = _extract_json_dict(advisor_out)

    if critic_json and not bool(critic_json.get("approved", True)):
        revision_prompt = (
            "Revise the candidate JSON output using critic feedback. Return strict JSON schema v1 only.\n\n"
            f"Original user request:\n{query}\n\n"
            f"Original advisor output:\n{_clip(advisor_out)}\n\n"
            f"Critic feedback:\n{json.dumps(critic_json, indent=2)}\n\n"
            f"Architect evidence:\n{_clip(architect_out)}\n\n"
            f"Quant evidence:\n{_clip(quant_out)}"
        )
        revised_out = await asyncio.wait_for(run_query(advisor, revision_prompt), timeout=timeout_sec)
        revised_json = _extract_json_dict(revised_out)
        if revised_json:
            advisor_json = revised_json
        else:
            advisor_out = revised_out

    if advisor_json:
        advisor_json.setdefault("schema_version", "v1")
        advisor_json["validation"] = {
            "critic_approved": bool(critic_json.get("approved", True)) if critic_json else None,
            "critic_issues": critic_json.get("issues", []) if critic_json else [],
            "uncertainty_score": critic_json.get("uncertainty_score"),
            "confidence_reason": critic_json.get("confidence_reason"),
        }
        if "uncertainty_score" not in advisor_json and critic_json.get("uncertainty_score") is not None:
            advisor_json["uncertainty_score"] = critic_json.get("uncertainty_score")
        if "confidence_reason" not in advisor_json and critic_json.get("confidence_reason"):
            advisor_json["confidence_reason"] = critic_json.get("confidence_reason")
        return json.dumps(advisor_json)

    # Fallback when advisor output is not strict JSON.
    fallback = {
        "schema_version": "v1",
        "situation": ["Parallel workflow completed but structured synthesis failed."],
        "quant_results": [
            "Architect and Quant agents executed in parallel.",
            "Advisor output was not valid JSON; using fallback payload.",
        ],
        "risk_rating": "ELEVATED",
        "actions": ["Re-run with simpler prompt or strict schema enforcement."],
        "monitoring_triggers": ["Check validation issues in notes."],
        "evidence_used": ["architect_output", "quant_output", "critic_output"],
        "notes": f"advisor_unstructured={_clip(advisor_out, 280)} | critic={_clip(critic_out, 280)}",
        "insufficient_data": True,
        "uncertainty_score": 0.75,
        "confidence_reason": "Structured synthesis failed validation.",
    }
    return json.dumps(fallback)


async def run_full_scenario(client, ticker: str, shock_pct: int, date: str = "2025-12-01") -> str:
    """Run a complete what-if scenario using the orchestrator.

    Args:
        client: Agent Framework client.
        ticker: Stock ticker to shock.
        shock_pct: Shock percentage (e.g., 40 for 40% crash).
        date: Network date.

    Returns:
        Full analysis text from the orchestrator.
    """
    orchestrator = create_orchestrator(client)
    query = (
        f"What happens if {ticker} crashes by {shock_pct}% on {date}? "
        f"Analyze the network, simulate the contagion, and provide risk mitigation advice."
    )
    return await run_query(orchestrator, query)
