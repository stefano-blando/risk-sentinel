"""
RiskSentinel — Orchestrator
Two modes:
1. Workflow mode: sequential pipeline Architect → Quant → Advisor (for full scenarios)
2. Single-agent mode: route to one agent based on query type (for quick questions)
3. Control-plane workflow mode: deterministic state machine + Planner/Workers/Critic

Uses Microsoft Agent Framework's agent-as-tool pattern for composition.
"""

from .architect import create_architect_agent
from .advisor import create_advisor_agent
from .critic import create_critic_agent
from .simulator import create_quant_agent
from .control_plane import RoleModelRouter, run_control_plane_workflow
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


async def run_parallel_workflow(client, query: str, timeout_sec: int = 45) -> str:
    """Run control-plane workflow with explicit state machine and critic hard gate."""
    try:
        from src.utils.azure_config import get_agent_framework_chat_client, get_settings

        settings = get_settings()
        router = RoleModelRouter(
            planner=settings.AZURE_OPENAI_FALLBACK_DEPLOYMENT,
            worker=settings.AZURE_OPENAI_FALLBACK_DEPLOYMENT,
            advisor=settings.AZURE_OPENAI_DEPLOYMENT,
            critic=settings.AZURE_OPENAI_FALLBACK_DEPLOYMENT,
        )
        return await run_control_plane_workflow(
            client,
            query,
            timeout_sec=timeout_sec,
            client_factory=lambda dep: get_agent_framework_chat_client(deployment_name=dep),
            model_router=router,
        )
    except Exception:
        return await run_control_plane_workflow(client, query, timeout_sec=timeout_sec)


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
