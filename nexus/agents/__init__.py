from nexus.agents.base import AgentOutput, BaseAgent
from nexus.agents.bear import BearAgent
from nexus.agents.bull import BullAgent
from nexus.agents.coordinator import CoordinatorAgent
from nexus.agents.event import EventAgent
from nexus.agents.execution import ExecutionAgent
from nexus.agents.fundamental import FundamentalAgent
from nexus.agents.macro import MacroAgent
from nexus.agents.market_data import MarketDataAgent
from nexus.agents.portfolio import PortfolioAgent
from nexus.agents.quantitative import QuantitativeAgent
from nexus.agents.rl_agent import RLAgent
from nexus.agents.risk import RiskAgent
from nexus.agents.sentiment import SentimentAgent
from nexus.agents.technical import TechnicalAgent

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "market_data": MarketDataAgent,
    "technical": TechnicalAgent,
    "fundamental": FundamentalAgent,
    "quantitative": QuantitativeAgent,
    "sentiment": SentimentAgent,
    "macro": MacroAgent,
    "event": EventAgent,
    "rl_agent": RLAgent,
    "bull": BullAgent,
    "bear": BearAgent,
    "risk": RiskAgent,
    "portfolio": PortfolioAgent,
    "execution": ExecutionAgent,
    "coordinator": CoordinatorAgent,
}

__all__ = [
    "AGENT_REGISTRY",
    "AgentOutput",
    "BaseAgent",
    "BearAgent",
    "BullAgent",
    "CoordinatorAgent",
    "EventAgent",
    "ExecutionAgent",
    "FundamentalAgent",
    "MacroAgent",
    "MarketDataAgent",
    "PortfolioAgent",
    "QuantitativeAgent",
    "RLAgent",
    "RiskAgent",
    "SentimentAgent",
    "TechnicalAgent",
]
