from __future__ import annotations

from string import Template
from typing import Any


SYSTEM_BASE = (
    "You are a senior quantitative analyst at an autonomous hedge fund. "
    "You analyze financial data with precision and provide structured JSON responses. "
    "All numerical values must be precise to 4 decimal places where applicable. "
    "Never fabricate data. If information is insufficient, state confidence as 0."
)


TECHNICAL_ANALYSIS_PROMPT = Template(
    "Analyze the following technical indicators for $ticker.\n\n"
    "Price Data:\n$price_data\n\n"
    "Technical Indicators:\n$indicators\n\n"
    "Provide your analysis as JSON with these exact fields:\n"
    "{\n"
    '  "signal": "BUY" | "SELL" | "HOLD",\n'
    '  "strength": 0.0 to 1.0,\n'
    '  "support_level": float,\n'
    '  "resistance_level": float,\n'
    '  "trend": "bullish" | "bearish" | "neutral",\n'
    '  "key_patterns": [list of detected chart patterns],\n'
    '  "reasoning": "string"\n'
    "}"
)


FUNDAMENTAL_ANALYSIS_PROMPT = Template(
    "Analyze the fundamental data for $ticker.\n\n"
    "Financial Statements:\n$financials\n\n"
    "Key Metrics:\n$metrics\n\n"
    "Industry Comparisons:\n$industry\n\n"
    "Provide your analysis as JSON with these exact fields:\n"
    "{\n"
    '  "fair_value": float,\n'
    '  "upside_pct": float,\n'
    '  "downside_pct": float,\n'
    '  "quality_score": 0.0 to 1.0,\n'
    '  "moat_rating": "wide" | "narrow" | "none",\n'
    '  "growth_outlook": "strong" | "moderate" | "weak" | "declining",\n'
    '  "reasoning": "string"\n'
    "}"
)


SENTIMENT_ANALYSIS_PROMPT = Template(
    "Analyze market sentiment for $ticker based on the following data.\n\n"
    "News Headlines:\n$news\n\n"
    "Social Media Data:\n$social\n\n"
    "Analyst Ratings:\n$analyst\n\n"
    "Provide your analysis as JSON with these exact fields:\n"
    "{\n"
    '  "sentiment": "very_bullish" | "bullish" | "neutral" | "bearish" | "very_bearish",\n'
    '  "score": -1.0 to 1.0,\n'
    '  "volume_indicator": "high" | "normal" | "low",\n'
    '  "key_topics": [list of key discussion topics],\n'
    '  "catalyst_events": [list of upcoming catalysts],\n'
    '  "reasoning": "string"\n'
    "}"
)


QUANTITATIVE_ANALYSIS_PROMPT = Template(
    "Perform quantitative analysis on $ticker.\n\n"
    "Factor Data:\n$factors\n\n"
    "Statistical Metrics:\n$statistics\n\n"
    "Correlation Matrix:\n$correlations\n\n"
    "Provide your analysis as JSON with these exact fields:\n"
    "{\n"
    '  "alpha_score": -3.0 to 3.0,\n'
    '  "factor_breakdown": {\n'
    '    "momentum": float,\n'
    '    "value": float,\n'
    '    "quality": float,\n'
    '    "volatility": float,\n'
    '    "size": float\n'
    "  },\n"
    '  "sharpe_estimate": float,\n'
    '  "var_95": float,\n'
    '  "reasoning": "string"\n'
    "}"
)


MACRO_ANALYSIS_PROMPT = Template(
    "Analyze macroeconomic conditions and their impact on $ticker.\n\n"
    "Economic Indicators:\n$indicators\n\n"
    "Fed Policy:\n$fed_policy\n\n"
    "Global Events:\n$global_events\n\n"
    "Provide your analysis as JSON with these exact fields:\n"
    "{\n"
    '  "macro_outlook": "favorable" | "neutral" | "unfavorable",\n'
    '  "rate_sensitivity": -1.0 to 1.0,\n'
    '  "sector_rotation_signal": "overweight" | "neutral" | "underweight",\n'
    '  "tail_risks": [list of macro tail risks],\n'
    '  "reasoning": "string"\n'
    "}"
)


EVENT_ANALYSIS_PROMPT = Template(
    "Analyze upcoming events and catalysts for $ticker.\n\n"
    "Calendar Events:\n$events\n\n"
    "Historical Event Impact:\n$history\n\n"
    "Provide your analysis as JSON with these exact fields:\n"
    "{\n"
    '  "events": [\n'
    "    {\n"
    '      "event": "string",\n'
    '      "date": "YYYY-MM-DD",\n'
    '      "expected_impact": -1.0 to 1.0,\n'
    '      "confidence": 0.0 to 1.0\n'
    "    }\n"
    "  ],\n"
    '  "net_catalyst_score": -1.0 to 1.0,\n'
    '  "reasoning": "string"\n'
    "}"
)


BULL_CASE_PROMPT = Template(
    "You are the BULL advocate. Build the strongest possible bullish case for $ticker.\n\n"
    "All Agent Analyses:\n$analyses\n\n"
    "Current Price: $current_price\n"
    "Position Context:\n$context\n\n"
    "Provide your bull case as JSON with these exact fields:\n"
    "{\n"
    '  "conviction": 1 to 10,\n'
    '  "position_action": "buy" | "add" | "hold",\n'
    '  "position_size_pct": float,\n'
    '  "entry_price": float,\n'
    '  "price_target": float,\n'
    '  "catalyst_timeline_days": int,\n'
    '  "upside_scenarios": [\n'
    "    {\n"
    '      "scenario": "string",\n'
    '      "probability": 0.0 to 1.0,\n'
    '      "target_price": float\n'
    "    }\n"
    "  ],\n"
    '  "key_arguments": [list of bull arguments],\n'
    '  "reasoning": "string"\n'
    "}"
)


BEAR_CASE_PROMPT = Template(
    "You are the BEAR advocate. Build the strongest possible bearish case for $ticker.\n\n"
    "All Agent Analyses:\n$analyses\n\n"
    "Current Price: $current_price\n"
    "Position Context:\n$context\n\n"
    "Provide your bear case as JSON with these exact fields:\n"
    "{\n"
    '  "conviction": 1 to 10,\n'
    '  "action": "sell" | "reduce" | "avoid",\n'
    '  "downside_target": float,\n'
    '  "stop_loss": float,\n'
    '  "downside_scenarios": [\n'
    "    {\n"
    '      "scenario": "string",\n'
    '      "probability": 0.0 to 1.0,\n'
    '      "target_price": float\n'
    "    }\n"
    "  ],\n"
    '  "key_arguments": [list of bear arguments],\n'
    '  "reasoning": "string"\n'
    "}"
)


RISK_ASSESSMENT_PROMPT = Template(
    "Evaluate the risk of the proposed trade for $ticker.\n\n"
    "Proposed Action:\n$action\n\n"
    "Portfolio State:\n$portfolio\n\n"
    "Risk Limits:\n$limits\n\n"
    "Bull/Bear Analysis:\n$debate\n\n"
    "Provide your assessment as JSON with these exact fields:\n"
    "{\n"
    '  "decision": "APPROVE" | "VETO",\n'
    '  "adjusted_size_pct": float,\n'
    '  "adjusted_stop_loss": float,\n'
    '  "risk_score": 0.0 to 1.0,\n'
    '  "position_var": float,\n'
    '  "portfolio_impact": {\n'
    '    "new_exposure_pct": float,\n'
    '    "sector_concentration": float,\n'
    '    "correlation_risk": float\n'
    "  },\n"
    '  "veto_reasons": [list if vetoed],\n'
    '  "conditions": [list of conditions for approval],\n'
    '  "reasoning": "string"\n'
    "}"
)


PORTFOLIO_DECISION_PROMPT = Template(
    "Make the final portfolio decision for $ticker.\n\n"
    "All Analyses:\n$analyses\n\n"
    "Bull Case:\n$bull_case\n\n"
    "Bear Case:\n$bear_case\n\n"
    "Risk Assessment:\n$risk\n\n"
    "Current Portfolio:\n$portfolio\n\n"
    "Kelly Criterion Data:\n$kelly\n\n"
    "Provide your decision as JSON with these exact fields:\n"
    "{\n"
    '  "action": "buy" | "sell" | "hold" | "close",\n'
    '  "size_pct": float,\n'
    '  "limit_price": float | null,\n'
    '  "stop_loss": float,\n'
    '  "take_profit": float,\n'
    '  "order_type": "market" | "limit" | "stop_limit",\n'
    '  "urgency": "immediate" | "patient" | "opportunistic",\n'
    '  "kelly_fraction": float,\n'
    '  "expected_return": float,\n'
    '  "reasoning": "string"\n'
    "}"
)


EXECUTION_PROMPT = Template(
    "Plan the execution strategy for the following trade.\n\n"
    "Trade Order:\n$order\n\n"
    "Market Conditions:\n$market\n\n"
    "Liquidity Data:\n$liquidity\n\n"
    "Provide your execution plan as JSON with these exact fields:\n"
    "{\n"
    '  "strategy": "VWAP" | "TWAP" | "ICEBERG" | "MARKET",\n'
    '  "urgency": "high" | "medium" | "low",\n'
    '  "num_slices": int,\n'
    '  "time_horizon_minutes": int,\n'
    '  "max_participation_rate": float,\n'
    '  "expected_slippage_bps": float,\n'
    '  "reasoning": "string"\n'
    "}"
)


RL_AGENT_PROMPT = Template(
    "Evaluate the reinforcement learning model output for $ticker.\n\n"
    "Model Prediction:\n$prediction\n\n"
    "Feature Importance:\n$features\n\n"
    "Historical Accuracy:\n$accuracy\n\n"
    "Provide your evaluation as JSON with these exact fields:\n"
    "{\n"
    '  "ml_signal": "BUY" | "SELL" | "HOLD",\n'
    '  "confidence": 0.0 to 1.0,\n'
    '  "predicted_return": float,\n'
    '  "model_uncertainty": float,\n'
    '  "feature_drivers": [top 5 features driving the prediction],\n'
    '  "reasoning": "string"\n'
    "}"
)


COORDINATOR_PROMPT = Template(
    "Synthesize all agent analyses and determine the final trading decision.\n\n"
    "Ticker: $ticker\n"
    "Current Price: $current_price\n\n"
    "Agent Outputs:\n$agent_outputs\n\n"
    "Consensus Threshold: $threshold\n\n"
    "Provide your coordination result as JSON with these exact fields:\n"
    "{\n"
    '  "consensus_reached": true | false,\n'
    '  "consensus_signal": "BUY" | "SELL" | "HOLD",\n'
    '  "consensus_confidence": 0.0 to 1.0,\n'
    '  "agent_agreement_pct": float,\n'
    '  "dissenting_agents": [list of agents that disagree],\n'
    '  "recommended_action": {\n'
    '    "side": "buy" | "sell" | "hold",\n'
    '    "size_pct": float,\n'
    '    "confidence": 0.0 to 1.0\n'
    "  },\n"
    '  "reasoning": "string"\n'
    "}"
)


PROMPT_REGISTRY: dict[str, Template] = {
    "technical": TECHNICAL_ANALYSIS_PROMPT,
    "fundamental": FUNDAMENTAL_ANALYSIS_PROMPT,
    "sentiment": SENTIMENT_ANALYSIS_PROMPT,
    "quantitative": QUANTITATIVE_ANALYSIS_PROMPT,
    "macro": MACRO_ANALYSIS_PROMPT,
    "event": EVENT_ANALYSIS_PROMPT,
    "bull": BULL_CASE_PROMPT,
    "bear": BEAR_CASE_PROMPT,
    "risk": RISK_ASSESSMENT_PROMPT,
    "portfolio": PORTFOLIO_DECISION_PROMPT,
    "execution": EXECUTION_PROMPT,
    "rl_agent": RL_AGENT_PROMPT,
    "coordinator": COORDINATOR_PROMPT,
}


def render_prompt(agent_type: str, **kwargs: Any) -> str:
    template = PROMPT_REGISTRY.get(agent_type)
    if not template:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return template.safe_substitute(**kwargs)


def get_system_prompt(agent_type: str) -> str:
    role_additions: dict[str, str] = {
        "technical": " You specialize in technical analysis and chart pattern recognition.",
        "fundamental": " You specialize in fundamental analysis and company valuation.",
        "sentiment": " You specialize in market sentiment analysis and behavioral finance.",
        "quantitative": " You specialize in quantitative factor models and statistical arbitrage.",
        "macro": " You specialize in macroeconomic analysis and cross-asset correlations.",
        "event": " You specialize in event-driven strategies and catalyst analysis.",
        "bull": " You are the BULL advocate. Always present the strongest possible bullish case.",
        "bear": " You are the BEAR advocate. Always present the strongest possible bearish case.",
        "risk": " You are a risk manager focused on capital preservation and drawdown control.",
        "portfolio": " You are a portfolio manager making final allocation decisions using Kelly criterion.",
        "execution": " You specialize in optimal trade execution and minimizing market impact.",
        "rl_agent": " You evaluate machine learning model outputs and assess model reliability.",
        "coordinator": " You synthesize all agent analyses into a unified trading decision.",
    }
    return SYSTEM_BASE + role_additions.get(agent_type, "")
