#!/usr/bin/env python3
"""
NEXUS Full Market Analysis — Captures all agent conversations for proof.
Runs the complete 14-agent pipeline and saves structured output.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("NEXUS_ENV", "development")


async def run_full_analysis(tickers: list[str]) -> dict:
    """Run full NEXUS trading graph for each ticker and return structured results."""
    from nexus.core.config import get_config, reset_config
    from nexus.orchestration.graph import TradingGraph
    from nexus.orchestration.router import LLMRouter
    from nexus.agents.base import AgentOutput

    reset_config()
    config = get_config()

    print("=" * 70)
    print(f"  NEXUS FULL MARKET ANALYSIS — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model: gemini-2.5-flash (Google Gemini free tier)")
    print(f"  Tickers: {', '.join(tickers)}")
    print("=" * 70)

    router = LLMRouter()
    graph = TradingGraph(router=router, config=config)

    all_results = {}

    for ticker in tickers:
        print(f"\n{'─' * 70}")
        print(f"  ANALYZING: {ticker}")
        print(f"{'─' * 70}")

        start = time.monotonic()
        state = await graph.run(ticker=ticker, initial_data=None, capital=100_000)
        elapsed = time.monotonic() - start

        # Extract agent conversations
        agent_conversations = {}
        agent_outputs = state.get("agent_outputs", {})
        for agent_name, output in agent_outputs.items():
            if isinstance(output, AgentOutput):
                agent_conversations[agent_name] = {
                    "agent_name": output.agent_name,
                    "agent_type": output.agent_type,
                    "model_used": output.model_used or "N/A (no LLM)",
                    "raw_llm_response": output.raw_response[:2000] if output.raw_response else "",
                    "parsed_output": output.parsed_output,
                    "signal": output.signal.model_dump() if output.signal else None,
                    "confidence": output.confidence,
                    "input_tokens": output.input_tokens,
                    "output_tokens": output.output_tokens,
                    "llm_cost_usd": output.llm_cost_usd,
                    "latency_ms": output.latency_ms,
                    "error": output.error,
                }

                # Print each agent's conversation
                status = "✓" if not output.error else "✗"
                signal_str = ""
                if output.signal:
                    signal_str = f" → {output.signal.signal_type.value} ({output.confidence:.0%})"
                print(f"  {status} {agent_name:<16} {output.latency_ms:>7.0f}ms  {output.input_tokens:>5}in/{output.output_tokens:>5}out{signal_str}")
                if output.error:
                    print(f"    └─ ERROR: {output.error[:100]}")

        # Summary
        recommendation = state.get("recommendation", "HOLD")
        confidence = state.get("confidence", 0)
        reasoning = state.get("reasoning", "N/A")
        cost = state.get("total_llm_cost", 0)
        errors = state.get("errors", [])

        print(f"\n  ┌─ RESULT {'─' * 58}")
        print(f"  │ Recommendation: {recommendation}")
        print(f"  │ Confidence:     {confidence:.1%}")
        print(f"  │ Reasoning:      {reasoning[:120]}")
        print(f"  │ LLM Cost:       ${cost:.6f}")
        print(f"  │ Time:           {elapsed:.1f}s")
        print(f"  │ API Calls:      {sum(1 for a in agent_conversations.values() if a['model_used'] != 'N/A (no LLM)')}")
        if errors:
            print(f"  │ Errors:         {len(errors)}")
            for e in errors[:5]:
                print(f"  │   └─ {e[:80]}")
        print(f"  └{'─' * 67}")

        all_results[ticker] = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "recommendation": recommendation,
            "confidence": confidence,
            "reasoning": reasoning,
            "exposure": state.get("exposure", 0),
            "current_price": state.get("current_price", 0),
            "total_llm_cost_usd": cost,
            "total_latency_ms": state.get("total_latency_ms", 0),
            "elapsed_seconds": round(elapsed, 2),
            "api_calls": sum(1 for a in agent_conversations.values() if a["model_used"] != "N/A (no LLM)"),
            "errors": errors,
            "consensus": state.get("consensus", {}),
            "risk_assessment": state.get("risk_assessment", {}),
            "bull_case": state.get("bull_case", {}),
            "bear_case": state.get("bear_case", {}),
            "portfolio_decision": state.get("portfolio_decision", {}),
            "execution_plan": state.get("execution_plan", {}),
            "agent_conversations": agent_conversations,
        }

    # Router stats
    print(f"\n{'=' * 70}")
    print("  ROUTER STATS")
    stats = router.stats
    print(f"  Total cost:    ${stats['total_cost_usd']:.6f}")
    print(f"  Calls by tier: {stats['calls_by_tier']}")
    print(f"  Active tiers:  {stats['available_tiers']}")
    print(f"{'=' * 70}")

    return {
        "run_metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": "gemini-2.5-flash",
            "provider": "google-gemini",
            "tickers": tickers,
            "total_cost_usd": sum(r["total_llm_cost_usd"] for r in all_results.values()),
            "total_api_calls": sum(r["api_calls"] for r in all_results.values()),
            "router_stats": stats,
        },
        "analyses": all_results,
    }


def main():
    # Major market tickers for today's analysis
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]

    if len(sys.argv) > 1:
        tickers = [t.strip().upper() for t in sys.argv[1].split(",")]

    result = asyncio.run(run_full_analysis(tickers))

    # Save to file for GitHub proof
    output_dir = Path(__file__).parent.parent / "logs"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"full_analysis_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n  Full analysis saved to: {output_file}")
    print(f"  Use this file as GitHub proof of agent conversations.\n")

    return result


if __name__ == "__main__":
    main()
