#!/usr/bin/env python3
"""
Generate a comprehensive GitHub-ready markdown report.

Combines:
1. Live market data from yfinance (real-time)
2. Live fundamentals from yfinance
3. Technical indicators computed by NEXUS pipeline
4. Existing Gemini agent conversation logs (proof of 14-agent pipeline)
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ.setdefault("NEXUS_ENV", "development")


async def fetch_live_data(tickers: list[str]) -> dict:
    """Fetch real market data + fundamentals + technicals via NEXUS pipeline."""
    from nexus.data.pipeline import DataPipeline
    
    pipeline = DataPipeline()
    results = {}
    
    for ticker in tickers:
        print(f"  Fetching {ticker}...", end=" ", flush=True)
        try:
            full = await pipeline.get_full_analysis(ticker, lookback_days=90)
            md = full.get("market_data", {})
            fund = full.get("fundamentals", {})
            
            # Also get raw bars for price changes
            bars = await pipeline.get_market_data(
                ticker,
                start=datetime.now() - timedelta(days=90),
                end=datetime.now(),
                include_technicals=False,
                include_features=False,
            )
            
            closes = bars["close"].values if not bars.empty else []
            volumes = bars["volume"].values if not bars.empty and "volume" in bars.columns else []
            n = len(closes)
            
            current = float(closes[-1]) if n > 0 else 0
            change_1d = round((current / float(closes[-2]) - 1) * 100, 2) if n >= 2 else 0
            change_5d = round((current / float(closes[-6]) - 1) * 100, 2) if n >= 6 else 0
            change_20d = round((current / float(closes[-21]) - 1) * 100, 2) if n >= 21 else 0
            high_52w = float(closes[-min(n, 252):].max()) if n > 0 else 0
            low_52w = float(closes[-min(n, 252):].min()) if n > 0 else 0
            avg_vol = int(volumes[-20:].mean()) if len(volumes) >= 20 else 0
            
            results[ticker] = {
                "price": round(current, 2),
                "change_1d": change_1d,
                "change_5d": change_5d,
                "change_20d": change_20d,
                "high_52w": round(high_52w, 2),
                "low_52w": round(low_52w, 2),
                "avg_volume_20d": avg_vol,
                "bars_count": n,
                "technicals": md.get("technical_summary", {}),
                "fundamentals": fund,
            }
            print(f"${current:.2f} ({change_1d:+.2f}%)")
        except Exception as e:
            print(f"ERROR: {e}")
            results[ticker] = {"price": 0, "error": str(e)}
    
    return results


def generate_report(live_data: dict, analysis_log: dict | None) -> str:
    """Generate the full markdown report."""
    now = datetime.now()
    
    report = []
    report.append(f"# NEXUS Financial Agent — Market Analysis Report")
    report.append(f"**Date:** {now.strftime('%B %d, %Y at %H:%M %Z')}")
    report.append(f"**Model:** Google Gemini 2.5 Flash (Free Tier)")
    report.append(f"**Pipeline:** 14-agent multi-agent system with sequential execution")
    report.append("")
    report.append("---")
    report.append("")
    
    # Live Market Data Table
    report.append("## Live Market Data (via yfinance)")
    report.append("")
    report.append("| Ticker | Price | 1D Chg | 5D Chg | 20D Chg | 52W High | 52W Low | Avg Vol (20d) |")
    report.append("|--------|-------|--------|--------|---------|----------|---------|---------------|")
    
    for ticker, data in live_data.items():
        if data.get("error"):
            report.append(f"| {ticker} | ERROR | - | - | - | - | - | - |")
            continue
        report.append(
            f"| **{ticker}** | ${data['price']:.2f} | {data['change_1d']:+.2f}% | "
            f"{data['change_5d']:+.2f}% | {data['change_20d']:+.2f}% | "
            f"${data['high_52w']:.2f} | ${data['low_52w']:.2f} | "
            f"{data['avg_volume_20d']:,} |"
        )
    report.append("")
    
    # Technical Indicators
    report.append("## Technical Indicators (computed by NEXUS pipeline)")
    report.append("")
    report.append("| Ticker | RSI(14) | MACD | ADX | BB% | ATR | Vol Ratio | Volatility(20) |")
    report.append("|--------|---------|------|-----|-----|-----|-----------|----------------|")
    
    for ticker, data in live_data.items():
        tech = data.get("technicals", {})
        if not tech:
            report.append(f"| {ticker} | - | - | - | - | - | - | - |")
            continue
        report.append(
            f"| **{ticker}** | {tech.get('rsi_14', 0):.1f} | {tech.get('macd', 0):.2f} | "
            f"{tech.get('adx', 0):.1f} | {tech.get('bb_pct', 0):.2f} | "
            f"{tech.get('atr', 0):.2f} | {tech.get('volume_ratio', 0):.2f} | "
            f"{tech.get('volatility_20', 0):.4f} |"
        )
    report.append("")
    
    # Fundamentals
    report.append("## Fundamentals (via yfinance)")
    report.append("")
    report.append("| Ticker | Revenue | Net Income | EPS | ROE | Gross Margin | D/E |")
    report.append("|--------|---------|------------|-----|-----|--------------|-----|")
    
    for ticker, data in live_data.items():
        fund = data.get("fundamentals", {})
        if not fund:
            report.append(f"| {ticker} | - | - | - | - | - | - |")
            continue
        rev = fund.get("revenue", 0) or 0
        ni = fund.get("net_income", 0) or 0
        report.append(
            f"| **{ticker}** | ${rev/1e9:.1f}B | ${ni/1e9:.1f}B | "
            f"${fund.get('eps', 0):.2f} | {(fund.get('roe', 0) or 0)*100:.1f}% | "
            f"{(fund.get('gross_margin', 0) or 0)*100:.1f}% | "
            f"{fund.get('debt_to_equity', 0):.2f} |"
        )
    report.append("")
    
    # Agent Pipeline Proof
    report.append("---")
    report.append("")
    report.append("## Agent Pipeline Proof (Gemini 2.5 Flash)")
    report.append("")
    report.append("The following shows a complete run of the 14-agent NEXUS pipeline.")
    report.append("Each agent independently analyzes the market and produces structured JSON output.")
    report.append("")
    
    if analysis_log:
        meta = analysis_log.get("run_metadata", {})
        report.append(f"- **Run Timestamp:** {meta.get('timestamp', 'N/A')}")
        report.append(f"- **Model:** {meta.get('model', 'N/A')}")
        report.append(f"- **Total API Calls:** {meta.get('total_api_calls', 0)}")
        report.append(f"- **Cost:** ${meta.get('total_cost_usd', 0):.6f} (free tier)")
        report.append("")
        
        # Agent conversation details
        analyses = analysis_log.get("analyses", {})
        for ticker, result in analyses.items():
            conversations = result.get("agent_conversations", {})
            report.append(f"### {ticker} — Agent Conversations")
            report.append("")
            report.append(f"**Result:** {result.get('recommendation', 'N/A')} "
                         f"(confidence: {result.get('confidence', 0):.0%})")
            report.append(f"**Elapsed:** {result.get('elapsed_seconds', 0):.1f}s | "
                         f"**API Calls:** {result.get('api_calls', 0)} | "
                         f"**Errors:** {len(result.get('errors', []))}")
            report.append("")
            
            # Agent summary table
            report.append("| # | Agent | Type | Model | Tokens (in/out) | Latency | Signal | Confidence |")
            report.append("|---|-------|------|-------|-----------------|---------|--------|------------|")
            
            for i, (agent_name, conv) in enumerate(conversations.items(), 1):
                model = conv.get("model_used", "N/A")
                in_tok = conv.get("input_tokens", 0)
                out_tok = conv.get("output_tokens", 0)
                lat = conv.get("latency_ms", 0)
                sig = conv.get("signal", {})
                signal_type = sig.get("signal_type", "-") if sig else "-"
                conf = conv.get("confidence", 0)
                error = "ERROR" if conv.get("error") else ""
                report.append(
                    f"| {i} | **{agent_name}** | {conv.get('agent_type', '-')} | "
                    f"{model} | {in_tok}/{out_tok} | {lat:.0f}ms | "
                    f"{signal_type} {error} | {conf:.0%} |"
                )
            report.append("")
            
            # Selected agent responses (show key agents)
            key_agents = ["technical", "fundamental", "bull", "bear", "coordinator", "risk", "portfolio"]
            for agent_name in key_agents:
                conv = conversations.get(agent_name, {})
                if not conv:
                    continue
                parsed = conv.get("parsed_output", {})
                if not parsed:
                    continue
                
                report.append(f"<details>")
                report.append(f"<summary><b>{agent_name.upper()}</b> Agent Response</summary>")
                report.append("")
                report.append("```json")
                report.append(json.dumps(parsed, indent=2, default=str))
                report.append("```")
                report.append("")
                
                reasoning = parsed.get("reasoning", "")
                if reasoning:
                    report.append(f"> **Reasoning:** {reasoning[:300]}{'...' if len(reasoning) > 300 else ''}")
                    report.append("")
                report.append("</details>")
                report.append("")
    
    # Architecture summary
    report.append("---")
    report.append("")
    report.append("## Pipeline Architecture")
    report.append("")
    report.append("```")
    report.append("┌─────────────────────────────────────────────────────────────┐")
    report.append("│                    NEXUS Trading Graph                      │")
    report.append("├─────────────────────────────────────────────────────────────┤")
    report.append("│  Phase 1: DATA FETCH                                       │")
    report.append("│    └─ MarketDataAgent (no LLM)  →  price/volume/indicators │")
    report.append("│                                                             │")
    report.append("│  Phase 2: RESEARCH (sequential, 7s cooldown)               │")
    report.append("│    ├─ TechnicalAgent   →  RSI, MACD, support/resistance    │")
    report.append("│    ├─ FundamentalAgent →  fair value, quality, moat        │")
    report.append("│    ├─ SentimentAgent   →  news/social sentiment            │")
    report.append("│    ├─ MacroAgent       →  Fed policy, sector rotation      │")
    report.append("│    ├─ EventAgent       →  catalysts, calendar events       │")
    report.append("│    └─ QuantitativeAgent→  alpha, Sharpe, VaR              │")
    report.append("│                                                             │")
    report.append("│  Phase 3: DEBATE                                           │")
    report.append("│    ├─ BullAgent        →  strongest bullish case           │")
    report.append("│    └─ BearAgent        →  strongest bearish case           │")
    report.append("│                                                             │")
    report.append("│  Phase 4: COORDINATE                                       │")
    report.append("│    └─ CoordinatorAgent →  consensus signal + confidence    │")
    report.append("│                                                             │")
    report.append("│  Phase 5: RISK CHECK                                       │")
    report.append("│    └─ RiskAgent        →  approve/veto + position sizing   │")
    report.append("│                                                             │")
    report.append("│  Phase 6: EXECUTE (if approved)                            │")
    report.append("│    ├─ PortfolioAgent   →  Kelly sizing + order params      │")
    report.append("│    └─ ExecutionAgent   →  algo strategy + slippage est.    │")
    report.append("└─────────────────────────────────────────────────────────────┘")
    report.append("```")
    report.append("")
    
    # Run instructions
    report.append("## How to Run")
    report.append("")
    report.append("```bash")
    report.append("# Set your Gemini API key")
    report.append("export GEMINI_API_KEY='your-key-here'")
    report.append("")
    report.append("# Run full analysis for specific tickers")
    report.append('python scripts/run_full_analysis.py "AAPL,MSFT,NVDA,TSLA,GOOGL"')
    report.append("")
    report.append("# Output saved to logs/full_analysis_YYYYMMDD_HHMMSS.json")
    report.append("```")
    report.append("")
    report.append(f"---")
    report.append(f"*Generated by NEXUS Financial Agent on {now.strftime('%Y-%m-%d %H:%M:%S')}*")
    
    return "\n".join(report)


def main():
    tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]
    
    print("=" * 60)
    print("  NEXUS Report Generator")
    print("=" * 60)
    
    # 1. Fetch live market data
    print("\n  Fetching live market data...")
    live_data = asyncio.run(fetch_live_data(tickers))
    
    # 2. Load existing analysis log
    analysis_log = None
    log_dir = Path(__file__).parent.parent / "logs"
    log_files = sorted(log_dir.glob("full_analysis_*.json"), reverse=True)
    if log_files:
        print(f"\n  Loading existing analysis: {log_files[0].name}")
        with open(log_files[0]) as f:
            analysis_log = json.load(f)
    
    # 3. Generate report
    print("\n  Generating report...")
    report = generate_report(live_data, analysis_log)
    
    # 4. Save
    output_file = Path(__file__).parent.parent / "MARKET_ANALYSIS.md"
    output_file.write_text(report, encoding="utf-8")
    print(f"\n  Report saved to: {output_file}")
    
    # Also save as a timestamped log
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"market_report_{ts}.md"
    log_dir.mkdir(exist_ok=True)
    log_file.write_text(report, encoding="utf-8")
    print(f"  Also saved to: {log_file}")
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
