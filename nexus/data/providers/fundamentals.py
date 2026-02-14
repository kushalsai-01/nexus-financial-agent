from __future__ import annotations

import asyncio
import re
from datetime import date
from typing import Any

import httpx
import pandas as pd

from nexus.core.config import get_config
from nexus.core.exceptions import DataFetchError
from nexus.core.logging import get_logger
from nexus.core.types import Fundamental
from nexus.data.providers.base import BaseFundamentalsProvider

logger = get_logger("data.providers.fundamentals")

_CIK_CACHE: dict[str, str] = {}

_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"


class FundamentalsProvider(BaseFundamentalsProvider):
    def __init__(self) -> None:
        config = get_config()
        super().__init__(cache_ttl=config.data.fundamentals.cache_ttl)
        self._base_url = config.data.fundamentals.sec_base_url
        self._user_agent = config.data.fundamentals.user_agent
        self._lookback = config.data.fundamentals.filings_lookback
        self._headers = {
            "User-Agent": self._user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    async def _get_cik(self, ticker: str) -> str:
        ticker_upper = ticker.upper()
        if ticker_upper in _CIK_CACHE:
            return _CIK_CACHE[ticker_upper]

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    _TICKER_CIK_URL, headers=self._headers
                )
                response.raise_for_status()
                data = response.json()

            for entry in data.values():
                t = entry.get("ticker", "").upper()
                cik = str(entry.get("cik_str", "")).zfill(10)
                _CIK_CACHE[t] = cik
                if t == ticker_upper:
                    return cik

            raise DataFetchError(f"CIK not found for {ticker}")
        except httpx.HTTPError as e:
            raise DataFetchError(f"Failed to fetch CIK mapping: {e}") from e

    async def fetch_financials(
        self, ticker: str, periods: int = 4
    ) -> pd.DataFrame:
        cache_key = f"financials:{ticker}:{periods}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        cik = await self._get_cik(ticker)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                url = f"{self._base_url}/api/xbrl/companyfacts/CIK{cik}.json"
                response = await client.get(url, headers=self._headers)
                response.raise_for_status()
                data = response.json()

            facts = data.get("facts", {}).get("us-gaap", {})
            records = self._extract_financial_records(facts, periods)

            df = pd.DataFrame(records)
            if not df.empty:
                df = df.sort_values("fiscal_date", ascending=False).head(periods)
            self._cache.set(cache_key, df)
            logger.info(f"Fetched {len(df)} financial periods for {ticker}")
            return df
        except httpx.HTTPError as e:
            raise DataFetchError(
                f"Failed to fetch SEC filings for {ticker}: {e}"
            ) from e

    def _extract_financial_records(
        self, facts: dict[str, Any], periods: int
    ) -> list[dict[str, Any]]:
        metric_mapping = {
            "Revenues": "revenue",
            "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
            "NetIncomeLoss": "net_income",
            "EarningsPerShareBasic": "eps",
            "StockholdersEquity": "equity",
            "Assets": "total_assets",
            "Liabilities": "total_liabilities",
            "OperatingIncomeLoss": "operating_income",
            "GrossProfit": "gross_profit",
        }

        period_data: dict[str, dict[str, Any]] = {}

        for sec_key, local_key in metric_mapping.items():
            concept = facts.get(sec_key, {})
            units = concept.get("units", {})
            values = units.get("USD", units.get("USD/shares", []))

            for entry in values:
                form = entry.get("form", "")
                if form not in ("10-K", "10-Q"):
                    continue

                end_date = entry.get("end", "")
                if not end_date:
                    continue

                if end_date not in period_data:
                    period_data[end_date] = {
                        "fiscal_date": end_date,
                        "period": "annual" if form == "10-K" else "quarterly",
                    }
                period_data[end_date][local_key] = entry.get("val")

        records = sorted(
            period_data.values(),
            key=lambda r: r["fiscal_date"],
            reverse=True,
        )
        return records[:periods]

    async def fetch_ratios(self, ticker: str) -> dict[str, float]:
        cache_key = f"ratios:{ticker}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        df = await self.fetch_financials(ticker, periods=1)
        if df.empty:
            return {}

        row = df.iloc[0]
        ratios: dict[str, float] = {}

        revenue = row.get("revenue")
        net_income = row.get("net_income")
        equity = row.get("equity")
        total_assets = row.get("total_assets")
        total_liabilities = row.get("total_liabilities")
        gross_profit = row.get("gross_profit")
        operating_income = row.get("operating_income")

        if revenue and net_income:
            ratios["net_margin"] = net_income / revenue
        if revenue and gross_profit:
            ratios["gross_margin"] = gross_profit / revenue
        if revenue and operating_income:
            ratios["operating_margin"] = operating_income / revenue
        if equity and net_income:
            ratios["roe"] = net_income / equity
        if total_assets and net_income:
            ratios["roa"] = net_income / total_assets
        if equity and total_liabilities:
            ratios["debt_to_equity"] = total_liabilities / equity
        if total_assets and total_liabilities:
            ratios["current_ratio"] = (
                (total_assets - total_liabilities) / total_liabilities
                if total_liabilities
                else 0.0
            )

        self._cache.set(cache_key, ratios)
        logger.info(f"Computed {len(ratios)} ratios for {ticker}")
        return ratios

    async def fetch_fundamental(self, ticker: str) -> Fundamental | None:
        df = await self.fetch_financials(ticker, periods=1)
        if df.empty:
            return None

        row = df.iloc[0]
        ratios = await self.fetch_ratios(ticker)

        return Fundamental(
            ticker=ticker,
            period=row.get("period", "unknown"),
            fiscal_date=date.fromisoformat(row["fiscal_date"]),
            revenue=row.get("revenue"),
            net_income=row.get("net_income"),
            eps=row.get("eps"),
            roe=ratios.get("roe"),
            roa=ratios.get("roa"),
            debt_to_equity=ratios.get("debt_to_equity"),
            gross_margin=ratios.get("gross_margin"),
            operating_margin=ratios.get("operating_margin"),
        )

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self._base_url}/api/xbrl/companyfacts/CIK0000320193.json",
                    headers=self._headers,
                )
                return response.status_code == 200
        except Exception:
            return False
