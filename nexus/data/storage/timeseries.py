from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from nexus.core.config import get_config
from nexus.core.exceptions import DataStorageError
from nexus.core.logging import get_logger

logger = get_logger("data.storage.timeseries")


class TimeseriesStorage:
    def __init__(self) -> None:
        config = get_config()
        self._url = config.storage.influxdb.url
        self._token = config.storage.influxdb.token
        self._org = config.storage.influxdb.org
        self._bucket = config.storage.influxdb.bucket
        self._timeout = config.storage.influxdb.timeout
        self._client: InfluxDBClient | None = None
        self._write_api = None
        self._query_api = None

    def _get_client(self) -> InfluxDBClient:
        if self._client is None:
            self._client = InfluxDBClient(
                url=self._url,
                token=self._token,
                org=self._org,
                timeout=self._timeout,
            )
            self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
            self._query_api = self._client.query_api()
        return self._client

    def write_market_data(
        self,
        ticker: str,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        vwap: float | None = None,
    ) -> None:
        try:
            self._get_client()
            point = (
                Point("market_data")
                .tag("ticker", ticker)
                .field("open", open_)
                .field("high", high)
                .field("low", low)
                .field("close", close)
                .field("volume", volume)
                .time(timestamp, WritePrecision.S)
            )
            if vwap is not None:
                point = point.field("vwap", vwap)
            self._write_api.write(bucket=self._bucket, record=point)
        except Exception as e:
            raise DataStorageError(f"Failed to write market data: {e}") from e

    def write_batch(self, points: list[dict[str, Any]]) -> int:
        try:
            self._get_client()
            influx_points = []
            for p in points:
                point = (
                    Point(p.get("measurement", "market_data"))
                    .tag("ticker", p["ticker"])
                    .field("open", p["open"])
                    .field("high", p["high"])
                    .field("low", p["low"])
                    .field("close", p["close"])
                    .field("volume", p["volume"])
                    .time(p["timestamp"], WritePrecision.S)
                )
                influx_points.append(point)
            self._write_api.write(bucket=self._bucket, record=influx_points)
            logger.info(f"Wrote {len(influx_points)} points to InfluxDB")
            return len(influx_points)
        except Exception as e:
            raise DataStorageError(f"Failed to write batch: {e}") from e

    def query_market_data(
        self,
        ticker: str,
        start: datetime,
        end: datetime | None = None,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            self._get_client()
            end = end or datetime.utcnow()
            field_filter = ""
            if fields:
                field_conditions = " or ".join(
                    [f'r["_field"] == "{f}"' for f in fields]
                )
                field_filter = f"|> filter(fn: (r) => {field_conditions})"

            query = f"""
                from(bucket: "{self._bucket}")
                |> range(start: {start.isoformat()}Z, stop: {end.isoformat()}Z)
                |> filter(fn: (r) => r["_measurement"] == "market_data")
                |> filter(fn: (r) => r["ticker"] == "{ticker}")
                {field_filter}
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            """

            tables = self._query_api.query(query, org=self._org)
            results: list[dict[str, Any]] = []
            for table in tables:
                for record in table.records:
                    results.append(
                        {
                            "timestamp": record.get_time(),
                            "ticker": record.values.get("ticker", ticker),
                            "open": record.values.get("open"),
                            "high": record.values.get("high"),
                            "low": record.values.get("low"),
                            "close": record.values.get("close"),
                            "volume": record.values.get("volume"),
                        }
                    )
            return results
        except Exception as e:
            raise DataStorageError(f"Failed to query market data: {e}") from e

    def write_metric(
        self,
        measurement: str,
        tags: dict[str, str],
        fields: dict[str, float],
        timestamp: datetime | None = None,
    ) -> None:
        try:
            self._get_client()
            point = Point(measurement)
            for k, v in tags.items():
                point = point.tag(k, v)
            for k, v in fields.items():
                point = point.field(k, v)
            if timestamp:
                point = point.time(timestamp, WritePrecision.S)
            self._write_api.write(bucket=self._bucket, record=point)
        except Exception as e:
            raise DataStorageError(f"Failed to write metric: {e}") from e

    def query_metrics(
        self,
        measurement: str,
        tags: dict[str, str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        try:
            self._get_client()
            start = start or datetime.utcnow() - timedelta(hours=24)
            end = end or datetime.utcnow()

            tag_filter = ""
            if tags:
                conditions = " and ".join(
                    [f'r["{k}"] == "{v}"' for k, v in tags.items()]
                )
                tag_filter = f"|> filter(fn: (r) => {conditions})"

            query = f"""
                from(bucket: "{self._bucket}")
                |> range(start: {start.isoformat()}Z, stop: {end.isoformat()}Z)
                |> filter(fn: (r) => r["_measurement"] == "{measurement}")
                {tag_filter}
            """

            tables = self._query_api.query(query, org=self._org)
            results: list[dict[str, Any]] = []
            for table in tables:
                for record in table.records:
                    results.append(
                        {
                            "timestamp": record.get_time(),
                            "field": record.get_field(),
                            "value": record.get_value(),
                            **{
                                k: v
                                for k, v in record.values.items()
                                if k not in ("_time", "_field", "_value", "_measurement")
                            },
                        }
                    )
            return results
        except Exception as e:
            raise DataStorageError(f"Failed to query metrics: {e}") from e

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            return client.ping()
        except Exception:
            return False

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            self._write_api = None
            self._query_api = None
