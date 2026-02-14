from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator
from uuid import uuid4

from nexus.core.logging import get_logger

logger = get_logger("monitoring.trace")


class Span:
    def __init__(
        self,
        name: str,
        trace_id: str,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.span_id = uuid4().hex[:16]
        self.name = name
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.metadata = metadata or {}
        self.start_time = datetime.now()
        self.end_time: datetime | None = None
        self.duration_ms: float = 0.0
        self.status: str = "running"
        self.input_data: dict[str, Any] = {}
        self.output_data: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self._start_mono = time.monotonic()

    def set_input(self, data: dict[str, Any]) -> None:
        self.input_data = data

    def set_output(self, data: dict[str, Any]) -> None:
        self.output_data = data

    def add_event(self, name: str, data: dict[str, Any] | None = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "data": data or {},
        })

    def end(self, status: str = "ok") -> None:
        self.end_time = datetime.now()
        self.duration_ms = (time.monotonic() - self._start_mono) * 1000
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "metadata": self.metadata,
            "input": self.input_data,
            "output": self.output_data,
            "events": self.events,
        }


class Trace:
    def __init__(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        self.trace_id = uuid4().hex
        self.name = name
        self.metadata = metadata or {}
        self.spans: list[Span] = []
        self.start_time = datetime.now()
        self.end_time: datetime | None = None
        self.total_cost_usd: float = 0.0
        self.total_tokens: int = 0

    def create_span(
        self,
        name: str,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        span = Span(name=name, trace_id=self.trace_id, parent_id=parent_id, metadata=metadata)
        self.spans.append(span)
        return span

    def end(self) -> None:
        self.end_time = datetime.now()
        for span in self.spans:
            if span.status == "running":
                span.end("ok")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata,
            "spans": [s.to_dict() for s in self.spans],
        }


class LangfuseTracer:
    def __init__(
        self,
        public_key: str = "",
        secret_key: str = "",
        host: str = "https://cloud.langfuse.com",
        enabled: bool = False,
    ) -> None:
        self.enabled = enabled
        self._client: Any = None
        self._host = host

        if enabled and public_key and secret_key:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                logger.info("Langfuse tracer initialized")
            except ImportError:
                logger.warning("langfuse package not installed, tracing disabled")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                self.enabled = False

    def create_trace(self, name: str, metadata: dict[str, Any] | None = None) -> Trace:
        trace = Trace(name=name, metadata=metadata)
        if self._client:
            try:
                self._client.trace(
                    id=trace.trace_id,
                    name=name,
                    metadata=metadata or {},
                )
            except Exception as e:
                logger.error(f"Failed to create Langfuse trace: {e}")
        return trace

    def log_generation(
        self,
        trace: Trace,
        name: str,
        model: str,
        prompt: str,
        completion: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        span = trace.create_span(name=name, metadata={"model": model, **(metadata or {})})
        span.set_input({"prompt": prompt[:500]})
        span.set_output({"completion": completion[:500]})
        span.end()
        span.duration_ms = latency_ms

        trace.total_cost_usd += cost_usd
        trace.total_tokens += input_tokens + output_tokens

        if self._client:
            try:
                self._client.generation(
                    trace_id=trace.trace_id,
                    name=name,
                    model=model,
                    input=prompt[:2000],
                    output=completion[:2000],
                    usage={"input": input_tokens, "output": output_tokens},
                    metadata=metadata or {},
                )
            except Exception as e:
                logger.error(f"Failed to log generation to Langfuse: {e}")

        return span

    def log_tool_call(
        self,
        trace: Trace,
        tool_name: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        latency_ms: float = 0.0,
    ) -> Span:
        span = trace.create_span(name=f"tool:{tool_name}")
        span.set_input(input_data)
        span.set_output(output_data)
        span.end()
        span.duration_ms = latency_ms

        if self._client:
            try:
                self._client.span(
                    trace_id=trace.trace_id,
                    name=f"tool:{tool_name}",
                    input=input_data,
                    output=output_data,
                )
            except Exception as e:
                logger.error(f"Failed to log tool call to Langfuse: {e}")

        return span

    def end_trace(self, trace: Trace) -> None:
        trace.end()
        if self._client:
            try:
                self._client.trace(
                    id=trace.trace_id,
                    metadata={
                        "total_cost_usd": trace.total_cost_usd,
                        "total_tokens": trace.total_tokens,
                    },
                )
            except Exception as e:
                logger.error(f"Failed to end Langfuse trace: {e}")

    def flush(self) -> None:
        if self._client:
            try:
                self._client.flush()
            except Exception:
                pass

    @contextmanager
    def trace_context(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[Trace, None, None]:
        trace = self.create_trace(name, metadata)
        try:
            yield trace
        except Exception as e:
            for span in trace.spans:
                if span.status == "running":
                    span.end("error")
            raise
        finally:
            self.end_trace(trace)


class LangSmithTracer:
    def __init__(
        self,
        api_key: str = "",
        project: str = "nexus",
        enabled: bool = False,
    ) -> None:
        self.enabled = enabled
        self._client: Any = None

        if enabled and api_key:
            try:
                from langsmith import Client
                self._client = Client(api_key=api_key)
                logger.info("LangSmith tracer initialized")
            except ImportError:
                logger.warning("langsmith package not installed")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize LangSmith: {e}")
                self.enabled = False

    def log_run(
        self,
        name: str,
        run_type: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self._client:
            return
        try:
            self._client.create_run(
                name=name,
                run_type=run_type,
                inputs=inputs,
                outputs=outputs,
                extra=extra or {},
                project_name="nexus",
            )
        except Exception as e:
            logger.error(f"Failed to log LangSmith run: {e}")


class TracingManager:
    _instance: TracingManager | None = None

    def __init__(self) -> None:
        self._langfuse: LangfuseTracer | None = None
        self._langsmith: LangSmithTracer | None = None
        self._traces: list[Trace] = []
        self._max_traces: int = 10000

    @classmethod
    def get_instance(cls) -> TracingManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def configure_langfuse(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
    ) -> None:
        self._langfuse = LangfuseTracer(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            enabled=True,
        )

    def configure_langsmith(self, api_key: str, project: str = "nexus") -> None:
        self._langsmith = LangSmithTracer(api_key=api_key, project=project, enabled=True)

    @contextmanager
    def trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[Trace, None, None]:
        trace = Trace(name=name, metadata=metadata)
        self._traces.append(trace)
        if len(self._traces) > self._max_traces:
            self._traces = self._traces[-self._max_traces:]

        if self._langfuse:
            lf_trace = self._langfuse.create_trace(name, metadata)
            trace.trace_id = lf_trace.trace_id

        try:
            yield trace
        finally:
            trace.end()
            if self._langfuse:
                self._langfuse.end_trace(trace)

    def get_recent_traces(self, limit: int = 100) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self._traces[-limit:]]
