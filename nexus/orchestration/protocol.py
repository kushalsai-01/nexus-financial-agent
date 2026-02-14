from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from uuid import uuid4

from nexus.core.logging import get_logger

logger = get_logger("orchestration.protocol")


class MCPServer:
    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._resources: dict[str, dict[str, Any]] = {}

    def register_tool(self, name: str, description: str, handler: Any, schema: dict[str, Any] | None = None) -> None:
        self._tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "input_schema": schema or {},
        }
        logger.info(f"MCP tool registered: {name}")

    def register_resource(self, uri: str, name: str, description: str, handler: Any) -> None:
        self._resources[uri] = {
            "uri": uri,
            "name": name,
            "description": description,
            "handler": handler,
        }

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in self._tools.values()
        ]

    def list_resources(self) -> list[dict[str, Any]]:
        return [
            {
                "uri": r["uri"],
                "name": r["name"],
                "description": r["description"],
            }
            for r in self._resources.values()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"MCP tool not found: {name}")

        handler = tool["handler"]
        args = arguments or {}

        start = time.monotonic()
        try:
            result = handler(**args)
            if hasattr(result, "__await__"):
                result = await result
            latency = (time.monotonic() - start) * 1000
            logger.info(f"MCP tool {name} executed in {latency:.0f}ms")
            return result
        except Exception as e:
            logger.error(f"MCP tool {name} failed: {e}")
            raise

    async def read_resource(self, uri: str) -> Any:
        resource = self._resources.get(uri)
        if not resource:
            raise ValueError(f"MCP resource not found: {uri}")

        handler = resource["handler"]
        try:
            result = handler()
            if hasattr(result, "__await__"):
                result = await result
            return result
        except Exception as e:
            logger.error(f"MCP resource {uri} read failed: {e}")
            raise


class A2AMessage:
    def __init__(
        self,
        sender: str,
        receiver: str,
        content: dict[str, Any],
        message_type: str = "analysis",
        correlation_id: str | None = None,
    ) -> None:
        self.id = str(uuid4())
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.message_type = message_type
        self.correlation_id = correlation_id or str(uuid4())
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "message_type": self.message_type,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
        }


class A2ABroker:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[A2AMessage]] = {}
        self._history: list[A2AMessage] = []
        self._subscribers: dict[str, list[Any]] = {}

    def register_agent(self, agent_name: str) -> None:
        if agent_name not in self._queues:
            self._queues[agent_name] = asyncio.Queue()

    async def send(self, message: A2AMessage) -> None:
        self._history.append(message)
        receiver = message.receiver

        if receiver == "*":
            for name, queue in self._queues.items():
                if name != message.sender:
                    await queue.put(message)
        elif receiver in self._queues:
            await self._queues[receiver].put(message)
        else:
            logger.warning(f"A2A receiver not found: {receiver}")

        for callback in self._subscribers.get(message.message_type, []):
            try:
                result = callback(message)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.error(f"A2A subscriber error: {e}")

    async def receive(self, agent_name: str, timeout: float = 30.0) -> A2AMessage | None:
        if agent_name not in self._queues:
            return None
        try:
            return await asyncio.wait_for(self._queues[agent_name].get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def subscribe(self, message_type: str, callback: Any) -> None:
        if message_type not in self._subscribers:
            self._subscribers[message_type] = []
        self._subscribers[message_type].append(callback)

    def get_history(self, agent_name: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        messages = self._history
        if agent_name:
            messages = [m for m in messages if m.sender == agent_name or m.receiver == agent_name]
        return [m.to_dict() for m in messages[-limit:]]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "registered_agents": list(self._queues.keys()),
            "total_messages": len(self._history),
            "pending_messages": {name: q.qsize() for name, q in self._queues.items()},
        }
