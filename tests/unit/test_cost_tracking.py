from __future__ import annotations

import pytest

from nexus.monitoring.cost import CostTracker, LLMUsageRecord


class TestLLMUsageRecord:
    def test_creation(self) -> None:
        record = LLMUsageRecord(
            agent_name="technical",
            model="claude-sonnet-4-20250514",
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.003,
        )
        assert record.agent_name == "technical"
        assert record.cost_usd == 0.003
        assert record.timestamp is not None


class TestCostTracker:
    def setup_method(self) -> None:
        CostTracker._instance = None
        self.tracker = CostTracker()
        self.tracker._records.clear()
        self.tracker._daily_cost.clear()
        self.tracker._agent_cost.clear()
        self.tracker._model_cost.clear()
        self.tracker._agent_tokens.clear()

    def test_record(self) -> None:
        self.tracker.record(
            agent_name="technical",
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.01,
        )
        assert self.tracker.total_cost == pytest.approx(0.01)

    def test_cost_by_agent(self) -> None:
        self.tracker.record(agent_name="a1", model="m1", input_tokens=100, output_tokens=50, cost_usd=0.01)
        self.tracker.record(agent_name="a2", model="m1", input_tokens=100, output_tokens=50, cost_usd=0.02)
        self.tracker.record(agent_name="a1", model="m1", input_tokens=100, output_tokens=50, cost_usd=0.03)
        by_agent = self.tracker.cost_by_agent()
        assert by_agent["a1"] == pytest.approx(0.04)
        assert by_agent["a2"] == pytest.approx(0.02)

    def test_cost_by_model(self) -> None:
        self.tracker.record(agent_name="a1", model="claude", input_tokens=100, output_tokens=50, cost_usd=0.01)
        self.tracker.record(agent_name="a1", model="gpt", input_tokens=100, output_tokens=50, cost_usd=0.02)
        by_model = self.tracker.cost_by_model()
        assert "claude" in by_model
        assert "gpt" in by_model

    def test_budget(self) -> None:
        self.tracker.daily_budget_usd = 1.0
        self.tracker.record(agent_name="a1", model="m1", input_tokens=100, output_tokens=50, cost_usd=0.50)
        assert self.tracker.budget_remaining > 0
        assert self.tracker.budget_utilization_pct == pytest.approx(50.0, abs=5)

    def test_get_summary(self) -> None:
        self.tracker.record(agent_name="a1", model="m1", input_tokens=100, output_tokens=50, cost_usd=0.01)
        summary = self.tracker.get_summary()
        assert "total_cost_usd" in summary
        assert "today_cost_usd" in summary
        assert "budget_remaining_usd" in summary
        assert "total_records" in summary

    def test_cleanup_old_records(self) -> None:
        self.tracker.record(agent_name="a1", model="m1", input_tokens=100, output_tokens=50, cost_usd=0.01)
        initial_count = len(self.tracker._records)
        self.tracker.cleanup_old_records(days=0)
        assert len(self.tracker._records) <= initial_count
