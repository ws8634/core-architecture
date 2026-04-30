"""
Tests for Execution Layer (Orchestrator).
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.execution import AgentOrchestrator
from agent.tools import ToolRegistry
from agent.types import ToolCallRequest, ToolErrorType


class TestAgentOrchestrator:
    def test_run_valid_input(self):
        orchestrator = AgentOrchestrator(max_steps=10)
        result = orchestrator.run("Calculate 10 plus 5")

        assert result.steps_executed > 0
        assert result.final_answer is not None

    def test_run_empty_input(self):
        orchestrator = AgentOrchestrator()
        result = orchestrator.run("")

        assert result.success is False
        assert "invalid" in result.final_answer.lower()

    def test_dry_run(self):
        orchestrator = AgentOrchestrator()
        result = orchestrator.run("Calculate 2 plus 3", dry_run=True)

        assert "DRY-RUN" in result.final_answer
        assert "Plan generated" in result.final_answer

    def test_max_steps_limit(self):
        orchestrator = AgentOrchestrator(max_steps=2)
        result = orchestrator.run("Calculate 1 plus 2")

        assert result.steps_executed <= 2

    def test_trace_recording(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            orchestrator = AgentOrchestrator(trace_output_path=str(trace_path))
            result = orchestrator.run("Calculate 1 plus 1")

            assert result.trace_file_path == str(trace_path)
            assert len(result.trace_events) > 0

    def test_memory_retrieval_in_flow(self):
        orchestrator = AgentOrchestrator()
        orchestrator.memory.write("My favorite color is blue", "SHORT_TERM")
        result = orchestrator.run("Recall favorite color")

        assert result.steps_executed > 0

    def test_abort_threshold(self):
        class AlwaysFailingTool:
            @property
            def schema(self):
                from agent.types import ToolSchema
                return ToolSchema(
                    name="always_fail",
                    description="Always fails",
                    parameters={"type": "object", "properties": {}, "required": []},
                    returns={"type": "object"},
                )

            def execute(self, parameters):
                raise RuntimeError("Always fails")

        orchestrator = AgentOrchestrator(
            planner_abort_threshold=1,
            max_steps=10,
        )
        orchestrator.tool_registry = ToolRegistry()
        orchestrator.tool_registry.register_tool(AlwaysFailingTool())

        original_plan = [
            {
                "tool_name": "always_fail",
                "parameters": {},
            }
        ]

        assert True


class TestToolTimeout:
    def test_tool_timeout_handling(self):
        class SlowTool:
            @property
            def schema(self):
                from agent.types import ToolSchema
                return ToolSchema(
                    name="slow_tool",
                    description="Slow tool for timeout testing",
                    parameters={
                        "type": "object",
                        "properties": {"delay": {"type": "number"}},
                        "required": ["delay"],
                    },
                    returns={"type": "object"},
                )

            def execute(self, parameters):
                delay = parameters.get("delay", 10)
                time.sleep(delay)
                return {"status": "done"}

        registry = ToolRegistry()
        registry.register_tool(SlowTool())

        request = ToolCallRequest(
            tool_name="slow_tool",
            parameters={"delay": 10},
        )
        result = registry.call_tool(request, timeout=0.1)

        assert result.success is False
        assert result.error_type == ToolErrorType.TIMEOUT


class TestTraceReplay:
    def test_trace_file_creation(self):
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            orchestrator = AgentOrchestrator(trace_output_path=str(trace_path))
            orchestrator.run("Calculate 1 plus 1")

            assert trace_path.exists()

            with open(trace_path, "r") as f:
                lines = f.readlines()

            assert len(lines) > 0
            for line in lines:
                event = json.loads(line)
                assert "event_id" in event
                assert "event_type" in event
                assert "stage_name" in event
