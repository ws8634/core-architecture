"""
Tests for Tool Layer.
"""

import pytest

from agent.tools import (
    CalculatorTool,
    ToolParameterValidator,
    ToolRegistry,
    UnstableAPITool,
)
from agent.types import ToolCallRequest, ToolErrorType


class TestToolParameterValidator:
    def test_validate_missing_required(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        }
        params = {"a": 1}
        is_valid, error = ToolParameterValidator.validate(schema, params)
        assert is_valid is False
        assert "Missing" in error

    def test_validate_wrong_type(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
            },
            "required": ["value"],
        }
        params = {"value": "not a number"}
        is_valid, error = ToolParameterValidator.validate(schema, params)
        assert is_valid is False
        assert "must be a number" in error

    def test_validate_string_min_length(self):
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 5},
            },
            "required": ["text"],
        }
        params = {"text": "abc"}
        is_valid, error = ToolParameterValidator.validate(schema, params)
        assert is_valid is False
        assert "too short" in error

    def test_validate_number_range(self):
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 120},
            },
            "required": ["age"],
        }
        params = {"age": 150}
        is_valid, error = ToolParameterValidator.validate(schema, params)
        assert is_valid is False
        assert "too large" in error

    def test_validate_valid(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        params = {"name": "John", "age": 30}
        is_valid, error = ToolParameterValidator.validate(schema, params)
        assert is_valid is True
        assert error is None


class TestCalculatorTool:
    def test_add(self):
        tool = CalculatorTool()
        result = tool.execute({"operation": "add", "a": 2, "b": 3})
        assert result["result"] == 5

    def test_subtract(self):
        tool = CalculatorTool()
        result = tool.execute({"operation": "subtract", "a": 10, "b": 3})
        assert result["result"] == 7

    def test_multiply(self):
        tool = CalculatorTool()
        result = tool.execute({"operation": "multiply", "a": 4, "b": 5})
        assert result["result"] == 20

    def test_divide(self):
        tool = CalculatorTool()
        result = tool.execute({"operation": "divide", "a": 10, "b": 2})
        assert result["result"] == 5

    def test_divide_by_zero(self):
        tool = CalculatorTool()
        with pytest.raises(ValueError, match="Division by zero"):
            tool.execute({"operation": "divide", "a": 10, "b": 0})

    def test_unknown_operation(self):
        tool = CalculatorTool()
        with pytest.raises(ValueError, match="Unknown operation"):
            tool.execute({"operation": "unknown", "a": 1, "b": 2})

    def test_schema(self):
        tool = CalculatorTool()
        schema = tool.schema
        assert schema.name == "calculator"
        assert "required" in schema.parameters


class TestUnstableAPITool:
    def test_force_success(self):
        tool = UnstableAPITool()
        result = tool.execute({"query": "test", "force_success": True})
        assert "Successfully processed" in result["response"]
        assert result["query"] == "test"

    def test_force_error(self):
        tool = UnstableAPITool()
        with pytest.raises(RuntimeError, match="Forced API error"):
            tool.execute({"query": "test", "force_error": True})

    def test_forced_timeout(self):
        tool = UnstableAPITool(base_timeout=0.1)
        with pytest.raises(TimeoutError, match="Forced timeout"):
            tool.execute({"query": "test", "force_timeout": True})


class TestToolRegistry:
    def test_register_and_list_tools(self):
        registry = ToolRegistry()
        calculator = CalculatorTool()
        registry.register_tool(calculator)

        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "calculator"

    def test_get_tool_schema(self):
        registry = ToolRegistry()
        calculator = CalculatorTool()
        registry.register_tool(calculator)

        schema = registry.get_tool_schema("calculator")
        assert schema is not None
        assert schema.name == "calculator"

    def test_get_nonexistent_tool_schema(self):
        registry = ToolRegistry()
        schema = registry.get_tool_schema("nonexistent")
        assert schema is None

    def test_call_tool_success(self):
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())

        request = ToolCallRequest(
            tool_name="calculator",
            parameters={"operation": "add", "a": 2, "b": 3},
        )
        result = registry.call_tool(request)

        assert result.success is True
        assert result.data["result"] == 5

    def test_call_tool_not_found(self):
        registry = ToolRegistry()
        request = ToolCallRequest(
            tool_name="nonexistent",
            parameters={},
        )
        result = registry.call_tool(request)

        assert result.success is False
        assert result.error_type == ToolErrorType.TOOL_NOT_FOUND

    def test_call_tool_parameter_error(self):
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())

        request = ToolCallRequest(
            tool_name="calculator",
            parameters={"operation": "add"},
        )
        result = registry.call_tool(request)

        assert result.success is False
        assert result.error_type == ToolErrorType.PARAMETER_ERROR

    def test_call_tool_runtime_error(self):
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())

        request = ToolCallRequest(
            tool_name="calculator",
            parameters={"operation": "divide", "a": 10, "b": 0},
        )
        result = registry.call_tool(request)

        assert result.success is False
        assert result.error_type == ToolErrorType.RUNTIME_ERROR
        assert "Division by zero" in result.error
