"""
Tool Layer - Tool registry, execution framework, and built-in tools.
"""

import random
import re
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from agent.types import (
    BaseTool,
    ToolCallRequest,
    ToolErrorType,
    ToolResult,
    ToolSchema,
)


T = TypeVar("T")


def timeout_decorator(timeout_seconds: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except FutureTimeoutError:
                executor.shutdown(wait=False, cancel_futures=True)
                raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")
            finally:
                executor.shutdown(wait=True)
        return wrapper
    return decorator


class ToolParameterValidator:
    @staticmethod
    def validate(schema: dict[str, Any], parameters: dict[str, Any]) -> tuple[bool, Optional[str]]:
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for param in required:
            if param not in parameters:
                return False, f"Missing required parameter: {param}"

        for param_name, param_value in parameters.items():
            if param_name not in properties:
                continue

            prop_schema = properties[param_name]
            expected_type = prop_schema.get("type")

            if expected_type == "string":
                if not isinstance(param_value, str):
                    return False, f"Parameter '{param_name}' must be a string"
                min_length = prop_schema.get("minLength")
                max_length = prop_schema.get("maxLength")
                if min_length is not None and len(param_value) < min_length:
                    return False, f"Parameter '{param_name}' too short (min {min_length})"
                if max_length is not None and len(param_value) > max_length:
                    return False, f"Parameter '{param_name}' too long (max {max_length})"

            elif expected_type == "number":
                if not isinstance(param_value, (int, float)):
                    return False, f"Parameter '{param_name}' must be a number"
                minimum = prop_schema.get("minimum")
                maximum = prop_schema.get("maximum")
                if minimum is not None and param_value < minimum:
                    return False, f"Parameter '{param_name}' too small (min {minimum})"
                if maximum is not None and param_value > maximum:
                    return False, f"Parameter '{param_name}' too large (max {maximum})"

            elif expected_type == "integer":
                if not isinstance(param_value, int) or isinstance(param_value, bool):
                    return False, f"Parameter '{param_name}' must be an integer"
                minimum = prop_schema.get("minimum")
                maximum = prop_schema.get("maximum")
                if minimum is not None and param_value < minimum:
                    return False, f"Parameter '{param_name}' too small (min {minimum})"
                if maximum is not None and param_value > maximum:
                    return False, f"Parameter '{param_name}' too large (max {maximum})"

            elif expected_type == "boolean":
                if not isinstance(param_value, bool):
                    return False, f"Parameter '{param_name}' must be a boolean"

            elif expected_type == "array":
                if not isinstance(param_value, list):
                    return False, f"Parameter '{param_name}' must be an array"

            elif expected_type == "object":
                if not isinstance(param_value, dict):
                    return False, f"Parameter '{param_name}' must be an object"

        return True, None


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        self._tools[tool.schema.name] = tool

    def get_tool_schema(self, tool_name: str) -> Optional[ToolSchema]:
        tool = self._tools.get(tool_name)
        return tool.schema if tool else None

    def list_tools(self) -> list[ToolSchema]:
        return [tool.schema for tool in self._tools.values()]

    def call_tool(
        self, request: ToolCallRequest, timeout: float = 30.0
    ) -> ToolResult:
        start_time = time.time()

        tool = self._tools.get(request.tool_name)
        if not tool:
            execution_time_ms = (time.time() - start_time) * 1000
            return ToolResult(
                success=False,
                error=f"Tool not found: {request.tool_name}",
                error_type=ToolErrorType.TOOL_NOT_FOUND,
                execution_time_ms=execution_time_ms,
            )

        is_valid, validation_error = ToolParameterValidator.validate(
            tool.schema.parameters, request.parameters
        )
        if not is_valid:
            execution_time_ms = (time.time() - start_time) * 1000
            return ToolResult(
                success=False,
                error=validation_error,
                error_type=ToolErrorType.PARAMETER_ERROR,
                execution_time_ms=execution_time_ms,
            )

        try:
            @timeout_decorator(timeout)
            def execute_with_timeout() -> Any:
                return tool.execute(request.parameters)

            result = execute_with_timeout()
            execution_time_ms = (time.time() - start_time) * 1000

            return ToolResult(
                success=True,
                data=result,
                execution_time_ms=execution_time_ms,
            )

        except TimeoutError:
            execution_time_ms = (time.time() - start_time) * 1000
            return ToolResult(
                success=False,
                error=f"Tool execution timed out after {timeout} seconds",
                error_type=ToolErrorType.TIMEOUT,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return ToolResult(
                success=False,
                error=str(e),
                error_type=ToolErrorType.RUNTIME_ERROR,
                execution_time_ms=execution_time_ms,
            )


class CalculatorTool(BaseTool):
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="calculator",
            description="A simple calculator for performing basic arithmetic operations. Use this for numerical calculations.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The arithmetic operation to perform",
                    },
                    "a": {
                        "type": "number",
                        "description": "First operand",
                    },
                    "b": {
                        "type": "number",
                        "description": "Second operand",
                    },
                },
                "required": ["operation", "a", "b"],
            },
            returns={
                "type": "object",
                "properties": {
                    "result": {"type": "number"},
                    "operation": {"type": "string"},
                },
            },
        )

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        operation = parameters["operation"]
        a = float(parameters["a"])
        b = float(parameters["b"])

        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return {"result": result, "operation": operation}


class UnstableAPITool(BaseTool):
    def __init__(
        self,
        failure_probability: float = 0.3,
        timeout_probability: float = 0.2,
        max_sleep_seconds: float = 2.0,
        base_timeout: float = 5.0,
    ):
        self.failure_probability = failure_probability
        self.timeout_probability = timeout_probability
        self.max_sleep_seconds = max_sleep_seconds
        self.base_timeout = base_timeout

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="unstable_api",
            description="A simulated unstable external API that may timeout, throw errors, or respond slowly. Use for testing error handling.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to send to the API",
                        "minLength": 1,
                    },
                    "force_success": {
                        "type": "boolean",
                        "description": "If true, forces the API to succeed (for testing)",
                        "default": False,
                    },
                    "force_timeout": {
                        "type": "boolean",
                        "description": "If true, forces the API to timeout (for testing)",
                        "default": False,
                    },
                    "force_error": {
                        "type": "boolean",
                        "description": "If true, forces the API to return an error (for testing)",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
            returns={
                "type": "object",
                "properties": {
                    "response": {"type": "string"},
                    "query": {"type": "string"},
                    "latency_ms": {"type": "number"},
                },
            },
        )

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        query = parameters["query"]
        force_success = parameters.get("force_success", False)
        force_timeout = parameters.get("force_timeout", False)
        force_error = parameters.get("force_error", False)

        start_time = time.time()

        sleep_time = random.uniform(0.1, self.max_sleep_seconds)

        if force_timeout:
            time.sleep(self.base_timeout + 1)
            raise TimeoutError("Forced timeout")

        if force_error:
            raise RuntimeError("Forced API error: Internal Server Error (500)")

        if not force_success:
            if random.random() < self.timeout_probability:
                time.sleep(self.base_timeout + 1)
                raise TimeoutError("API request timed out")

            if random.random() < self.failure_probability:
                raise RuntimeError(f"API error: Service Unavailable (503) - Query: {query}")

        time.sleep(sleep_time)

        latency_ms = (time.time() - start_time) * 1000

        return {
            "response": f"Successfully processed: {query}",
            "query": query,
            "latency_ms": round(latency_ms, 2),
        }


class MemoryReadWriteTool(BaseTool):
    def __init__(self, memory_layer: Any):
        self._memory_layer = memory_layer

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="memory_rw",
            description="Read from and write to the agent's memory system. Use this to store information for later retrieval or recall previously stored information.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write"],
                        "description": "The memory operation to perform",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to memory (required for write operation)",
                    },
                    "query": {
                        "type": "string",
                        "description": "Query to search memory (required for read operation)",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["short_term", "long_term"],
                        "description": "Type of memory to access",
                        "default": "short_term",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return for read operations",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["operation"],
            },
            returns={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "operation": {"type": "string"},
                    "results": {"type": "array"},
                    "memory_id": {"type": "string"},
                },
            },
        )

    def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        from agent.types import MemoryType

        operation = parameters["operation"]
        memory_type_str = parameters.get("memory_type", "short_term")
        memory_type = (
            MemoryType.LONG_TERM if memory_type_str == "long_term" else MemoryType.SHORT_TERM
        )

        if operation == "write":
            content = parameters.get("content", "")
            if not content:
                raise ValueError("Content is required for write operation")

            item = self._memory_layer.write(
                content=content,
                memory_type=memory_type,
                metadata={"tool_written": True},
            )
            return {
                "success": True,
                "operation": "write",
                "memory_id": item.id,
                "results": [],
            }

        elif operation == "read":
            query = parameters.get("query", "")
            limit = parameters.get("limit", 5)

            results = self._memory_layer.retrieve(
                query=query,
                memory_type=memory_type,
                limit=limit,
            )

            return {
                "success": True,
                "operation": "read",
                "results": [
                    {
                        "id": item.id,
                        "content": item.content,
                        "created_at": item.created_at.isoformat(),
                    }
                    for item in results
                ],
            }

        else:
            raise ValueError(f"Unknown operation: {operation}")
