"""
Core type definitions for the Agent architecture.
All layers communicate using these types to maintain loose coupling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Literal, Optional, Protocol, TypeVar, Union

# Type variables
T = TypeVar("T")


class ActionType(Enum):
    CALL_TOOL = auto()
    WRITE_MEMORY = auto()
    FINAL_ANSWER = auto()
    REQUEST_MORE_INFO = auto()


class TraceEventType(Enum):
    PERCEPTION_START = auto()
    PERCEPTION_END = auto()
    MEMORY_RETRIEVAL_START = auto()
    MEMORY_RETRIEVAL_END = auto()
    PLANNING_START = auto()
    PLANNING_END = auto()
    TOOL_CALL_START = auto()
    TOOL_CALL_END = auto()
    MEMORY_WRITE_START = auto()
    MEMORY_WRITE_END = auto()
    EXECUTION_STEP = auto()
    EXECUTION_FINISH = auto()


class ToolErrorType(Enum):
    PARAMETER_ERROR = auto()
    TIMEOUT = auto()
    RUNTIME_ERROR = auto()
    TOOL_NOT_FOUND = auto()


class MemoryType(Enum):
    SHORT_TERM = auto()
    LONG_TERM = auto()


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any]


@dataclass
class ToolCallRequest:
    tool_name: str
    parameters: dict[str, Any]


@dataclass
class ToolResult:
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    error_type: Optional[ToolErrorType] = None
    execution_time_ms: float = 0.0


@dataclass
class Action:
    action_type: ActionType
    tool_call: Optional[ToolCallRequest] = None
    memory_write: Optional[dict[str, Any]] = None
    content: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    steps: list[Action]
    max_retries: int = 3
    abort_threshold: int = 3
    is_complete: bool = False


@dataclass
class UserInput:
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    memory_retrieval_results: list[dict[str, Any]] = field(default_factory=list)
    available_tools: list[ToolSchema] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskInput:
    user_input: UserInput
    context: Context
    normalized_input: str
    validation_status: str


@dataclass
class MemoryItem:
    id: str
    content: str
    memory_type: MemoryType
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"


@dataclass
class TraceEvent:
    event_id: str
    event_type: TraceEventType
    stage_name: str
    timestamp: datetime
    input_summary: str
    output_summary: str
    execution_time_ms: float
    memory_hit: bool = False
    tool_called: bool = False
    tool_name: Optional[str] = None
    tool_parameters: Optional[dict[str, Any]] = None
    tool_result: Optional[str] = None
    tool_error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    success: bool
    final_answer: Optional[str] = None
    failure_reason: Optional[str] = None
    trace_events: list[TraceEvent] = field(default_factory=list)
    trace_file_path: Optional[str] = None
    steps_executed: int = 0


class PerceptionProtocol(Protocol):
    def process(
        self, user_input: UserInput, context: Context
    ) -> tuple[TaskInput, list[TraceEvent]]:
        ...


class PlanningProtocol(Protocol):
    def plan(
        self, task_input: TaskInput, context: Context, max_retries: int = 3
    ) -> tuple[Plan, list[TraceEvent]]:
        ...

    def replan(
        self,
        original_plan: Plan,
        last_action_result: Any,
        failure_count: int,
    ) -> tuple[Plan, list[TraceEvent]]:
        ...


class ToolRegistryProtocol(Protocol):
    def register_tool(self, tool: "BaseTool") -> None:
        ...

    def get_tool_schema(self, tool_name: str) -> Optional[ToolSchema]:
        ...

    def list_tools(self) -> list[ToolSchema]:
        ...

    def call_tool(
        self, request: ToolCallRequest, timeout: float = 30.0
    ) -> ToolResult:
        ...


class MemoryProtocol(Protocol):
    def write(
        self, content: str, memory_type: MemoryType, metadata: Optional[dict[str, Any]] = None
    ) -> MemoryItem:
        ...

    def retrieve(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        ...

    def retrieve_by_time(
        self,
        start_time: datetime,
        end_time: datetime,
        memory_type: Optional[MemoryType] = None,
    ) -> list[MemoryItem]:
        ...


class BaseTool(ABC):
    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        ...

    @abstractmethod
    def execute(self, parameters: dict[str, Any]) -> Any:
        ...
