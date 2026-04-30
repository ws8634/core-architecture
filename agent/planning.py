"""
Planning Layer - Task decomposition and plan generation.
Supports multi-step plans, retry strategies, and abort conditions.
"""

import re
import time
import uuid
from typing import Any, Optional

from agent.types import (
    Action,
    ActionType,
    Context,
    Plan,
    TaskInput,
    TraceEvent,
    TraceEventType,
    ToolSchema,
)


class SimplePlanner:
    def __init__(self, max_retries: int = 3, abort_threshold: int = 3):
        self.max_retries = max_retries
        self.abort_threshold = abort_threshold

    def plan(
        self, task_input: TaskInput, context: Context, max_retries: int = 3
    ) -> tuple[Plan, list[TraceEvent]]:
        events: list[TraceEvent] = []
        start_time = time.time()

        input_summary = task_input.normalized_input[:100] if task_input.normalized_input else "INVALID"
        events.append(
            self._create_event(
                TraceEventType.PLANNING_START,
                "Planning-Start",
                input_summary,
                "Generating execution plan",
            )
        )

        if "INVALID" in task_input.validation_status:
            plan = Plan(
                steps=[
                    Action(
                        action_type=ActionType.FINAL_ANSWER,
                        content=f"Input validation failed: {task_input.validation_status}",
                    )
                ],
                max_retries=max_retries,
                abort_threshold=self.abort_threshold,
                is_complete=True,
            )
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            events.append(
                self._create_event(
                    TraceEventType.PLANNING_END,
                    "Planning-End",
                    input_summary,
                    f"Plan generated: 1 step (validation failed)",
                    execution_time_ms=execution_time_ms,
                    metadata={"step_count": 1},
                )
            )
            return plan, events

        steps = self._generate_steps(task_input, context)
        plan = Plan(
            steps=steps,
            max_retries=max_retries,
            abort_threshold=self.abort_threshold,
            is_complete=False,
        )

        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000

        steps_summary = ", ".join(
            f"{i+1}:{s.action_type.name}" for i, s in enumerate(steps[:5])
        )
        if len(steps) > 5:
            steps_summary += f"... ({len(steps)} total)"

        events.append(
            self._create_event(
                TraceEventType.PLANNING_END,
                "Planning-End",
                input_summary,
                f"Plan generated: {steps_summary}",
                execution_time_ms=execution_time_ms,
                metadata={
                    "step_count": len(steps),
                    "max_retries": max_retries,
                    "abort_threshold": self.abort_threshold,
                },
            )
        )

        return plan, events

    def replan(
        self,
        original_plan: Plan,
        last_action_result: Any,
        failure_count: int,
    ) -> tuple[Plan, list[TraceEvent]]:
        events: list[TraceEvent] = []
        start_time = time.time()

        events.append(
            self._create_event(
                TraceEventType.PLANNING_START,
                "Re-Planning-Start",
                f"Failure count: {failure_count}",
                "Re-evaluating plan after failure",
            )
        )

        if failure_count >= original_plan.abort_threshold:
            new_steps = [
                Action(
                    action_type=ActionType.FINAL_ANSWER,
                    content=(
                        f"Execution aborted after {failure_count} consecutive failures. "
                        f"Last error: {last_action_result}"
                    ),
                    metadata={"aborted": True, "failure_count": failure_count},
                )
            ]
            new_plan = Plan(
                steps=new_steps,
                max_retries=original_plan.max_retries,
                abort_threshold=original_plan.abort_threshold,
                is_complete=True,
            )
        else:
            retry_action = Action(
                action_type=ActionType.REQUEST_MORE_INFO,
                content=f"Action failed, retrying (attempt {failure_count + 1})",
                metadata={
                    "retry": True,
                    "failure_count": failure_count,
                    "last_error": str(last_action_result),
                },
            )
            new_steps = [retry_action] + original_plan.steps
            new_plan = Plan(
                steps=new_steps,
                max_retries=original_plan.max_retries,
                abort_threshold=original_plan.abort_threshold,
                is_complete=False,
            )

        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000

        events.append(
            self._create_event(
                TraceEventType.PLANNING_END,
                "Re-Planning-End",
                f"Failure count: {failure_count}",
                f"Re-plan complete: {len(new_plan.steps)} steps, aborted={new_plan.is_complete}",
                execution_time_ms=execution_time_ms,
                metadata={
                    "failure_count": failure_count,
                    "aborted": new_plan.is_complete,
                },
            )
        )

        return new_plan, events

    def _generate_steps(self, task_input: TaskInput, context: Context) -> list[Action]:
        steps: list[Action] = []
        normalized_input = task_input.normalized_input.lower()
        available_tools = context.available_tools
        tool_names = {t.name for t in available_tools}

        memory_results = context.memory_retrieval_results
        if memory_results:
            pass

        if any(kw in normalized_input for kw in ["calculate", "compute", "math", "加", "减", "乘", "除"]):
            numbers = self._extract_numbers(normalized_input)
            operation = self._extract_operation(normalized_input)

            if numbers and len(numbers) >= 2:
                steps.append(
                    Action(
                        action_type=ActionType.CALL_TOOL,
                        tool_call={
                            "tool_name": "calculator",
                            "parameters": {
                                "operation": operation,
                                "a": numbers[0],
                                "b": numbers[1],
                            },
                        },
                    )
                )

        if any(kw in normalized_input for kw in ["api", "search", "query", "查找", "查询"]):
            query = self._extract_query(normalized_input)
            if "unstable_api" in tool_names:
                steps.append(
                    Action(
                        action_type=ActionType.CALL_TOOL,
                        tool_call={
                            "tool_name": "unstable_api",
                            "parameters": {
                                "query": query or normalized_input,
                                "force_success": False,
                            },
                        },
                    )
                )

        if any(kw in normalized_input for kw in ["remember", "save", "store", "记住", "保存"]):
            if "memory_rw" in tool_names:
                content = self._extract_content_to_remember(normalized_input)
                steps.append(
                    Action(
                        action_type=ActionType.CALL_TOOL,
                        tool_call={
                            "tool_name": "memory_rw",
                            "parameters": {
                                "operation": "write",
                                "content": content or normalized_input,
                                "memory_type": "short_term",
                            },
                        },
                    )
                )

        if any(kw in normalized_input for kw in ["recall", "find", "retrieve", "回忆", "查找"]):
            if "memory_rw" in tool_names:
                query = self._extract_query(normalized_input)
                steps.append(
                    Action(
                        action_type=ActionType.CALL_TOOL,
                        tool_call={
                            "tool_name": "memory_rw",
                            "parameters": {
                                "operation": "read",
                                "query": query or "",
                                "memory_type": "short_term",
                                "limit": 5,
                            },
                        },
                    )
                )

        if not steps:
            if "unstable_api" in tool_names:
                steps.append(
                    Action(
                        action_type=ActionType.CALL_TOOL,
                        tool_call={
                            "tool_name": "unstable_api",
                            "parameters": {
                                "query": normalized_input,
                                "force_success": True,
                            },
                        },
                    )
                )

        steps.append(
            Action(
                action_type=ActionType.FINAL_ANSWER,
                content="",
            )
        )

        return steps

    def _extract_numbers(self, text: str) -> list[float]:
        matches = re.findall(r"-?\d+\.?\d*", text)
        return [float(m) for m in matches]

    def _extract_operation(self, text: str) -> str:
        if any(kw in text for kw in ["add", "plus", "+", "加"]):
            return "add"
        elif any(kw in text for kw in ["subtract", "minus", "-", "减"]):
            return "subtract"
        elif any(kw in text for kw in ["multiply", "times", "*", "乘"]):
            return "multiply"
        elif any(kw in text for kw in ["divide", "/", "除以", "除"]):
            return "divide"
        return "add"

    def _extract_query(self, text: str) -> Optional[str]:
        patterns = [
            r"(?:search|query|find|look for|查找|查询|搜索)\s+(.+?)(?:\?|$|,|\.)",
            r"(?:about|关于|for)\s+(.+?)(?:\?|$|,|\.)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_content_to_remember(self, text: str) -> Optional[str]:
        patterns = [
            r"(?:remember|save|store|记住|保存)\s+(.+?)(?:$|,|\.)",
            r"(?:that|that:)\s+(.+?)(?:$|,|\.)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _create_event(
        self,
        event_type: TraceEventType,
        stage_name: str,
        input_summary: str,
        output_summary: str,
        execution_time_ms: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> TraceEvent:
        return TraceEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            stage_name=stage_name,
            timestamp=__import__("datetime").datetime.now(),
            input_summary=input_summary,
            output_summary=output_summary,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
        )
