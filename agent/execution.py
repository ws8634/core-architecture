"""
Execution Layer - Orchestrator that coordinates all layers.
Implements: perception -> memory retrieval -> planning -> tool calling -> memory write -> output.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agent.memory import MemoryLayer
from agent.perception import PerceptionLayer
from agent.planning import SimplePlanner
from agent.tools import CalculatorTool, MemoryReadWriteTool, ToolRegistry, UnstableAPITool
from agent.tracing import TraceRecorder, TraceReplayer
from agent.types import (
    Action,
    ActionType,
    Context,
    ExecutionResult,
    MemoryType,
    TaskInput,
    TraceEvent,
    TraceEventType,
    ToolCallRequest,
    UserInput,
)


class AgentOrchestrator:
    def __init__(
        self,
        memory_persistent_path: Optional[str] = None,
        trace_output_path: Optional[str] = None,
        max_steps: int = 10,
        tool_timeout: float = 30.0,
        planner_max_retries: int = 3,
        planner_abort_threshold: int = 3,
    ):
        self.max_steps = max_steps
        self.tool_timeout = tool_timeout

        self.memory = MemoryLayer(persistent_path=memory_persistent_path)
        self.perception = PerceptionLayer()
        self.planner = SimplePlanner(
            max_retries=planner_max_retries,
            abort_threshold=planner_abort_threshold,
        )

        self.tool_registry = ToolRegistry()
        self.tool_registry.register_tool(CalculatorTool())
        self.tool_registry.register_tool(
            UnstableAPITool(
                failure_probability=0.3,
                timeout_probability=0.2,
                max_sleep_seconds=1.0,
                base_timeout=2.0,
            )
        )
        self.tool_registry.register_tool(MemoryReadWriteTool(self.memory))

        self.trace_recorder = TraceRecorder(output_file=trace_output_path)

    def run(
        self,
        user_input_text: str,
        dry_run: bool = False,
        conversation_history: Optional[list[dict]] = None,
    ) -> ExecutionResult:
        start_time = time.time()
        steps_executed = 0
        consecutive_failures = 0
        current_plan: Optional[Any] = None
        plan_step_index = 0
        final_answer: Optional[str] = None
        failure_reason: Optional[str] = None

        conversation_history = conversation_history or []

        try:
            user_input = UserInput(text=user_input_text)
            context = Context(
                conversation_history=conversation_history,
                memory_retrieval_results=[],
                available_tools=self.tool_registry.list_tools(),
            )

            task_input, perception_events = self.perception.process(user_input, context)
            self.trace_recorder.add_events(perception_events)
            steps_executed += 1

            if "INVALID" in task_input.validation_status:
                final_answer = f"Input validation failed: {task_input.validation_status}"
                failure_reason = task_input.validation_status
                return self._build_result(
                    success=False,
                    final_answer=final_answer,
                    failure_reason=failure_reason,
                    steps_executed=steps_executed,
                    start_time=start_time,
                )

            retrieval_events = []
            retrieval_start = time.time()
            self.trace_recorder.record(
                event_type=TraceEventType.MEMORY_RETRIEVAL_START,
                stage_name="Memory-Retrieval-Start",
                input_summary=f"Query: {task_input.normalized_input[:50]}",
                output_summary="Searching memory for relevant information...",
            )

            memory_results = self.memory.retrieve(
                query=task_input.normalized_input,
                limit=5,
            )

            memory_hit = len(memory_results) > 0
            retrieval_results = [
                {
                    "id": item.id,
                    "content": item.content,
                    "memory_type": item.memory_type.name,
                    "created_at": item.created_at.isoformat(),
                }
                for item in memory_results
            ]

            retrieval_time_ms = (time.time() - retrieval_start) * 1000

            self.trace_recorder.record(
                event_type=TraceEventType.MEMORY_RETRIEVAL_END,
                stage_name="Memory-Retrieval-End",
                input_summary=f"Query: {task_input.normalized_input[:50]}",
                output_summary=f"Found {len(memory_results)} relevant memory items",
                execution_time_ms=retrieval_time_ms,
                memory_hit=memory_hit,
                metadata={"result_count": len(memory_results)},
            )

            context.memory_retrieval_results = retrieval_results

            current_plan, planning_events = self.planner.plan(
                task_input, context, max_retries=self.planner.max_retries
            )
            self.trace_recorder.add_events(planning_events)
            steps_executed += 1

            if dry_run:
                plan_summary = self._format_plan_for_dry_run(current_plan)
                final_answer = (
                    f"[DRY-RUN] Plan generated:\n{plan_summary}\n\n"
                    f"No actual tools were called. Use 'run' to execute."
                )
                return self._build_result(
                    success=True,
                    final_answer=final_answer,
                    steps_executed=steps_executed,
                    start_time=start_time,
                )

            plan_step_index = 0
            while steps_executed < self.max_steps and plan_step_index < len(current_plan.steps):
                action = current_plan.steps[plan_step_index]
                steps_executed += 1

                self.trace_recorder.record(
                    event_type=TraceEventType.EXECUTION_STEP,
                    stage_name=f"Execute-Step-{plan_step_index + 1}",
                    input_summary=f"Action: {action.action_type.name}",
                    output_summary="Executing action...",
                )

                if action.action_type == ActionType.CALL_TOOL:
                    result, success = self._execute_tool_call(action)

                    if not success:
                        consecutive_failures += 1
                        if consecutive_failures >= self.planner.abort_threshold:
                            final_answer = (
                                f"Execution stopped after {consecutive_failures} consecutive failures. "
                                f"Last error: {result}"
                            )
                            failure_reason = f"Aborted after {consecutive_failures} failures"
                            break
                        else:
                            current_plan, replan_events = self.planner.replan(
                                current_plan, result, consecutive_failures
                            )
                            self.trace_recorder.add_events(replan_events)
                            plan_step_index = 0
                            continue
                    else:
                        consecutive_failures = 0

                        if isinstance(result, dict) and "result" in result:
                            formatted_result = f"Calculation result: {result['result']}"
                            action.content = formatted_result
                        elif isinstance(result, dict) and "response" in result:
                            action.content = result["response"]
                        elif isinstance(result, dict) and "results" in result:
                            memory_items = result.get("results", [])
                            if memory_items:
                                contents = [item.get("content", "") for item in memory_items]
                                action.content = f"Found {len(memory_items)} memory items: {'; '.join(contents)}"
                            else:
                                action.content = "No matching memories found."

                elif action.action_type == ActionType.WRITE_MEMORY:
                    write_start = time.time()
                    self.trace_recorder.record(
                        event_type=TraceEventType.MEMORY_WRITE_START,
                        stage_name="Memory-Write-Start",
                        input_summary=f"Writing to memory: {str(action.memory_write)[:50]}",
                        output_summary="Writing to memory...",
                    )

                    memory_data = action.memory_write or {}
                    content = memory_data.get("content", str(action.content or ""))
                    mem_type = (
                        MemoryType.LONG_TERM
                        if memory_data.get("persist", False)
                        else MemoryType.SHORT_TERM
                    )

                    self.memory.write(content=content, memory_type=mem_type)

                    write_time_ms = (time.time() - write_start) * 1000
                    self.trace_recorder.record(
                        event_type=TraceEventType.MEMORY_WRITE_END,
                        stage_name="Memory-Write-End",
                        input_summary=f"Content length: {len(content)}",
                        output_summary="Memory write complete",
                        execution_time_ms=write_time_ms,
                    )
                    consecutive_failures = 0

                elif action.action_type == ActionType.FINAL_ANSWER:
                    if not action.content and plan_step_index > 0:
                        prev_action = current_plan.steps[plan_step_index - 1]
                        if prev_action.content:
                            action.content = prev_action.content
                        else:
                            action.content = "Task completed successfully."

                    final_answer = action.content
                    break

                elif action.action_type == ActionType.REQUEST_MORE_INFO:
                    if action.content:
                        pass
                    plan_step_index += 1
                    continue

                plan_step_index += 1

            if steps_executed >= self.max_steps:
                final_answer = (
                    f"Execution stopped after reaching max_steps ({self.max_steps}). "
                    f"Last progress: {final_answer or 'No answer generated'}"
                )
                failure_reason = f"Max steps ({self.max_steps}) exceeded"

        except Exception as e:
            final_answer = f"Unexpected error during execution: {str(e)}"
            failure_reason = str(e)

        return self._build_result(
            success=(failure_reason is None),
            final_answer=final_answer,
            failure_reason=failure_reason,
            steps_executed=steps_executed,
            start_time=start_time,
        )

    def _execute_tool_call(self, action: Action) -> tuple[Any, bool]:
        tool_call_data = action.tool_call
        if not tool_call_data:
            return "No tool call specified", False

        tool_name = tool_call_data.get("tool_name", "")
        parameters = tool_call_data.get("parameters", {})

        call_start = time.time()
        self.trace_recorder.record(
            event_type=TraceEventType.TOOL_CALL_START,
            stage_name=f"Tool-Call-Start-{tool_name}",
            input_summary=f"Tool: {tool_name}, Params: {str(parameters)[:100]}",
            output_summary="Calling tool...",
            tool_called=True,
            tool_name=tool_name,
            tool_parameters=parameters,
        )

        request = ToolCallRequest(tool_name=tool_name, parameters=parameters)
        result = self.tool_registry.call_tool(request, timeout=self.tool_timeout)

        call_time_ms = (time.time() - call_start) * 1000

        tool_result_str = str(result.data)[:200] if result.data else ""
        tool_error_str = result.error if result.error else None

        self.trace_recorder.record(
            event_type=TraceEventType.TOOL_CALL_END,
            stage_name=f"Tool-Call-End-{tool_name}",
            input_summary=f"Tool: {tool_name}",
            output_summary=(
                f"Success: {result.success}, "
                f"Time: {call_time_ms:.2f}ms, "
                f"Result: {tool_result_str[:100] if tool_result_str else 'N/A'}"
            ),
            execution_time_ms=call_time_ms,
            tool_called=True,
            tool_name=tool_name,
            tool_parameters=parameters,
            tool_result=tool_result_str if result.success else None,
            tool_error=tool_error_str,
        )

        if result.success:
            return result.data, True
        else:
            return result.error, False

    def _build_result(
        self,
        success: bool,
        final_answer: Optional[str],
        steps_executed: int,
        start_time: float,
        failure_reason: Optional[str] = None,
    ) -> ExecutionResult:
        self.trace_recorder.record(
            event_type=TraceEventType.EXECUTION_FINISH,
            stage_name="Execution-Finish",
            input_summary=f"Steps: {steps_executed}",
            output_summary=(
                f"Success: {success}, "
                f"Answer: {(final_answer or '')[:100]}..."
            ),
        )

        return ExecutionResult(
            success=success,
            final_answer=final_answer,
            failure_reason=failure_reason,
            trace_events=self.trace_recorder.events,
            trace_file_path=(
                str(self.trace_recorder._output_file)
                if self.trace_recorder._output_file
                else None
            ),
            steps_executed=steps_executed,
        )

    def _format_plan_for_dry_run(self, plan: Any) -> str:
        lines = []
        for i, step in enumerate(plan.steps, 1):
            lines.append(f"\nStep {i}: {step.action_type.name}")
            if step.tool_call:
                lines.append(f"  Tool: {step.tool_call.get('tool_name')}")
                lines.append(f"  Parameters: {step.tool_call.get('parameters')}")
            if step.memory_write:
                lines.append(f"  Memory Write: {step.memory_write}")
            if step.content:
                lines.append(f"  Content: {step.content[:100]}")
        return "".join(lines)
