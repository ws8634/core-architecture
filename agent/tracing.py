"""
Tracing system for recording execution flows.
Supports both console output and JSONL file storage.
"""

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from agent.types import TraceEvent, TraceEventType


class TraceRecorder:
    def __init__(self, output_file: Optional[str] = None):
        self._events: list[TraceEvent] = []
        self._output_file = Path(output_file) if output_file else None

    @property
    def events(self) -> list[TraceEvent]:
        return self._events.copy()

    def record(
        self,
        event_type: TraceEventType,
        stage_name: str,
        input_summary: str,
        output_summary: str,
        execution_time_ms: float = 0.0,
        memory_hit: bool = False,
        tool_called: bool = False,
        tool_name: Optional[str] = None,
        tool_parameters: Optional[dict] = None,
        tool_result: Optional[str] = None,
        tool_error: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> TraceEvent:
        event = TraceEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            stage_name=stage_name,
            timestamp=datetime.now(),
            input_summary=input_summary,
            output_summary=output_summary,
            execution_time_ms=execution_time_ms,
            memory_hit=memory_hit,
            tool_called=tool_called,
            tool_name=tool_name,
            tool_parameters=tool_parameters,
            tool_result=tool_result,
            tool_error=tool_error,
            metadata=metadata or {},
        )
        self._events.append(event)
        self._write_event(event)
        self._print_event(event)
        return event

    def _write_event(self, event: TraceEvent) -> None:
        if not self._output_file:
            return
        event_dict = asdict(event)
        event_dict["event_type"] = event.event_type.name
        event_dict["timestamp"] = event.timestamp.isoformat()
        with open(self._output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")

    def _print_event(self, event: TraceEvent) -> None:
        timestamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
        status = "✓" if "error" not in event.output_summary.lower() else "✗"
        
        tool_info = ""
        if event.tool_called and event.tool_name:
            tool_info = f" [Tool: {event.tool_name}]"
        
        memory_info = ""
        if event.memory_hit:
            memory_info = " [Memory Hit]"
        
        print(
            f"[{timestamp}] {status} {event.stage_name}{tool_info}{memory_info}"
        )
        if event.tool_error:
            print(f"    Error: {event.tool_error}")

    def add_events(self, events: list[TraceEvent]) -> None:
        for event in events:
            self._events.append(event)
            self._write_event(event)
            self._print_event(event)

    def clear(self) -> None:
        self._events = []


class TraceReplayer:
    @staticmethod
    def replay(trace_file: str) -> list[dict]:
        events = []
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        return events

    @staticmethod
    def format_summary(events: list[dict]) -> str:
        if not events:
            return "No trace events found."

        lines = []
        lines.append("=" * 80)
        lines.append("TRACE EXECUTION SUMMARY")
        lines.append("=" * 80)

        total_time = sum(e.get("execution_time_ms", 0) for e in events)
        tool_calls = sum(1 for e in events if e.get("tool_called"))
        memory_hits = sum(1 for e in events if e.get("memory_hit"))
        errors = sum(1 for e in events if e.get("tool_error"))

        lines.append(f"Total events: {len(events)}")
        lines.append(f"Total execution time: {total_time:.2f}ms")
        lines.append(f"Tool calls: {tool_calls}")
        lines.append(f"Memory hits: {memory_hits}")
        lines.append(f"Errors: {errors}")
        lines.append("")
        lines.append("=" * 80)
        lines.append("EVENT DETAILS")
        lines.append("=" * 80)

        for i, event in enumerate(events, 1):
            event_type = event.get("event_type", "UNKNOWN")
            stage = event.get("stage_name", "Unknown")
            time_ms = event.get("execution_time_ms", 0)
            
            lines.append(f"\n[{i}] {event_type} - {stage}")
            lines.append(f"    Time: {event.get('timestamp', 'N/A')}")
            lines.append(f"    Duration: {time_ms:.2f}ms")
            lines.append(f"    Input: {event.get('input_summary', 'N/A')[:100]}")
            lines.append(f"    Output: {event.get('output_summary', 'N/A')[:100]}")

            if event.get("tool_called"):
                lines.append(f"    Tool: {event.get('tool_name', 'N/A')}")
                if event.get("tool_parameters"):
                    lines.append(f"    Params: {json.dumps(event['tool_parameters'])[:100]}")
                if event.get("tool_error"):
                    lines.append(f"    Error: {event['tool_error']}")

            if event.get("memory_hit"):
                lines.append(f"    Memory: Hit")

        return "\n".join(lines)
