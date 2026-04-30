"""
Perception Layer - Handles input validation and normalization.
Converts raw user input into structured TaskInput objects.
"""

import re
import time
import uuid
from typing import Optional

from agent.types import (
    Context,
    TaskInput,
    TraceEvent,
    TraceEventType,
    UserInput,
)


class InputValidationError(Exception):
    pass


class InputNormalizer:
    MAX_INPUT_LENGTH = 4096
    ILLEGAL_PATTERNS = [
        re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"),
    ]

    @classmethod
    def validate(cls, text: str, max_length: Optional[int] = None) -> tuple[bool, Optional[str]]:
        if not text or not text.strip():
            return False, "Input is empty or contains only whitespace"

        actual_max_length = max_length if max_length is not None else cls.MAX_INPUT_LENGTH
        if len(text) > actual_max_length:
            return (
                False,
                f"Input exceeds maximum length ({len(text)} > {actual_max_length})",
            )

        for pattern in cls.ILLEGAL_PATTERNS:
            if pattern.search(text):
                return False, "Input contains illegal control characters"

        return True, None

    @classmethod
    def normalize(cls, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @classmethod
    def truncate(cls, text: str, max_length: int = 100) -> str:
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."


class PerceptionLayer:
    def __init__(self, max_input_length: Optional[int] = None):
        self.max_input_length = max_input_length or InputNormalizer.MAX_INPUT_LENGTH

    def process(
        self, user_input: UserInput, context: Context
    ) -> tuple[TaskInput, list[TraceEvent]]:
        events: list[TraceEvent] = []
        start_time = time.time()

        input_summary = InputNormalizer.truncate(user_input.text)
        events.append(
            self._create_event(
                TraceEventType.PERCEPTION_START,
                "Perception-Start",
                input_summary,
                "Starting input processing",
            )
        )

        is_valid, error_msg = InputNormalizer.validate(user_input.text, self.max_input_length)

        if not is_valid:
            normalized = ""
            status = f"INVALID: {error_msg}"
        else:
            normalized = InputNormalizer.normalize(user_input.text)
            status = "VALID"

        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000

        output_summary = (
            f"Status: {status}, Normalized: {InputNormalizer.truncate(normalized)}"
        )

        events.append(
            self._create_event(
                TraceEventType.PERCEPTION_END,
                "Perception-End",
                input_summary,
                output_summary,
                execution_time_ms=execution_time_ms,
                metadata={
                    "validation_status": status,
                    "input_length": len(user_input.text),
                    "normalized_length": len(normalized),
                },
            )
        )

        task_input = TaskInput(
            user_input=user_input,
            context=context,
            normalized_input=normalized,
            validation_status=status,
        )

        return task_input, events

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
