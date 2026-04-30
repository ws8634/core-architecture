"""
Tests for Perception Layer.
"""

import pytest

from agent.perception import InputNormalizer, PerceptionLayer
from agent.types import Context, UserInput


class TestInputNormalizer:
    def test_validate_empty_input(self):
        is_valid, error = InputNormalizer.validate("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_validate_whitespace_only(self):
        is_valid, error = InputNormalizer.validate("   \t\n  ")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_validate_valid_input(self):
        is_valid, error = InputNormalizer.validate("Hello, world!")
        assert is_valid is True
        assert error is None

    def test_validate_too_long_input(self):
        long_input = "x" * (InputNormalizer.MAX_INPUT_LENGTH + 100)
        is_valid, error = InputNormalizer.validate(long_input)
        assert is_valid is False
        assert "exceeds" in error.lower()

    def test_normalize_whitespace(self):
        result = InputNormalizer.normalize("  hello   world  ")
        assert result == "hello world"

    def test_normalize_newlines(self):
        result = InputNormalizer.normalize("hello\nworld\n\n!")
        assert result == "hello world !"

    def test_truncate_short(self):
        result = InputNormalizer.truncate("short text", max_length=50)
        assert result == "short text"

    def test_truncate_long(self):
        result = InputNormalizer.truncate("a" * 200, max_length=100)
        assert len(result) == 103
        assert result.endswith("...")


class TestPerceptionLayer:
    def test_process_valid_input(self):
        layer = PerceptionLayer()
        user_input = UserInput(text="Calculate 1 + 2")
        context = Context()

        task_input, events = layer.process(user_input, context)

        assert "VALID" in task_input.validation_status
        assert task_input.normalized_input == "Calculate 1 + 2"
        assert len(events) > 0

    def test_process_invalid_input(self):
        layer = PerceptionLayer()
        user_input = UserInput(text="")
        context = Context()

        task_input, events = layer.process(user_input, context)

        assert "INVALID" in task_input.validation_status
        assert task_input.normalized_input == ""
        assert len(events) > 0

    def test_custom_max_length(self):
        layer = PerceptionLayer(max_input_length=10)
        user_input = UserInput(text="This is definitely longer than 10 characters")
        context = Context()

        task_input, events = layer.process(user_input, context)

        assert "INVALID" in task_input.validation_status
