"""
Tests for Memory Layer.
"""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agent.memory import MemoryLayer, SimpleRetriever
from agent.types import MemoryItem, MemoryType


class TestSimpleRetriever:
    def test_calculate_score_exact_match(self):
        query = "hello world"
        content = "hello world this is a test"
        score = SimpleRetriever.calculate_score(query, content)
        assert score > 0

    def test_calculate_score_no_match(self):
        query = "xyz abc"
        content = "hello world"
        score = SimpleRetriever.calculate_score(query, content)
        assert score == 0.0

    def test_calculate_score_partial_match(self):
        query = "hello"
        content = "hello world"
        score = SimpleRetriever.calculate_score(query, content)
        assert score > 0

    def test_retrieve_empty_query(self):
        items = [
            MemoryItem(
                id="1",
                content="first item",
                memory_type=MemoryType.SHORT_TERM,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        ]
        results = SimpleRetriever.retrieve("", items, limit=5)
        assert len(results) == 1

    def test_retrieve_with_matches(self):
        items = [
            MemoryItem(
                id="1",
                content="apple banana cherry",
                memory_type=MemoryType.SHORT_TERM,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
            MemoryItem(
                id="2",
                content="dog elephant fox",
                memory_type=MemoryType.SHORT_TERM,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
            MemoryItem(
                id="3",
                content="apple grape kiwi",
                memory_type=MemoryType.SHORT_TERM,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]
        results = SimpleRetriever.retrieve("apple", items, limit=5)
        assert len(results) == 2
        assert results[0].content in ["apple banana cherry", "apple grape kiwi"]


class TestMemoryLayer:
    def test_write_short_term(self):
        memory = MemoryLayer()
        item = memory.write("test content", MemoryType.SHORT_TERM)

        assert item.id is not None
        assert item.content == "test content"
        assert item.memory_type == MemoryType.SHORT_TERM

    def test_write_long_term(self):
        memory = MemoryLayer()
        item = memory.write("test content", MemoryType.LONG_TERM)

        assert item.memory_type == MemoryType.LONG_TERM

    def test_retrieve_short_term_only(self):
        memory = MemoryLayer()
        memory.write("apple red", MemoryType.SHORT_TERM)
        memory.write("banana yellow", MemoryType.LONG_TERM)

        results = memory.retrieve("apple", memory_type=MemoryType.SHORT_TERM)
        assert len(results) == 1
        assert results[0].content == "apple red"

    def test_retrieve_long_term_only(self):
        memory = MemoryLayer()
        memory.write("apple red", MemoryType.SHORT_TERM)
        memory.write("banana yellow", MemoryType.LONG_TERM)

        results = memory.retrieve("banana", memory_type=MemoryType.LONG_TERM)
        assert len(results) == 1
        assert results[0].content == "banana yellow"

    def test_retrieve_both_types(self):
        memory = MemoryLayer()
        memory.write("apple red", MemoryType.SHORT_TERM)
        memory.write("apple green", MemoryType.LONG_TERM)

        results = memory.retrieve("apple")
        assert len(results) == 2

    def test_retrieve_by_time(self):
        memory = MemoryLayer()
        now = datetime.now()
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)

        memory.write("test content", MemoryType.SHORT_TERM)
        time.sleep(0.01)

        results = memory.retrieve_by_time(past, future)
        assert len(results) >= 1

    def test_get_all(self):
        memory = MemoryLayer()
        memory.write("item1", MemoryType.SHORT_TERM)
        memory.write("item2", MemoryType.LONG_TERM)

        all_items = memory.get_all()
        assert len(all_items) == 2

    def test_clear_short_term(self):
        memory = MemoryLayer()
        memory.write("short", MemoryType.SHORT_TERM)
        memory.write("long", MemoryType.LONG_TERM)

        memory.clear(MemoryType.SHORT_TERM)
        short_items = memory.get_all(MemoryType.SHORT_TERM)
        long_items = memory.get_all(MemoryType.LONG_TERM)

        assert len(short_items) == 0
        assert len(long_items) == 1

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "memory.json"

            memory1 = MemoryLayer(persistent_path=str(memory_path))
            memory1.write("persistent content", MemoryType.LONG_TERM)

            memory2 = MemoryLayer(persistent_path=str(memory_path))
            results = memory2.retrieve("persistent")

            assert len(results) == 1
            assert results[0].content == "persistent content"

    def test_schema_version(self):
        memory = MemoryLayer()
        item = memory.write("test", MemoryType.SHORT_TERM)
        assert item.schema_version == "1.0"
