"""
Memory Layer - Manages short-term (session) and long-term (persistent) memory.
"""

import json
import re
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from agent.types import MemoryItem, MemoryType


class SimpleRetriever:
    @staticmethod
    def calculate_score(query: str, content: str) -> float:
        query_lower = query.lower()
        content_lower = content.lower()

        query_words = set(re.findall(r"\w+", query_lower))
        content_words = set(re.findall(r"\w+", content_lower))

        if not query_words:
            return 0.0

        exact_match_count = sum(
            1 for word in query_words if word in content_words
        )

        substring_match_score = 0.0
        for word in query_words:
            if word in content_lower:
                substring_match_score += 0.5

        combined_score = exact_match_count + substring_match_score
        max_possible = len(query_words) * 1.5
        normalized_score = combined_score / max_possible if max_possible > 0 else 0.0

        return normalized_score

    @staticmethod
    def retrieve(
        query: str, items: list[MemoryItem], limit: int = 10
    ) -> list[MemoryItem]:
        if not query.strip():
            return items[:limit]

        scored = []
        for item in items:
            score = SimpleRetriever.calculate_score(query, item.content)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]


class MemoryLayer:
    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        persistent_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self._short_term: dict[str, MemoryItem] = {}
        self._long_term: dict[str, MemoryItem] = {}
        self._persistent_path = Path(persistent_path) if persistent_path else None
        self._session_id = session_id or str(uuid.uuid4())

        if self._persistent_path:
            self._persistent_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def write(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[dict] = None,
    ) -> MemoryItem:
        item_id = str(uuid.uuid4())
        now = datetime.now()

        item = MemoryItem(
            id=item_id,
            content=content,
            memory_type=memory_type,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            schema_version=self.SCHEMA_VERSION,
        )

        if memory_type == MemoryType.SHORT_TERM:
            self._short_term[item_id] = item
        else:
            self._long_term[item_id] = item
            self._save_to_disk()

        return item

    def retrieve(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        candidates: list[MemoryItem] = []

        if memory_type is None or memory_type == MemoryType.SHORT_TERM:
            candidates.extend(self._short_term.values())
        if memory_type is None or memory_type == MemoryType.LONG_TERM:
            candidates.extend(self._long_term.values())

        return SimpleRetriever.retrieve(query, candidates, limit)

    def retrieve_by_time(
        self,
        start_time: datetime,
        end_time: datetime,
        memory_type: Optional[MemoryType] = None,
    ) -> list[MemoryItem]:
        candidates: list[MemoryItem] = []

        if memory_type is None or memory_type == MemoryType.SHORT_TERM:
            candidates.extend(self._short_term.values())
        if memory_type is None or memory_type == MemoryType.LONG_TERM:
            candidates.extend(self._long_term.values())

        results = [
            item
            for item in candidates
            if start_time <= item.created_at <= end_time
        ]
        results.sort(key=lambda x: x.created_at)
        return results

    def get_all(
        self, memory_type: Optional[MemoryType] = None
    ) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        if memory_type is None or memory_type == MemoryType.SHORT_TERM:
            items.extend(self._short_term.values())
        if memory_type is None or memory_type == MemoryType.LONG_TERM:
            items.extend(self._long_term.values())
        return items

    def clear(self, memory_type: Optional[MemoryType] = None) -> None:
        if memory_type is None or memory_type == MemoryType.SHORT_TERM:
            self._short_term.clear()
        if memory_type is None or memory_type == MemoryType.LONG_TERM:
            self._long_term.clear()
            if self._persistent_path and self._persistent_path.exists():
                self._persistent_path.unlink()

    def _save_to_disk(self) -> None:
        if not self._persistent_path:
            return

        data = {
            "schema_version": self.SCHEMA_VERSION,
            "session_id": self._session_id,
            "saved_at": datetime.now().isoformat(),
            "long_term_memories": [
                self._memory_to_dict(item) for item in self._long_term.values()
            ],
        }

        with open(self._persistent_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_from_disk(self) -> None:
        if not self._persistent_path or not self._persistent_path.exists():
            return

        with open(self._persistent_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        schema_version = data.get("schema_version", "1.0")
        if schema_version != self.SCHEMA_VERSION:
            return

        for item_data in data.get("long_term_memories", []):
            try:
                item = self._dict_to_memory(item_data)
                self._long_term[item.id] = item
            except (KeyError, ValueError):
                continue

    def _memory_to_dict(self, item: MemoryItem) -> dict:
        result = asdict(item)
        result["memory_type"] = item.memory_type.name
        result["created_at"] = item.created_at.isoformat()
        result["updated_at"] = item.updated_at.isoformat()
        return result

    def _dict_to_memory(self, data: dict) -> MemoryItem:
        return MemoryItem(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType[data["memory_type"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
            schema_version=data.get("schema_version", self.SCHEMA_VERSION),
        )
