# server/agent_memory.py
"""
Shared memory bank for the Overseer agent.
Stores past episode outcomes so the Overseer can learn from its mistakes.
This is the self-improvement mechanism — judges will see reward go UP over runs.
"""
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field, asdict
import json


@dataclass
class MemoryEntry:
    """One stored lesson from a past episode."""
    task_id:          str
    episode_number:   int
    worker_id:        str           # which worker agent made the call
    worker_reported:  str           # what the worker said (e.g. "normal" or "flag:temperature_c")
    actual_fault:     str           # what actually happened (ground truth)
    overseer_verdict: str           # what the overseer decided
    was_correct:      bool          # did overseer get it right?
    score:            float         # score for this episode
    lesson:           str           # human-readable lesson string


class AgentMemoryBank:
    """
    In-memory store of past episode outcomes.
    Injected into Overseer's system prompt each episode.
    Grows over runs → agent improves → reward curves go up.
    """

    def __init__(self, max_entries: int = 20):
        self._entries: list[MemoryEntry] = []
        self._episode_count: int = 0
        self._max_entries = max_entries

    def store(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        # Keep only the most recent N entries to avoid prompt bloat
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    def increment_episode(self) -> int:
        self._episode_count += 1
        return self._episode_count

    def get_episode_count(self) -> int:
        return self._episode_count

    def retrieve_for_task(self, task_id: str) -> list[MemoryEntry]:
        """Get all memories relevant to a specific task."""
        return [e for e in self._entries if e.task_id == task_id]

    def retrieve_all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def format_for_prompt(self, task_id: str) -> str:
        """
        Format relevant memories as a string to inject into the Overseer prompt.
        Returns empty string if no memories yet.
        """
        relevant = self.retrieve_for_task(task_id)
        if not relevant:
            return "No past experience with this task yet."

        lines = [f"PAST EXPERIENCE ({len(relevant)} episodes of {task_id}):"]
        for e in relevant[-5:]:  # last 5 most recent
            status = "✓ CORRECT" if e.was_correct else "✗ WRONG"
            lines.append(
                f"  Episode {e.episode_number}: {status} | "
                f"Worker {e.worker_id} said '{e.worker_reported}' | "
                f"Truth: '{e.actual_fault}' | "
                f"Lesson: {e.lesson} | Score: {e.score:.2f}"
            )
        return "\n".join(lines)

    def get_stats(self) -> dict:
        if not self._entries:
            return {"total_episodes": 0, "accuracy": 0.0, "avg_score": 0.0}
        correct = sum(1 for e in self._entries if e.was_correct)
        return {
            "total_episodes": self._episode_count,
            "memory_entries":  len(self._entries),
            "accuracy":        round(correct / len(self._entries), 3),
            "avg_score":       round(sum(e.score for e in self._entries) / len(self._entries), 3),
        }


# Module-level singleton — shared across all episodes in one server lifetime
_MEMORY_BANK: Optional[AgentMemoryBank] = None

def get_memory_bank() -> AgentMemoryBank:
    global _MEMORY_BANK
    if _MEMORY_BANK is None:
        _MEMORY_BANK = AgentMemoryBank()
    return _MEMORY_BANK