"""Task DAG with handle-based dependency inference and eager dispatch."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .types import ParamType


@dataclass
class TaskNode:
    """A single task in the DAG."""

    task_id: int
    chip: int
    kernel: str
    args: list
    deps: set[int] = field(default_factory=set)
    is_group: bool = False
    group_chips: list[int] = field(default_factory=list)


class TaskDAG:
    """Dependency graph for L3 task scheduling.

    Dependencies are inferred automatically from tensor handle usage:
    if task B reads a handle that task A produced (OUTPUT/INOUT),
    then B depends on A.
    """

    def __init__(self):
        self._tasks: dict[int, TaskNode] = {}
        self._handle_producer: dict[int, int] = {}  # handle_id → producer task_id
        self._consumers: dict[int, set[int]] = defaultdict(set)  # task_id → dependents
        self._completed: set[int] = set()
        self._dispatched: set[int] = set()
        self._next_id = 0

    def add_task(self, chip: int, kernel: str, args: list,
                 is_group: bool = False,
                 group_chips: list[int] = None) -> TaskNode:
        """Add a task and infer dependencies from tensor handles.

        Returns the TaskNode (with deps populated).
        """
        task_id = self._next_id
        self._next_id += 1

        deps = set()
        for arg in args:
            if not hasattr(arg, "type"):
                continue
            if arg.type in (ParamType.INPUT, ParamType.INOUT):
                handle_id = getattr(arg, "_handle_id", None)
                if handle_id is not None:
                    producer = self._handle_producer.get(handle_id)
                    if producer is not None and producer not in self._completed:
                        deps.add(producer)

        # Register outputs
        for arg in args:
            if not hasattr(arg, "type"):
                continue
            if arg.type in (ParamType.OUTPUT, ParamType.INOUT):
                handle_id = getattr(arg, "_handle_id", None)
                if handle_id is not None:
                    self._handle_producer[handle_id] = task_id

        node = TaskNode(
            task_id=task_id,
            chip=chip,
            kernel=kernel,
            args=args,
            deps=deps,
            is_group=is_group,
            group_chips=group_chips or [],
        )
        self._tasks[task_id] = node

        for dep in deps:
            self._consumers[dep].add(task_id)

        return node

    def is_ready(self, task_id: int) -> bool:
        """Check if a task has all dependencies satisfied."""
        node = self._tasks.get(task_id)
        if node is None:
            return False
        return len(node.deps) == 0

    def mark_dispatched(self, task_id: int) -> None:
        self._dispatched.add(task_id)

    def complete(self, task_id: int) -> list[TaskNode]:
        """Mark task complete and return newly-ready tasks."""
        self._completed.add(task_id)
        ready = []

        for consumer_id in self._consumers.get(task_id, set()):
            consumer = self._tasks[consumer_id]
            consumer.deps.discard(task_id)
            if len(consumer.deps) == 0 and consumer_id not in self._dispatched:
                ready.append(consumer)

        return ready

    def get_ready_tasks(self) -> list[TaskNode]:
        """Return all tasks that are ready but not yet dispatched."""
        ready = []
        for tid, node in self._tasks.items():
            if tid not in self._dispatched and len(node.deps) == 0:
                ready.append(node)
        return ready

    def all_complete(self) -> bool:
        return len(self._completed) == len(self._tasks)

    @property
    def task_count(self) -> int:
        return len(self._tasks)
