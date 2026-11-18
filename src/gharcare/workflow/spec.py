"""A tiny declarative graph definition with two interchangeable executors:
  * LangGraphRunner  - compiles the spec into a LangGraph StateGraph (used when langgraph is installed)
  * LocalRunner      - a minimal built-in executor with the same semantics (offline / tests)
Node logic is plain Python, so behaviour is identical and fully testable either way."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable

from gharcare.errors import WorkflowError

END = "__end__"
Node = Callable[[dict], dict]


@dataclass
class WorkflowSpec:
    entry: str
    nodes: dict[str, Node]
    edges: dict[str, str] = field(default_factory=dict)                                   # node -> next node | END
    branches: dict[str, tuple[Callable[[dict], str], dict[str, str]]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.entry not in self.nodes:
            raise WorkflowError(f"entry node {self.entry!r} is not defined")
        for name in self.nodes:
            if (name in self.edges) == (name in self.branches):
                raise WorkflowError(f"node {name!r} must have exactly one outgoing edge or branch")
        targets = list(self.edges.values()) + [t for _, m in self.branches.values() for t in m.values()]
        for target in targets:
            if target != END and target not in self.nodes:
                raise WorkflowError(f"edge points to unknown node {target!r}")


class LocalRunner:
    def __init__(self, spec: WorkflowSpec, max_steps: int = 30) -> None:
        spec.validate()
        self._spec, self._max_steps = spec, max_steps

    def invoke(self, state: dict) -> dict:
        state, node = dict(state), self._spec.entry
        for _ in range(self._max_steps):
            state.update(self._spec.nodes[node](state) or {})
            if node in self._spec.branches:
                router, mapping = self._spec.branches[node]
                key = router(state)
                if key not in mapping:
                    raise WorkflowError(f"router for {node!r} returned unknown key {key!r}")
                nxt = mapping[key]
            else:
                nxt = self._spec.edges[node]
            if nxt == END:
                return state
            node = nxt
        raise WorkflowError(f"workflow exceeded {self._max_steps} steps (loop?)")


class LangGraphRunner:
    def __init__(self, spec: WorkflowSpec, state_schema: type) -> None:
        spec.validate()
        try:
            from langgraph.graph import END as LG_END, START, StateGraph
        except ImportError as exc:
            raise WorkflowError("LangGraph backend needs: pip install langgraph") from exc
        graph = StateGraph(state_schema)
        for name, fn in spec.nodes.items():
            graph.add_node(name, fn)
        graph.add_edge(START, spec.entry)
        for source, target in spec.edges.items():
            graph.add_edge(source, LG_END if target == END else target)
        for source, (router, mapping) in spec.branches.items():
            graph.add_conditional_edges(source, router, {k: (LG_END if v == END else v) for k, v in mapping.items()})
        self._compiled = graph.compile()

    def invoke(self, state: dict) -> dict:
        return dict(self._compiled.invoke(state))


def langgraph_available() -> bool:
    return importlib.util.find_spec("langgraph") is not None


def make_runner(spec: WorkflowSpec, state_schema: type, backend: str = "auto") -> Any:
    if backend == "auto":
        backend = "langgraph" if langgraph_available() else "local"
    if backend == "local":
        return LocalRunner(spec), "local"
    if backend == "langgraph":
        return LangGraphRunner(spec, state_schema), "langgraph"
    raise WorkflowError(f"unknown backend {backend!r} (use 'auto', 'local' or 'langgraph')")
