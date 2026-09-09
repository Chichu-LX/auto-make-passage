from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .schemas import WorkflowState


class WorkflowGraphState(TypedDict):
    state: str
    target_state: str


class LangGraphWorkflowTopology:
    """Executable conditional guard for every Lead-selected Runtime transition."""

    def __init__(
        self,
        edges: Iterable[tuple[WorkflowState, WorkflowState]],
    ) -> None:
        self.edges = frozenset(edges)
        graph = StateGraph(WorkflowGraphState)
        graph.add_node("transition_guard", self._identity)
        for state in WorkflowState:
            graph.add_node(state.value, self._identity)
        graph.add_edge(START, "transition_guard")
        graph.add_conditional_edges(
            "transition_guard",
            self._route_selected_target,
            {state.value: state.value for state in WorkflowState},
        )
        for state in WorkflowState:
            graph.add_edge(state.value, END)
        self.compiled = graph.compile()

    @staticmethod
    def _identity(state: WorkflowGraphState) -> WorkflowGraphState:
        return state

    def _route_selected_target(self, state: WorkflowGraphState) -> str:
        try:
            source = WorkflowState(state["state"])
            target = WorkflowState(state["target_state"])
        except (KeyError, ValueError) as exc:
            raise ValueError("LangGraph received an invalid transition state") from exc
        if (source, target) not in self.edges:
            raise ValueError(
                f"LangGraph topology rejects transition {source.value} -> {target.value}"
            )
        return target.value

    def assert_transition(
        self, source: WorkflowState, target: WorkflowState
    ) -> None:
        self.compiled.invoke(
            {"state": source.value, "target_state": target.value}
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "engine": "langgraph",
            "node_count": len(WorkflowState),
            "guard_node_count": 1,
            "edge_count": len(self.edges),
            "entry": "transition_guard",
            "routing": "conditional",
        }
