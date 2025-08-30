# search_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Callable, Optional, Protocol, Tuple, List
import heapq, math, time

import sys
from collections.abc import Mapping, Set as _Set

def _deep_getsizeof(o, _seen=None):
    """Approximate deep size in bytes of a Python object graph.
    Only used to estimate memory overhead of seen_states."""
    if _seen is None:
        _seen = set()
    oid = id(o)
    if oid in _seen:
        return 0
    _seen.add(oid)
    size = sys.getsizeof(o)
    if isinstance(o, Mapping):
        for k, v in o.items():
            size += _deep_getsizeof(k, _seen)
            size += _deep_getsizeof(v, _seen)
    elif isinstance(o, (list, tuple, set, frozenset)) or isinstance(o, _Set):
        for i in o:
            size += _deep_getsizeof(i, _seen)
    elif hasattr(o, "__dict__"):
        size += _deep_getsizeof(vars(o), _seen)
    elif hasattr(o, "__slots__"):
        for s in o.__slots__:
            if hasattr(o, s):
                size += _deep_getsizeof(getattr(o, s), _seen)
    return size
# ---------- Core abstractions ----------

class StateLike(Protocol):
    """Your state must be hashable + comparable for dict/set usage."""
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

Action = Any  # keep generic; e.g., 'CLICK', 'NO_CLICK', 0/1, etc.

class SearchProblem(Protocol):
    """Algorithm-agnostic problem interface."""
    def initial_state(self) -> StateLike: ...
    def is_goal(self, s: StateLike) -> bool: ...
    def successors(self, s: StateLike) -> Iterable[Tuple[Action, StateLike, float]]:
        """Yield (action, next_state, step_cost)."""
        ...

HeuristicFn = Callable[[StateLike], float]

@dataclass
class Node:
    state: StateLike
    g: float
    h: float
    parent: Optional["Node"] = None
    action: Optional[Action] = None

    @property
    def f(self) -> float:
        return self.g + self.h

def reconstruct_path(n: Node) -> List[Tuple[StateLike, Optional[Action]]]:
    path: List[Tuple[StateLike, Optional[Action]]] = []
    while n:
        path.append((n.state, n.action))
        n = n.parent  # type: ignore
    path.reverse()
    # First tuple's action is the action that led to initial (usually None)
    return path

@dataclass
class SearchResult:
    found: bool
    path: List[Tuple[StateLike, Optional[Action]]]
    cost: float
    nodes_expanded: int
    nodes_generated: int
    duplicate_nodes: int         # NEW: number of duplicate states generated
    seen_bytes_peak: int          # NEW: peak deep size (bytes) of seen_states during this run
    open_size_peak: int          # NEW: largest “OPEN” size (PQ length for A*/UCS; recursion depth for IDA*/ILBFS)

    elapsed_sec: float
    algorithm: str


# ---------- A*-EARLY (goal test on generation, conservative stop rule) ----------

def astar_early(problem: SearchProblem, h: HeuristicFn,
                tie_break: Callable[[Tuple[float, float, int], Tuple[float, float, int]], bool] = None
                ) -> SearchResult:
    """
    A*-EARLY with an incumbent solution U; we stop when OPEN.min_f >= U.
    Tie-breaker (optional): prefer smaller g on equal f.
    """
    t0 = time.perf_counter()
    nodes_expanded = 0
    nodes_generated = 0

    s0 = problem.initial_state()
    n0 = Node(s0, g=0.0, h=h(s0), parent=None, action=None)
    nodes_generated += 1
    duplicate_nodes = 0        # NEW
    seen_states = {s0}         # NEW: track all states generated this run
    seen_bytes_peak = _deep_getsizeof(seen_states)   # NEW


    # (f, g, counter, Node) – smaller f first; then smaller g; then FIFO by counter
    counter = 0
    open_heap: List[Tuple[float, float, int, Node]] = [(n0.f, n0.g, counter, n0)]
    open_size_peak = len(open_heap)  # NEW

    best_g = {s0: 0.0}

    U = math.inf       # incumbent cost (best solution found so far)
    best_goal: Optional[Node] = None

    while open_heap:
        # Early-stop condition: when the best possible f on OPEN is >= U
        fmin, _, _, _ = open_heap[0]
        if fmin >= U:
            break

        _, _, _, node = heapq.heappop(open_heap)
        # Skip stale entries
        if node.g > best_g.get(node.state, math.inf):
            continue

        nodes_expanded += 1

        # Expand
        for action, s2, cost in problem.successors(node.state):
            g2 = node.g + cost
            # EARLY GOAL TEST: check on generation
            if problem.is_goal(s2):
                if g2 < U:
                    U = g2
                    best_goal = Node(s2, g=g2, h=0.0, parent=node, action=action)
                continue  # do not insert goal children; rely on U/pruning

            if g2 >= best_g.get(s2, math.inf):
                continue  # not better

            h2 = h(s2)
            # NEW: duplicate counting
            if s2 in seen_states:
                duplicate_nodes += 1
            seen_states.add(s2)
            cur_seen = _deep_getsizeof(seen_states)          # NEW
            if cur_seen > seen_bytes_peak:
                seen_bytes_peak = cur_seen
            n2 = Node(s2, g=g2, h=h2, parent=node, action=action)
            nodes_generated += 1

            if n2.f >= U:
                continue  # surplus; prune against incumbent U

            counter += 1
            heapq.heappush(open_heap, (n2.f, n2.g, counter, n2))
            if len(open_heap) > open_size_peak:
                open_size_peak = len(open_heap)
            best_g[s2] = g2

    t1 = time.perf_counter()
    if best_goal:
        return SearchResult(
            found=True,
            path=reconstruct_path(best_goal),
            cost=best_goal.g,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            duplicate_nodes=duplicate_nodes, 
            seen_bytes_peak= seen_bytes_peak,  # NEW
            open_size_peak=open_size_peak,

            elapsed_sec=(t1 - t0),
            algorithm="A*-EARLY",
        )
    else:
        return SearchResult(False, [], math.inf, nodes_expanded, nodes_generated, duplicate_nodes,seen_bytes_peak,open_size_peak,  (t1 - t0), "A*-EARLY", )


# ---------- Uniform-Cost Search (Dijkstra) ----------

def ucs(problem: SearchProblem, h: HeuristicFn | None = None) -> SearchResult:
    """
    Uniform-Cost Search (no heuristic). Uses step costs from the problem.
    Note: For Flappy1D, NO_CLICK should have a small positive cost (e.g., 0.0001)
    so UCS makes progress and distinguishes paths.
    """
    t0 = time.perf_counter()
    nodes_expanded = 0
    nodes_generated = 0

    s0 = problem.initial_state()
    n0 = Node(s0, g=0.0, h=0.0, parent=None, action=None)
    nodes_generated += 1
    duplicate_nodes = 0
    seen_states = {s0}
    seen_bytes_peak = _deep_getsizeof(seen_states)


    # (g, counter, Node) – smaller g first; then FIFO by counter
    counter = 0
    open_heap: List[Tuple[float, int, Node]] = [(n0.g, counter, n0)]
    open_size_peak = len(open_heap)  # NEW

    best_g = {s0: 0.0}

    while open_heap:
        _, _, node = heapq.heappop(open_heap)
        # Skip stale entries
        if node.g > best_g.get(node.state, math.inf):
            continue
        # Goal on pop is optimal for UCS
        if problem.is_goal(node.state):
            t1 = time.perf_counter()
            return SearchResult(
                found=True,
                path=reconstruct_path(node),
                cost=node.g,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                duplicate_nodes=duplicate_nodes,
                seen_bytes_peak= seen_bytes_peak,  # NEW
                open_size_peak=open_size_peak,

                elapsed_sec=(t1 - t0),
                algorithm="UCS",
            )

        nodes_expanded += 1
        for action, s2, cost in problem.successors(node.state):
            g2 = node.g + cost
            if g2 >= best_g.get(s2, math.inf):
                continue
            if s2 in seen_states:
                duplicate_nodes += 1
            seen_states.add(s2)
            cur_seen = _deep_getsizeof(seen_states)          # NEW
            if cur_seen > seen_bytes_peak:
                seen_bytes_peak = cur_seen
            n2 = Node(s2, g=g2, h=0.0, parent=node, action=action)
            nodes_generated += 1
            counter += 1
            heapq.heappush(open_heap, (g2, counter, n2))
            if len(open_heap) > open_size_peak:
                open_size_peak = len(open_heap)
            best_g[s2] = g2

    t1 = time.perf_counter()
    return SearchResult(False, [], math.inf, nodes_expanded, nodes_generated,duplicate_nodes, seen_bytes_peak,open_size_peak, (t1 - t0), "UCS")

# ---------- IDA* (iterative deepening on f = g + h) ----------

def idastar(problem: SearchProblem, h: HeuristicFn) -> SearchResult:
    t0 = time.perf_counter()
    nodes_expanded_total = 0
    nodes_generated_total = 0

    s0 = problem.initial_state()
    h0 = h(s0)
    nodes_generated_total += 1
    duplicate_nodes_total = 0         # NEW
    seen_states = {s0}
    seen_bytes_peak = _deep_getsizeof(seen_states)   # NEW
    stack_size_peak = 1  # root on stack


    # Depth-first contour search
    def dfs(node: Node, bound: float,  depth: int) -> Tuple[float, Optional[Node], int, int]:
        nonlocal nodes_expanded_total, nodes_generated_total,stack_size_peak
        if depth > stack_size_peak:
            stack_size_peak = depth
        f = node.f
        if f > bound:
            return f, None, 0, 0
        if problem.is_goal(node.state):
            return f, node, 0, 0

        min_next_bound = math.inf
        expanded_here = 1
        nodes_expanded_total += 1

        for action, s2, cost in problem.successors(node.state):
            g2 = node.g + cost
            h2 = h(s2)
            nodes_generated_total += 1
                        # NEW: duplicate counting
            nonlocal duplicate_nodes_total, seen_bytes_peak
            if s2 in seen_states:
                duplicate_nodes_total += 1
            seen_states.add(s2)
            cur_seen = _deep_getsizeof(seen_states)          # NEW
            if cur_seen > seen_bytes_peak:
                seen_bytes_peak = cur_seen
            child = Node(s2, g2, h2, parent=node, action=action)

            t, sol, e_cnt, g_cnt = dfs(child, bound, depth + 1)
            expanded_here += e_cnt
            if sol is not None:
                return t, sol, expanded_here, 0
            min_next_bound = min(min_next_bound, t)

        return min_next_bound, None, expanded_here, 0

    bound = h0  # initial f-bound
    root = Node(s0, 0.0, h0, parent=None, action=None)
    while True:
        t, sol, e_cnt, g_cnt = dfs(root, bound, 1)
        if sol is not None:
            t1 = time.perf_counter()
            return SearchResult(
                found=True,
                path=reconstruct_path(sol),
                cost=sol.g,
                nodes_expanded=nodes_expanded_total,
                nodes_generated=nodes_generated_total,
                duplicate_nodes= duplicate_nodes_total,
                seen_bytes_peak= seen_bytes_peak,  # NEW
                open_size_peak=stack_size_peak,

                elapsed_sec=(t1 - t0),
                algorithm="IDA*",
            )
        if t == math.inf:
            t1 = time.perf_counter()
            return SearchResult(False, [], math.inf, nodes_expanded_total, nodes_generated_total, duplicate_nodes_total, seen_bytes_peak,stack_size_peak, (t1 - t0), "IDA*")
        bound = t  # raise threshold


# ---------- ILBFS (non-recursive RBFS-equivalent; linear memory best-first) ----------

def ilbfs(problem: SearchProblem, h: HeuristicFn) -> SearchResult:
    """
    Recursive Best-First Search (RBFS).
    Uses f-bound recursion with linear memory; returns optimal solution if one exists.
    """
    t0 = time.perf_counter()
    nodes_expanded = 0
    nodes_generated = 0

    s0 = problem.initial_state()
    root = Node(s0, g=0.0, h=h(s0), parent=None, action=None)
    nodes_generated += 1
    duplicate_nodes = 0
    seen_states = {s0}
    seen_bytes_peak = _deep_getsizeof(seen_states)   # NEW
    stack_size_peak = 1  # root on stack


    # Return list of (child_node, backed_up_f)
    def expand(n: Node) -> List[Tuple[Node, float]]:
        nonlocal nodes_expanded, nodes_generated, duplicate_nodes, seen_bytes_peak
        nodes_expanded += 1
        kids: List[Tuple[Node, float]] = []
        for action, s2, cost in problem.successors(n.state):
            g2 = n.g + cost
            if s2 in seen_states:
                duplicate_nodes += 1
            seen_states.add(s2)
            cur_seen = _deep_getsizeof(seen_states)          # NEW
            if cur_seen > seen_bytes_peak:
                seen_bytes_peak = cur_seen
            n2 = Node(s2, g=g2, h=h(s2), parent=n, action=action)
            nodes_generated += 1
            f_backed = max(n2.f, n.f)
            kids.append((n2, f_backed))
        kids.sort(key=lambda t: (t[1], t[0].g))
        return kids

    def rbfs(node: Node, f_limit: float, depth: int) -> Tuple[Optional[Node], float]:
        nonlocal stack_size_peak
        if depth > stack_size_peak:
            stack_size_peak = depth
        # Goal check
        if problem.is_goal(node.state):
            return node, node.f
        # Expand current node
        children = expand(node)
        if not children:
            return None, math.inf
        while True:
            children.sort(key=lambda t: (t[1], t[0].g))
            best_node, best_f = children[0]
            alt_f = children[1][1] if len(children) > 1 else math.inf
            if best_f > f_limit:
                return None, best_f
            result, new_best_f = rbfs(best_node, min(f_limit, alt_f), depth + 1)
            # Update backed-up f for best child in-place
            children[0] = (best_node, new_best_f)
            if result is not None:
                return result, new_best_f

    result, _ = rbfs(root, math.inf, 1)
    t1 = time.perf_counter()
    if result is not None:
        return SearchResult(
            found=True,
            path=reconstruct_path(result),
            cost=result.g,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            duplicate_nodes=duplicate_nodes,
            seen_bytes_peak= seen_bytes_peak,
            open_size_peak=stack_size_peak,

            elapsed_sec=(t1 - t0),
            algorithm="RBFS",
        )
    else:
        return SearchResult(False, [], math.inf, nodes_expanded, nodes_generated, duplicate_nodes, seen_bytes_peak,stack_size_peak, (t1 - t0), "RBFS")
