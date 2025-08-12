# search_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Callable, Optional, Protocol, Tuple, List
import heapq, math, time

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

    # (f, g, counter, Node) – smaller f first; then smaller g; then FIFO by counter
    counter = 0
    open_heap: List[Tuple[float, float, int, Node]] = [(n0.f, n0.g, counter, n0)]
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
            n2 = Node(s2, g=g2, h=h2, parent=node, action=action)
            nodes_generated += 1

            if n2.f >= U:
                continue  # surplus; prune against incumbent U

            counter += 1
            heapq.heappush(open_heap, (n2.f, n2.g, counter, n2))
            best_g[s2] = g2

    t1 = time.perf_counter()
    if best_goal:
        return SearchResult(
            found=True,
            path=reconstruct_path(best_goal),
            cost=best_goal.g,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            elapsed_sec=(t1 - t0),
            algorithm="A*-EARLY",
        )
    else:
        return SearchResult(False, [], math.inf, nodes_expanded, nodes_generated, (t1 - t0), "A*-EARLY")


# ---------- IDA* (iterative deepening on f = g + h) ----------

def idastar(problem: SearchProblem, h: HeuristicFn) -> SearchResult:
    t0 = time.perf_counter()
    nodes_expanded_total = 0
    nodes_generated_total = 0

    s0 = problem.initial_state()
    h0 = h(s0)
    nodes_generated_total += 1

    # Depth-first contour search
    def dfs(node: Node, bound: float) -> Tuple[float, Optional[Node], int, int]:
        nonlocal nodes_expanded_total, nodes_generated_total
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
            child = Node(s2, g2, h2, parent=node, action=action)

            t, sol, e_cnt, g_cnt = dfs(child, bound)
            expanded_here += e_cnt
            if sol is not None:
                return t, sol, expanded_here, 0
            min_next_bound = min(min_next_bound, t)

        return min_next_bound, None, expanded_here, 0

    bound = h0  # initial f-bound
    root = Node(s0, 0.0, h0, parent=None, action=None)
    while True:
        t, sol, _, _ = dfs(root, bound)
        if sol is not None:
            t1 = time.perf_counter()
            return SearchResult(
                found=True,
                path=reconstruct_path(sol),
                cost=sol.g,
                nodes_expanded=nodes_expanded_total,
                nodes_generated=nodes_generated_total,
                elapsed_sec=(t1 - t0),
                algorithm="IDA*",
            )
        if t == math.inf:
            t1 = time.perf_counter()
            return SearchResult(False, [], math.inf, nodes_expanded_total, nodes_generated_total, (t1 - t0), "IDA*")
        bound = t  # raise threshold


# ---------- ILBFS (non-recursive RBFS-equivalent; linear memory best-first) ----------

def ilbfs(problem: SearchProblem, h: HeuristicFn) -> SearchResult:
    """
    Linear-memory best-first (RBFS-equivalent).
    Maintains a single path stack; each path node carries an F-bound (the best
    alternative below it). Expands best child; when best f exceeds its bound,
    backtracks and updates the parent's backed-up F.
    """
    t0 = time.perf_counter()
    nodes_expanded = 0
    nodes_generated = 0

    s0 = problem.initial_state()
    root = Node(s0, g=0.0, h=h(s0), parent=None, action=None)
    nodes_generated += 1

    # Stack of (node, F_bound)
    stack: List[Tuple[Node, float]] = [(root, math.inf)]

    # For each expanded node, cache its children (ordered by f) and their current f values.
    children_cache: dict[int, List[Node]] = {}

    def expand(n: Node) -> List[Node]:
        nonlocal nodes_expanded, nodes_generated
        nodes_expanded += 1
        kids: List[Node] = []
        for action, s2, cost in problem.successors(n.state):
            g2 = n.g + cost
            n2 = Node(s2, g=g2, h=h(s2), parent=n, action=action)
            nodes_generated += 1
            kids.append(n2)
        kids.sort(key=lambda c: (c.f, c.g))  # best-first; tie-break smaller g
        return kids

    while stack:
        node, bound = stack[-1]

        # Goal check at the tip
        if problem.is_goal(node.state):
            t1 = time.perf_counter()
            return SearchResult(
                found=True,
                path=reconstruct_path(node),
                cost=node.g,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                elapsed_sec=(t1 - t0),
                algorithm="ILBFS",
            )

        # Get or generate children list
        key = id(node)
        if key not in children_cache:
            children_cache[key] = expand(node)
        kids = children_cache[key]

        if not kids:
            # Dead end: backtrack by popping
            stack.pop()
            if not stack:
                break
            # On backtrack, update parent's best alternative (no-op here; handled via kids sorting)
            continue

        # Best and alternative f-values
        best = kids[0]
        alt_f = kids[1].f if len(kids) > 1 else math.inf

        # If best exceeds current bound, backtrack (propagate new bound = best.f)
        if best.f > bound:
            stack.pop()
            if not stack:
                break
            # Update parent's bound to min(previous_alt, best.f)
            parent, parent_bound = stack[-1]
            new_bound = min(parent_bound, best.f)
            stack[-1] = (parent, new_bound)
            continue

        # Otherwise, dive into best child with a new bound: min(bound, alt_f)
        stack.append((best, min(bound, alt_f)))
        # Remove it from the siblings list so that, upon backtrack, the next best is ready
        kids.pop(0)

    t1 = time.perf_counter()
    return SearchResult(False, [], math.inf, nodes_expanded, nodes_generated, (t1 - t0), "ILBFS")
