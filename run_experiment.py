# run_experiment.py
from search_core import astar_early, idastar, ilbfs
from flappy_1d import Flappy1D, clicks_heuristic_factory, BirdState

def run_once(algo_name: str):
    # Example level
    y0, x0 = 6, 50
    L, U = 1, 6      # gap [L, U], but we require y >= L+1 while x >= 0
    y_min, y_max = 0, 6

    problem = Flappy1D(y0=y0, x0=x0, lower_edge_L=L, upper_edge_U=U,
                       y_min=y_min, y_max=y_max)
    h = clicks_heuristic_factory(L)

    if algo_name == "astar":
        res = astar_early(problem, h)
    elif algo_name == "idastar":
        res = idastar(problem, h)
    elif algo_name == "ilbfs":
        res = ilbfs(problem, h)
    else:
        raise ValueError("algo_name must be one of: astar | idastar | ilbfs")

    print(f"[{res.algorithm}] found={res.found} cost={res.cost:.0f} "
          f"expanded={res.nodes_expanded} generated={res.nodes_generated} "
          f"time={res.elapsed_sec:.6f}s")
    if res.found:
        print("Path (state, action):")
        for s, a in res.path:
            action_str = "CLICK" if a == 1 else "NO CLICK" if a == 0 else None
            print(f"  {s}  via {action_str}")
if __name__ == "__main__":
    for algo in ("astar", "idastar", "ilbfs"):
        run_once(algo)
