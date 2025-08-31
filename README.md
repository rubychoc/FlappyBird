# 🐥FlappyBird🐥 — Optimal Search in a 1D Flappy World

This repository explores classical search algorithms on a simplified, discrete **Flappy Bird** environment. The world is a vertical integer grid with a sequence of pipes. At each step the bird either **CLICKs** (up) or **NOCLICKs** (down). The goal is to pass all pipes with **minimum clicks** while avoiding collisions.

We implement and compare the following search algorithms, and study two ways to plan across multiple pipes:

- Algorithms: `A*-early`, `IDA*`, `UCS`, `RBFS`
- Planning strategies:
  - **Greedy** — solve each pipe independently and pass the terminal height to the next.
  - **Global coordination** — choose terminal heights jointly across pipes (via a small DP) so later pipes remain reachable without increasing total clicks.

---

## 🔗 Live Demo (Visualizer)

- Open locally: `flappy_1d_visualizer_standalone.html` (double-click)
- Hosted (if GitHub Pages is enabled): **https://rubychoc.github.io/FlappyBird/**

---

## 📁 Repository Structure

```text
FlappyBird/
├─ search_core.py                  # Core search engines: A*-early, IDA*, UCS, ILBFS (RBFS-style)
├─ flappy_1d.py                    # Environment: state, successors, heuristic (clicks_heuristic_factory)
├─ multi_pipe.py                   # Greedy vs global-coordination multi-pipe planner (+ metrics)
├─ run_multi.py                    # Batch runner: runs experiments and prints aggregated stats
├─ flappy_1d_visualizer_standalone.html  # Browser visualizer for single/multi-pipe courses
└─ README.md
