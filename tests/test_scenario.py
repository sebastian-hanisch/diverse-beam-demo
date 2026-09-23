import numpy as np
import pytest

import dbs_algorithm as A
import dbs_graph as G
import dbs_scenario as S


def _bfs_connected(graph, start, goal):
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        if u == goal:
            return True
        for v in graph.neighbors[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return goal in seen


@pytest.mark.parametrize("seed", range(30))
def test_start_and_goal_are_always_connected(seed):
    inst = S.grid_instance(side=10, obstacle_pct=35, seed=seed)
    assert _bfs_connected(inst.graph, inst.start, inst.goal)


@pytest.mark.parametrize("seed", range(10))
def test_edge_weights_match_euclidean_distance_between_endpoints(seed):
    inst = S.grid_instance(side=10, obstacle_pct=20, seed=seed)
    xy = inst.graph.xy
    for u in range(inst.graph.n):
        for v, w in zip(inst.graph.neighbors[u], inst.graph.weights[u]):
            assert w == pytest.approx(float(np.hypot(*(xy[u] - xy[v]))), abs=1e-9)


def test_higher_obstacle_percent_blocks_more_cells_on_average():
    low = [len(S.grid_instance(side=14, obstacle_pct=5, seed=s).blocked_xy) for s in range(10)]
    high = [len(S.grid_instance(side=14, obstacle_pct=35, seed=s).blocked_xy) for s in range(10)]
    assert np.mean(high) > np.mean(low)


def test_zero_obstacle_percent_leaves_every_cell_open():
    inst = S.grid_instance(side=10, obstacle_pct=0, seed=1)
    assert len(inst.blocked_xy) == 0 and inst.graph.n == 100


def test_all_edge_weights_are_positive():
    inst = S.grid_instance(side=12, obstacle_pct=30, seed=7)
    assert all(w > 0 for u in range(inst.graph.n) for w in inst.graph.weights[u])


def test_the_grid_instance_has_no_explicit_heuristic_or_labels():
    inst = S.grid_instance(side=6, obstacle_pct=10, seed=1)
    assert inst.h is None and inst.labels is None


def test_only_the_corridor_trap_is_hand_built():
    assert set(S.HAND_BUILT) == {"corridor"}


def test_corridor_instance_has_the_expected_shape_and_a_connected_start_goal():
    inst = S.HAND_BUILT["corridor"]()
    assert inst.graph.n == 9 and len(inst.labels) == 9 and inst.labels[0] == "S" and inst.labels[-1] == "Z"
    assert inst.start == 0 and inst.goal == 8 and _bfs_connected(inst.graph, inst.start, inst.goal)
    assert len(inst.blocked_xy) == 0 and len(inst.h) == 9 and inst.h[inst.goal] == 0
    assert sum(len(n) for n in inst.graph.neighbors) == 2 * len(S.CORRIDOR_EDGES)


def test_corridor_heuristic_is_admissible_against_uniform_cost_distances():
    inst = S.HAND_BUILT["corridor"]()
    for node in range(inst.graph.n):
        assert inst.h[node] <= A.uniform_cost_search(inst.graph, node, inst.goal).cost + 1e-9
    assert A.uniform_cost_search(inst.graph, inst.start, inst.goal).cost == 8.0


def test_path_cost_matches_a_hand_walked_corridor_path():
    inst = S.HAND_BUILT["corridor"]()
    assert G.path_cost(inst.graph, [0, 2, 5, 6, 8]) == pytest.approx(8.0)
    assert G.path_cost(inst.graph, [0, 1, 3, 7, 8]) == pytest.approx(12.0)
