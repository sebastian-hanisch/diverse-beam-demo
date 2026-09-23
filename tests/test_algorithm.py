"""Die zentrale Korrektheits-Kette für Diverse Beam Search: gültige Pfade je Gruppe, Kosten nie unter dem Optimum (gegen Brute-Force),
G = 1 == Beam Search, Gruppe 1 == Beam Search der Breite b' für JEDES λ (Paper-Garantie), λ = 0 macht alle Gruppen identisch, beste
Gruppe nie schlechter als Gruppe 1, die Strafsemantik (n_fresh) direkt an den Schicht-Aufzeichnungen, Buchführung, die handgebaute
Zwei-Korridore-Falle und die Kopie der Beam-Search-Zahlen."""

import numpy as np
import pytest

import dbs_algorithm as A
import dbs_graph as G
import dbs_scenario as S

EPS = 1e-9
HUGE_PENALTY = 1e6
INF = float("inf")


def _brute_force_shortest_cost(graph, start, goal):
    best = None
    stack = [(start, [start], 0.0)]
    while stack:
        node, path, cost = stack.pop()
        if node == goal:
            if best is None or cost < best:
                best = cost
            continue
        for v, w in zip(graph.neighbors[node], graph.weights[node]):
            if v not in path:
                stack.append((v, path + [v], cost + w))
    return best


def _dbs(inst, groups, width, penalty):
    return A.diverse_beam_search(inst.graph, inst.start, inst.goal, groups, width, penalty, h=inst.h)


def _assert_valid(graph, start, goal, res):
    if res.failed:
        assert res.path == [] and res.cost == INF
        return
    assert res.path[0] == start and res.path[-1] == goal and len(set(res.path)) == len(res.path)
    for u, v in zip(res.path[:-1], res.path[1:]):
        assert v in graph.neighbors[u]
    assert G.path_cost(graph, res.path) == pytest.approx(res.cost, abs=1e-6)


# --- Gültigkeit und Untergrenze ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("penalty", [0.0, 0.5, 4.0])
@pytest.mark.parametrize("groups,width", [(1, 3), (2, 1), (3, 2), (4, 1), (2, 3)])
@pytest.mark.parametrize("seed", range(5))
def test_every_group_returns_a_valid_simple_path_with_recomputed_cost(seed, groups, width, penalty):
    inst = S.grid_instance(side=7, obstacle_pct=20, seed=seed)
    result = _dbs(inst, groups, width, penalty)
    assert len(result.groups) == groups
    for res in result.groups:
        _assert_valid(inst.graph, inst.start, inst.goal, res)


@pytest.mark.parametrize("groups,width,penalty", [(2, 1, 1.0), (3, 2, 0.5), (4, 1, 4.0)])
@pytest.mark.parametrize("seed", range(12))
def test_no_group_beats_the_brute_force_optimum(seed, groups, width, penalty):
    inst = S.grid_instance(side=5, obstacle_pct=15, seed=seed)
    optimum = _brute_force_shortest_cost(inst.graph, inst.start, inst.goal)
    for res in _dbs(inst, groups, width, penalty).groups:
        if not res.failed:
            assert res.cost >= optimum - 1e-6


def test_diverse_beam_search_is_deterministic():
    inst = S.grid_instance(side=8, obstacle_pct=20, seed=3)
    r1, r2 = _dbs(inst, 3, 2, 0.5), _dbs(inst, 3, 2, 0.5)
    assert [g.path for g in r1.groups] == [g.path for g in r2.groups] and r1.per_layer == r2.per_layer and r1.expansions == r2.expansions


def test_invalid_arguments_are_rejected():
    inst = S.grid_instance(side=5, obstacle_pct=0, seed=1)
    for args in ((0, 2, 0.5), (2, 0, 0.5), (2, 2, -0.1)):
        with pytest.raises(ValueError):
            _dbs(inst, *args)


def test_start_equals_goal_and_unreachable_goal():
    inst = S.grid_instance(side=5, obstacle_pct=0, seed=1)
    trivial = A.diverse_beam_search(inst.graph, inst.start, inst.start, 3, 2, 0.5)
    assert all(g.path == [inst.start] and g.cost == 0.0 and not g.failed for g in trivial.groups) and trivial.expansions == 0
    graph = G.from_edges(4, [(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 1, 1.0), (2, 3, 1.0)])
    result = A.diverse_beam_search(graph, 0, 3, 3, 2, 0.5)
    assert result.failed and result.best is None and all(g.failed and g.path == [] and g.cost == INF for g in result.groups)


def test_chain_graph_is_solved_by_every_group():
    n = 7
    graph = G.from_edges(n, [(i, 0) for i in range(n)], [(i, i + 1, 1.0) for i in range(n - 1)])
    result = A.diverse_beam_search(graph, 0, n - 1, 3, 1, 4.0)
    assert result.failed is False and all(g.path == list(range(n)) for g in result.groups)


# --- Paper-Garantien: G = 1, Gruppe 1, λ = 0, beste Gruppe -----------------------------------------------------------------------------------


@pytest.mark.parametrize("width", [1, 2, 4, 7])
@pytest.mark.parametrize("penalty", [0.0, 1.0, 8.0])
@pytest.mark.parametrize("seed", range(10))
def test_one_group_is_exactly_beam_search(seed, width, penalty):
    inst = S.grid_instance(side=9, obstacle_pct=20, seed=seed)
    one = _dbs(inst, 1, width, penalty).groups[0]
    beam = A.beam_search(inst.graph, inst.start, inst.goal, width, "f")
    assert (one.path, one.failed, one.expansions) == (beam.path, beam.failed, beam.expansions)
    assert one.cost == beam.cost


@pytest.mark.parametrize("penalty", [0.0, 0.25, 0.5, 1.0, 4.0, 8.0])
@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("seed", range(20))
def test_group_one_is_beam_search_of_the_group_width_for_every_penalty(seed, width, penalty):
    inst = S.grid_instance(side=10, obstacle_pct=15 + seed % 3 * 10, seed=seed)
    first = _dbs(inst, 3, width, penalty).groups[0]
    beam = A.beam_search(inst.graph, inst.start, inst.goal, width, "f")
    assert (first.path, first.failed, first.expansions) == (beam.path, beam.failed, beam.expansions)


@pytest.mark.parametrize("groups,width", [(2, 1), (3, 2), (4, 1)])
@pytest.mark.parametrize("seed", range(20))
def test_zero_penalty_makes_all_groups_identical_to_beam_search(seed, groups, width):
    inst = S.grid_instance(side=10, obstacle_pct=25, seed=seed)
    result = _dbs(inst, groups, width, 0.0)
    beam = A.beam_search(inst.graph, inst.start, inst.goal, width, "f")
    for res in result.groups:
        assert (res.path, res.failed, res.layer, res.expansions) == (beam.path, beam.failed, result.groups[0].layer, beam.expansions)


@pytest.mark.parametrize("penalty", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("groups,width", [(2, 1), (3, 2), (4, 1), (2, 3)])
@pytest.mark.parametrize("seed", range(25))
def test_the_best_group_is_never_worse_than_group_one(seed, groups, width, penalty):
    inst = S.grid_instance(side=10, obstacle_pct=15 + seed % 3 * 10, seed=seed)
    result = _dbs(inst, groups, width, penalty)
    first = result.groups[0]
    if not first.failed:
        assert result.best is not None and result.best.cost <= first.cost + EPS
    if result.best is not None:
        assert all(result.best.cost <= g.cost + EPS for g in result.solved)


# --- Strafsemantik direkt an den Schicht-Aufzeichnungen ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("groups,width", [(2, 1), (3, 2), (4, 2)])
@pytest.mark.parametrize("seed", range(15))
def test_a_huge_penalty_only_overlaps_when_there_are_too_few_fresh_candidates(seed, groups, width):
    inst = S.grid_instance(side=10, obstacle_pct=20, seed=seed)
    result = _dbs(inst, groups, width, HUGE_PENALTY)
    checked = 0
    for rec in result.per_layer[1:]:
        for g in range(1, groups):
            beam = rec[g]["beam"]
            if not beam:
                continue
            earlier = {s for h in range(g) for s in rec[h]["beam"]}
            forced = len(set(beam) & earlier)
            assert forced == len(beam) - min(width, rec[g]["n_fresh"]), (g, beam, rec[g])
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("seed", range(10))
def test_group_one_beams_are_identical_for_every_penalty_layer_by_layer(seed):
    inst = S.grid_instance(side=10, obstacle_pct=20, seed=seed)
    off, on = _dbs(inst, 3, 2, 0.0), _dbs(inst, 3, 2, 2.0)
    depth = off.groups[0].layer
    assert [r[0]["beam"] for r in off.per_layer[:depth + 1]] == [r[0]["beam"] for r in on.per_layer[:depth + 1]]
    assert off.groups[0].path == on.groups[0].path


# --- Buchführung -------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("penalty", [0.0, 0.5, 4.0])
@pytest.mark.parametrize("groups,width", [(2, 1), (3, 2), (4, 3)])
@pytest.mark.parametrize("seed", range(8))
def test_bookkeeping_beam_sizes_expansions_and_finished_groups(seed, groups, width, penalty):
    inst = S.grid_instance(side=9, obstacle_pct=25, seed=seed)
    result = _dbs(inst, groups, width, penalty)
    total = 0
    for g, res in enumerate(result.groups):
        sizes = [len(rec[g]["beam"]) for rec in result.per_layer]
        assert max(sizes) <= width
        expanded = sum(sizes[:res.layer])
        assert res.expansions == expanded
        total += expanded
        assert all(size == 0 for size in sizes[res.layer:])                 # fertige oder gescheiterte Gruppen stoppen
        if not res.failed:
            assert len(res.path) == res.layer + 1                            # Ziel wird in Schicht `layer` erzeugt
            assert len(res.frontier) == sizes[res.layer - 1]
            for p in res.frontier:
                assert p[0] == inst.start and len(p) == res.layer and p[-1] in result.per_layer[res.layer - 1][g]["beam"]
    assert result.expansions == total


# --- handgebaute Zwei-Korridore-Falle ---------------------------------------------------------------------------------------------------------


def test_corridor_trap_beam_search_loses_the_better_corridor_and_diverse_beam_search_finds_it():
    inst = S.HAND_BUILT["corridor"]()
    ucs = A.uniform_cost_search(inst.graph, inst.start, inst.goal)
    beams = [A.beam_search(inst.graph, inst.start, inst.goal, w, "f", h=inst.h) for w in (1, 2, 3)]
    assert (ucs.cost, [b.cost for b in beams]) == (8.0, [9.0, 11.0, 8.0])
    result = _dbs(inst, 2, 1, 1.0)
    assert [g.cost for g in result.groups] == [9.0, 8.0] and result.best.cost == 8.0 and result.best.cost < beams[1].cost
    assert result.groups[0].path == beams[0].path and result.best.path == ucs.path


@pytest.mark.parametrize("penalty", [0.0, 0.25])
def test_corridor_trap_needs_enough_diversity(penalty):
    inst = S.HAND_BUILT["corridor"]()
    assert [g.cost for g in _dbs(inst, 2, 1, penalty).groups] == [9.0, 9.0]


def test_corridor_second_group_is_pushed_off_the_first_groups_state_in_layer_one():
    inst = S.HAND_BUILT["corridor"]()
    layer1 = _dbs(inst, 2, 1, 1.0).per_layer[1]
    assert layer1[0]["beam"] == [1] and layer1[1]["beam"] == [2] and layer1[1]["n_fresh"] == 1


# --- Kopie treu und geerbte Eigenschaften -----------------------------------------------------------------------------------------------------


def test_beam_search_copy_reproduces_the_beam_search_demo_numbers():
    inst = S.grid_instance(side=12, obstacle_pct=15, seed=35)
    f3 = A.beam_search(inst.graph, inst.start, inst.goal, 3, "f")
    h8 = A.beam_search(inst.graph, inst.start, inst.goal, 8, "h")
    assert f3.cost == pytest.approx(175.90, abs=0.01) and f3.expansions == 63
    assert h8.cost == pytest.approx(175.90, abs=0.01) and h8.expansions == 118


def test_inherited_heuristic_is_admissible_and_a_star_is_optimal():
    for seed in range(10):
        inst = S.grid_instance(side=7, obstacle_pct=20, seed=seed)
        h = A.heuristic(inst.graph.xy, inst.goal)
        ucs = A.uniform_cost_search(inst.graph, inst.start, inst.goal)
        for node in range(inst.graph.n):
            assert h[node] <= A.uniform_cost_search(inst.graph, node, inst.goal).cost + EPS
        assert A.a_star(inst.graph, inst.start, inst.goal).cost == pytest.approx(ucs.cost, abs=EPS)


def test_mean_edge_length_is_the_mean_of_all_directed_weights():
    inst = S.HAND_BUILT["corridor"]()
    assert A.mean_edge_length(inst.graph) == pytest.approx(np.mean([w for _u, _v, w in S.CORRIDOR_EDGES]), abs=EPS)
    assert _dbs(inst, 2, 1, 1.0).unit == pytest.approx(A.mean_edge_length(inst.graph), abs=EPS)
