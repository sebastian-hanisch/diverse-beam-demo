"""Unabhängiges Orakel für Diverse Beam Search: (1) eine pfadbasierte Neuimplementierung aus der Spezifikation (Gruppen nacheinander je
Schicht, Hamming-Strafe = λ × mittlere Kantenlänge × Zahl früherer Gruppen mit demselben Zustand in dieser Schicht) muss auf Zufallsgraphen
(reelle und ganzzahlige Gewichte, auch unzulässige Heuristiken) und Rastern für jede Gruppe Pfad, Kosten, Scheitern, Schicht, Expansionen,
Strahlliste sowie die Schicht-Aufzeichnungen (Strahl, Kandidaten, frische Kandidaten) identisch liefern; (2) bei unbegrenzter Breite liefert jede
Gruppe den billigsten Pfad unter allen Pfaden mit minimaler Kantenzahl (Orakel: networkx-BFS-Schichten + Programmierung über die Schichten);
(3) die Zwei-Korridore-Falle von Hand."""

import math
import random

import numpy as np
import pytest

import dbs_algorithm as A
import dbs_graph as Gr
import dbs_scenario as S

nx = pytest.importorskip("networkx")


def _adj(g):
    return {u: dict(zip(g.neighbors[u], g.weights[u])) for u in range(g.n)}


def _ref_dbs(adj, s, t, groups, width, lam, h):
    ws = [w for u in adj for w in adj[u].values()]
    unit = sum(ws) / len(ws)
    beams = [[(s, (s,))] for _ in range(groups)]
    closed = [set() for _ in range(groups)]
    live = [True] * groups
    res = [None] * groups
    nexp = [0] * groups
    layers = []
    depth = 0

    def cost_of(path):
        return sum(adj[a][b] for a, b in zip(path, path[1:]))

    while any(live):
        depth += 1
        chosen = [set() for _ in range(groups)]
        rec = []
        for gi in range(groups):
            if not live[gi]:
                rec.append((tuple(), 0, 0, False))
                continue
            b = beams[gi]
            closed[gi].update(node for node, _p in b)
            nexp[gi] += len(b)
            cand = {}
            for node, path in b:
                for v in adj[node]:
                    if v in closed[gi]:
                        continue
                    c = cost_of(path) + adj[node][v]
                    if v not in cand or c < cand[v][0]:
                        cand[v] = (c, path + (v,))
            if not cand:
                res[gi] = (True, [], math.inf, depth, nexp[gi], [])
                live[gi] = False
                rec.append((tuple(), 0, 0, True))
                continue
            if t in cand:
                res[gi] = (False, list(cand[t][1]), cand[t][0], depth, nexp[gi], [list(p) for _n, p in b])
                live[gi] = False
                rec.append((tuple(), len(cand), len(cand), True))
                continue
            prev = chosen[:gi]
            keep = sorted(cand, key=lambda v: (cand[v][0] + h[v] + lam * unit * sum(v in c for c in prev), v))[:width]
            fresh = sum(1 for v in cand if all(v not in c for c in prev))
            chosen[gi] = set(keep)
            beams[gi] = [(v, cand[v][1]) for v in keep]
            rec.append((tuple(keep), len(cand), fresh, True))
        layers.append(rec)
    return res, layers


def _check(g, s, t, groups, width, lam, h):
    got = A.diverse_beam_search(g, s, t, groups, width, lam, h=np.asarray(h, dtype=float))
    res, layers = _ref_dbs(_adj(g), s, t, groups, width, lam, list(h))
    for grp, (failed, path, cost, layer, exps, frontier) in zip(got.groups, res):
        assert grp.failed == failed and grp.path == path and grp.layer == layer and grp.expansions == exps
        assert [list(p) for p in grp.frontier] == frontier
        if not failed:
            assert grp.cost == pytest.approx(cost, abs=1e-9)
    mine = [[(tuple(r["beam"]), r["n_candidates"], r["n_fresh"], r["active"]) for r in layer] for layer in got.per_layer[1:]]
    assert mine == layers
    assert got.expansions == sum(grp.expansions for grp in got.groups)


def _random_graph(rng, n, mode):
    pts = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(n)]
    edges = [(rng.randrange(v), v) for v in range(1, n)]
    edges += [tuple(rng.sample(range(n), 2)) for _ in range(rng.randrange(0, 2 * n))]
    es = []
    for u, v in edges:
        if mode == "int":
            w = float(rng.randint(1, 5))
        elif mode == "admissible":
            w = math.dist(pts[u], pts[v]) * (1 + rng.random())
        else:
            w = rng.uniform(0.3, 6.0)
        es.append((u, v, w))
    g = Gr.from_edges(n, pts, es)
    if mode == "int":
        h = [float(rng.randint(0, 6)) for _ in range(n)]
    elif mode == "admissible":
        h = [math.dist(pts[v], pts[n - 1]) for v in range(n)]
    else:
        h = [rng.uniform(0, 6) for _ in range(n)]
    return g, h


def test_matches_path_based_reimplementation_on_random_graphs():
    rng = random.Random(606)
    for it in range(300):
        n = rng.randint(3, 12)
        g, h = _random_graph(rng, n, ("admissible", "free", "int")[it % 3])
        _check(g, 0, n - 1, rng.choice([1, 2, 3, 4, 5]), rng.choice([1, 1, 2, 3, 4]), rng.choice([0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 8.0, 1e6]), h)


def test_matches_path_based_reimplementation_on_grids():
    for seed in range(200000, 200060):
        inst = S.grid_instance(3 + seed % 8, (0, 10, 20, 30, 40)[seed % 5], seed)
        h = np.hypot(*(inst.graph.xy - inst.graph.xy[inst.goal]).T)
        for groups, width, lam in ((3, 2, 0.5), (2, 1, 1.0), (4, 1, 4.0), (1, 4, 2.0)):
            _check(inst.graph, inst.start, inst.goal, groups, width, lam, h)


def test_unlimited_width_gives_cheapest_path_among_fewest_edges_in_every_group():
    rng = random.Random(7)
    for it in range(150):
        n = rng.randint(3, 12)
        g, h = _random_graph(rng, n, ("int", "admissible")[it % 2])
        H = nx.Graph()
        for u in range(g.n):
            for v, w in zip(g.neighbors[u], g.weights[u]):
                H.add_edge(u, v, weight=w)
        dist = nx.single_source_shortest_path_length(H, 0)
        best = {0: 0.0}
        for level in range(1, dist[n - 1] + 1):
            for v in (x for x in H if dist.get(x) == level):
                best[v] = min(best[u] + H[u][v]["weight"] for u in H.neighbors(v) if dist.get(u) == level - 1)
        res = A.diverse_beam_search(g, 0, n - 1, 3, 100, rng.choice([0.0, 1.0, 5.0]), h=np.asarray(h, dtype=float))
        for grp in res.groups:
            assert not grp.failed and grp.layer == dist[n - 1] and grp.cost == pytest.approx(best[n - 1], abs=1e-9)


def test_two_corridor_trap_by_hand():
    inst = S.corridor_instance()
    # Optimum 8 über S-B-E-F-Z (Knoten 0-2-5-6-8: 5 + 1 + 1 + 1); Gruppe 1 folgt der Heuristik zu D und nimmt den Umweg über B (S-A-D-B-E-F-Z, Kosten 9).
    res = A.diverse_beam_search(inst.graph, 0, 8, 2, 1, 1.0, h=inst.h)
    assert [(g.path, g.cost) for g in res.groups] == [([0, 1, 4, 2, 5, 6, 8], 9.0), ([0, 2, 5, 6, 8], 8.0)]
    res0 = A.diverse_beam_search(inst.graph, 0, 8, 2, 1, 0.0, h=inst.h)
    assert [g.cost for g in res0.groups] == [9.0, 9.0]
