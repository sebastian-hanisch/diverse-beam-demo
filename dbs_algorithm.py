"""Suchkerne - `_search`, `greedy_best_first`, `uniform_cost_search`, `a_star`, `beam_search` (wortgleich aus monobeam-demo/beam-search-demo, dort
korrektheitsgeprüft; `beam_search` mit optionalem expliziten Heuristik-Vektor `h`) und NEU `diverse_beam_search`.

Diverse Beam Search (Vijayakumar et al., arXiv 1610.02424 / AAAI 2018, Algorithmus 1): das Strahlbudget B wird in `groups` Gruppen zu je
`group_width` Plätzen geteilt (B = groups * group_width). Schicht für Schicht wird jede Gruppe der Reihe nach mit einem normalen Beam-Schritt
fortgesetzt - aber die Bewertung eines Kandidaten wird um einen Diversitätsterm verschlechtert (Hamming): `penalty` * Kantenlänge * Zahl der
früheren Gruppen, die denselben Zustand in DIESER Schicht schon gewählt haben. Gruppe 1 hat nie eine Strafe (ist also exakt ein Beam Search der
Breite `group_width`; das Paper folgert: nie schlechter als Beam Search der Breite B/G). Jede Gruppe liefert ihre EIGENE Lösung (Pfad) - hier
`groups` alternative Routen. Die Gruppen laufen schichtweise im Gleichschritt; eine Gruppe endet, wenn sie das Ziel erzeugt (kleinstes g der
Schicht) oder keine Kandidaten mehr hat, und übt danach keine Strafe mehr auf tiefere Schichten aus. Die Längeneinheit `Kantenlänge` ist das
mittlere Kantengewicht des Graphen. Deterministisch: Tie-Break nach Zustandsindex."""

import heapq
from dataclasses import dataclass, field

import numpy as np


def heuristic(xy, goal):
    """Euklidischer Abstand jedes Knotens zum Ziel - vektorisiert. Bei echten Kantengewichten (siehe
    `beam_scenario.py`) automatisch zulässig (Dreiecksungleichung)."""
    return np.hypot(*(xy - xy[goal]).T)


@dataclass
class SearchResult:
    path: list                  # Knotenfolge Start..Ziel, oder [] falls kein Pfad existiert
    cost: float                 # Summe der Kantengewichte entlang des Pfades
    expansions: int             # Zahl der expandierten Knoten (Effizienz-Kennzahl dieses Stücks)
    order: list = field(default_factory=list)     # Reihenfolge der expandierten Knoten (für die Schritt-Visualisierung)
    stored: int = 0             # Speicher-Kennzahl: gespeicherte Knoten am Ende (A*/UCS/GBFS: entdeckte Knoten; IDA*: max. Pfadtiefe)


def _search(graph, start, goal, priority_fn, relax=True):
    """`priority_fn(node, g_cost) -> float` bestimmt die Warteschlangen-Priorität. `g_cost` ist der bislang
    aufgelaufene Pfadwert zu `node` (für Uniform-Cost-Search gebraucht, von Greedy Best-First ignoriert).

    `relax`: ob ein noch nicht expandierter, aber schon entdeckter Knoten einen GÜNSTIGEREN Elternknoten
    bekommt, sobald ein billigerer Weg zu ihm gefunden wird (klassische Dijkstra-Relaxation - für
    Uniform-Cost-Search nötig, damit es tatsächlich optimal bleibt). Bei `relax=False` behält ein Knoten für
    immer den ERSTEN gefundenen Elternknoten (echtes "kein Backtracking" - der Kern der GBFS-Schwäche: eine
    Relaxation hier würde den gemessenen Qualitätsverlust künstlich kleinrechnen, da GBFS dann doch beiläufig
    von g(n) profitieren würde, obwohl es g(n) laut Definition komplett ignoriert)."""
    counter = 0
    frontier = [(priority_fn(start, 0.0), counter, start, 0.0)]
    came_from = {start: None}
    g_cost = {start: 0.0}
    visited = set()
    order = []

    while frontier:
        _priority, _c, node, g = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        if node == goal:
            path = []
            cur = node
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return SearchResult(path, g, len(order), order, len(g_cost))
        for v, w in zip(graph.neighbors[node], graph.weights[node]):
            if v in visited:
                continue
            g_v = g + w
            is_new = v not in g_cost
            if is_new or (relax and g_v < g_cost[v]):
                g_cost[v] = g_v
                came_from[v] = node
                counter += 1
                heapq.heappush(frontier, (priority_fn(v, g_v), counter, v, g_v))

    return SearchResult([], float("inf"), len(order), order, len(g_cost))


def greedy_best_first(graph, start, goal):
    h = heuristic(graph.xy, goal)
    return _search(graph, start, goal, lambda node, g: h[node], relax=False)


def uniform_cost_search(graph, start, goal):
    return _search(graph, start, goal, lambda node, g: g, relax=True)


def a_star(graph, start, goal):
    h = heuristic(graph.xy, goal)
    return _search(graph, start, goal, lambda node, g: g + h[node], relax=True)


@dataclass
class BeamResult(SearchResult):
    failed: bool = False
    layers: int = 0
    per_layer: list = field(default_factory=list)    # [(Strahl, verworfene Kandidaten), ...] je Schicht
    peak_width: int = 0                              # größte Kandidatenzahl einer Schicht vor dem Beschneiden


def beam_search(graph, start, goal, width, rank="h", h=None):
    if width < 1:
        raise ValueError("width muss mindestens 1 sein")
    if rank not in ("h", "f"):
        raise ValueError("rank muss 'h' oder 'f' sein")
    if h is None:
        h = heuristic(graph.xy, goal)
    neighbors, weights = graph.neighbors, graph.weights
    beam = [start]
    g = {start: 0.0}
    parent = {start: None}
    expanded = set()
    discovered = {start}
    order = []
    per_layer = []
    peak = 1

    def build_path(node):
        path = []
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    if start == goal:
        return BeamResult([start], 0.0, 0, [], 1, False, 0, [], 1)

    while True:
        cand_g, cand_parent = {}, {}
        current = list(beam)
        for node in beam:
            expanded.add(node)
            order.append(node)
        for node in beam:
            for v, w in zip(neighbors[node], weights[node]):
                if v in expanded:
                    continue
                g_v = g[node] + w
                if v not in cand_g or g_v < cand_g[v]:
                    cand_g[v] = g_v
                    cand_parent[v] = node
        discovered.update(cand_g)
        peak = max(peak, len(cand_g))
        if goal in cand_g:
            parent[goal] = cand_parent[goal]
            per_layer.append((current, []))
            return BeamResult(build_path(goal), cand_g[goal], len(order), order, len(discovered), False, len(per_layer), per_layer, peak)
        if not cand_g:
            per_layer.append((current, []))
            return BeamResult([], float("inf"), len(order), order, len(discovered), True, len(per_layer), per_layer, peak)
        if rank == "h":
            ranked = sorted(cand_g, key=lambda v: (h[v], v))
        else:
            ranked = sorted(cand_g, key=lambda v: (cand_g[v] + h[v], v))
        beam, dropped = ranked[:width], ranked[width:]
        per_layer.append((current, dropped))
        for v in beam:
            g[v] = cand_g[v]
            parent[v] = cand_parent[v]


@dataclass
class GroupResult:
    path: list
    cost: float
    failed: bool
    layer: int              # Schicht, in der die Gruppe das Ziel erzeugt hat bzw. gescheitert ist
    expansions: int
    frontier: list = field(default_factory=list)    # Pfade (Start bis Knoten) aller Strahlknoten der Schicht VOR dem Zielfund - die "Strahlliste" der Gruppe


@dataclass
class DiverseResult:
    groups: list                                    # [GroupResult] in Gruppenreihenfolge
    per_layer: list = field(default_factory=list)   # je Schicht: [{"beam": [Zustände], "n_candidates": int, "n_fresh": int, "active": bool} je Gruppe]
    expansions: int = 0
    unit: float = 1.0                               # Kantenlänge (mittleres Kantengewicht), Einheit der Strafe

    @property
    def solved(self):
        return [g for g in self.groups if not g.failed]

    @property
    def failed(self):
        return not self.solved

    @property
    def best(self):
        """Beste (billigste) Gruppenlösung oder None."""
        return min(self.solved, key=lambda g: g.cost) if self.solved else None


def mean_edge_length(graph):
    ws = [w for row in graph.weights for w in row]
    return float(sum(ws) / len(ws)) if ws else 1.0


def diverse_beam_search(graph, start, goal, groups, group_width, penalty=0.0, h=None):
    if groups < 1 or group_width < 1:
        raise ValueError("groups und group_width müssen mindestens 1 sein")
    if penalty < 0:
        raise ValueError("penalty muss >= 0 sein")
    if h is None:
        h = heuristic(graph.xy, goal)
    neighbors, weights = graph.neighbors, graph.weights
    unit = mean_edge_length(graph)
    inf = float("inf")
    if start == goal:
        one = [{"beam": [start], "n_candidates": 0, "n_fresh": 0, "active": True} for _ in range(groups)]
        return DiverseResult([GroupResult([start], 0.0, False, 0, 0) for _ in range(groups)], [one], 0, unit)

    beams = [[start] for _ in range(groups)]
    gcost = [{start: 0.0} for _ in range(groups)]
    parent = [{start: None} for _ in range(groups)]
    expanded = [set() for _ in range(groups)]
    active = [True] * groups
    results = [None] * groups
    exp_g = [0] * groups
    per_layer = [[{"beam": [start], "n_candidates": 0, "n_fresh": 0, "active": True} for _ in range(groups)]]
    layer = 0

    def build_path(g, node):
        path = []
        while node is not None:
            path.append(node)
            node = parent[g][node]
        path.reverse()
        return path

    while any(active):
        layer += 1
        chosen = [set() for _ in range(groups)]
        record = []
        for g in range(groups):
            if not active[g]:
                record.append({"beam": [], "n_candidates": 0, "n_fresh": 0, "active": False})
                continue
            beam = beams[g]
            for node in beam:
                expanded[g].add(node)
            exp_g[g] += len(beam)
            cand_g, cand_parent = {}, {}
            for node in beam:
                for v, w in zip(neighbors[node], weights[node]):
                    if v in expanded[g]:
                        continue
                    g_v = gcost[g][node] + w
                    if v not in cand_g or g_v < cand_g[v]:
                        cand_g[v] = g_v
                        cand_parent[v] = node
            if not cand_g:
                results[g] = GroupResult([], inf, True, layer, exp_g[g])
                active[g] = False
                record.append({"beam": [], "n_candidates": 0, "n_fresh": 0, "active": True})
                continue
            if goal in cand_g:
                parent[g][goal] = cand_parent[goal]
                results[g] = GroupResult(build_path(g, goal), cand_g[goal], False, layer, exp_g[g], [build_path(g, n) for n in beam])
                active[g] = False
                record.append({"beam": [], "n_candidates": len(cand_g), "n_fresh": len(cand_g), "active": True})
                continue
            earlier = chosen[:g]
            counts = {v: sum(1 for c in earlier if v in c) for v in cand_g} if (penalty > 0 and g > 0) else {}

            def key(v):
                return (cand_g[v] + float(h[v]) + penalty * unit * counts.get(v, 0), v)

            kept = sorted(cand_g, key=key)[:group_width]
            n_fresh = sum(1 for v in cand_g if not any(v in c for c in earlier))
            for v in kept:
                gcost[g][v] = cand_g[v]
                parent[g][v] = cand_parent[v]
            chosen[g] = set(kept)
            beams[g] = kept
            record.append({"beam": list(kept), "n_candidates": len(cand_g), "n_fresh": n_fresh, "active": True})
        per_layer.append(record)
    return DiverseResult(results, per_layer, sum(exp_g), unit)
