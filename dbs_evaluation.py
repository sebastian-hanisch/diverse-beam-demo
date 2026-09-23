"""Auswertung: was bringt die Aufteilung des Strahlbudgets in Gruppen mit Diversitätsstrafe? Vergleich von Diverse Beam Search (G Gruppen zu je b'
Plätzen, Gesamtbreite B = G * b') mit Beam Search gleicher Gesamtbreite B (eine Lösung), Beam Search der Gruppenbreite b' (= Gruppe 1, die nie
eine Strafe bekommt), A* (Expansions-Referenz) und Uniform-Cost (Optimum) auf demselben Graphen. Alle Kennzahlen über die 50 festen Instanzen
(Seeds 200000-200049; billig, robuster für Anteile als 5 Instanzen), Median mit 10./90. Perzentil.

- **Lücke der besten Gruppenlösung** = 100 * (Kosten - Optimum) / Optimum (nur gelöste Läufe), daneben Scheiter-Quote (DBS: ALLE Gruppen gescheitert)
  und der Anteil gescheiterter Gruppen; **besser/gleich/schlechter** gegen Beam Search Breite B bei gleicher Gesamtbreite.
- **Streuung** = mittlere paarweise Entfernung aller Strahlknoten einer Schicht (in Kantenlängen), gemittelt über die Schichten (Knoten in mehreren
  Gruppen zählen mehrfach, Entfernung 0).
- **Überlappung** = mittlere paarweise Jaccard-Ähnlichkeit der inneren Knotenmengen (ohne Start/Ziel) der gelösten Gruppenlösungen (1 = identisch,
  0 = disjunkt), **verschiedene Lösungen** = Zahl verschiedener Pfade unter den gelösten Gruppen, **brauchbare Alternativen** = verschiedene Pfade unter den gelösten Gruppen mit
  Lücke <= 10 %.
- **Strahlliste-Redundanz** = Überlappung der Pfade aller Strahlknoten der Schicht VOR dem Zielfund (die "Liste fast identischer Sequenzen" des
  Papers), gepoolt über alle Gruppen; Vergleich mit einem einzelnen Beam Search der Breite B (eine Gruppe, B Plätze).
- **Breiten-Alternativen** (Basis ohne Strafe): die Lösungen von Beam Search mit den Breiten b', 2b', ..., G*b' - wie verschieden sind sie?
- **Aufwand** = Expansionen DBS / Expansionen Beam Search Breite B."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations

import numpy as np

import dbs_algorithm as A
import dbs_constants as C
import dbs_scenario as S

INF = float("inf")
USABLE_GAP = 10.0


@dataclass(frozen=True)
class Settings:
    network: str = "grid"
    side: int = C.DEFAULT_SIDE
    obstacle_pct: int = C.DEFAULT_OBSTACLE
    seed: int = C.DEFAULT_SEED
    groups: int = C.DEFAULT_GROUPS
    group_width: int = C.DEFAULT_GROUP_WIDTH
    penalty: float = C.DEFAULT_PENALTY

    @property
    def total_width(self):
        return self.groups * self.group_width


@lru_cache(maxsize=512)
def _grid(side, obstacle_pct, seed):
    return S.grid_instance(side, obstacle_pct, seed)


def instance_of(settings):
    if settings.network == "grid":
        return _grid(settings.side, settings.obstacle_pct, settings.seed)
    return S.HAND_BUILT[settings.network]()


# --- Diversitätsmaße (einzeln testbar) -------------------------------------------------------------------------------------------------------


def jaccard(path_a, path_b, exclude=()):
    """Jaccard-Ähnlichkeit der Knotenmengen zweier Pfade (ohne die Knoten in `exclude`, z. B. Start und Ziel)."""
    a, b = set(path_a) - set(exclude), set(path_b) - set(exclude)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean_overlap(paths, exclude=()):
    """Mittlere paarweise Überlappung; NaN bei weniger als zwei Pfaden."""
    pairs = list(combinations(paths, 2))
    return float(np.mean([jaccard(a, b, exclude) for a, b in pairs])) if pairs else float("nan")


def layer_spread(xy, states, unit):
    """Mittlere paarweise Entfernung (in Kantenlängen) der Strahlknoten einer Schicht; NaN bei weniger als zwei Knoten."""
    if len(states) < 2:
        return float("nan")
    p = xy[list(states)]
    d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    n = len(states)
    return float(d.sum() / (n * (n - 1)) / unit)


def mean_spread(xy, layers, unit):
    """Über die Schichten gemittelte Streuung; `layers` = Liste von Zustandslisten je Schicht."""
    vals = [layer_spread(xy, states, unit) for states in layers]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


# --- Analyse -----------------------------------------------------------------------------------------------------------------------------------


@dataclass
class Analysis:
    settings: Settings
    inst: object
    dbs: A.DiverseResult
    beam_b: A.BeamResult            # Breite B = groups * group_width
    beam_g: A.BeamResult            # Breite group_width (= Gruppe 1)
    astar: object
    ucs: A.SearchResult

    def _gap(self, cost):
        return 100.0 * (cost - self.ucs.cost) / self.ucs.cost

    @property
    def best_gap(self):
        b = self.dbs.best
        return float("nan") if b is None else self._gap(b.cost)

    @property
    def beam_b_gap(self):
        return float("nan") if self.beam_b.failed else self._gap(self.beam_b.cost)

    @property
    def beam_g_gap(self):
        return float("nan") if self.beam_g.failed else self._gap(self.beam_g.cost)

    @property
    def group_gaps(self):
        return [None if g.failed else self._gap(g.cost) for g in self.dbs.groups]

    @property
    def verdict(self):
        """Beste Gruppenlösung gegen Beam Search Breite B: besser / gleich / schlechter; None, wenn einer ohne Lösung."""
        b = self.dbs.best
        if b is None or self.beam_b.failed:
            return None
        if b.cost < self.beam_b.cost - 1e-9:
            return "besser"
        if b.cost > self.beam_b.cost + 1e-9:
            return "schlechter"
        return "gleich"

    @property
    def groups_failed_share(self):
        return 100.0 * sum(g.failed for g in self.dbs.groups) / len(self.dbs.groups)

    @property
    def solved_paths(self):
        return [g.path for g in self.dbs.groups if not g.failed]

    @property
    def overlap(self):
        return mean_overlap(self.solved_paths, exclude=(self.inst.start, self.inst.goal))

    @property
    def distinct(self):
        return len({tuple(p) for p in self.solved_paths})

    @property
    def usable(self):
        paths = {tuple(g.path) for g, gap in zip(self.dbs.groups, self.group_gaps) if gap is not None and gap <= USABLE_GAP}
        return len(paths)

    @property
    def spread_dbs(self):
        layers = [[s for grp in rec for s in grp["beam"]] for rec in self.dbs.per_layer]
        return mean_spread(self.inst.graph.xy, layers, self.dbs.unit)

    @property
    def spread_beam(self):
        layers = [list(beam) for beam, _dropped in self.beam_b.per_layer]
        return mean_spread(self.inst.graph.xy, layers, self.dbs.unit)

    @property
    def frontier_overlap(self):
        """Überlappung der Strahlpfade (Strahlknoten der Schicht vor dem Zielfund), gepoolt über alle Gruppen."""
        paths = [p for g in self.dbs.groups for p in g.frontier]
        return mean_overlap(paths, exclude=(self.inst.start, self.inst.goal))

    @property
    def beam_frontier_overlap(self):
        """Dasselbe für einen einzelnen Beam Search der Breite B (eine Gruppe mit B Plätzen)."""
        one = A.diverse_beam_search(self.inst.graph, self.inst.start, self.inst.goal, 1, self.settings.total_width, 0.0, h=self.inst.h)
        return mean_overlap(one.groups[0].frontier, exclude=(self.inst.start, self.inst.goal))

    @property
    def width_alt_overlap(self):
        """Überlappung der Lösungen von Beam Search mit den Breiten b', 2b', ..., G*b' (Alternativen OHNE Diversitätsstrafe)."""
        st = self.settings
        paths = []
        for j in range(1, st.groups + 1):
            r = A.beam_search(self.inst.graph, self.inst.start, self.inst.goal, j * st.group_width, "f", h=self.inst.h)
            if not r.failed:
                paths.append(r.path)
        return mean_overlap(paths, exclude=(self.inst.start, self.inst.goal))

    @property
    def width_alt_distinct(self):
        st = self.settings
        paths = set()
        for j in range(1, st.groups + 1):
            r = A.beam_search(self.inst.graph, self.inst.start, self.inst.goal, j * st.group_width, "f", h=self.inst.h)
            if not r.failed:
                paths.add(tuple(r.path))
        return len(paths)

    @property
    def exp_ratio(self):
        return self.dbs.expansions / self.beam_b.expansions


@lru_cache(maxsize=512)
def _references(network, side, obstacle_pct, seed):
    inst = instance_of(Settings(network=network, side=side, obstacle_pct=obstacle_pct, seed=seed))
    g, s, t, h = inst.graph, inst.start, inst.goal, inst.h
    return A.a_star(g, s, t) if h is None else None, A.uniform_cost_search(g, s, t)


def analyse(settings):
    inst = instance_of(settings)
    g, s, t, h = inst.graph, inst.start, inst.goal, inst.h
    dbs = A.diverse_beam_search(g, s, t, settings.groups, settings.group_width, settings.penalty, h=h)
    beam_b = A.beam_search(g, s, t, settings.total_width, "f", h=h)
    beam_g = A.beam_search(g, s, t, settings.group_width, "f", h=h)
    astar, ucs = _references(settings.network, settings.side, settings.obstacle_pct, settings.seed)
    return Analysis(settings, inst, dbs, beam_b, beam_g, astar, ucs)


# --- Sweeps ------------------------------------------------------------------------------------------------------------------------------------


def _stats(values):
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.percentile(values, 10)), float(np.percentile(values, 90))


def run_config(base, seeds=C.POP_SEEDS, **changes):
    s0 = replace(base, **changes)
    rows = [analyse(replace(s0, seed=seed)) for seed in seeds]
    n = len(rows)
    out = {
        "n_runs": n,
        "dbs_failed_share": 100.0 * sum(r.dbs.failed for r in rows) / n,
        "beam_b_failed_share": 100.0 * sum(r.beam_b.failed for r in rows) / n,
        "beam_g_failed_share": 100.0 * sum(r.beam_g.failed for r in rows) / n,
        "groups_failed_share": float(np.mean([r.groups_failed_share for r in rows])),
        "only_dbs_share": 100.0 * sum((not r.dbs.failed) and r.beam_b.failed for r in rows) / n,
        "only_beam_share": 100.0 * sum(r.dbs.failed and not r.beam_b.failed for r in rows) / n,
        "dbs_optimal_share": 100.0 * sum((not r.dbs.failed) and r.dbs.best.cost <= r.ucs.cost + 1e-9 for r in rows) / n,
        "beam_b_optimal_share": 100.0 * sum((not r.beam_b.failed) and r.beam_b.cost <= r.ucs.cost + 1e-9 for r in rows) / n,
    }
    verdicts = [r.verdict for r in rows if r.verdict is not None]
    out["n_compared"] = len(verdicts)
    for v in ("besser", "gleich", "schlechter"):
        out[f"verdict_{v}"] = sum(x == v for x in verdicts)
    for key, values in (
        ("best_gap", [r.best_gap for r in rows]),
        ("beam_b_gap", [r.beam_b_gap for r in rows]),
        ("beam_g_gap", [r.beam_g_gap for r in rows]),
        ("spread_dbs", [r.spread_dbs for r in rows]),
        ("spread_beam", [r.spread_beam for r in rows]),
        ("overlap", [r.overlap for r in rows]),
        ("distinct", [float(r.distinct) for r in rows]),
        ("usable", [float(r.usable) for r in rows]),
        ("frontier_overlap", [r.frontier_overlap for r in rows]),
        ("beam_frontier_overlap", [r.beam_frontier_overlap for r in rows]),
        ("width_alt_overlap", [r.width_alt_overlap for r in rows]),
        ("width_alt_distinct", [float(r.width_alt_distinct) for r in rows]),
        ("exp_ratio", [r.exp_ratio for r in rows]),
    ):
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = _stats(values)
    return out


def penalty_profile(settings):
    """Für EINE Instanz: die Lücken der Gruppen, Überlappung und Zahl verschiedener Routen über alle Stufen von λ."""
    rows = []
    for p in C.PENALTIES:
        a = analyse(replace(settings, penalty=p))
        rows.append({"penalty": p, "gaps": a.group_gaps, "overlap": a.overlap, "distinct": a.distinct})
    return rows


SWEEP_VALUES = {"penalty": C.PENALTIES, "groups": C.GROUPS_OPTIONS, "group_width": C.GROUP_WIDTHS, "obstacle_pct": C.OBSTACLE_SWEEP, "side": C.SCALING_SIDES}
SWEEP_LABELS = {"penalty": "Diversitätsstärke λ (Kantenlängen)", "groups": "Gruppen G", "group_width": "Breite je Gruppe b'", "obstacle_pct": "Hindernisdichte (%)", "side": "Rastergröße (Seitenlänge)"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]
