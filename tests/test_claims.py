"""Jede Zahl aus README, App-Texten und dbs_constants.py, nachgerechnet über die echten Auswertungsfunktionen (ev.sweep/ev.run_config) auf den 50
festen Instanzen (Seeds 200000-200049), Rastergröße 12, Hindernisdichte 15 %, G = 3 Gruppen zu b' = 2 Plätzen, λ = 0.5, sofern nicht anders
angegeben. Toleranzen sind bewusst großzügig gegenüber Rundung (die Instanzen sind deterministisch), aber enger als jede Aussage."""

from functools import lru_cache

import pytest

import dbs_constants as C
import dbs_evaluation as ev

BASE = ev.Settings()


@lru_cache(maxsize=None)
def _rows(param):
    return ev.sweep(param, BASE)


def _col(param, key):
    return [r[key] for r in _rows(param)]


def _at(param, value):
    return next(r for r in _rows(param) if r["value"] == value)


def _close(values, expected, abs_tol):
    assert len(values) == len(expected)
    for v, e in zip(values, expected):
        assert v == pytest.approx(e, abs=abs_tol), (values, expected)


DEFAULT = lambda: _at("penalty", 0.5)


# --- Frage 1: Top-1 gegen Beam Search gleicher Gesamtbreite -----------------------------------------------------------------------------------


def test_default_top_one_is_never_better_and_never_optimal_more_often_than_beam_of_the_total_width():
    r = DEFAULT()
    assert (r["verdict_besser"], r["verdict_gleich"], r["verdict_schlechter"], r["n_compared"]) == (0, 27, 23, 50)
    assert r["dbs_optimal_share"] == 54.0 and r["beam_b_optimal_share"] == 100.0 and r["best_gap"] == 0.0 and r["beam_b_gap"] == 0.0


def test_better_never_occurs_in_any_of_the_28_sweep_settings():
    rows = [r for param in ev.SWEEP_VALUES for r in _rows(param)]
    assert len(rows) == 28 and all(r["verdict_besser"] == 0 for r in rows)


def test_beam_of_the_group_width_versus_dbs_best_of_three():
    r = DEFAULT()
    assert r["beam_g_gap"] == pytest.approx(0.27, abs=0.01) and r["beam_g_failed_share"] == 2.0 and r["best_gap"] == 0.0 and r["dbs_failed_share"] == 0.0


# --- Frage 2: Vielfalt und brauchbare Alternativen (λ) --------------------------------------------------------------------------------------


def test_penalty_sweep_overlap_distinct_and_usable_routes():
    _close(_col("penalty", "overlap"), [1.00, 0.60, 0.37, 0.25, 0.24, 0.25, 0.26], 0.01)
    assert _col("penalty", "distinct") == [1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    assert _col("penalty", "usable") == [1.0, 2.0, 3.0, 2.0, 1.0, 1.0, 1.0]
    assert _col("penalty", "value") == [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def test_penalty_sweep_overlap_is_flat_from_lambda_one_on():
    flat = _col("penalty", "overlap")[3:]
    assert max(flat) - min(flat) < 0.03


def test_penalty_sweep_failed_group_share_grows_with_lambda():
    _close(_col("penalty", "groups_failed_share"), [2.0, 5.33, 6.67, 7.33, 9.33, 12.67, 14.0], 0.05)
    assert _col("penalty", "groups_failed_share") == sorted(_col("penalty", "groups_failed_share"))


def test_penalty_sweep_verdict_counts():
    assert [r["verdict_schlechter"] for r in _rows("penalty")] == [29, 19, 23, 28, 28, 28, 28]
    assert [r["n_compared"] for r in _rows("penalty")] == [49, 49, 50, 50, 50, 50, 50]


# --- Frage 3: Klumpen des Strahls -------------------------------------------------------------------------------------------------------------


def test_frontier_redundancy_beam_versus_dbs_and_spatial_spread():
    _close(_col("penalty", "frontier_overlap"), [0.90, 0.59, 0.45, 0.36, 0.35, 0.35, 0.37], 0.01)
    assert all(v == pytest.approx(0.53, abs=0.01) for v in _col("penalty", "beam_frontier_overlap"))
    _close(_col("penalty", "spread_dbs"), [0.86, 1.29, 1.68, 2.04, 2.26, 2.28, 2.32], 0.02)
    assert all(v == pytest.approx(3.27, abs=0.02) for v in _col("penalty", "spread_beam")) and max(_col("penalty", "spread_dbs")) < 3.27


def test_alternatives_without_a_penalty_are_near_duplicates():
    r = DEFAULT()
    assert r["width_alt_overlap"] == pytest.approx(0.91, abs=0.01) and r["width_alt_distinct"] == 2.0 and r["distinct"] == 3.0


# --- Frage 4: Aufwand -------------------------------------------------------------------------------------------------------------------------


def test_expansion_ratio_by_number_of_groups():
    _close(_col("groups", "exp_ratio"), [1.00, 1.12, 1.26, 1.47, 2.12, 2.85], 0.02)
    assert _col("penalty", "exp_ratio")[2] == pytest.approx(1.26, abs=0.01)


def test_routes_and_overlap_by_number_of_groups():
    assert _col("groups", "distinct") == [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    assert _col("groups", "usable") == [1.0, 2.0, 3.0, 3.0, 4.0, 5.0]
    _close(_col("groups", "overlap")[1:], [0.56, 0.37, 0.36, 0.31, 0.30], 0.01)
    assert _at("groups", 1)["overlap"] != _at("groups", 1)["overlap"]                       # NaN: eine Lösung hat keine Überlappung


# --- Sensitivität b' ------------------------------------------------------------------------------------------------------------------------


def test_group_width_one_is_greedy_runs_with_a_penalty():
    r = _at("group_width", 1)
    assert r["best_gap"] == pytest.approx(4.28, abs=0.02) and r["beam_b_gap"] == 0.0 and r["groups_failed_share"] == 28.0 and r["dbs_failed_share"] == 10.0
    assert (r["verdict_besser"], r["verdict_gleich"], r["verdict_schlechter"], r["n_compared"]) == (0, 3, 42, 45) and r["dbs_optimal_share"] == 6.0


def test_group_width_three_and_four():
    r3, r4 = _at("group_width", 3), _at("group_width", 4)
    assert r3["overlap"] == pytest.approx(0.70, abs=0.01) and (r3["verdict_gleich"], r3["verdict_schlechter"]) == (43, 7) and r3["dbs_optimal_share"] == 86.0
    assert r3["exp_ratio"] == pytest.approx(1.53, abs=0.01)
    assert r4["overlap"] == 1.0 and r4["distinct"] == 1.0 and r4["exp_ratio"] == pytest.approx(1.87, abs=0.01)


# --- Hindernisse und Größe -----------------------------------------------------------------------------------------------------------------


def test_obstacle_sweep_failure_shares_and_overlap():
    assert _col("obstacle_pct", "dbs_failed_share") == [0.0, 0.0, 2.0, 12.0, 16.0]
    assert _col("obstacle_pct", "beam_b_failed_share") == [0.0, 0.0, 0.0, 2.0, 2.0]
    assert _col("obstacle_pct", "beam_g_failed_share") == [0.0, 0.0, 6.0, 20.0, 48.0]
    _close(_col("obstacle_pct", "overlap"), [0.25, 0.30, 0.49, 0.78, 0.84], 0.01)
    assert _col("obstacle_pct", "only_dbs_share") == [0.0, 0.0, 0.0, 0.0, 2.0] and _col("obstacle_pct", "only_beam_share") == [0.0, 0.0, 2.0, 10.0, 16.0]


def test_size_sweep_worse_counts_gaps_and_effort():
    assert _col("side", "verdict_schlechter") == [10, 11, 23, 29, 33]
    _close(_col("side", "best_gap"), [0.0, 0.0, 0.0, 0.12, 0.53], 0.01)
    _close(_col("side", "exp_ratio"), [1.53, 1.36, 1.26, 1.22, 1.18], 0.01)


# --- Konstanten ------------------------------------------------------------------------------------------------------------------------------


def test_constants_are_consistent_with_the_measured_sweeps():
    assert C.POP_SEEDS[0] == 200000 and len(C.POP_SEEDS) == 50
    assert list(C.PENALTIES) == _col("penalty", "value") and list(C.GROUPS_OPTIONS) == _col("groups", "value")
    assert list(C.GROUP_WIDTHS) == _col("group_width", "value") and list(C.OBSTACLE_SWEEP) == _col("obstacle_pct", "value") and list(C.SCALING_SIDES) == _col("side", "value")
