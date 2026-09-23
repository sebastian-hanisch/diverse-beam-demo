"""Auswertung: Diversitätsmaße einzeln, Analysis-Felder gegen unabhängige Neuberechnung, run_config/sweep/penalty_profile."""

import math

import numpy as np
import pytest

import dbs_algorithm as A
import dbs_constants as C
import dbs_evaluation as ev
import dbs_scenario as S


# --- Diversitätsmaße einzeln --------------------------------------------------------------------------------------------------------------------


def test_jaccard_identical_disjoint_and_partial_paths():
    assert ev.jaccard([0, 1, 2, 5], [0, 1, 2, 5]) == 1.0
    assert ev.jaccard([0, 1, 2, 5], [0, 3, 4, 5], exclude=(0, 5)) == 0.0
    assert ev.jaccard([0, 1, 2, 5], [0, 2, 3, 5], exclude=(0, 5)) == pytest.approx(1 / 3)
    assert ev.jaccard([0, 5], [0, 5], exclude=(0, 5)) == 1.0                       # beide ohne innere Knoten


def test_mean_overlap_averages_over_all_pairs_and_is_nan_below_two_paths():
    paths = [[0, 1, 2, 9], [0, 1, 3, 9], [0, 4, 5, 9]]
    expected = np.mean([ev.jaccard(a, b, (0, 9)) for i, a in enumerate(paths) for b in paths[i + 1:]])
    assert ev.mean_overlap(paths, (0, 9)) == pytest.approx(expected) and expected == pytest.approx((1 / 3 + 0 + 0) / 3)
    assert math.isnan(ev.mean_overlap([paths[0]])) and math.isnan(ev.mean_overlap([]))


def test_layer_spread_of_one_node_is_nan_and_of_two_nodes_is_their_distance_in_edge_lengths():
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 10.0]])
    assert math.isnan(ev.layer_spread(xy, [1], 2.0))
    assert ev.layer_spread(xy, [0, 1], 2.0) == pytest.approx(5.0 / 2.0)
    assert ev.layer_spread(xy, [0, 1, 2], 1.0) == pytest.approx((5.0 + 10.0 + np.hypot(3, 6)) / 3)
    assert ev.layer_spread(xy, [1, 1], 1.0) == 0.0                                  # derselbe Zustand in zwei Gruppen


def test_mean_spread_ignores_layers_with_a_single_node():
    xy = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert ev.mean_spread(xy, [[0], [0, 1], [1]], 1.0) == pytest.approx(5.0)
    assert math.isnan(ev.mean_spread(xy, [[0], [1]], 1.0))


# --- Analysis -----------------------------------------------------------------------------------------------------------------------------------


def test_default_settings_come_from_the_constants():
    s = ev.Settings()
    assert (s.groups, s.group_width, s.penalty, s.seed) == (C.DEFAULT_GROUPS, C.DEFAULT_GROUP_WIDTH, C.DEFAULT_PENALTY, C.DEFAULT_SEED)
    assert s.total_width == 6 and C.DEFAULT_GROUPS in C.GROUPS_OPTIONS and C.DEFAULT_GROUP_WIDTH in C.GROUP_WIDTHS and C.DEFAULT_PENALTY in C.PENALTIES


@pytest.mark.parametrize("seed", [44, 200046, 200011])
def test_analysis_fields_match_an_independent_recomputation(seed):
    s = ev.Settings(seed=seed)
    a = ev.analyse(s)
    inst = S.grid_instance(s.side, s.obstacle_pct, seed)
    ucs = A.uniform_cost_search(inst.graph, inst.start, inst.goal).cost
    dbs = A.diverse_beam_search(inst.graph, inst.start, inst.goal, s.groups, s.group_width, s.penalty)
    beam_b = A.beam_search(inst.graph, inst.start, inst.goal, s.total_width, "f")
    assert a.ucs.cost == pytest.approx(ucs)
    assert [None if g.failed else g.cost for g in dbs.groups] == [None if g is None else pytest.approx(ucs * (1 + g / 100)) for g in a.group_gaps]
    if not beam_b.failed:
        assert a.beam_b_gap == pytest.approx(100 * (beam_b.cost - ucs) / ucs) and a.exp_ratio == pytest.approx(dbs.expansions / beam_b.expansions)
    assert a.best_gap == pytest.approx(100 * (dbs.best.cost - ucs) / ucs) if dbs.best else math.isnan(a.best_gap)


def test_verdict_besser_gleich_schlechter_and_none_without_a_path():
    a = ev.analyse(ev.Settings(seed=44))
    assert a.verdict == "gleich"
    a2 = ev.analyse(ev.Settings(network="corridor", groups=2, group_width=1, penalty=1.0))
    assert a2.verdict == "besser" and a2.best_gap == 0.0 and a2.beam_b_gap == pytest.approx(37.5)
    assert ev.analyse(ev.Settings(seed=200002, obstacle_pct=40)).verdict is None                       # Beam Breite 6 scheitert
    worse = ev.analyse(ev.Settings(seed=200046))
    assert worse.verdict == "schlechter"


def test_diversity_fields_on_the_default_instance():
    a = ev.analyse(ev.Settings(seed=44))
    assert a.distinct == 3 and a.usable == 3 and 0.2 < a.overlap < 0.35 and a.width_alt_distinct == 1 and a.width_alt_overlap == 1.0
    zero = ev.analyse(ev.Settings(seed=44, penalty=0.0))
    assert zero.distinct == 1 and zero.overlap == 1.0 and zero.usable == 1 and zero.groups_failed_share == 0.0


def test_usable_counts_distinct_routes_only_and_respects_the_gap_threshold():
    a = ev.analyse(ev.Settings(seed=200011, penalty=4.0))
    assert a.group_gaps[1] is None and a.group_gaps[2] > ev.USABLE_GAP and a.usable == 1 and a.distinct == 2 and a.groups_failed_share == pytest.approx(100 / 3)


def test_frontier_overlap_is_defined_for_solved_runs_and_equal_to_beam_when_one_group():
    a = ev.analyse(ev.Settings(seed=44, groups=1, group_width=6))
    assert a.frontier_overlap == pytest.approx(a.beam_frontier_overlap) and a.exp_ratio == 1.0
    assert 0.0 <= ev.analyse(ev.Settings(seed=44)).frontier_overlap <= 1.0


def test_astar_reference_is_only_present_without_an_explicit_heuristic():
    assert ev.analyse(ev.Settings()).astar is not None and ev.analyse(ev.Settings(network="corridor")).astar is None


# --- run_config / sweep / penalty_profile -------------------------------------------------------------------------------------------------------


def test_run_config_keys_counts_and_ranges_on_a_small_population():
    seeds = C.POP_SEEDS[:8]
    r = ev.run_config(ev.Settings(), seeds=seeds)
    assert r["n_runs"] == 8 and r["n_compared"] == sum(r[f"verdict_{v}"] for v in ("besser", "gleich", "schlechter"))
    for key in ("best_gap", "overlap", "distinct", "usable", "exp_ratio", "spread_dbs", "frontier_overlap"):
        assert r[f"{key}_lo"] <= r[key] <= r[f"{key}_hi"] or math.isnan(r[key])
    for key in ("dbs_failed_share", "beam_b_failed_share", "beam_g_failed_share", "groups_failed_share", "only_dbs_share", "only_beam_share", "dbs_optimal_share"):
        assert 0.0 <= r[key] <= 100.0


def test_run_config_overrides_settings_and_uses_the_given_seeds():
    seeds = C.POP_SEEDS[:6]
    a = ev.run_config(ev.Settings(seed=1), seeds=seeds, penalty=0.0)
    b = ev.run_config(ev.Settings(seed=999), seeds=seeds, penalty=0.0)
    assert a == b and a["overlap"] == 1.0


def test_sweep_returns_one_row_per_value_and_penalty_zero_row_has_full_overlap():
    rows = ev.sweep("penalty", ev.Settings(), values=(0.0, 1.0))
    assert [r["value"] for r in rows] == [0.0, 1.0] and rows[0]["overlap"] == 1.0 and rows[1]["overlap"] < 1.0
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS) == {"penalty", "groups", "group_width", "obstacle_pct", "side"}


def test_penalty_profile_has_one_row_per_penalty_and_group_one_is_constant():
    rows = ev.penalty_profile(ev.Settings(seed=44))
    assert [r["penalty"] for r in rows] == list(C.PENALTIES)
    assert len({r["gaps"][0] for r in rows}) == 1 and all(len(r["gaps"]) == 3 for r in rows)
    assert rows[0]["distinct"] == 1 and rows[0]["overlap"] == 1.0 and rows[2]["distinct"] == 3
