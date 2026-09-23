"""Presets: Vollständigkeit, gültige Werte, der Median der Lücke der besten Gruppenlösung bleibt bei den Raster-Presets in der gemessenen
Spannweite über die 50 festen Instanzen (vollständig deterministisch), und jedes Preset zeigt, was sein Name und sein Hilfetext sagen."""

import pytest

import dbs_constants as C
import dbs_evaluation as ev
import dbs_presets as P


def _settings(p):
    return ev.Settings(network=p["network"], side=p["side"], obstacle_pct=p["obstacle_pct"], seed=p["seed"],
                       groups=p["groups"], group_width=p["group_width"], penalty=p["penalty"])


def _analyse(name):
    return ev.analyse(_settings(C.PRESETS[name]))


def _gaps(a):
    return [None if g is None else round(g, 1) for g in a.group_gaps]


def test_every_preset_has_help_and_the_grid_presets_a_band():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 9
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]
    grid_with_band = {"Standardfall (Voreinstellung)", "Keine Strafe (λ = 0)", "Zu starke Strafe (λ = 4)", "Gruppenbreite 1 (G = B)", "Größere Gruppen (b' = 4)", "Selbst Beam(6) scheitert"}
    assert set(C.PRESET_EXPECTED_BANDS) == grid_with_band and grid_with_band <= {n for n, p in C.PRESETS.items() if p["network"] == "grid"}


def test_preset_values_are_valid_members_of_the_controls():
    for p in C.PRESETS.values():
        assert p["network"] in C.NETWORKS and p["groups"] in C.GROUPS_OPTIONS and p["group_width"] in C.GROUP_WIDTHS and p["penalty"] in C.PENALTIES
        assert C.SIDE_MIN <= p["side"] <= C.SIDE_MAX and C.OBSTACLE_MIN <= p["obstacle_pct"] <= C.OBSTACLE_MAX and 0 <= p["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert _settings(C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


@pytest.mark.parametrize("name", list(C.PRESET_EXPECTED_BANDS))
def test_grid_preset_median_gap_stays_in_its_measured_band(name):
    lo, hi = C.PRESET_EXPECTED_BANDS[name]
    row = ev.run_config(_settings(C.PRESETS[name]))
    assert lo <= row["best_gap"] <= hi, row["best_gap"]


def test_standard_preset_gives_three_routes_where_beam_search_of_every_width_gives_one():
    a = _analyse("Standardfall (Voreinstellung)")
    assert _gaps(a) == [0.0, 3.3, 4.1] and a.distinct == 3 and a.overlap == pytest.approx(0.27, abs=0.01)
    assert a.width_alt_distinct == 1 and a.verdict == "gleich" and a.beam_b_gap == pytest.approx(0.0, abs=1e-9)
    assert (a.dbs.expansions, a.beam_b.expansions, a.astar.expansions) == (129, 108, 94)


def test_corridor_preset_beam_finds_eleven_and_the_second_group_finds_the_optimum():
    a = _analyse("Zwei-Korridore-Falle")
    assert (a.ucs.cost, a.beam_b.cost, a.beam_g.cost, a.dbs.best.cost) == (8.0, 11.0, 9.0, 8.0)
    assert [g.cost for g in a.dbs.groups] == [9.0, 8.0] and a.verdict == "besser" and (a.dbs.expansions, a.beam_b.expansions) == (10, 7)


def test_no_penalty_preset_gives_three_identical_routes():
    a = _analyse("Keine Strafe (λ = 0)")
    assert a.distinct == 1 and a.overlap == 1.0 and _gaps(a) == [0.0, 0.0, 0.0] and (a.dbs.expansions, a.beam_b.expansions) == (129, 108)


def test_too_strong_penalty_preset_has_a_failing_group_and_a_poor_third_route():
    a = _analyse("Zu starke Strafe (λ = 4)")
    assert _gaps(a) == [0.0, None, 17.4] and a.distinct == 2 and a.usable == 1 and a.overlap == pytest.approx(0.10, abs=0.01)


def test_group_width_one_preset_numbers():
    a = _analyse("Gruppenbreite 1 (G = B)")
    assert _gaps(a) == [1.0, None, 21.8, 24.3] and a.beam_b_gap == pytest.approx(0.0, abs=1e-9) and a.verdict == "schlechter"


def test_wide_groups_preset_all_groups_converge_on_one_route():
    a = _analyse("Größere Gruppen (b' = 4)")
    assert a.distinct == 1 and a.overlap == 1.0 and (a.dbs.expansions, a.beam_b.expansions) == (241, 125)


def test_beam_of_the_group_width_fails_where_a_later_group_finds_a_route():
    a = _analyse("Beam(2) scheitert, DBS findet")
    assert a.beam_g.failed and _gaps(a) == [None, None, 2.0] and a.beam_b_gap == pytest.approx(0.0, abs=1e-9) and a.verdict == "schlechter"


def test_even_beam_of_the_total_width_fails_on_the_dense_obstacle_preset():
    a = _analyse("Selbst Beam(6) scheitert")
    assert a.beam_b.failed and a.beam_g.failed and _gaps(a) == [None, None, 0.0] and a.verdict is None


def test_many_alternatives_preset_numbers():
    a = _analyse("Viele Alternativen (G = 8)")
    assert a.distinct == 8 and a.usable == 5 and (a.dbs.expansions, a.beam_b.expansions) == (351, 125)
    assert min(g for g in a.group_gaps if g is not None) == 0.0 and 11.0 < max(a.group_gaps) < 12.5


def test_bounds_and_permalink_constants():
    assert P.bounds("side_slider") == (C.SIDE_MIN, C.SIDE_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_permalink_casters_accept_members_and_reject_everything_else():
    c = P.SETTING_SPECS
    assert c["network_select"].caster("corridor") == "corridor" and c["groups_select"].caster("6") == 6 and c["group_width_select"].caster("3") == 3
    assert c["penalty_select"].caster("0.5") == 0.5 and c["penalty_select"].caster("0") == 0.0
    for key, bad in (("network_select", "trap"), ("groups_select", "5"), ("group_width_select", "5"), ("penalty_select", "0.3"), ("penalty_select", "x")):
        with pytest.raises(ValueError):
            c[key].caster(bad)
