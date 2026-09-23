"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt und jede Schicht, beide Instanz-Typen, gescheiterte Läufe, Randwerte,
Würfel-Knopf, Permalink-Grenzen, Instanzwechsel, Sweeps und Vergleich auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import dbs_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=180)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if step != 1:
        at.select_slider(key="dbs_step").set_value(step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label.startswith(label))


def test_default_run_has_no_exception_and_shows_the_four_metrics():
    at = _run()
    _ok(at)
    assert {"DBS: Lücke", "Beam (B = 6): Lücke", "Vergleich", "Überlappung"} <= {m.label for m in at.metric}
    assert _metric(at, "DBS: Lücke") == "0.00 %" and _metric(at, "Beam (B = 6)") == "0.00 %" and _metric(at, "Vergleich") == "gleich" and _metric(at, "Überlappung") == "0.27"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["network_select"] == p["network"] and at.session_state["groups_select"] == p["groups"]
    assert at.session_state["group_width_select"] == p["group_width"] and at.session_state["penalty_select"] == p["penalty"]
    assert at.metric


@pytest.mark.parametrize("step", [1, 2, 3])
def test_every_step_runs_for_every_network(step):
    for network in C.NETWORKS:
        at = _run(network_select=network, step=step)
        _ok(at)
        assert at.get("plotly_chart") and at.session_state["dbs_step"] == step


@pytest.mark.parametrize("preset", ["Selbst Beam(6) scheitert", "Gruppenbreite 1 (G = B)", "Zu starke Strafe (λ = 4)"])
@pytest.mark.parametrize("step", [1, 2, 3])
def test_failed_groups_render_in_every_step_and_are_flagged(preset, step):
    p = C.PRESETS[preset]
    at = _run(seed_input=p["seed"], obstacle_slider=p["obstacle_pct"], groups_select=p["groups"], group_width_select=p["group_width"], penalty_select=p["penalty"], step=step)
    _ok(at)
    if step == 3:
        assert any("gescheitert" in line for m in at.markdown for line in m.value.splitlines() if line.startswith("| "))
    if preset == "Selbst Beam(6) scheitert":
        assert _metric(at, "Beam (B = 6)") == "kein Pfad" and _metric(at, "Vergleich") == "-"


def test_layer_slider_walks_through_all_layers_and_survives_an_instance_change():
    at = _run(step=2)
    _ok(at)
    slider = at.slider(key="dbs_layer")
    slider.set_value(slider.max).run()
    _ok(at)
    assert at.session_state["dbs_layer"] == slider.max
    at.session_state["network_select"] = "corridor"                       # weniger Schichten: gespeicherter Wert wird geklemmt
    at.run()
    _ok(at)
    assert at.session_state["dbs_layer"] <= 5


@pytest.mark.parametrize("kw", [
    dict(side_slider=C.SIDE_MIN), dict(side_slider=C.SIDE_MAX), dict(obstacle_slider=C.OBSTACLE_MIN), dict(obstacle_slider=C.OBSTACLE_MAX),
    dict(groups_select=C.GROUPS_OPTIONS[0]), dict(groups_select=C.GROUPS_OPTIONS[-1], group_width_select=C.GROUP_WIDTHS[-1]),
    dict(group_width_select=C.GROUP_WIDTHS[0]), dict(penalty_select=C.PENALTIES[0]), dict(penalty_select=C.PENALTIES[-1]),
    dict(groups_select=1, group_width_select=1), dict(groups_select=8, group_width_select=1, obstacle_slider=C.OBSTACLE_MAX),
])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))
    _ok(_run(step=2, **kw))
    _ok(_run(step=3, **kw))


def test_penalty_changes_the_result_on_the_matching_preset():
    off = _run(penalty_select=0.0)
    on = _run()
    assert _metric(off, "Überlappung") == "1.00" and _metric(on, "Überlappung") == "0.27"


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["side"] = "9999"
    at.query_params["obstacle"] = "9999"
    at.query_params["groups"] = "5"
    at.query_params["gwidth"] = "7"
    at.query_params["lambda"] = "0.3"
    at.query_params["network"] = "trap"
    at.run()
    _ok(at)
    assert at.session_state["side_slider"] == C.SIDE_MAX and at.session_state["obstacle_slider"] == C.OBSTACLE_MAX
    assert at.session_state["groups_select"] == C.DEFAULT_GROUPS and at.session_state["group_width_select"] == C.DEFAULT_GROUP_WIDTH
    assert at.session_state["penalty_select"] == C.DEFAULT_PENALTY and at.session_state["network_select"] == "grid"


def test_permalink_accepts_valid_group_width_and_penalty():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["groups"] = "4"
    at.query_params["gwidth"] = "1"
    at.query_params["lambda"] = "2.0"
    at.query_params["network"] = "corridor"
    at.run()
    _ok(at)
    assert (at.session_state["groups_select"], at.session_state["group_width_select"], at.session_state["penalty_select"]) == (4, 1, 2.0)
    assert at.session_state["network_select"] == "corridor"


def test_sidebar_hides_grid_only_controls_for_the_corridor_trap_but_keeps_the_diversity_controls():
    at = _run(network_select="corridor")
    _ok(at)
    assert not any(s.key == "side_slider" for s in at.slider)
    assert {"groups_select", "group_width_select", "penalty_select"} <= {s.key for s in at.select_slider}


def test_changing_the_instance_while_on_step_two_does_not_crash():
    at = _run(step=2, side_slider=12)
    _ok(at)
    at.session_state["side_slider"] = C.SIDE_MIN
    at.run()
    _ok(at)
    at.session_state["network_select"] = "corridor"
    at.run()
    _ok(at)


def test_total_width_is_shown_in_the_sidebar():
    at = _run(groups_select=4, group_width_select=3)
    assert any("B = 4 × 3 = 12" in c.value for c in at.sidebar.caption)


@pytest.mark.parametrize("param", ["penalty", "groups", "group_width", "obstacle_pct", "side"])
@pytest.mark.parametrize("metric", ["gap", "overlap", "routes", "shares", "verdict", "expansion"])
def test_sweeps_run_on_demand_for_every_metric(param, metric):
    at = _run(side_slider=8, sweep_metric=metric)
    at.selectbox(key="sweep_select").set_value(param).run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_comparison_experiment_runs_on_demand_and_shows_its_metrics():
    at = _run(side_slider=8)
    next(b for b in at.button if b.key == "cfg_start").click().run()
    _ok(at)
    labels = {m.label for m in at.metric}
    assert {"besser / gleich / schlechter", "nur DBS findet einen Pfad", "nur Beam (B) findet einen Pfad", "Optimal-Anteil DBS / Beam (B)"} <= labels


def test_lambda_profile_is_drawn_on_the_main_page():
    at = _run()
    assert any("Wie wirkt λ auf diese Instanz?" in m.value for m in at.markdown)


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Wer setzt an" in m.value and "DBS liefert die bessere Top-1-Lösung" in m.value for m in at.markdown)
