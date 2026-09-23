"""Diverse Beam Search - bessere Lösung oder nur Alternativen? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Siebtes Stück der Heuristische-Baumsuche-Linie der "Konzepte"-Reihe, Fortsetzung von Beam Search (beam-search-demo): ein Beam Search liefert
EINE Lösung, und seine Strahlplätze laufen oft auf dieselbe Route zu. Diverse Beam Search (Vijayakumar et al., AAAI 2018) teilt das
Strahlbudget in Gruppen und bestraft, was eine frühere Gruppe schon gewählt hat - jede Gruppe liefert eine eigene Route. Bringt das eine
bessere Lösung, oder nur Alternativen, und was kosten sie? Muss gemessen werden, nicht angenommen.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import streamlit as st

import dbs_constants as C
from dbs_evaluation import SWEEP_LABELS, Settings, analyse, jaccard, penalty_profile, run_config, sweep
from dbs_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from dbs_visualization import (
    BEAM_COLOR,
    build_alternatives,
    build_instance,
    build_lambda_profile,
    build_layer_maps,
    build_share_bars,
    build_sweep,
    build_verdict_bars,
    group_color,
)

st.set_page_config(page_title="Diverse Beam Search – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _profile(settings):
    return penalty_profile(settings)


@st.cache_data(show_spinner=False)
def _config(settings):
    return run_config(settings)


def _fmt_cost(result):
    return "gescheitert" if result.failed else f"{result.cost:.2f}"


def _max_depth(per_layer):
    return max((depth for depth, rec in enumerate(per_layer) if any(r["beam"] for r in rec)), default=0)


st.title("🌈 Diverse Beam Search – bessere Lösung oder nur Alternativen?")
st.markdown(
    """
**Siebtes Stück der Heuristische-Baumsuche-Linie** - die zweite Fortsetzung von Beam Search. Beam Search liefert **eine** Lösung,
und die B Plätze seines Strahls laufen oft auf dieselbe Route zu: eine Liste fast identischer Kandidaten, gerechnet für den
Preis von B.

**Diverse Beam Search** (Vijayakumar et al., AAAI 2018) teilt das Strahlbudget B in **G Gruppen** zu je b' = B / G Plätzen. Die
Gruppen laufen Schicht für Schicht nacheinander; die Bewertung eines Kandidaten wird um **λ × (Zahl früherer Gruppen, die diesen
Zustand in dieser Schicht schon gewählt haben)** verschlechtert. Jede Gruppe liefert ihre **eigene Route**. Gruppe 1 wird nie
bestraft - sie ist ein gewöhnlicher Beam Search der Breite b'. Das Paper verspricht Vielfalt bei geringem Zusatzaufwand und eine
nie schlechtere Lösung als Beam Search der Breite b'. Hier wird gemessen, was davon auf dem Raster gilt.
"""
)
st.caption(
    "Setzt auf [beam-search-demo](https://github.com/sebastian-hanisch/beam-search-demo) auf (derselbe Graph, dieselben Instanzen; Beam Search "
    "mit f-Rang als Vergleich). Noch nicht gebautes Geschwister: Monte Carlo Tree Search (MCTS)."
)

with st.expander("So funktioniert Diverse Beam Search", expanded=True):
    st.markdown(
        """
1. **Gruppen und Schichten:** wie Beam Search Schicht für Schicht (Schicht = Kantenzahl vom Start). Das Budget B = G × b' ist auf G Gruppen zu je b' Plätzen aufgeteilt.
2. **Reihenfolge:** in jeder Schicht setzt Gruppe 1 ihren Strahl fort (Kandidaten nach f = g + h, die b' besten bleiben), dann Gruppe 2 usw.
3. **Diversitätsstrafe (Hamming):** für Gruppe g wird jeder Kandidat um **λ × Kantenlänge × (Zahl früherer Gruppen, die ihn in dieser Schicht gewählt haben)** schlechter bewertet. λ = 0 heißt: G unabhängige gleiche Beam Searches.
4. **Ergebnis:** jede Gruppe endet, wenn sie das Ziel erzeugt (kleinstes g der Schicht) oder keine Kandidaten mehr hat - **G Routen**, die beste zählt als Top-1.
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:5], preset_names[5:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    network = st.radio("Instanz", options=list(C.NETWORKS), format_func=lambda n: C.NETWORK_LABELS[n], key="network_select",
                        help="Die handgebaute Falle hat eine explizite Heuristik und zeigt den Mechanismus an neun Knoten - Rastergröße/Hindernisdichte/Seed wirken dort nicht.")
    if network == "grid":
        side = st.slider("Rastergröße (Seitenlänge)", *bounds("side_slider"), key="side_slider",
                          help="Bei größeren Rastern ist die Gruppenlösung öfter schlechter als Beam Search gleicher Gesamtbreite (Größe 8: 10 von 50 Instanzen, Größe 16: 33 von 50).")
        obstacle_pct = st.slider("Hindernisdichte [%]", *bounds("obstacle_slider"), key="obstacle_slider", step=C.OBSTACLE_STEP,
                                  help="Mit vielen Hindernissen scheitern Gruppen häufiger (bei 40 %: DBS ohne Pfad in 16 % der Instanzen, Beam Search Breite 6 in 2 %) und die Routen fallen zusammen.")
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        side, obstacle_pct, seed = C.DEFAULT_SIDE, C.DEFAULT_OBSTACLE, C.DEFAULT_SEED
    groups = st.select_slider("Gruppen G", options=list(C.GROUPS_OPTIONS), key="groups_select", help="Anzahl der Gruppen = Anzahl der gelieferten Routen.")
    group_width = st.select_slider("Plätze je Gruppe b'", options=list(C.GROUP_WIDTHS), key="group_width_select",
                                   help="Breite jeder Gruppe. Gesamtbreite B = G × b' - der Vergleich mit Beam Search läuft immer bei diesem B.")
    penalty = st.select_slider("Diversitätsstärke λ (Kantenlängen)", options=list(C.PENALTIES), key="penalty_select", format_func=lambda p: f"{p:g}",
                               help="Strafe je früherer Gruppe, die einen Zustand schon gewählt hat, in mittleren Kantenlängen. 0 = keine Strafe.")
    st.caption(f"Gesamtbreite **B = {groups} × {group_width} = {groups * group_width}**")

sync_query_params({"network_select": network, "side_slider": int(side), "obstacle_slider": int(obstacle_pct), "seed_input": int(seed),
                   "groups_select": int(groups), "group_width_select": int(group_width), "penalty_select": float(penalty)})

settings = Settings(network, int(side), int(obstacle_pct), int(seed), int(groups), int(group_width), float(penalty))
with st.spinner("Rechne..."):
    a = _analysis(settings)
dbs, beam_b = a.dbs, a.beam_b
hand = network != "grid"
B = settings.total_width

# --- Diverse Beam Search in Aktion -------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Diverse Beam Search in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Strahl je Schicht", 3: "3 · Alternativen"}
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="dbs_step", format_func=lambda s: STEP_LABELS[s])

if step == 1:
    if hand:
        st.markdown(f"**{a.inst.graph.n} Knoten** (Start S, Ziel Z), Kantengewichte an den Kanten, Heuristik h(n) am Knoten")
    else:
        st.markdown(f"**{a.inst.graph.n} Zellen** ({len(a.inst.blocked_xy)} Hindernisse), Start (grün) und Ziel (rot)")
    st.plotly_chart(build_instance(a.inst), width="stretch", key="s1_map")
elif step == 2:
    n_depth = max(_max_depth(dbs.per_layer), len(beam_b.per_layer) - 1)
    if n_depth > 1:
        if "dbs_layer" in st.session_state:
            st.session_state["dbs_layer"] = min(max(1, int(st.session_state["dbs_layer"])), n_depth)
        layer = st.slider("Schicht", 1, n_depth, key="dbs_layer", help="Schicht = Kantenzahl vom Start. Links Beam Search der Gesamtbreite B, rechts die Gruppen.")
    else:
        layer = 1
    b_nodes = beam_b.per_layer[layer][0] if layer < len(beam_b.per_layer) else []
    recs = dbs.per_layer[layer] if layer < len(dbs.per_layer) else []
    lines = []
    for g, rec in enumerate(recs):
        if not rec["active"] or (not rec["beam"] and rec["n_candidates"] == 0):
            lines.append(f"Gruppe {g + 1}: fertig oder gescheitert")
        elif not rec["beam"]:
            lines.append(f"Gruppe {g + 1}: erzeugt hier das Ziel")
        else:
            lines.append(f"Gruppe {g + 1}: {len(rec['beam'])} von {rec['n_candidates']} Kandidaten, davon {rec['n_fresh']} von keiner früheren Gruppe gewählt")
    chosen = [s for rec in recs for s in rec["beam"]]
    st.markdown(
        f"**Schicht {layer} von {n_depth}:** Beam Search (B = {B}) hält {len(b_nodes)} Knoten"
        + (" (Suche beendet)" if layer >= len(beam_b.per_layer) else "")
        + f", die Gruppen zusammen {len(chosen)} Plätze auf {len(set(chosen))} verschiedenen Zuständen."
    )
    st.plotly_chart(build_layer_maps(a.inst, beam_b.per_layer, dbs.per_layer, layer, settings.groups), width="stretch", key=f"s2_map_{layer}")
    st.caption(" · ".join(lines))
    st.caption(
        f"Expansionen insgesamt: DBS {dbs.expansions}, Beam Search (B = {B}) {beam_b.expansions}"
        + (f", A\\* {a.astar.expansions}" if a.astar is not None else "")
        + ". Jede Gruppe expandiert ihren eigenen Strahl - es gibt nichts zu teilen."
    )
else:
    solved_paths = [g.path for g in dbs.groups]
    st.plotly_chart(build_alternatives(a.inst, solved_paths, a.ucs.path, None if beam_b.failed else beam_b.path), width="stretch", key="s3_map")
    rows = ["| Gruppe | Kosten | Lücke zum Optimum | Überlappung mit Gruppe 1 | Schicht |", "|---|---|---|---|---|"]
    ex = (a.inst.start, a.inst.goal)
    first = dbs.groups[0]
    for g, (res, gap) in enumerate(zip(dbs.groups, a.group_gaps)):
        if res.failed:
            rows.append(f"| {g + 1} | gescheitert | - | - | {res.layer} |")
        else:
            ov = "-" if first.failed or g == 0 else f"{jaccard(res.path, first.path, ex):.2f}"
            rows.append(f"| {g + 1} | {res.cost:.2f} | {gap:.2f} % | {ov} | {res.layer} |")
    st.markdown("\n".join(rows))
    parts = [f"**Optimum:** {a.ucs.cost:.2f}", f"**Beam (B = {B}):** {_fmt_cost(beam_b)}", f"**Beam (b' = {settings.group_width}):** {_fmt_cost(a.beam_g)}"]
    if dbs.failed:
        st.warning(" – ".join(parts) + " – keine Gruppe fand einen Pfad.")
    else:
        st.markdown(" – ".join(parts))

st.markdown("---")

# --- Ergebnis ----------------------------------------------------------------------------------------------------------------------------------

st.markdown(f"## 🎯 Diverse Beam Search gegen Beam Search bei Gesamtbreite B = {B}")
st.caption(
    "**Lücke:** Kosten der besten Gruppenlösung gegenüber dem Optimum (Uniform-Cost). **Vergleich:** beste Gruppenlösung gegen Beam Search derselben "
    "Gesamtbreite (nur wenn beide einen Pfad fanden). **Überlappung:** mittlere Jaccard-Ähnlichkeit der Knotenmengen (ohne Start und Ziel) der "
    "gelösten Gruppenlösungen, 1 = identisch. **Routen:** Zahl verschiedener Pfade, darunter die mit Lücke ≤ 10 %."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("DBS: Lücke", "kein Pfad" if dbs.failed else f"{a.best_gap:.2f} %", delta=f"{dbs.expansions} Expansionen", delta_color="off")
m2.metric(f"Beam (B = {B}): Lücke", "kein Pfad" if beam_b.failed else f"{a.beam_b_gap:.2f} %", delta=f"{beam_b.expansions} Expansionen", delta_color="off")
m3.metric("Vergleich", "-" if a.verdict is None else a.verdict, delta="DBS gegen Beam" if a.verdict is not None else "ohne Pfad", delta_color="off")
m4.metric("Überlappung", "-" if a.overlap != a.overlap else f"{a.overlap:.2f}", delta=f"{a.distinct} Routen", delta_color="off")
st.caption(f"Die beste Gruppe ist die mit den kleinsten Kosten. Verschiedene Routen: {a.distinct} von {settings.groups}, davon **{a.usable} brauchbar** (Lücke ≤ 10 %).")

st.markdown("---")

# --- λ-Profil ----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎚️ Wie wirkt λ auf diese Instanz?")
st.caption(
    "Lücke jeder Gruppe für jede Stufe von λ bei sonst gleichen Einstellungen. Gruppe 1 bleibt für jedes λ dieselbe (sie wird nie bestraft); "
    "ein x oberhalb der Kurven heißt: diese Gruppe fand keinen Pfad."
)
profile = _profile(settings)
st.plotly_chart(build_lambda_profile(profile, settings.groups), width="stretch", key="lambda_profile")

st.markdown("---")

# --- Sweeps ------------------------------------------------------------------------------------------------------------------------------------

if network == "grid":
    st.subheader("📐 Wie hängen Lücke, Vielfalt und Aufwand von den Reglern ab?")
    sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
    metric = st.radio("Kennzahl", options=["gap", "overlap", "routes", "shares", "verdict", "expansion"],
                       format_func=lambda k: {"gap": "Lücke (%)", "overlap": "Überlappung", "routes": "Routen", "shares": "Scheitern (%)",
                                              "verdict": "besser / gleich / schlechter", "expansion": "Aufwand"}[k],
                       key="sweep_metric", horizontal=True)
    base_sweep = replace(settings, seed=0)
    if st.button("Sweep über 50 feste Instanzen berechnen (kann eine Minute dauern)", key="sweep_start"):
        st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
    if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
        with st.spinner("Rechne den Sweep über 50 feste Instanzen..."):
            rows_sweep = _sweep(sweep_param, base_sweep)
        label = SWEEP_LABELS[sweep_param]
        if metric == "gap":
            st.plotly_chart(build_sweep(rows_sweep, label, [("best_gap", "DBS: beste Gruppe", group_color(0)), ("beam_b_gap", "Beam (Breite B)", BEAM_COLOR),
                                                            ("beam_g_gap", "Beam (Breite b')", "#888888")], "Lücke zum Optimum (%)"), width="stretch", key="sweep_gap")
        elif metric == "overlap":
            st.plotly_chart(build_sweep(rows_sweep, label, [("overlap", "DBS: Lösungen der Gruppen", group_color(0)), ("width_alt_overlap", "Beam mit Breite b', 2b', ... (ohne Strafe)", "#888888"),
                                                            ("frontier_overlap", "DBS: Strahllisten-Pfade", group_color(1)), ("beam_frontier_overlap", "Beam (Breite B): Strahllisten-Pfade", BEAM_COLOR)],
                                        "Überlappung (Jaccard)"), width="stretch", key="sweep_overlap")
        elif metric == "routes":
            st.plotly_chart(build_sweep(rows_sweep, label, [("distinct", "verschiedene Routen (DBS)", group_color(0)), ("usable", "davon brauchbar (Lücke ≤ 10 %)", group_color(2)),
                                                            ("width_alt_distinct", "verschiedene Routen, Beam mit Breite b', 2b', ...", "#888888")], "Routen (Anzahl)"), width="stretch", key="sweep_routes")
        elif metric == "shares":
            st.plotly_chart(build_share_bars(rows_sweep, label), width="stretch", key="sweep_shares")
        elif metric == "verdict":
            st.plotly_chart(build_verdict_bars(rows_sweep, label), width="stretch", key="sweep_verdict")
        else:
            st.plotly_chart(build_sweep(rows_sweep, label, [("exp_ratio", "Expansionen DBS / Beam (Breite B)", group_color(0))], "Expansionen-Verhältnis", ref_line=1.0, ref_label="so viel wie Beam"),
                            width="stretch", key="sweep_exp")
        st.caption(
            "Median über 50 feste Instanzen (Seeds 200000–200049), Band = 10. bis 90. Perzentil. Lücken nur über gelöste Läufe; das Scheitern steht im Balken-Modus. "
            "Die 50 Instanzen weichen von den sonst üblichen 5 ab: Anteile brauchen mehr Instanzen."
        )

    st.markdown("---")

    st.subheader("🔬 Wie oft ist die beste Gruppenlösung besser als Beam Search gleicher Gesamtbreite?")
    st.caption("Über 50 feste Instanzen bei den aktuellen Einstellungen: beste Gruppenlösung gegen Beam Search der Gesamtbreite B - besser, gleich, schlechter, und wer allein eine Lösung fand.")
    key_cfg = replace(settings, seed=0)
    if st.button("Vergleich über 50 feste Instanzen messen", key="cfg_start"):
        st.session_state["cfg_done"] = st.session_state.get("cfg_done", set()) | {key_cfg}
    if key_cfg in st.session_state.get("cfg_done", set()):
        with st.spinner("Rechne 50 Instanzen..."):
            cfg = _config(key_cfg)
        st.plotly_chart(build_verdict_bars([{"value": f"G={settings.groups}, b'={settings.group_width}, λ={settings.penalty:g}", **cfg}], "Einstellung"), width="stretch", key="cfg_verdict")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("besser / gleich / schlechter", f"{cfg['verdict_besser']} / {cfg['verdict_gleich']} / {cfg['verdict_schlechter']}", delta=f"von {cfg['n_compared']} verglichenen", delta_color="off")
        c2.metric("nur DBS findet einen Pfad", f"{cfg['only_dbs_share']:.0f} %", delta="Beam (B) scheitert", delta_color="off")
        c3.metric("nur Beam (B) findet einen Pfad", f"{cfg['only_beam_share']:.0f} %", delta="DBS scheitert", delta_color="off")
        c4.metric("Optimal-Anteil DBS / Beam (B)", f"{cfg['dbs_optimal_share']:.0f} / {cfg['beam_b_optimal_share']:.0f} %", delta="aller Instanzen", delta_color="off")
        st.caption(f"Instanzen: Rastergröße {settings.side}, Hindernisdichte {settings.obstacle_pct} %.")

    st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **DBS liefert die bessere Top-1-Lösung** | Nicht gegen Beam Search gleicher Gesamtbreite: die beste Gruppenlösung war in keiner von 50 Instanzen besser als Beam Search der Breite 6 (27 gleich, 23 schlechter; Optimal-Anteil 54 % gegen 100 %) und über alle 28 Einstellungen der fünf Sweeps kam "besser" kein einziges Mal vor. Gegen Beam Search der **Gruppenbreite** stimmt es: Gruppe 1 IST dieser Beam Search, die beste Gruppe ist nie schlechter (im Test exakt geprüft). | Beam Stack Search / A\\* (finden das Optimum) |
| **Die Gruppen liefern brauchbare Alternativen** | Nur bei mittlerem λ: bei λ = 0.5 sind im Median alle drei Routen verschieden und höchstens 10 % länger als das Optimum, ab λ = 2 nur noch eine - die anderen sind schlechter. Bei λ = 0 sind alle drei identisch (Überlappung 1.0). Ab λ = 1 flacht die Überlappung bei etwa 0.25 ab. | Eine gezielte k-kürzeste-Pfade-Suche (Yen; hier nicht gebaut) |
| **Vielfalt ist umsonst** | Der Aufwand beträgt 1.26x die Expansionen von Beam Search gleicher Gesamtbreite (G = 3) und steigt mit G auf 2.85x (G = 8). Der Anteil gescheiterter Gruppen steigt mit λ von 2 % (λ = 0) auf 14 % (λ = 8): die Strafe treibt Gruppen in Sackgassen. Bei 40 % Hindernissen scheitert DBS in 16 % der Instanzen, Beam Search Breite 6 in 2 %. | - |
| **Der Strahl klumpt sich räumlich** | Nicht gestützt: die räumliche Streuung der Strahlknoten ist bei Beam Search Breite 6 mit 3.27 Kantenlängen größer als bei DBS (0.86 bis 2.32). Redundant sind die Strahlpfade (gemeinsame Präfixe): Überlappung 0.53 bei Beam Search, 0.35 bis 0.37 bei DBS mit λ ≥ 2. | - |
| **Breitere Gruppen sind besser** | Nicht für die Vielfalt: bei b' = 4 (G = 3) laufen alle drei Gruppen auf dieselbe Route (Überlappung 1.0, Aufwand 1.87x). Bei b' = 1 (G = B) scheitern 28 % der Gruppen und die beste hat 4.3 % Lücke (Beam Search Breite 3: 0.0 %). | - |
| **Synthetische Instanzen, Hamming-Vielfalt** | Ein Raster mit Jitter, Vierer-Nachbarschaft, keine gerichteten Kanten, dazu eine handgebaute Falle. Vielfalt wird nur als Gleichheit des Zustands in derselben Schicht bestraft (Hamming); n-Gramm- und Einbettungs-Varianten des Papers, Stochastic Beam Search und ein Vergleich mit k-kürzesten Pfaden sind nicht gebaut. | Weitere Vielfaltsmaße (hier nicht gebaut) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gruppen.** Das Budget $B = G \cdot b'$ ist in Gruppen $1, \dots, G$ zu je $b'$ Plätzen geteilt. In Schicht $t$ ist $Y_g^{(t)}$ der Strahl
der Gruppe $g$ (höchstens $b'$ Zustände), $C_g^{(t)}$ ihre Kandidatenmenge (Nachfolger von $Y_g^{(t-1)}$, je Zustand kleinstes $g$-Wert,
bereits von der eigenen Gruppe expandierte Zustände ausgeschlossen).

**Bewertung mit Diversitätsstrafe.**
$$\theta_g(v) = f(v) + \lambda \cdot \bar w \cdot \left|\{\, g' < g : v \in Y_{g'}^{(t)} \,\}\right|, \qquad f(v) = g(v) + h(v),$$
mit der mittleren Kantenlänge $\bar w$; $Y_g^{(t)}$ besteht aus den $b'$ Kandidaten mit kleinstem $\theta_g$ (Tie-Break: kleiner Zustandsindex). Für
$g = 1$ ist die Menge der früheren Gruppen leer, also $\theta_1 = f$: **Gruppe 1 ist Beam Search der Breite $b'$** (Paper-Garantie:
$\min_g \text{cost}_g \le \text{cost}_1$).

**Ende.** Eine Gruppe endet, wenn ihr Kandidatenpool das Ziel enthält (Kosten = kleinstes $g$ des Ziels) oder leer ist (gescheitert). Endet
eine Gruppe, übt sie auf tiefere Schichten keine Strafe mehr aus.

**Überlappung.** $J(P, Q) = |P \cap Q| / |P \cup Q|$ über die inneren Knotenmengen zweier Lösungen; Mittel über alle Paare gelöster Gruppen.

**Literatur.** Vijayakumar, A. K., Cogswell, M., Selvaraju, R. R., Sun, Q., Lee, S., Crandall, D., & Batra, D. (2018). *Diverse Beam Search
for Improved Description of Complex Scenes.* Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), 7371-7379 (Vorabdruck: *Diverse Beam
Search: Decoding Diverse Solutions from Neural Sequence Models*, arXiv:1610.02424).

Implementiert in `dbs_algorithm.py` (Suchkerne und `beam_search` aus der Beam-Search-Demo, `diverse_beam_search` neu),
`dbs_graph.py`/`dbs_scenario.py` (Graph, Raster und handgebaute Instanz), `dbs_evaluation.py` (Kennzahlen, Sweeps, Diversitätsmaße).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
