"""Plotly-Abbildungen: Instanz (Raster oder handgebauter Graph mit Knotennamen, Kantengewichten und Heuristik), Alternativen-Karte (alle
Gruppenlösungen in Gruppenfarben), Strahl je Schicht nebeneinander (Beam Search Breite B links, Diverse Beam Search rechts nach Gruppen
gefärbt), λ-Profil einer Instanz, Sweeps, Anteile, Vergleichs-Balken. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen
nicht zoomen."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

NODE_COLOR = "#4c78a8"
BLOCKED_COLOR = "#9d755d"
OPT_COLOR = "#4c78a8"
BEAM_COLOR = "#333333"
GROUP_COLORS = ("#e45756", "#f58518", "#54a24b", "#7b3fbf", "#b279a2", "#9d755d", "#17becf", "#bab0ac")
GREY = "rgba(120,120,120,0.55)"
INF = float("inf")


def group_color(g):
    return GROUP_COLORS[g % len(GROUP_COLORS)]


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.1):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _hand(inst):
    return inst.labels is not None


def _label(inst, node):
    if not _hand(inst):
        return str(node)
    return inst.labels[node] + (f" (h={inst.h[node]:g})" if inst.h is not None else "")


def _add(fig, trace, row=None, col=None):
    if row is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=col)


def _panel_base(fig, inst, row=None, col=None, node_alpha=0.3):
    """Hintergrund einer Karte: bei handgebauten Graphen Kanten mit Gewicht, sonst Raster-Zellen und Hindernisse."""
    xy = inst.graph.xy
    if _hand(inst):
        ex, ey, mx, my, mt = [], [], [], [], []
        for u in range(inst.graph.n):
            for v, w in zip(inst.graph.neighbors[u], inst.graph.weights[u]):
                if u < v:
                    ex += [xy[u, 0], xy[v, 0], None]
                    ey += [xy[u, 1], xy[v, 1], None]
                    mx.append((xy[u, 0] + xy[v, 0]) / 2)
                    my.append((xy[u, 1] + xy[v, 1]) / 2)
                    mt.append(f"{w:g}")
        _add(fig, go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(120,120,120,0.5)", width=1.5), hoverinfo="skip", showlegend=False), row, col)
        _add(fig, go.Scatter(x=mx, y=my, mode="text", text=mt, textfont=dict(size=10, color="#777"), hoverinfo="skip", showlegend=False), row, col)
        _add(fig, go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=list(inst.labels), textposition="top center", textfont=dict(size=10, color="#888"),
                             marker=dict(size=8, color="rgba(120,120,120,0.4)"), hoverinfo="skip", showlegend=False), row, col)
    else:
        _add(fig, go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers", marker=dict(size=5, color=f"rgba(76,120,168,{node_alpha})"), name="Noch nicht berührt", hoverinfo="skip", showlegend=False), row, col)
        if len(inst.blocked_xy):
            _add(fig, go.Scatter(x=inst.blocked_xy[:, 0], y=inst.blocked_xy[:, 1], mode="markers", marker=dict(size=6, symbol="square", color=BLOCKED_COLOR), name="Hindernis", showlegend=False), row, col)


def _nodes_trace(inst, nodes, name, color, size, symbol="circle", showlegend=True, texts=None, line_color="white"):
    xy = inst.graph.xy
    nodes = list(nodes)
    marker = dict(size=size, symbol=symbol, color=color, line=dict(width=1, color=line_color))
    if _hand(inst):
        text = texts if texts is not None else [inst.labels[n] for n in nodes]
        return go.Scatter(x=xy[nodes, 0] if nodes else [], y=xy[nodes, 1] if nodes else [], mode="markers+text", text=text, textposition="top center",
                          marker=marker, name=name, showlegend=showlegend, hoverinfo="skip")
    return go.Scatter(x=xy[nodes, 0] if nodes else [], y=xy[nodes, 1] if nodes else [], mode="markers+text" if texts else "markers", text=texts,
                      textposition="middle center", textfont=dict(size=8, color="white"), marker=marker, name=name, showlegend=showlegend, hoverinfo="skip")


def _start_goal(inst, showlegend=True):
    xy = inst.graph.xy
    return [
        go.Scatter(x=[xy[inst.start, 0]], y=[xy[inst.start, 1]], mode="markers", marker=dict(size=16, symbol="star", color="#2ca02c", line=dict(width=1, color="white")), name="Start", showlegend=showlegend, hoverinfo="skip"),
        go.Scatter(x=[xy[inst.goal, 0]], y=[xy[inst.goal, 1]], mode="markers", marker=dict(size=16, symbol="star", color="#d62728", line=dict(width=1, color="white")), name="Ziel", showlegend=showlegend, hoverinfo="skip"),
    ]


def _map_axes(fig, height, hand):
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    if not hand:
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return _base(fig, height)


def build_instance(inst):
    fig = go.Figure()
    _panel_base(fig, inst, node_alpha=0.9)
    if _hand(inst):
        rest = [n for n in range(inst.graph.n) if n not in (inst.start, inst.goal)]
        fig.add_trace(_nodes_trace(inst, rest, "Knoten", NODE_COLOR, 16, showlegend=False, texts=[_label(inst, n) for n in rest]))
        for t in _start_goal(inst):
            fig.add_trace(t)
        return _map_axes(fig, 340, True)
    xy = inst.graph.xy
    fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers", marker=dict(size=6, color=NODE_COLOR, line=dict(width=1, color="white")), name="Offene Zellen"))
    for t in _start_goal(inst):
        fig.add_trace(t)
    return _map_axes(fig, 460, False)


def build_layer_maps(inst, beam_layers, dbs_layers, layer, n_groups):
    """Schicht `layer` (0 = Start) nebeneinander: links Beam Search Breite B (Strahl der Schicht), rechts Diverse Beam Search (Strahl je Gruppe in
    Gruppenfarbe, Zahl = Gruppe; von der Gruppe mit der größten Nummer nach innen gestapelte Kreise, damit von mehreren Gruppen gewählte Zustände
    als konzentrische Ringe erkennbar bleiben)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Beam Search (Gesamtbreite B)", "Diverse Beam Search (Farbe = Gruppe)"), horizontal_spacing=0.04)
    for col in (1, 2):
        _panel_base(fig, inst, 1, col)
    beam_nodes = list(beam_layers[layer][0]) if layer < len(beam_layers) else []
    earlier_b = sorted({n for beam, _d in beam_layers[:layer] for n in beam})
    if earlier_b:
        fig.add_trace(_nodes_trace(inst, earlier_b, "Früher expandiert", GREY, 8, showlegend=True), row=1, col=1)
    if beam_nodes:
        fig.add_trace(_nodes_trace(inst, beam_nodes, "Strahl (Beam)", BEAM_COLOR, 13), row=1, col=1)
    earlier_d = sorted({s for rec in dbs_layers[:layer] for grp in rec for s in grp["beam"]})
    if earlier_d:
        fig.add_trace(_nodes_trace(inst, earlier_d, "Früher expandiert", GREY, 8, showlegend=False), row=1, col=2)
    step = min(3.0, 14.0 / max(1, n_groups - 1))
    if layer < len(dbs_layers):
        for g in range(n_groups):
            nodes = dbs_layers[layer][g]["beam"]
            if nodes:
                size = 10 + step * (n_groups - 1 - g)
                texts = [f"{inst.labels[n]} ({g + 1})" if _hand(inst) else str(g + 1) for n in nodes]
                fig.add_trace(_nodes_trace(inst, nodes, f"Gruppe {g + 1}", group_color(g), size, texts=texts), row=1, col=2)
    for i, t in enumerate(_start_goal(inst, showlegend=True)):
        fig.add_trace(t, row=1, col=1)
    for t in _start_goal(inst, showlegend=False):
        fig.add_trace(t, row=1, col=2)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    if not _hand(inst):
        fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1), xaxis2=dict(scaleanchor="y2", scaleratio=1))
    _base(fig, 400 if _hand(inst) else 450, legend_y=-0.14)
    fig.update_layout(margin=dict(l=10, r=10, t=34, b=10))
    return fig


def build_alternatives(inst, group_paths, opt_path, beam_path):
    """Alle Gruppenlösungen in Gruppenfarben (Gruppe 1 am breitesten, damit gemeinsame Abschnitte sichtbar bleiben), darunter das Optimum,
    darüber gepunktet die Lösung von Beam Search mit der Gesamtbreite B."""
    fig = go.Figure()
    _panel_base(fig, inst)
    xy = inst.graph.xy
    if _hand(inst):
        rest = [n for n in range(inst.graph.n) if n not in (inst.start, inst.goal)]
        fig.add_trace(_nodes_trace(inst, rest, "Knoten", NODE_COLOR, 12, showlegend=False, texts=[_label(inst, n) for n in rest]))
    if opt_path:
        p = xy[opt_path]
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", line=dict(color="rgba(76,120,168,0.35)", width=15), name="Optimum (Uniform-Cost)"))
    n = len(group_paths)
    for g, path in enumerate(group_paths):
        if not path:
            continue
        p = xy[path]
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", line=dict(color=group_color(g), width=3 + 1.6 * (n - 1 - g)), name=f"Gruppe {g + 1}"))
    if beam_path:
        p = xy[beam_path]
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", line=dict(color=BEAM_COLOR, width=2, dash="dot"), name="Beam Search (Gesamtbreite B)"))
    for t in _start_goal(inst):
        fig.add_trace(t)
    return _map_axes(fig, 340 if _hand(inst) else 460, _hand(inst))


def build_lambda_profile(rows, n_groups):
    """Für EINE Instanz: Lücke jeder Gruppe über λ; gescheiterte Gruppen als x oberhalb der Kurven."""
    xs = [r["penalty"] for r in rows]
    finite = [g for r in rows for g in r["gaps"] if g is not None]
    top = (max(finite) if finite else 1.0) * 1.15 + 0.5
    fig = go.Figure()
    for g in range(n_groups):
        ok = [(x, r["gaps"][g]) for x, r in zip(xs, rows) if r["gaps"][g] is not None]
        if ok:
            fig.add_trace(go.Scatter(x=[str(x) for x, _ in ok], y=[y for _, y in ok], mode="lines+markers", line=dict(color=group_color(g), width=2.5), name=f"Gruppe {g + 1}"))
        bad = [str(x) for x, r in zip(xs, rows) if r["gaps"][g] is None]
        if bad:
            fig.add_trace(go.Scatter(x=bad, y=[top] * len(bad), mode="markers", marker=dict(size=10, symbol="x", color=group_color(g), line=dict(width=2, color=group_color(g))), name=f"Gruppe {g + 1}: gescheitert"))
    fig.update_xaxes(title_text="Diversitätsstärke λ (Kantenlängen)", type="category")
    fig.update_yaxes(title_text="Lücke zum Optimum (%)")
    return _base(fig, 340, legend_y=-0.35)


def build_sweep(rows, param_label, series, y_label, ref_line=None, ref_label=None):
    """`series` = [(key, Name, Farbe)]: Median als Linie, 10. bis 90. Perzentil als Band (`<key>_lo`/`<key>_hi`)."""
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    for key, name, color in series:
        ys = [r[key] for r in rows]
        lo = [r[f"{key}_lo"] for r in rows]
        hi = [r[f"{key}_hi"] for r in rows]
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], mode="lines", fill="toself", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.13)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=color, width=2.5), name=name))
    if ref_line is not None:
        fig.add_hline(y=ref_line, line=dict(color="#888", dash="dash", width=1.5), annotation_text=ref_label, annotation_position="top left")
    fig.update_xaxes(title_text=param_label, type="category")
    fig.update_yaxes(title_text=y_label)
    return _base(fig, 360, legend_y=-0.3)


def build_share_bars(rows, param_label):
    """Anteil der Läufe ohne Lösung: DBS (alle Gruppen), Beam Search Breite B, Beam Search Breite b', und Anteil gescheiterter Gruppen."""
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    for key, name, color in (("dbs_failed_share", "DBS (alle Gruppen ohne Pfad)", group_color(0)), ("groups_failed_share", "DBS: Anteil gescheiterter Gruppen", group_color(1)),
                             ("beam_b_failed_share", "Beam Search (Breite B)", BEAM_COLOR), ("beam_g_failed_share", "Beam Search (Breite b')", GREY)):
        fig.add_trace(go.Bar(x=xs, y=[r[key] for r in rows], name=name, marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text=param_label, type="category")
    fig.update_yaxes(title_text="Anteil (%)", range=[0, 100])
    return _base(fig, 340, legend_y=-0.4)


def build_verdict_bars(rows, param_label):
    """Beste Gruppenlösung gegen Beam Search gleicher Gesamtbreite: Anteil besser / gleich / schlechter (nur Instanzen, in denen beide einen Pfad fanden)."""
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    for key, name, color in (("besser", "DBS besser", "#54a24b"), ("gleich", "gleich", "#bab0ac"), ("schlechter", "DBS schlechter", "#e45756")):
        fig.add_trace(go.Bar(x=xs, y=[100.0 * r[f"verdict_{key}"] / r["n_compared"] if r["n_compared"] else 0 for r in rows], name=name, marker_color=color))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text=param_label, type="category")
    fig.update_yaxes(title_text="Anteil der verglichenen Instanzen (%)", range=[0, 100])
    return _base(fig, 340, legend_y=-0.35)
