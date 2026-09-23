"""Konstanten der Diverse-Beam-Search-Demo: Raster-Geometrie (wortgleich zur Monobeam-Demo), Regler, gemessene Werte, Presets."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
JITTER = 0.35                    # Lageabweichung je Zelle, Anteil des Zellenabstands

SIDE_MIN, SIDE_MAX, DEFAULT_SIDE, SIDE_STEP = 5, 25, 12, 1     # Rastergröße (Zellen je Kante)
OBSTACLE_MIN, OBSTACLE_MAX, DEFAULT_OBSTACLE, OBSTACLE_STEP = 0, 40, 15, 5   # Prozent gesperrte Zellen
SEED_MAX = 999999
DEFAULT_SEED = 44

GROUPS_OPTIONS = (1, 2, 3, 4, 6, 8)            # Gruppen G
GROUP_WIDTHS = (1, 2, 3, 4, 6)                 # Plätze je Gruppe b'; Gesamtbreite B = G * b'
DEFAULT_GROUPS = 3                             # MUSS Mitglied sein (st.select_slider snappt sonst still)
DEFAULT_GROUP_WIDTH = 2
PENALTIES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)   # Diversitätsstärke λ in Kantenlängen
DEFAULT_PENALTY = 0.5
NETWORKS = ("grid", "corridor")
NETWORK_LABELS = {"grid": "Raster", "corridor": "Zwei-Korridore-Falle (handgebaut)"}

OBSTACLE_SWEEP = (0, 10, 20, 30, 40)
SCALING_SIDES = (8, 10, 12, 14, 16)
POP_SEEDS = tuple(range(200000, 200050))       # 50 feste Instanzen (Abweichung von der 5-Seed-Konvention: Anteile brauchen mehr Instanzen)

# --- Gemessene Werte (MEDIAN über 50 feste Instanzen, Seeds 200000-200049; Rastergröße 12, Hindernisdichte 15 %, G = 3 Gruppen zu
# --- b' = 2 Plätzen (Gesamtbreite B = 6), λ = 0.5 Kantenlängen; 2026-09-23, alle Werte über ev.run_config/ev.sweep nachgerechnet,
# --- s. tests/test_claims.py). Vergleich immer gegen Beam Search f-Rang MIT DER GLEICHEN GESAMTBREITE B (eine Lösung) und gegen
# --- Beam Search der Breite b' (= Gruppe 1, die nie eine Strafe bekommt). Lücke nur über GELÖSTE Läufe. ---
# ZENTRALE FRAGE 1 - bringt DBS eine bessere Top-1-Lösung als Beam Search gleicher Gesamtbreite? NEIN. Die beste Gruppenlösung war in
#   keiner von 50 Instanzen besser als Beam Search der Breite 6 (27 gleich, 23 schlechter), und über alle 28 Einstellungen der
#   fünf Sweeps (λ, G, b', Hindernisse, Größe) kam "besser" kein einziges Mal vor; Optimal-Anteil DBS 54 %, Beam(6) 100 %.
#   Gegen die GRUPPENBREITE gerechnet stimmt die Paper-Aussage: Gruppe 1 IST Beam(b'), die beste Gruppe ist nie schlechter (im Test
#   exakt geprüft); Beam(2) hat Lücke 0.27 % und scheitert in 2 % der Instanzen, DBS best-of-3 hat Lücke 0.00 % und scheitert nie.
# ZENTRALE FRAGE 2 - wie vielfältig und wie gut sind die G Alternativen? Überlappung (Jaccard der Knotenmengen ohne Start/Ziel)
#   λ = 0/0.25/0.5/1/2/4/8: 1.00/0.60/0.37/0.25/0.24/0.25/0.26 - fällt bis λ = 1 und bleibt dann flach (das Paper nennt DBS robust
#   gegen λ; hier gilt das ab λ = 1). Verschiedene Routen 1/2/3/3/3/3/3; BRAUCHBARE verschiedene Routen (Lücke <= 10 %)
#   1/2/3/2/1/1/1: nur bei λ = 0.5 sind alle drei Alternativen brauchbar, ab λ = 2 nur noch eine - die übrigen sind mehr als 10 %
#   schlechter. Anteil gescheiterter GRUPPEN 2.0/5.3/6.7/7.3/9.3/12.7/14.0 %: Vielfalt kostet Erfolgsquote, weil eine Gruppe,
#   der die frei gewählten Zustände weggestraft werden, in eine Sackgasse laufen kann.
# ZENTRALE FRAGE 3 - klumpt sich der Strahl? Die Strahllisten-Pfade (Strahlknoten der Schicht vor dem Zielfund) überlappen bei einem
#   einzelnen Beam(6) mit 0.53, bei DBS mit 0.90/0.59/0.45/0.36/0.35/0.35/0.37 (λ wie oben): ab λ >= 1 sinkt die Redundanz auf 0.35-0.36.
#   Die RÄUMLICHE Streuung der Strahlknoten (in Kantenlängen) ist aber bei Beam(6) mit 3.27 GRÖSSER als bei DBS (0.86 bis 2.32 über
#   λ) - die Vorab-Vermutung "Strahlplätze klumpen im Raum" wird hier NICHT gestützt; die Redundanz zeigt sich in gemeinsamen
#   Pfad-Präfixen, nicht in räumlicher Nähe. Alternativen OHNE Strafe: Beam mit den Breiten 2/4/6 liefert im Median 2 verschiedene
#   Routen mit Überlappung 0.91 (Beinahe-Duplikate); DBS bei λ = 0.5: 3 Routen, Überlappung 0.37.
# ZENTRALE FRAGE 4 - Aufwand: Expansionen DBS / Beam(6) = 1.26 (Median) bei G = 3, wächst mit G auf 1.12/1.26/1.47/2.12/2.85
#   für G = 2/3/4/6/8 - jede Gruppe expandiert ihren eigenen Strahl, es gibt nichts zu teilen. Beispiel Seed 44: 129 gegen 108
#   Expansionen (A*: 94).
# SENSITIVITÄT b' (G = 3, λ = 0.5): b' = 1 (G = B, drei gierige Läufe mit Strafe): Lücke der besten Gruppe 4.28 % (Beam(3) 0.00 %),
#   28 % der Gruppen und 10 % der Instanzen ohne Lösung, DBS schlechter als Beam(3) in 42 von 45; b' = 3: 43 gleich / 7 schlechter,
#   Überlappung 0.70, 86 % optimal; b' = 4: Überlappung 1.00 (alle Gruppen laufen auf DIESELBE Route zusammen, trotz Strafe),
#   Aufwand 1.87x. Größere Gruppen sind also nicht vielfältiger, sondern konvergieren.
# SENSITIVITÄT G (λ = 0.5, b' = 2): verschiedene Routen 1/2/3/4/5/7 für G = 1/2/3/4/6/8, brauchbare 1/2/3/3/4/5, Überlappung
#   nan/0.56/0.37/0.36/0.31/0.30.
# HINDERNISSE (0/10/20/30/40 %): Anteil der Instanzen ohne Lösung DBS 0/0/2/12/16 % gegen Beam(6) 0/0/0/2/2 % und Beam(2) 0/0/6/20/48 %;
#   Überlappung 0.25/0.30/0.49/0.78/0.84 (die Wege verengen sich, die Alternativen fallen zusammen). Nur bei 40 % gab es EINE Instanz
#   (Seed 200002), an der DBS eine Lösung fand und Beam(6) keine, aber 8 Instanzen mit umgekehrtem Ergebnis.
# GRÖSSE (8-16): beste Gruppe schlechter als Beam(6) in 10/11/23/29/33 von 50 Instanzen, Lücke 0/0/0/0.12/0.53 %, Aufwand 1.53/1.36/
#   1.26/1.22/1.18x.
# HANDGEBAUTE ZWEI-KORRIDORE-FALLE (9 Knoten, explizite zulässige Heuristik): Optimum 8; Beam Breite 1 findet 9, Breite 2 sogar 11 (in Schicht 2
#   behalten D und C die beiden Plätze, der Einstieg E in den unteren Korridor fliegt aus dem Strahl); DBS mit G = 2, b' = 1, λ = 1
#   findet mit Gruppe 2 die 8, Gruppe 1 (= Beam(1)) die 9. Bei λ = 0 und λ = 0.25 wird die zweite Gruppe wieder 9.

PRESETS = {
    "Standardfall (Voreinstellung)": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 44, "groups": 3, "group_width": 2, "penalty": 0.5},
    "Zwei-Korridore-Falle": {"network": "corridor", "side": 12, "obstacle_pct": 15, "seed": 44, "groups": 2, "group_width": 1, "penalty": 1.0},
    "Keine Strafe (λ = 0)": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 44, "groups": 3, "group_width": 2, "penalty": 0.0},
    "Zu starke Strafe (λ = 4)": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 200011, "groups": 3, "group_width": 2, "penalty": 4.0},
    "Gruppenbreite 1 (G = B)": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 35, "groups": 4, "group_width": 1, "penalty": 1.0},
    "Größere Gruppen (b' = 4)": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 44, "groups": 3, "group_width": 4, "penalty": 0.5},
    "Beam(2) scheitert, DBS findet": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 200046, "groups": 3, "group_width": 2, "penalty": 0.5},
    "Selbst Beam(6) scheitert": {"network": "grid", "side": 12, "obstacle_pct": 40, "seed": 200002, "groups": 3, "group_width": 2, "penalty": 0.5},
    "Viele Alternativen (G = 8)": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 44, "groups": 8, "group_width": 2, "penalty": 0.5},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Rastergröße 12, 15 % Hindernisse, G = 3 Gruppen zu 2 Plätzen, λ = 0.5. Seed 44: drei verschiedene Routen mit 0.0 / 3.3 / 4.1 % Lücke (Überlappung 0.27), während Beam Search mit den Breiten 2, 4 und 6 dieselbe eine Route findet. Beam(6) ist hier auch optimal - DBS liefert Alternativen, kein besseres Top-1.",
    "Zwei-Korridore-Falle": "Handgebauter Graph mit expliziter Heuristik: Beam Search Breite 2 findet nur 11 (Optimum 8), weil der Einstieg in den unteren Korridor aus dem Strahl fliegt. DBS mit zwei Gruppen zu einem Platz bestraft der zweiten Gruppe den von der ersten gewählten Zustand, sie nimmt den anderen Korridor und findet 8.",
    "Keine Strafe (λ = 0)": "Seed 44 ohne Diversitätsstrafe: alle drei Gruppen laufen denselben Beam Search und liefern dieselbe Route (Überlappung 1.0). Das Strahlbudget ist dreifach ausgegeben, gewonnen ist nichts.",
    "Zu starke Strafe (λ = 4)": "Seed 200011: Gruppe 1 findet den optimalen Pfad, Gruppe 2 findet gar keinen (die Strafe treibt sie in eine Sackgasse), Gruppe 3 einen mit 17.4 % Lücke. Vielfalt kostet Erfolgsquote und Qualität.",
    "Gruppenbreite 1 (G = B)": "Seed 35, vier Gruppen zu einem Platz (G = B = 4), λ = 1: Gruppen 1/3/4 liegen 1.0 / 21.8 / 24.3 % über dem Optimum, Gruppe 2 scheitert. Beam(4) findet hier den optimalen Pfad. Vier gierige Läufe mit Strafe erkunden viel, beuten wenig aus.",
    "Größere Gruppen (b' = 4)": "Seed 44 mit vier Plätzen je Gruppe: trotz Strafe finden alle drei Gruppen dieselbe Route (Überlappung 1.0) - je breiter eine Gruppe, desto weniger kann die Strafe sie umlenken. Aufwand 241 gegen 125 Expansionen bei Beam(12).",
    "Beam(2) scheitert, DBS findet": "Seed 200046: Beam Search der Breite 2 (= Gruppe 1) findet keinen Pfad, ebenso Gruppe 2 - Gruppe 3 findet einen mit 2.0 % Lücke. Beam(6) findet hier allerdings den optimalen Pfad.",
    "Selbst Beam(6) scheitert": "Seed 200002 mit 40 % Hindernissen: Beam Search der Breite 6 findet keinen Pfad, DBS mit 3 x 2 Plätzen findet mit Gruppe 3 den optimalen. Das einzige solche Beispiel unter 50 Instanzen; umgekehrt scheitert DBS in 8.",
    "Viele Alternativen (G = 8)": "Seed 44 mit acht Gruppen zu zwei Plätzen: acht verschiedene Routen mit Lücken von 0 bis 12 %, fünf davon unter 10 %. Aufwand 351 gegen 125 Expansionen bei Beam(16): jede Gruppe kostet ihren eigenen Strahl.",
}
# Beobachtete Spannweite des MEDIANS der Lücke der besten Gruppenlösung (%) über die 50 festen Instanzen (mit Sicherheitsabstand),
# nur für die Raster-Presets (die handgebaute Instanz ist ein fester Graph, ihre Zahlen stehen als Tests). Auf den Seed des Presets
# selbst wirken die Bänder nicht - sie gelten für die Einstellung, gemittelt über die Instanzen.
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": (0.0, 0.8),
    "Keine Strafe (λ = 0)": (0.0, 1.0),
    "Zu starke Strafe (λ = 4)": (0.0, 1.0),
    "Gruppenbreite 1 (G = B)": (2.5, 7.5),
    "Größere Gruppen (b' = 4)": (0.0, 0.3),
    "Selbst Beam(6) scheitert": (0.0, 0.8),
}
