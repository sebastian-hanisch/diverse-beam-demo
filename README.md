# Diverse Beam Search – bessere Lösung oder nur Alternativen? – Streamlit-Demo

Siebtes Stück der **Heuristische-Baumsuche-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning" - die zweite Fortsetzung von [beam-search-demo](../beam-search-demo): Beam Search liefert **eine** Lösung, und die B Plätze seines Strahls laufen oft auf dieselbe Route zu. **Diverse Beam Search** (Vijayakumar, Cogswell, Selvaraju, Sun, Lee, Crandall & Batra, AAAI 2018) teilt das Strahlbudget B in **G Gruppen** zu je b' = B / G Plätzen. Die Gruppen laufen Schicht für Schicht nacheinander; die Bewertung eines Kandidaten wird um **λ × (Zahl früherer Gruppen, die diesen Zustand in dieser Schicht schon gewählt haben)** verschlechtert. Jede Gruppe liefert ihre **eigene Route**; Gruppe 1 wird nie bestraft - sie ist ein gewöhnlicher Beam Search der Breite b'.

**Einordnung in die Linie:** derselbe Graph, dieselben Instanzen und dieselben 50 festen Vergleichsinstanzen wie in [beam-search-demo](../beam-search-demo) und [monobeam-demo](../monobeam-demo) (das dortige `beam_search` ist wortgleich kopiert und reproduziert dessen Zahlen, als Test hinterlegt); Beam Search mit f-Rang, A\* und Uniform-Cost dienen als Vergleichsgrößen. Neu ist `diverse_beam_search` mit schichtweise gestaffelten Gruppen und Hamming-Diversitätsstrafe, dazu eine handgebaute Zwei-Korridore-Falle.

```
Greedy Best-First Search (Wurzel)                                                          [gebaut]
 ├─ Beam Search → {Diverse Beam Search, Monobeam}   [Beam Search gebaut, Monobeam gebaut, Diverse Beam Search = DIESES STÜCK]
 ├─ A* → Iterative Deepening A* (IDA*)                                                     [gebaut]
 └─ Monte Carlo Tree Search (MCTS)                                                         [nicht gebaut]
Beam Search + A* → Beam Stack Search (Konvergenzpunkt)                                     [gebaut]
```

Ergebnis in Kürze: Die Vorab-Hypothese des Papers "**bessere Lösung bei geringem Zusatzaufwand**" hält auf dem Raster **nicht**, die zweite - "**Vielfalt**" - schon, mit Preis. Die beste Gruppenlösung war in **keiner** von 50 Instanzen besser als Beam Search **gleicher Gesamtbreite** (27 gleich, 23 schlechter; Optimal-Anteil **54 %** gegen **100 %**), und über alle 28 Einstellungen der fünf Sweeps kam "besser" kein einziges Mal vor. Was DBS liefert, sind **Alternativen**: drei verschiedene Routen mit Überlappung 0.37, während Beam Search mit den Breiten 2/4/6 im Median nur 2 Beinahe-Duplikate (Überlappung 0.91) erzeugt - bei **1.26x** den Expansionen von Beam Search gleicher Gesamtbreite. Die Vielfalt hat aber ein enges Fenster: nur bei λ = 0.5 sind alle drei Routen im Median höchstens 10 % länger als das Optimum.

| Frage | Ergebnis (Rastergröße 12, Hindernisdichte 15 %, G = 3 Gruppen zu b' = 2 Plätzen, Gesamtbreite B = 6, λ = 0.5 Kantenlängen, sofern nicht anders angegeben; **Median** über 50 feste Instanzen, Seeds 200000–200049; vollständig deterministisch; Lücke nur über gelöste Läufe) |
|---|---|
| **Bessere Top-1-Lösung als Beam Search gleicher Gesamtbreite?** | ❌ nein: besser/gleich/schlechter **0/27/23** von 50, Optimal-Anteil DBS 54 %, Beam(6) 100 %; "besser" kommt in **keiner** der 28 Sweep-Einstellungen vor |
| **... als Beam Search der Gruppenbreite?** | ✅ ja, aber trivial: Gruppe 1 IST Beam(b') (im Test exakt geprüft, für jedes λ); Beam(2) hat Lücke 0.27 % und scheitert in 2 % der Instanzen, DBS best-of-3 hat 0.00 % und scheitert nie |
| **Wie vielfältig sind die Alternativen? (λ)** | Überlappung λ = 0/0.25/0.5/1/2/4/8: **1.00/0.60/0.37/0.25/0.24/0.25/0.26**; verschiedene Routen 1/2/3/3/3/3/3 - das Paper nennt DBS robust gegen λ, hier gilt das ab λ = 1 (flache Überlappung) |
| **... wie viele brauchbar?** | ⚠️ verschiedene Routen mit Lücke ≤ 10 %: **1/2/3/2/1/1/1** - nur bei λ = 0.5 alle drei, ab λ = 2 nur noch eine |
| **... was kostet sie an Erfolg?** | ⚠️ Anteil gescheiterter **Gruppen** 2.0/5.3/6.7/7.3/9.3/12.7/14.0 % (λ wie oben): die Strafe treibt Gruppen in Sackgassen |
| **Klumpt sich der Strahl?** | ⚠️ **räumlich nicht**: Streuung der Strahlknoten (in Kantenlängen) bei Beam(6) **3.27**, bei DBS 0.86 bis 2.32. Redundant sind die **Strahlpfade** (gemeinsame Präfixe): Überlappung 0.53 bei Beam(6), 0.35 bis 0.37 bei DBS mit λ ≥ 2 (0.90 bei λ = 0) |
| **Alternativen ohne Strafe?** | Beam mit den Breiten 2/4/6: im Median **2** verschiedene Routen, Überlappung **0.91** - Beinahe-Duplikate; DBS bei λ = 0.5: 3 Routen, 0.37 |
| **Aufwand** | Expansionen DBS / Beam(6) **1.26**; bei G = 2/3/4/6/8: 1.12/1.26/1.47/2.12/2.85 - jede Gruppe expandiert ihren eigenen Strahl |
| **Mehr Routen (G)** | verschiedene Routen 1/2/3/4/5/7, brauchbare 1/2/3/3/4/5 für G = 1/2/3/4/6/8 |
| **Breite je Gruppe (b')** | ⚠️ b' = 1 (G = B, drei gierige Läufe mit Strafe): Lücke der besten Gruppe **4.28 %** (Beam(3): 0.00 %), 28 % der Gruppen und 10 % der Instanzen ohne Lösung, DBS schlechter in **42 von 45**; b' = 3: Überlappung 0.70, 86 % optimal; **b' = 4: Überlappung 1.00, alle Gruppen laufen auf dieselbe Route zusammen**, Aufwand 1.87x |
| **Hindernisse (0/10/20/30/40 %)** | ⚠️ Instanzen ohne Lösung: DBS 0/0/2/12/16 %, Beam(6) 0/0/0/2/2 %, Beam(2) 0/0/6/20/48 %; Überlappung 0.25/0.30/0.49/0.78/0.84 (die Wege verengen sich, die Alternativen fallen zusammen); nur bei 40 % **eine** Instanz, an der DBS eine Lösung fand und Beam(6) keine, aber 8 mit umgekehrtem Ergebnis |
| **Größe (8-16)** | beste Gruppe schlechter als Beam(6) in 10/11/23/29/33 von 50 Instanzen, Lücke 0/0/0/0.12/0.53 %, Aufwand 1.53/1.36/1.26/1.22/1.18x |
| **Handgebaute Zwei-Korridore-Falle** | Optimum 8; Beam Breite 1/2/3 = **9/11/8**; DBS mit G = 2, b' = 1, λ = 1: Gruppen **9 / 8**; bei λ = 0 und 0.25 wird die zweite Gruppe wieder 9 |

## Was die Demo zeigt

1. **Diverse Beam Search in Aktion** (Schritt-Slider): **Instanz** → **Strahl je Schicht** (Schicht-Slider; links Beam Search der Gesamtbreite B, rechts die Gruppen in Gruppenfarbe, von mehreren Gruppen gewählte Zustände als konzentrische Ringe; darunter je Gruppe: Kandidaten und wie viele davon keine frühere Gruppe gewählt hat) → **Alternativen** (alle Gruppenrouten in Gruppenfarben über dem Optimum, Beam Search gepunktet; Tabelle mit Kosten, Lücke, Überlappung mit Gruppe 1).
2. **Vergleich bei Gesamtbreite B:** Lücke der besten Gruppe, Lücke von Beam Search, Vergleich (besser / gleich / schlechter), Überlappung, verschiedene und brauchbare Routen.
3. **🎚️ λ-Profil der Instanz:** die Lücke jeder Gruppe über alle λ-Stufen.
4. **📐 Sweeps** über λ, Gruppen, Breite je Gruppe, Hindernisdichte und Rastergröße (Median, 10.–90. Perzentil-Band): Lücke, Überlappung, Routen, Scheitern, besser / gleich / schlechter, Aufwand; 50 feste Instanzen ab Seed 200000.
5. **🔬 Vergleich über 50 Instanzen** bei den aktuellen Einstellungen: besser / gleich / schlechter, wer allein eine Lösung findet, Optimal-Anteil.
6. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an".

Regler: Instanz (Raster / **Zwei-Korridore-Falle**), Rastergröße (5–25), Hindernisdichte (0–40 %), Seed der Instanz (+ 🎲), **Gruppen G** (1–8), **Plätze je Gruppe b'** (1–6; die Gesamtbreite B = G × b' wird angezeigt), **Diversitätsstärke λ** (0–8 Kantenlängen). Kein Zufall im Kern, kein Ketten-Seed, kein Bewertungsbudget - vollständig deterministisch.

## Messwerte der Presets

| Preset | Instanz | Ergebnis |
|---|---|---|
| Standardfall (Voreinstellung) | Seed 44, G = 3, b' = 2, λ = 0.5 | drei Routen mit **0.0 / 3.3 / 4.1 %** Lücke, Überlappung 0.27; Beam mit den Breiten 2/4/6 findet dieselbe eine Route; 129 gegen 108 Expansionen (A\*: 94) |
| Zwei-Korridore-Falle | handgebaut, G = 2, b' = 1, λ = 1 | Beam(2) 11, DBS 8 (Optimum 8) |
| Keine Strafe (λ = 0) | Seed 44 | drei identische Routen (Überlappung 1.0) |
| Zu starke Strafe (λ = 4) | Seed 200011 | Gruppen 0.0 % / gescheitert / 17.4 % |
| Gruppenbreite 1 (G = B) | Seed 35, G = 4, λ = 1 | 1.0 / gescheitert / 21.8 / 24.3 %; Beam(4) optimal |
| Größere Gruppen (b' = 4) | Seed 44 | alle Gruppen dieselbe Route; 241 gegen 125 Expansionen |
| Beam(2) scheitert, DBS findet | Seed 200046 | Gruppen gescheitert / gescheitert / 2.0 %; Beam(6) optimal |
| Selbst Beam(6) scheitert | Seed 200002, 40 % Hindernisse | Gruppen gescheitert / gescheitert / 0.0 %; Beam(6) und Beam(2) ohne Pfad |
| Viele Alternativen (G = 8) | Seed 44 | 8 verschiedene Routen mit 0 bis 12 % Lücke, 5 davon unter 10 %; 351 gegen 125 Expansionen |

Die einzelne Instanz weicht von den Medianen ab - die Mediane oben sind die belastbaren Zahlen; die Raster-Presets prüfen sich zusätzlich über die 50 festen Instanzen gegen eine gemessene Spannweite des Medians der Lücke, die Aussagen der Presets sind als eigene Tests hinterlegt (`tests/test_presets.py`).

## Modell und Verfahren

- **Instanz und Graph** (`dbs_scenario.py`, `dbs_graph.py`): das gestörte Raster mit Hindernissen aus [beam-search-demo](../beam-search-demo) (Kantengewicht = echter euklidischer Abstand); statt der dortigen Sackgassen-Falle eine **handgebaute Zwei-Korridore-Falle** (9 Knoten) mit expliziter, zulässiger Heuristik, per Skriptsuche über Gewichte und h konstruiert, dann festgeschrieben und gegen Uniform-Cost auf Zulässigkeit geprüft: die Heuristik lockt in den oberen Korridor, Beam(2) verliert in Schicht 2 den Einstieg in den unteren.
- **Suchkern** (`dbs_algorithm.py`): `_search`, `greedy_best_first`, `uniform_cost_search`, `a_star`, `beam_search` aus der Beam-Search-Demo; NEU `diverse_beam_search(graph, start, goal, groups, group_width, penalty, h)`: je Schicht wird jede Gruppe der Reihe nach fortgesetzt (Kandidaten = Nachfolger des eigenen Strahls, je Zustand kleinstes g, eigene schon expandierte Zustände ausgeschlossen; Schlüssel = g + h + λ × mittlere Kantenlänge × Zahl früherer Gruppen, die den Zustand in dieser Schicht gewählt haben; die b' besten bleiben, Tie-Break Zustandsindex). Eine Gruppe endet, wenn ihre Kandidaten das Ziel enthalten (kleinstes g) oder leer sind, und übt danach keine Strafe mehr aus.
- **Auswertung** (`dbs_evaluation.py`): Lücke ggü. Uniform-Cost nur über gelöste Läufe, daneben stets Scheiter-Quote (alle Gruppen / Anteil Gruppen) und Optimal-Anteil; Paarvergleich besser / gleich / schlechter gegen Beam(B); **Überlappung** = mittleres Jaccard der inneren Knotenmengen der gelösten Gruppenlösungen; **brauchbar** = verschiedene Pfade mit Lücke ≤ 10 %; **Strahllisten-Redundanz** und **räumliche Streuung** der Strahlknoten je Schicht; Aufwand als Expansionen DBS / Beam(B).

## Was nicht funktioniert hat / Grenzen

- **Vorab-Hypothese "bessere Top-1-Lösung bei geringem Zusatzaufwand" - WIDERLEGT für gleiche Gesamtbreite.** Beam Search der Breite 6 war in allen 50 Instanzen mindestens so gut und in 23 besser; nur gegenüber Beam Search der Gruppenbreite (= Gruppe 1) stimmt die Garantie, und sie folgt direkt aus der Konstruktion. Der Aufwand von 1.26x ist außerdem nicht "gering" gegenüber einem Beam Search, der dasselbe Budget für einen Strahl nutzt.
- **Vorab-Hypothese "Strahlplätze klumpen im Raum" - NICHT gestützt.** Beam(6) streut räumlich mehr (3.27 Kantenlängen) als DBS (0.86 bis 2.32); die Redundanz zeigt sich in gemeinsamen Pfad-Präfixen (Überlappung 0.53), nicht in räumlicher Nähe. Warum, wird nicht isoliert untersucht.
- **Vielfalt ist nicht automatisch brauchbar.** Die Überlappung flacht ab λ = 1 bei etwa 0.25 ab, aber die zusätzlichen Routen werden mit wachsendem λ schlechter (brauchbar 3/2/1/1/1 bei λ = 0.5/1/2/4/8) und die Gruppen scheitern häufiger (bis 14 %). Bei 40 % Hindernissen scheitert DBS in 16 % der Instanzen, Beam(6) in 2 %.
- **Breitere Gruppen sind nicht vielfältiger.** Bei b' = 4 laufen alle drei Gruppen trotz Strafe auf dieselbe Route zusammen (Überlappung 1.0); bei b' = 1 sind die Gruppen gierige Läufe, die oft scheitern.
- **Nicht gebaut:** n-Gramm- und Einbettungs-Vielfaltsmaße (nur Hamming auf Zuständen derselben Schicht), Stochastic Beam Search, Vergleich mit einer gezielten k-kürzesten-Pfade-Suche (Yen) als Alternativen-Erzeuger, Verwendung in einem neuronalen Decoder (das Paper zielt auf Sequenzmodelle; hier ist die Suche die klassische heuristische).
- **Abweichung von der Konvention:** 50 statt 5 feste Instanzen - Anteile und Überlappungen brauchen mehr Instanzen; die Instanzen sind dieselben 50 wie in den Geschwistern.
- **Synthetische Instanzen:** ein Raster mit Jitter, Vierer-Nachbarschaft, keine gerichteten Kanten, dazu eine handgebaute Falle. Andere Graphstrukturen wurden nicht gemessen.

## Verifikation

- **Gültiger Pfad je Gruppe** (zusammenhängend, Start bis Ziel, ohne Wiederholung, Kosten gegen unabhängige Neuberechnung); **Kosten nie unter dem Optimum** (gegen vollständige Enumeration auf kleinen Instanzen); Determinismus; ein nicht erreichbares Ziel scheitert für alle Gruppen.
- **Paper-Garantien direkt:** G = 1 == `beam_search` (Pfad, Kosten, Expansionen); **Gruppe 1 == Beam Search der Breite b' für jedes λ**; **λ = 0 ⇒ alle Gruppen identisch**; beste Gruppe nie schlechter als Gruppe 1.
- **Strafsemantik direkt** an den Schicht-Aufzeichnungen: bei riesigem λ überlappt eine Gruppe nur so weit, wie zu wenige "frische" Kandidaten (von keiner früheren Gruppe gewählt) existieren.
- **Buchführung:** je Gruppe und Schicht höchstens b' Zustände, Expansionen == Summe der Strahlgrößen, fertige Gruppen stoppen, Ziel-Schicht == Pfadkanten, Strahlpfade der letzten Schicht gültig; **Kopie treu:** `beam_search` reproduziert die Zahlen aus beam-search-demo.
- **Diversitätsmaße einzeln** (identische Pfade → 1, disjunkte → 0, Streuung eines Knotens → NaN) und die **Zwei-Korridore-Falle** (Beam 9/11/8, DBS 9/8, λ-Schwelle, zulässige Heuristik).
- **Alle Zahlen der App-Texte sind als Tests hinterlegt**, über dieselben Auswertungsfunktionen wie die App selbst (`ev.run_config`/`ev.sweep`), NIE über ein Ad-hoc-Skript; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt, jede Schicht, beide Instanz-Typen, gescheiterte Gruppen in jedem Schritt, Extremwerte, Würfel, Permalink-Grenzen inkl. Gruppen/Breite/λ, Instanzwechsel, Sweeps und Vergleich auf Abruf, Footer). 1264 Tests.

Literatur: Vijayakumar, A. K., Cogswell, M., Selvaraju, R. R., Sun, Q., Lee, S., Crandall, D., & Batra, D. (2018). *Diverse Beam Search for Improved Description of Complex Scenes.* Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), 7371-7379 (Vorabdruck: *Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models*, arXiv:1610.02424).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Instanz-Umschalter, Schritte (mit Schicht-Slider), Vergleich, λ-Profil, 📐 Sweeps, 🔬 Vergleich, 🚧 Grenzen, Mathe |
| `dbs_algorithm.py` | Suchkerne und `beam_search` (Kopie aus der Beam-Search-Demo) + `diverse_beam_search` |
| `dbs_graph.py`, `dbs_scenario.py` | Graph, Rasterinstanz (Kopie) und die handgebaute Zwei-Korridore-Falle |
| `dbs_constants.py` | Konstanten, Presets, gemessene Werte |
| `dbs_evaluation.py` | Lücke, Scheiter-Quote, Paarvergleich, Diversitätsmaße, Sweeps, λ-Profil |
| `dbs_presets.py`, `dbs_visualization.py` | Permalink/Presets, Plotly-Figuren (Alternativen-Karte, Schicht-Karten, λ-Profil, Sweeps, Balken) |
| `tests/` | Zentrale Korrektheitskette (Paper-Garantien, Strafsemantik), Szenario/Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
