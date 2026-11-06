# nachtlernen

Ein lokales Sprachmodell mit Gedächtnis, das nachts aus seinen eigenen Erinnerungen lernt.

Am Tag lebt das Modell in einer kleinen Textwelt und spricht mit Menschen. Alles, was es erlebt,
landet in einem gewichteten Gedächtnis. Nachts wählt es selbst aus, welche Erinnerungen fest zu ihm
gehören sollen. Daraus werden mehrere Varianten eines LoRA-Adapters trainiert, von vorsichtig bis
mutig. Ein Prüfstand misst Kohärenz und Innovation, und nur die innovativste Variante, die kohärent
bleibt, ersetzt den bisherigen Adapter. Das Grundmodell bleibt dabei unverändert; die Kette der
Adapter ist die "DNA" des Wesens.

Ziel ist, herauszufinden, ob ein solches System über Monate stabil bleibt und sich erkennbar von
einer Kopie unterscheidet, die dasselbe Gedächtnis hat, aber nie trainiert wurde.

**Stand:** Minimalfassung. Die ganze Schleife läuft und ist ohne Grafikkarte getestet (simuliertes
Modell, Trockentraining). Das echte Training auf der Grafikkarte ist geschrieben, aber noch nicht
auf Hardware gelaufen.

## Ausprobieren ohne Grafikkarte

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest
nachtlernen probelauf --tage 21 --ziel probe
nachtlernen --ordner probe status
cat probe/naechte/2027-07-09/bericht.md
```

Der Probelauf lässt ein simuliertes Modell drei Wochen lang in der Textwelt leben und jede Nacht
lernen. Das prüft die Mechanik (Auswahl, Mischung, Auslese, 1/5-Regel, Notbremse), nicht das
Lernen selbst.

## Aufbau

| Datei | Aufgabe | Abschnitt im Bauplan |
|---|---|---|
| `gedaechtnis.py` | Ereignisprotokoll in SQLite, Gewicht aus Wichtigkeit, Überraschung, Wiederholung; Abruf nach Ähnlichkeit mal Gewicht mal Aktualität; Verdichtung pro Tag; versioniertes Selbstbild und Ziele | Architektur, Phase 1 |
| `wesen.py` | Das wache Modell: spricht, handelt in der Welt, bewertet und speichert jedes Erlebnis | Architektur |
| `welt.py` | Textwelt (kleine Station mit Räumen, Dingen, Drohne, Aufgaben, auch unlösbaren). Jede Änderung wird als Fakt mit Antwort protokolliert | Sandbox-Spielwelt |
| `schlaf.py` | Auswahl der Erinnerungen durch das Modell (Obergrenze 200), Frage-Antwort-Paare und Ich-Erzählungen, Mischung mit höchstens 50 % Neuem | Phase 2 |
| `training.py` | Variantenfächer (Lernrate und Anteil Neues je nach Mut und sigma), QLoRA mit PEFT und TRL in eigenem Prozess pro Variante | Phase 2 und 3 |
| `pruefung.py` | Kohärenz K aus Erinnerung E, Selbstbild S, Widerspruchsfreiheit W, minus Drift D; Ankertest A; Innovation N aus Neuheit und Lernfortschritt | Prüfkriterien |
| `auslese.py` | Auslese, 1/5-Regel für sigma, Notbremse, Stammbaum aller Varianten | Grenze zwischen Kohärenz und Innovation |
| `nacht.py` | Die Nachtschleife von Selbstbild bis Nachtbericht | Phase 2 und 3 |
| `leitung.py` | Serielle Textleitung zwischen Gateway und abgekoppeltem KI-Rechner, nur Textzeilen, mit Protokoll | Chat-Zugang ohne Internet |
| `simulation.py` | Simuliertes Modell für Probeläufe und Tests | – |
| `cli.py` | Kommandozeile | – |

## Auf dem KI-Rechner

Voraussetzungen: Linux, CUDA, eine oder zwei RTX 3090, die Gewichte des Grundmodells und eines
Einbettungsmodells lokal auf der Platte (der Rechner selbst hat kein Internet).

```bash
pip install -e '.[training,einbettung,leitung]' vllm
cp beispiel.toml lauf.toml   # Pfade und Grundmodell eintragen
nachtlernen --konfig lauf.toml geburt \
  --selbstbild "Ich bin neu auf dieser Station und weiß noch wenig über mich." \
  --ziele "Die Station kennenlernen."
```

Die Dateien in `betrieb/` sind systemd-Einheiten für den Dauerbetrieb: Wach-Modell (vLLM auf Karte 1),
morgens ein Tag in der Textwelt, nachts die Nachtschleife (Training auf Karte 2) und die Textleitung.
Bei nur einer Karte stoppen `vor_training` und `nach_training` in der Konfiguration den Server während
des Trainings.

Wichtig: Das Grundmodell wird bei der Geburt festgelegt und darf sich danach nie ändern, sonst passen
die Adapter nicht mehr. `nachtlernen nacht` bricht ab, wenn es sich geändert hat.

Weitere Befehle:

- `nachtlernen sprechen` Gespräch im Terminal, mit `--kontrolle` als Kontroll-Kopie ohne Adapter.
- `nachtlernen blindtest` Fragebogen mit Antworten des gewachsenen Modells und der Kontroll-Kopie in
  verdeckter Reihenfolge, Schlüssel in einer eigenen Datei (Stufe-4-Kriterium 4).
- `nachtlernen gateway --port /dev/ttyUSB0` Gateway-Seite der Textleitung, liest Nachrichten von stdin.

## Was in der Minimalfassung noch fehlt

- **Ankertest:** 25 Fragen sind ein Platzhalter. Eine falsche Antwort sind schon 4 %, die 2-%-Grenze
  braucht mindestens 100 Fragen, besser ein Teil von lm-evaluation-harness.
- **Widerspruchsfreiheit W:** steht fest auf 1,0, bis ein kleines NLI-Modell eingebaut ist
  (Schnittstelle `widerspruch` in `Pruefer`).
- **Einbettung:** Ohne Konfiguration wird eine einfache Hash-Einbettung benutzt, die nur Wörter
  vergleicht. Für den echten Lauf ein Modell wie bge-m3 eintragen.
- **Training auf der Grafikkarte:** `training._qlora` ist noch nicht auf Hardware gelaufen. Das ist der
  erste Schritt auf dem echten Rechner (Phase 0: Probetraining mit 100 Beispielen).
- **Gateway mit Chat-Oberfläche:** Die Leitung steht, ein Matrix-Bot auf dem Gateway noch nicht.
- **Eigene Auslesekriterien des Modells** mit Schattenphase, Bewohner mit eigenem Modell, monatliche
  Stufe-4-Prüfung als Bericht.

## Lizenz

Apache-2.0 (vorläufig, siehe `pyproject.toml`).
