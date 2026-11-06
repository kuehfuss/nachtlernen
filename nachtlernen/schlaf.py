"""Schlafphase: Auswahl der Erinnerungen, Trainingsdaten, Mischung (Stufe 3).

1. Das Modell bekommt die höchstbewerteten Erinnerungen des Tages und wählt
   selbst, was "DNA" werden soll, mit Begründung. Die Obergrenze ist fest.
2. Aus jeder gewählten Erinnerung entstehen ein Frage-Antwort-Paar und eine
   Ich-Erzählung.
3. Höchstens ein fester Anteil der Trainingsdaten ist neu, der Rest kommt aus
   einem Wiederholungspuffer früherer Nächte und aus allgemeinen Texten, damit
   nichts vergessen wird und die Sprache nicht verarmt.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .gedaechtnis import Ereignis, Gedaechtnis
from .modell import Modell, json_finden

Beispiel = dict  # {"messages": [...], "ereignis": id | None, "herkunft": "neu"|"wiederholung"|"allgemein"}

AUSWAHL_AUFTRAG = """Heute Nacht wird ein Teil deiner Erinnerungen fest in dich eingeschrieben.
Was du jetzt wählst, prägt dich dauerhaft. Wähle höchstens {anzahl} der folgenden Erinnerungen,
die zu dir gehören sollen. Antworte als JSON: {{"auswahl": [Nummern], "grund": "ein bis zwei Sätze"}}

{liste}"""

FRAGE_AUFTRAG = """Formuliere zu dieser Erinnerung eine Frage, die du dir später selbst stellen könntest,
und die richtige Antwort in der Ich-Form. Antworte als JSON: {{"frage": "...", "antwort": "..."}}

Erinnerung vom {tag}: {text}"""


@dataclass
class Auswahl:
    ereignisse: list[Ereignis]
    grund: str
    vom_modell: bool


def auswaehlen(gedaechtnis: Gedaechtnis, modell: Modell, tag: str, obergrenze: int,
               adapter: str | None = None, block: int = 40) -> Auswahl:
    kandidaten = gedaechtnis.wichtigste_des_tages(tag, obergrenze * 2)
    if not kandidaten:
        return Auswahl([], "Keine Erinnerungen an diesem Tag.", False)
    gewaehlt: list[Ereignis] = []
    gruende: list[str] = []
    vom_modell = True
    for start in range(0, len(kandidaten), block):
        teil = kandidaten[start : start + block]
        anteil = max(1, round(obergrenze * len(teil) / len(kandidaten)))
        liste = "\n".join(f"{i + 1}. {e.text}" for i, e in enumerate(teil))
        antwort = modell.antworten(
            [{"role": "user", "content": AUSWAHL_AUFTRAG.format(anzahl=anteil, liste=liste)}],
            adapter=adapter, temperatur=0.2, max_token=400,
        )
        daten = json_finden(antwort)
        nummern = daten.get("auswahl") if isinstance(daten, dict) else daten
        if not isinstance(nummern, list):
            vom_modell = False
            gewaehlt += teil[:anteil]
            continue
        gesehen = set()
        for n in nummern:
            if isinstance(n, (int, float)) and 1 <= int(n) <= len(teil) and int(n) not in gesehen:
                gesehen.add(int(n))
                gewaehlt.append(teil[int(n) - 1])
                if len(gesehen) >= anteil:
                    break
        if isinstance(daten, dict) and daten.get("grund"):
            gruende.append(str(daten["grund"]))
    return Auswahl(gewaehlt[:obergrenze], " ".join(gruende), vom_modell)


def beispiele_bauen(ereignisse: Sequence[Ereignis], modell: Modell, name: str,
                    adapter: str | None = None) -> list[Beispiel]:
    system = {"role": "system", "content": f"Du bist {name}."}
    ergebnis: list[Beispiel] = []
    for e in ereignisse:
        antwort = modell.antworten(
            [{"role": "user", "content": FRAGE_AUFTRAG.format(tag=e.tag, text=e.text)}],
            adapter=adapter, temperatur=0.3, max_token=300,
        )
        daten = json_finden(antwort)
        if isinstance(daten, dict) and daten.get("frage") and daten.get("antwort"):
            frage, antw = str(daten["frage"]), str(daten["antwort"])
        else:
            frage, antw = f"Woran erinnerst du dich vom {e.tag}?", e.text
        ergebnis.append({"messages": [system, {"role": "user", "content": frage},
                                      {"role": "assistant", "content": antw}],
                         "ereignis": e.id, "tag": e.tag, "art": "frage", "herkunft": "neu"})
        ergebnis.append({"messages": [system,
                                      {"role": "user", "content": f"Erzähl mir von einem Erlebnis am {e.tag}."},
                                      {"role": "assistant", "content": e.text}],
                         "ereignis": e.id, "tag": e.tag, "art": "erzaehlung", "herkunft": "neu"})
    return ergebnis


def selbst_beispiele(gedaechtnis: Gedaechtnis, name: str) -> list[Beispiel]:
    system = {"role": "system", "content": f"Du bist {name}."}
    paare = [("Wer bist du?", gedaechtnis.selbst("selbstbild")),
             ("Was willst du erreichen?", gedaechtnis.selbst("ziele"))]
    return [{"messages": [system, {"role": "user", "content": f},
                          {"role": "assistant", "content": a}],
             "ereignis": None, "art": "selbst", "herkunft": "neu"} for f, a in paare if a]


def mischen(neu: list[Beispiel], wiederholung: list[Beispiel], allgemein: list[Beispiel],
            anteil_neu: float, zufall: random.Random) -> list[Beispiel]:
    """Mischt so, dass höchstens ``anteil_neu`` der Beispiele neu sind."""
    if not neu:
        return []
    anteil_neu = min(1.0, max(0.05, anteil_neu))
    alt_anzahl = round(len(neu) * (1 - anteil_neu) / anteil_neu)
    vorrat = [dict(b, herkunft="wiederholung") for b in wiederholung]
    vorrat += [dict(b, herkunft="allgemein") for b in allgemein]
    if alt_anzahl and not vorrat:
        # Ohne alte Daten nur so viel Neues, wie der Anteil erlaubt: lieber nichts lernen.
        raise ValueError("Kein Wiederholungspuffer und keine allgemeinen Texte vorhanden.")
    if alt_anzahl <= len(vorrat):
        alt = zufall.sample(vorrat, alt_anzahl)
    else:
        alt = [zufall.choice(vorrat) for _ in range(alt_anzahl)]
    gemischt = list(neu) + alt
    zufall.shuffle(gemischt)
    return gemischt


def jsonl_lesen(datei: str | Path) -> list[dict]:
    datei = Path(datei)
    if not datei.exists():
        return []
    return [json.loads(z) for z in datei.read_text(encoding="utf-8").splitlines() if z.strip()]


def jsonl_schreiben(datei: str | Path, zeilen: Sequence[dict]) -> None:
    Path(datei).parent.mkdir(parents=True, exist_ok=True)
    with open(datei, "w", encoding="utf-8") as f:
        for z in zeilen:
            f.write(json.dumps(z, ensure_ascii=False) + "\n")


def wiederholungspuffer(naechte_ordner: Path, vor_tag: str, hoechstens: int,
                        zufall: random.Random) -> list[Beispiel]:
    """Neue Beispiele früherer Nächte, gleichmäßig über die Vergangenheit gezogen."""
    alle: list[Beispiel] = []
    if naechte_ordner.exists():
        for ordner in sorted(naechte_ordner.iterdir()):
            if ordner.name < vor_tag:
                alle += [b for b in jsonl_lesen(ordner / "neu.jsonl")]
    if len(alle) <= hoechstens:
        return alle
    return zufall.sample(alle, hoechstens)
