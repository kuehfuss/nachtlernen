"""Textleitung zwischen Gateway und abgekoppeltem KI-Rechner.

Über die serielle Leitung gehen nur einzelne Textzeilen: je Nachricht eine
Zeile JSON mit ``von`` und ``text``, je Antwort eine Zeile mit ``antwort``.
Es gibt keine Befehle und keine Dateien. Beide Seiten kürzen zu lange Texte,
entfernen Steuerzeichen und protokollieren alles.

Die Funktionen arbeiten mit beliebigen Byte-Strömen (``readline``/``write``),
damit sie ohne serielle Hardware getestet werden können. Für die echte Leitung
``serielle_leitung`` verwenden (braucht ``pip install .[leitung]``).
"""

from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path
from typing import BinaryIO, Callable

MAX_ZEICHEN = 4000
MAX_ZEILE = 4 * MAX_ZEICHEN + 200  # JSON-Escapes können Text verlängern


def bereinigen(text: str, max_zeichen: int = MAX_ZEICHEN) -> str:
    erlaubt = []
    for zeichen in str(text):
        if zeichen in "\n\t" or unicodedata.category(zeichen)[0] != "C":
            erlaubt.append(zeichen)
    return "".join(erlaubt)[:max_zeichen]


def _protokoll(datei: Path | None, richtung: str, daten: dict) -> None:
    if datei is None:
        return
    datei.parent.mkdir(parents=True, exist_ok=True)
    with open(datei, "a", encoding="utf-8") as f:
        f.write(json.dumps({"zeit": time.time(), "richtung": richtung, **daten}, ensure_ascii=False) + "\n")


def _zeile_lesen(strom: BinaryIO) -> dict | None:
    roh = strom.readline(MAX_ZEILE)
    if not roh:
        return None
    if not roh.endswith(b"\n"):
        # Überlange Zeile: Rest bis zum Zeilenende verwerfen.
        while roh and not roh.endswith(b"\n"):
            roh = strom.readline(MAX_ZEILE)
        return {}
    try:
        daten = json.loads(roh.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
    return daten if isinstance(daten, dict) else {}


def _zeile_schreiben(strom: BinaryIO, daten: dict) -> None:
    strom.write((json.dumps(daten, ensure_ascii=False) + "\n").encode("utf-8"))
    strom.flush()


def ki_seite(lesen: BinaryIO, schreiben: BinaryIO, antworten: Callable[[str, str], str],
             protokoll: Path | None = None, max_nachrichten: int | None = None) -> int:
    """Läuft auf dem KI-Rechner: nimmt Nachrichten an und schickt Antworten zurück."""
    anzahl = 0
    while max_nachrichten is None or anzahl < max_nachrichten:
        daten = _zeile_lesen(lesen)
        if daten is None:
            break
        von = bereinigen(daten.get("von", "unbekannt"), 40) or "unbekannt"
        text = bereinigen(daten.get("text", ""))
        if not text:
            _zeile_schreiben(schreiben, {"antwort": "", "fehler": "leere oder ungültige Nachricht"})
            continue
        _protokoll(protokoll, "ein", {"von": von, "text": text})
        antwort = bereinigen(antworten(von, text))
        _protokoll(protokoll, "aus", {"an": von, "text": antwort})
        _zeile_schreiben(schreiben, {"antwort": antwort})
        anzahl += 1
    return anzahl


def gateway_senden(lesen: BinaryIO, schreiben: BinaryIO, von: str, text: str,
                   protokoll: Path | None = None) -> str:
    """Läuft auf dem Gateway: schickt eine Nachricht und wartet auf die Antwort."""
    nachricht = {"von": bereinigen(von, 40), "text": bereinigen(text)}
    _protokoll(protokoll, "aus", nachricht)
    _zeile_schreiben(schreiben, nachricht)
    daten = _zeile_lesen(lesen) or {}
    antwort = bereinigen(daten.get("antwort", ""))
    _protokoll(protokoll, "ein", {"antwort": antwort})
    return antwort


def serielle_leitung(port: str, baud: int = 115200, zeitlimit_s: float | None = None):
    import serial  # type: ignore

    return serial.Serial(port, baudrate=baud, timeout=zeitlimit_s)
