"""Konfiguration eines Laufs.

Alle Stellschrauben stehen hier an einer Stelle und werden aus einer TOML-Datei
geladen (siehe ``beispiel.toml``). Fehlende Werte nehmen die Vorgaben aus dem
Bauplan.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModellKonfig:
    # OpenAI-kompatibler Server (vLLM empfohlen, llama.cpp geht ohne Adapterwechsel).
    url: str = "http://127.0.0.1:8000"
    # Name des Grundmodells beim Server. Darf sich während eines Laufs NIE ändern,
    # sonst passen die Adapter nicht mehr ("die DNA beginnt von vorn").
    grundmodell: str = "Qwen/Qwen2.5-14B-Instruct"
    # Lokaler Pfad zu den Gewichten des Grundmodells, fürs Training.
    gewichte: str = ""
    # "vllm": Adapter werden zur Laufzeit geladen. "ohne": nur Grundmodell.
    adapter_modus: str = "vllm"
    max_token: int = 512
    temperatur: float = 0.7
    zeitlimit_s: float = 300.0


@dataclass
class GedaechtnisKonfig:
    # Wie viele Erinnerungen pro Antwort in den Kontext kommen.
    abruf_anzahl: int = 8
    # Halbwertszeit der Aktualität in Tagen.
    halbwertszeit_tage: float = 14.0
    # "hash" (ohne Abhängigkeiten, nur zum Testen) oder ein sentence-transformers-Name.
    einbettung: str = "hash"


@dataclass
class NachtKonfig:
    # Das "gewisse Maß": höchstens so viele Erinnerungen werden pro Nacht DNA.
    max_erinnerungen: int = 200
    # Höchstens dieser Anteil der Trainingsdaten ist neu, der Rest Wiederholung.
    max_anteil_neu: float = 0.5
    # Varianten pro Nacht (Minimalfassung: 3, voller Plan: 4 bis 8).
    varianten: int = 3
    # Grundlernrate; die Varianten fächern um sie herum auf.
    lernrate: float = 1e-4
    # LoRA-Rang. Wird bei der Geburt festgelegt und bleibt, weil jede Nacht
    # den Adapter der Vornacht fortsetzt.
    lora_rang: int = 32
    lora_alpha: int = 64
    max_laenge_token: int = 2048
    # Ankertest: fällt eine Variante um mehr als diesen Anteil, scheidet sie aus.
    max_ankerverlust: float = 0.02
    # Kohärenz darf höchstens so weit unter den bisherigen Adapter fallen.
    kohaerenz_toleranz: float = 0.03
    # Absolute Untergrenze der Kohärenz.
    kohaerenz_min: float = 0.3
    # Startwert der Schrittweite sigma (Breite des Fächers).
    sigma_start: float = 1.0
    sigma_min: float = 0.25
    sigma_max: float = 4.0
    # Nach wie vielen Nächten die 1/5-Regel sigma anpasst.
    sigma_fenster: int = 7
    # Fällt die Kohärenz so viele Nächte in Folge, greift die Notbremse.
    notbremse_naechte: int = 3
    # Gewichte der Kohärenz K aus Erinnerung E, Selbstbild S, Widerspruchsfreiheit W.
    gewicht_e: float = 0.5
    gewicht_s: float = 0.3
    gewicht_w: float = 0.2
    # Strafe für Drift D (Abstand zum Sieger der Vornacht).
    gewicht_drift: float = 0.2
    # Wie viele Fragen pro Prüfung höchstens gestellt werden.
    pruef_fragen: int = 40
    # Shell-Befehle vor und nach dem Training. Bei nur einer Grafikkarte
    # (Variante A) z. B. "systemctl stop nachtlernen-vllm" und "... start ...".
    vor_training: str = ""
    nach_training: str = ""


@dataclass
class Konfig:
    # Arbeitsordner des Laufs: Gedächtnis, Adapter, Stammbaum, Nachtberichte.
    ordner: str = "lauf"
    name: str = "Wesen"
    modell: ModellKonfig = field(default_factory=ModellKonfig)
    gedaechtnis: GedaechtnisKonfig = field(default_factory=GedaechtnisKonfig)
    nacht: NachtKonfig = field(default_factory=NachtKonfig)

    @property
    def pfad(self) -> Path:
        return Path(self.ordner)

    @classmethod
    def laden(cls, datei: str | Path | None) -> "Konfig":
        if datei is None:
            return cls()
        with open(datei, "rb") as f:
            roh = tomllib.load(f)
        return _fuellen(cls, roh)


def _fuellen(klasse: type, roh: dict[str, Any]):
    werte = {}
    bekannte = {f.name: f for f in fields(klasse)}
    for schluessel, wert in roh.items():
        if schluessel not in bekannte:
            raise ValueError(f"Unbekannte Einstellung: {schluessel}")
        standard = bekannte[schluessel].default_factory  # type: ignore[attr-defined]
        if isinstance(wert, dict) and callable(standard):
            unterklasse = type(standard())
            if is_dataclass(unterklasse):
                wert = _fuellen(unterklasse, wert)
        werte[schluessel] = wert
    return klasse(**werte)
