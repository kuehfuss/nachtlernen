"""Zugang zum Sprachmodell.

``ServerModell`` spricht mit einem lokalen OpenAI-kompatiblen Server (vLLM
oder llama.cpp) über 127.0.0.1, nur mit der Standardbibliothek. Mit vLLM
werden die nächtlichen Adapter zur Laufzeit geladen; dafür muss der Server mit
``--enable-lora`` und ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`` laufen.

``SkriptModell`` ist ein Ersatz ohne Grafikkarte für Tests und Trockenläufe.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Callable, Protocol, Sequence

Nachricht = dict[str, str]


class Modell(Protocol):
    def antworten(self, nachrichten: Sequence[Nachricht], *, adapter: str | None = None,
                  max_token: int | None = None, temperatur: float | None = None) -> str: ...

    def adapter_laden(self, name: str, pfad: str) -> None: ...


class ServerModell:
    def __init__(self, url: str, grundmodell: str, *, adapter_modus: str = "vllm",
                 max_token: int = 512, temperatur: float = 0.7, zeitlimit_s: float = 300.0):
        self.url = url.rstrip("/")
        self.grundmodell = grundmodell
        self.adapter_modus = adapter_modus
        self.max_token = max_token
        self.temperatur = temperatur
        self.zeitlimit_s = zeitlimit_s

    def _post(self, pfad: str, daten: dict) -> dict:
        anfrage = urllib.request.Request(
            self.url + pfad,
            data=json.dumps(daten).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(anfrage, timeout=self.zeitlimit_s) as antwort:
            text = antwort.read().decode()
        return json.loads(text) if text.strip().startswith("{") else {}

    def adapter_laden(self, name: str, pfad: str) -> None:
        """Lädt einen Adapter. Ist er schon geladen, meldet vLLM 400; das ist in Ordnung.

        Kein Zwischenspeicher auf unserer Seite: Nach einem Neustart des Servers
        (z. B. bei nur einer Grafikkarte) müssen alle Adapter neu geladen werden.
        """
        if self.adapter_modus != "vllm":
            return
        try:
            self._post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": pfad})
        except urllib.error.HTTPError as fehler:
            if fehler.code != 400:
                raise

    def antworten(self, nachrichten, *, adapter=None, max_token=None, temperatur=None) -> str:
        if adapter and self.adapter_modus != "vllm":
            raise RuntimeError("Adapterwechsel braucht adapter_modus = 'vllm'")
        daten = {
            "model": adapter or self.grundmodell,
            "messages": list(nachrichten),
            "max_tokens": max_token or self.max_token,
            "temperature": self.temperatur if temperatur is None else temperatur,
        }
        antwort = self._post("/v1/chat/completions", daten)
        return antwort["choices"][0]["message"]["content"]


class SkriptModell:
    """Antwortet über eine Funktion ``(nachrichten, adapter) -> str``."""

    def __init__(self, funktion: Callable[[Sequence[Nachricht], str | None], str] | None = None):
        self.funktion = funktion or (lambda n, a: "Ich weiß es nicht.")
        self.geladen: dict[str, str] = {}
        self.aufrufe: list[tuple[list[Nachricht], str | None]] = []

    def adapter_laden(self, name: str, pfad: str) -> None:
        self.geladen[name] = pfad

    def antworten(self, nachrichten, *, adapter=None, max_token=None, temperatur=None) -> str:
        self.aufrufe.append((list(nachrichten), adapter))
        return self.funktion(nachrichten, adapter)


# --- Hilfen zum robusten Auslesen von Modellantworten -------------------------

def json_finden(text: str):
    """Sucht das erste JSON-Objekt oder die erste JSON-Liste in einer Antwort."""
    for start, ende in (("{", "}"), ("[", "]")):
        i = text.find(start)
        while i != -1:
            j = text.rfind(ende)
            while j > i:
                try:
                    return json.loads(text[i : j + 1])
                except json.JSONDecodeError:
                    j = text.rfind(ende, i, j)
            i = text.find(start, i + 1)
    return None


def zahl_finden(text: str, standard: float, kleinste: float, groesste: float) -> float:
    m = re.search(r"-?\d+(?:[.,]\d+)?", text)
    if not m:
        return standard
    return min(groesste, max(kleinste, float(m.group().replace(",", "."))))
