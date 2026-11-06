"""Das wache Modell am Tag: spricht, handelt und schreibt alles ins Gedächtnis."""

from __future__ import annotations

import time
from typing import Sequence

from .gedaechtnis import Ereignis, Gedaechtnis
from .modell import Modell, zahl_finden

WICHTIGKEIT_FRAGE = (
    "Wie wichtig ist dieses Erlebnis für dich, auf einer Skala von 1 (belanglos) "
    "bis 10 (prägend)? Antworte nur mit der Zahl.\n\nErlebnis: {text}"
)


def systemtext(name: str, selbstbild: str, ziele: str, erinnerungen: Sequence[Ereignis]) -> str:
    teile = [f"Du bist {name}. Du lebst auf einem eigenen Rechner ohne Internet."]
    if selbstbild:
        teile.append("So siehst du dich selbst:\n" + selbstbild)
    if ziele:
        teile.append("Deine Ziele:\n" + ziele)
    if erinnerungen:
        zeilen = [f"- [{e.tag}] {e.text}" for e in sorted(erinnerungen, key=lambda e: e.zeit)]
        teile.append("Erinnerungen, die dir gerade einfallen:\n" + "\n".join(zeilen))
    teile.append("Antworte auf Deutsch, in eigenen Worten, und erfinde keine Erinnerungen.")
    return "\n\n".join(teile)


class Wesen:
    def __init__(self, name: str, gedaechtnis: Gedaechtnis, modell: Modell,
                 adapter: str | None = None, abruf_anzahl: int = 8,
                 mit_gedaechtnis: bool = True):
        self.name = name
        self.gedaechtnis = gedaechtnis
        self.modell = modell
        self.adapter = adapter
        self.abruf_anzahl = abruf_anzahl
        # False ergibt die "Kontroll-Kopie ohne Gedächtnis"; die eigentliche
        # Kontroll-Kopie hat Gedächtnis, aber adapter=None.
        self.mit_gedaechtnis = mit_gedaechtnis

    def _kontext(self, anlass: str, jetzt: float) -> str:
        g = self.gedaechtnis
        erinnerungen = g.abrufen(anlass, self.abruf_anzahl, jetzt=jetzt) if self.mit_gedaechtnis else []
        return systemtext(self.name, g.selbst("selbstbild"), g.selbst("ziele"), erinnerungen)

    def wichtigkeit(self, text: str) -> float:
        antwort = self.modell.antworten(
            [{"role": "user", "content": WICHTIGKEIT_FRAGE.format(text=text)}],
            adapter=self.adapter, max_token=8, temperatur=0.0,
        )
        return zahl_finden(antwort, 5.0, 1.0, 10.0)

    def erleben(self, text: str, *, quelle: str = "welt", zeit: float | None = None,
                wichtigkeit: float | None = None) -> int:
        if wichtigkeit is None:
            wichtigkeit = self.wichtigkeit(text)
        return self.gedaechtnis.merken(text, quelle=quelle, wichtigkeit=wichtigkeit, zeit=zeit)

    def sprechen(self, text: str, *, von: str = "Fabian", zeit: float | None = None,
                 merken: bool = True) -> str:
        jetzt = time.time() if zeit is None else zeit
        antwort = self.modell.antworten(
            [{"role": "system", "content": self._kontext(text, jetzt)},
             {"role": "user", "content": text}],
            adapter=self.adapter,
        )
        if merken:
            self.erleben(f"{von} sagte zu mir: {text}\nIch antwortete: {antwort}",
                         quelle=von, zeit=jetzt)
        return antwort

    def handeln(self, beobachtung: str, aktionen: Sequence[str], *, zeit: float | None = None) -> str:
        jetzt = time.time() if zeit is None else zeit
        auftrag = (
            f"{beobachtung}\n\nMögliche Handlungen: {', '.join(aktionen)}.\n"
            "Antworte mit genau einer Zeile: Handlung und Ziel, zum Beispiel 'gehe Werkstatt'. "
            "Danach darfst du in einer zweiten Zeile mit 'Grund:' begründen."
        )
        return self.modell.antworten(
            [{"role": "system", "content": self._kontext(beobachtung, jetzt)},
             {"role": "user", "content": auftrag}],
            adapter=self.adapter, max_token=120,
        )


def in_welt_leben(wesen: Wesen, welt, schritte: int, *, start: float | None = None,
                  schritt_s: float = 60.0) -> list[str]:
    """Lässt das Wesen ``schritte`` Handlungen in der Textwelt ausführen."""
    from .welt import AKTIONEN

    jetzt = time.time() if start is None else start
    protokoll = []
    for i in range(schritte):
        zeit = jetzt + i * schritt_s
        beobachtung = welt.beobachtung()
        entscheidung = wesen.handeln(beobachtung, AKTIONEN, zeit=zeit)
        erlebnis = welt.handeln(entscheidung)
        grund = next((z.split(":", 1)[1].strip() for z in entscheidung.splitlines()[1:]
                      if z.lower().startswith("grund:")), "")
        text = erlebnis + (f" (Ich wollte: {grund})" if grund else "")
        wesen.erleben(text, quelle="welt", zeit=zeit)
        protokoll.append(text)
    welt.speichern()
    return protokoll
