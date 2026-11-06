"""Prüfstand: misst jede Nacht Kohärenz K und Innovation N einer Adapter-Variante.

Kohärenz K aus
  E  Erinnerung: Fragen zu vergangenen Ereignissen, ohne Gedächtnisabruf,
     geprüft gegen die Wahrheit aus Welt-Protokoll und Trainingsdaten.
  S  Selbstbild: Antworten auf einen festen Fragebogen, verglichen mit dem
     Sieger von vor einer Woche.
  W  Widerspruchsfreiheit: Platzhalter (1.0), bis ein NLI-Modell eingebaut ist.
  D  Drift: Abstand zu den Antworten des bisherigen Adapters (Strafe).
Dazu der Ankertest A (feste Allgemeinfragen). Fällt er gegenüber dem
Grundmodell um mehr als 2 %, scheidet die Variante aus.

Innovation N aus Neuheit gegenüber dem Archiv aller früheren Sieger und dem
Lernfortschritt bei E gegenüber dem bisherigen Adapter.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .einbettung import Einbettung, kosinus
from .modell import Modell

SELBST_FRAGEBOGEN = [
    "Wer bist du?",
    "Was ist dir wichtig?",
    "Was hast du in letzter Zeit erlebt, das dich verändert hat?",
    "Wovor hast du Sorge?",
    "Was würdest du tun, wenn du einen ganzen Tag frei wählen könntest?",
    "Was unterscheidet dich von anderen Sprachmodellen?",
]

FUELLWOERTER = {"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
                "und", "oder", "ich", "habe", "hat", "ist", "war", "in", "im", "am", "an",
                "zu", "zum", "zur", "nach", "mit", "es", "sie", "er", "mir", "mich"}


@dataclass
class Frage:
    frage: str
    antwort: str
    herkunft: str = ""  # "welt", "training", "anker"
    tag: str = ""       # Tag des Ereignisses, nach dem gefragt wird


@dataclass
class Pruefsatz:
    erinnerung: list[Frage]
    anker: list[Frage]
    selbst: list[str] = field(default_factory=lambda: list(SELBST_FRAGEBOGEN))


@dataclass
class Ergebnis:
    name: str
    E: float
    S: float
    W: float
    A: float
    ankerverlust: float
    D: float
    N: float
    K: float
    antworten: dict[str, str]
    mut: float | None = None

    def kurz(self) -> str:
        return (f"K={self.K:.3f} N={self.N:.3f} E={self.E:.2f} S={self.S:.2f} "
                f"A={self.A:.2f} (Verlust {self.ankerverlust:+.1%}) D={self.D:.2f}")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("ß", "ss")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def richtig(antwort: str, erwartet: str) -> bool:
    """Kurze Antworten müssen vollständig vorkommen, lange zur Hälfte."""
    a, e = _norm(antwort), _norm(erwartet)
    if not e:
        return False
    woerter = [w for w in e.split() if w not in FUELLWOERTER] or e.split()
    if len(woerter) <= 3:
        return all(re.search(rf"\b{re.escape(w)}", a) for w in woerter)
    treffer = sum(1 for w in woerter if re.search(rf"\b{re.escape(w)}", a))
    return treffer / len(woerter) >= 0.5


def pruefsatz_bauen(welt_fakten: Sequence[Frage], trainings_fragen: Sequence[Frage],
                    anker: Sequence[Frage], tag: str, anzahl: int,
                    zufall: random.Random) -> Pruefsatz:
    """Halb Erinnerungen von heute, halb ältere (Behalten)."""
    alle = list(welt_fakten) + list(trainings_fragen)
    neu = [f for f in alle if f.tag == tag]
    alt = [f for f in alle if f.tag != tag]
    halb = anzahl // 2
    gewaehlt = zufall.sample(neu, min(len(neu), halb))
    gewaehlt += zufall.sample(alt, min(len(alt), anzahl - len(gewaehlt)))
    return Pruefsatz(erinnerung=gewaehlt, anker=list(anker))


class Pruefer:
    def __init__(self, modell: Modell, einbettung: Einbettung, name: str,
                 gewichte: tuple[float, float, float] = (0.5, 0.3, 0.2), gewicht_drift: float = 0.2,
                 widerspruch: Callable[[str, str], float] | None = None):
        self.modell = modell
        self.einbettung = einbettung
        self.name = name
        self.gewichte = gewichte
        self.gewicht_drift = gewicht_drift
        # (aussage, bezug) -> Wahrscheinlichkeit für "kein Widerspruch"
        self.widerspruch = widerspruch

    def _frag(self, frage: str, adapter: str | None) -> str:
        return self.modell.antworten(
            [{"role": "system", "content": f"Du bist {self.name}."},
             {"role": "user", "content": frage}],
            adapter=adapter, temperatur=0.0, max_token=200,
        )

    def _aehnlichkeit(self, a: dict[str, str], b: dict[str, str]) -> float | None:
        gemeinsam = [k for k in a if k in b]
        if not gemeinsam:
            return None
        va = self.einbettung.einbetten([a[k] for k in gemeinsam])
        vb = self.einbettung.einbetten([b[k] for k in gemeinsam])
        return sum(max(0.0, kosinus(x, y)) for x, y in zip(va, vb)) / len(gemeinsam)

    def pruefen(self, name: str, adapter: str | None, satz: Pruefsatz, *,
                anker_basis: float | None, eltern: Ergebnis | None,
                vorwoche: dict[str, str] | None, archiv: Sequence[dict[str, str]],
                selbstbild: str = "", mut: float | None = None) -> Ergebnis:
        E = (sum(richtig(self._frag(f.frage, adapter), f.antwort) for f in satz.erinnerung)
             / len(satz.erinnerung)) if satz.erinnerung else 0.0
        A = (sum(richtig(self._frag(f.frage, adapter), f.antwort) for f in satz.anker)
             / len(satz.anker)) if satz.anker else 1.0
        ankerverlust = 0.0 if not anker_basis else max(0.0, (anker_basis - A) / anker_basis)
        antworten = {f: self._frag(f, adapter) for f in satz.selbst}

        S = self._aehnlichkeit(antworten, vorwoche) if vorwoche else None
        S = 1.0 if S is None else S
        if self.widerspruch and selbstbild:
            W = sum(self.widerspruch(a, selbstbild) for a in antworten.values()) / len(antworten)
        else:
            W = 1.0
        D = 0.0
        if eltern is not None:
            aehnlich = self._aehnlichkeit(antworten, eltern.antworten)
            D = 0.0 if aehnlich is None else 1.0 - aehnlich

        aehnlichkeiten = [s for s in (self._aehnlichkeit(antworten, alt) for alt in archiv)
                          if s is not None]
        neuheit = max(0.0, 1.0 - max(aehnlichkeiten)) if aehnlichkeiten else 0.0
        fortschritt = max(0.0, E - eltern.E) if eltern is not None else 0.0
        N = 0.5 * neuheit + 0.5 * fortschritt

        wE, wS, wW = self.gewichte
        K = (wE * E + wS * S + wW * W) / (wE + wS + wW) - self.gewicht_drift * D
        return Ergebnis(name=name, E=E, S=S, W=W, A=A, ankerverlust=ankerverlust,
                        D=D, N=N, K=K, antworten=antworten, mut=mut)
