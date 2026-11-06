"""Text-Einbettungen für Gedächtnisabruf, Drift und Neuheit.

``HashEinbettung`` braucht keine Abhängigkeiten und ist deterministisch. Sie
erkennt nur Wortähnlichkeit, keine Bedeutung, und ist deshalb für Tests und
den Trockenlauf gedacht. Im echten Lauf ``SatzEinbettung`` mit einem
mehrsprachigen Modell wie bge-m3 verwenden.
"""

from __future__ import annotations

import hashlib
import math
import re
from array import array
from typing import Protocol, Sequence

Vektor = list[float]


class Einbettung(Protocol):
    dim: int

    def einbetten(self, texte: Sequence[str]) -> list[Vektor]: ...


def _woerter(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HashEinbettung:
    """Wörter und Buchstaben-Trigramme werden in feste Fächer gehasht."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _fach(self, merkmal: str) -> tuple[int, float]:
        h = hashlib.blake2b(merkmal.encode(), digest_size=8).digest()
        zahl = int.from_bytes(h, "little")
        return zahl % self.dim, (1.0 if (zahl >> 63) & 1 else -1.0)

    def einbetten(self, texte: Sequence[str]) -> list[Vektor]:
        ergebnis = []
        for text in texte:
            v = [0.0] * self.dim
            for wort in _woerter(text):
                i, s = self._fach("w:" + wort)
                v[i] += 2.0 * s
                gerahmt = f"#{wort}#"
                for k in range(len(gerahmt) - 2):
                    i, s = self._fach("t:" + gerahmt[k : k + 3])
                    v[i] += s
            ergebnis.append(normieren(v))
        return ergebnis


class SatzEinbettung:
    """Wrapper um sentence-transformers (optional installiert)."""

    def __init__(self, name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._modell = SentenceTransformer(name)
        self.dim = int(self._modell.get_sentence_embedding_dimension())

    def einbetten(self, texte: Sequence[str]) -> list[Vektor]:
        vs = self._modell.encode(list(texte), normalize_embeddings=True)
        return [list(map(float, v)) for v in vs]


def erstellen(name: str) -> Einbettung:
    if name == "hash":
        return HashEinbettung()
    return SatzEinbettung(name)


def normieren(v: Vektor) -> Vektor:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def kosinus(a: Vektor, b: Vektor) -> float:
    return sum(x * y for x, y in zip(a, b))


def zu_bytes(v: Vektor) -> bytes:
    return array("f", v).tobytes()


def aus_bytes(b: bytes) -> Vektor:
    a = array("f")
    a.frombytes(b)
    return list(a)
