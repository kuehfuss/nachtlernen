"""Gewichtetes Ereignisgedächtnis (Stufe 2).

Jedes Ereignis bekommt ein Gewicht aus Wichtigkeit (vom Modell, 1 bis 10),
Überraschung (wie wenig es den bisherigen Erinnerungen ähnelt) und
Wiederholung (wie oft es abgerufen wurde). Beim Abruf zählt zusätzlich die
Aktualität. Selbstbild und Ziele sind zwei versionierte Texte, die das Modell
selbst umschreibt.

Alles liegt in einer SQLite-Datei. Originale werden nie gelöscht; Tage werden
zu Zusammenfassungen verdichtet, die beim Abruf mitkonkurrieren.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

from .einbettung import Einbettung, aus_bytes, kosinus, zu_bytes

SCHEMA = """
CREATE TABLE IF NOT EXISTS ereignis (
    id INTEGER PRIMARY KEY,
    zeit REAL NOT NULL,
    tag TEXT NOT NULL,
    art TEXT NOT NULL,
    quelle TEXT NOT NULL,
    text TEXT NOT NULL,
    wichtigkeit REAL NOT NULL,
    ueberraschung REAL NOT NULL,
    wiederholungen INTEGER NOT NULL DEFAULT 0,
    zuletzt REAL,
    ebene INTEGER NOT NULL DEFAULT 0,
    quellen TEXT,
    vektor BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ereignis_tag ON ereignis(tag);
CREATE TABLE IF NOT EXISTS selbst (
    id INTEGER PRIMARY KEY,
    zeit REAL NOT NULL,
    art TEXT NOT NULL,
    text TEXT NOT NULL,
    grund TEXT
);
"""

SELBST_ARTEN = ("selbstbild", "ziele")


@dataclass
class Ereignis:
    id: int
    zeit: float
    tag: str
    art: str
    quelle: str
    text: str
    wichtigkeit: float
    ueberraschung: float
    wiederholungen: int
    ebene: int

    @property
    def gewicht(self) -> float:
        """Gewicht ohne Alter: Wichtigkeit mal Überraschung mal Wiederholung."""
        return (
            (self.wichtigkeit / 10.0)
            * (0.5 + self.ueberraschung)
            * (1.0 + 0.5 * math.log1p(self.wiederholungen))
        )


def tag_von(zeit: float) -> str:
    return datetime.fromtimestamp(zeit, tz=timezone.utc).date().isoformat()


class Gedaechtnis:
    def __init__(self, pfad: str | Path, einbettung: Einbettung, halbwertszeit_tage: float = 14.0):
        self.pfad = str(pfad)
        self.einbettung = einbettung
        self.halbwertszeit_s = halbwertszeit_tage * 86400.0
        self.db = sqlite3.connect(self.pfad)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def schliessen(self) -> None:
        self.db.close()

    # --- Schreiben -----------------------------------------------------------

    def merken(
        self,
        text: str,
        *,
        art: str = "erlebnis",
        quelle: str = "welt",
        wichtigkeit: float = 5.0,
        ueberraschung: float | None = None,
        zeit: float | None = None,
        ebene: int = 0,
        quellen: Iterable[int] | None = None,
    ) -> int:
        zeit = time.time() if zeit is None else zeit
        vektor = self.einbettung.einbetten([text])[0]
        if ueberraschung is None:
            ueberraschung = self._ueberraschung(vektor)
        wichtigkeit = min(10.0, max(1.0, float(wichtigkeit)))
        cur = self.db.execute(
            "INSERT INTO ereignis (zeit, tag, art, quelle, text, wichtigkeit, ueberraschung,"
            " ebene, quellen, vektor) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                zeit,
                tag_von(zeit),
                art,
                quelle,
                text,
                wichtigkeit,
                ueberraschung,
                ebene,
                json.dumps(list(quellen)) if quellen else None,
                zu_bytes(vektor),
            ),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def _ueberraschung(self, vektor: list[float], vergleich: int = 200) -> float:
        """1 minus die größte Ähnlichkeit zu den letzten Erinnerungen."""
        zeilen = self.db.execute(
            "SELECT vektor FROM ereignis ORDER BY id DESC LIMIT ?", (vergleich,)
        ).fetchall()
        if not zeilen:
            return 1.0
        aehnlichste = max(kosinus(vektor, aus_bytes(z["vektor"])) for z in zeilen)
        return float(min(1.0, max(0.0, 1.0 - aehnlichste)))

    # --- Lesen ---------------------------------------------------------------

    @staticmethod
    def _ereignis(z: sqlite3.Row) -> Ereignis:
        return Ereignis(
            id=z["id"],
            zeit=z["zeit"],
            tag=z["tag"],
            art=z["art"],
            quelle=z["quelle"],
            text=z["text"],
            wichtigkeit=z["wichtigkeit"],
            ueberraschung=z["ueberraschung"],
            wiederholungen=z["wiederholungen"],
            ebene=z["ebene"],
        )

    def holen(self, ids: Iterable[int]) -> list[Ereignis]:
        ids = list(ids)
        if not ids:
            return []
        platz = ",".join("?" * len(ids))
        zeilen = self.db.execute(f"SELECT * FROM ereignis WHERE id IN ({platz})", ids).fetchall()
        nach_id = {z["id"]: self._ereignis(z) for z in zeilen}
        return [nach_id[i] for i in ids if i in nach_id]

    def des_tages(self, tag: str, ebene: int = 0) -> list[Ereignis]:
        zeilen = self.db.execute(
            "SELECT * FROM ereignis WHERE tag = ? AND ebene = ? ORDER BY zeit", (tag, ebene)
        ).fetchall()
        return [self._ereignis(z) for z in zeilen]

    def wichtigste_des_tages(self, tag: str, anzahl: int) -> list[Ereignis]:
        ereignisse = self.des_tages(tag)
        ereignisse.sort(key=lambda e: e.gewicht, reverse=True)
        return ereignisse[:anzahl]

    def vor(self, tag: str, anzahl: int | None = None) -> list[Ereignis]:
        """Ereignisse (Ebene 0) aus Tagen vor ``tag``, neueste zuerst."""
        sql = "SELECT * FROM ereignis WHERE tag < ? AND ebene = 0 ORDER BY zeit DESC"
        if anzahl is not None:
            sql += f" LIMIT {int(anzahl)}"
        return [self._ereignis(z) for z in self.db.execute(sql, (tag,)).fetchall()]

    def anzahl(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM ereignis").fetchone()[0])

    def abrufen(self, anfrage: str, anzahl: int = 8, jetzt: float | None = None,
                zaehlen: bool = True) -> list[Ereignis]:
        """Ähnlichkeit mal Gewicht mal Aktualität (wie Generative Agents)."""
        jetzt = time.time() if jetzt is None else jetzt
        q = self.einbettung.einbetten([anfrage])[0]
        bewertet = []
        for z in self.db.execute("SELECT * FROM ereignis"):
            e = self._ereignis(z)
            aehnlich = max(0.0, kosinus(q, aus_bytes(z["vektor"])))
            alter = max(0.0, jetzt - (z["zuletzt"] or e.zeit))
            aktuell = 0.5 ** (alter / self.halbwertszeit_s)
            bewertet.append((aehnlich * (0.2 + e.gewicht) * (0.3 + aktuell), e))
        bewertet.sort(key=lambda p: p[0], reverse=True)
        treffer = [e for s, e in bewertet[:anzahl] if s > 0]
        if zaehlen and treffer:
            self.db.executemany(
                "UPDATE ereignis SET wiederholungen = wiederholungen + 1, zuletzt = ? WHERE id = ?",
                [(jetzt, e.id) for e in treffer],
            )
            self.db.commit()
        return treffer

    # --- Verdichtung ---------------------------------------------------------

    def verdichten(self, tag: str, zusammenfassen: Callable[[list[Ereignis]], str]) -> int | None:
        """Fasst einen Tag zu einer Erinnerung der Ebene 1 zusammen (einmal pro Tag)."""
        if self.des_tages(tag, ebene=1):
            return None
        ereignisse = self.des_tages(tag)
        if not ereignisse:
            return None
        text = zusammenfassen(ereignisse)
        zeit = max(e.zeit for e in ereignisse)
        return self.merken(
            text,
            art="zusammenfassung",
            quelle="gedaechtnis",
            wichtigkeit=max(e.wichtigkeit for e in ereignisse),
            ueberraschung=sum(e.ueberraschung for e in ereignisse) / len(ereignisse),
            zeit=zeit,
            ebene=1,
            quellen=[e.id for e in ereignisse],
        )

    # --- Selbstbild und Ziele ------------------------------------------------

    def selbst_setzen(self, art: str, text: str, grund: str = "", zeit: float | None = None) -> None:
        if art not in SELBST_ARTEN:
            raise ValueError(f"art muss eins von {SELBST_ARTEN} sein")
        if self.selbst(art) == text:
            return
        self.db.execute(
            "INSERT INTO selbst (zeit, art, text, grund) VALUES (?,?,?,?)",
            (time.time() if zeit is None else zeit, art, text, grund),
        )
        self.db.commit()

    def selbst(self, art: str) -> str:
        z = self.db.execute(
            "SELECT text FROM selbst WHERE art = ? ORDER BY id DESC LIMIT 1", (art,)
        ).fetchone()
        return z["text"] if z else ""

    def selbst_verlauf(self, art: str) -> list[tuple[float, str, str]]:
        return [
            (z["zeit"], z["text"], z["grund"] or "")
            for z in self.db.execute("SELECT * FROM selbst WHERE art = ? ORDER BY id", (art,))
        ]


def tage_rueckwaerts(tag: str, anzahl: int) -> list[str]:
    d = date.fromisoformat(tag)
    return [(d - timedelta(days=i)).isoformat() for i in range(anzahl)]
