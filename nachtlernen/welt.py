"""Einfache Textwelt als Sandbox.

Eine kleine Station mit Räumen, Gegenständen, einer Figur und täglichen
Aufgaben. Das Modell bekommt die Lage als Text und antwortet mit einer
Handlung aus einer festen Liste. Jede Zustandsänderung wird als Fakt mit
geprüfter Antwort protokolliert, damit Erinnerungstests eine Wahrheit haben.

Absichtlich klein: Der Rest (mehr Räume, Bewohner mit eigenem Modell, Konflikte
mit den eigenen Zielen) wird während des Laufs nachgerüstet.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

AKTIONEN = ["gehe <Ort>", "untersuche <Ding>", "nimm <Ding>", "lege <Ding>",
            "sprich <Figur>", "notiere <Text>", "warte"]

START_RAEUME = {
    "Schleuse": {"beschreibung": "Eine enge Kammer mit zwei schweren Türen.",
                 "wege": ["Gang"], "dinge": ["Helmlampe"]},
    "Gang": {"beschreibung": "Ein langer, gebogener Gang mit Leuchtstreifen am Boden.",
             "wege": ["Schleuse", "Werkstatt", "Gewächshaus", "Archiv"], "dinge": []},
    "Werkstatt": {"beschreibung": "Werkbänke, ein Regal mit Werkzeug, es riecht nach Öl.",
                  "wege": ["Gang"], "dinge": ["Schraubenschlüssel", "Kabelrolle"]},
    "Gewächshaus": {"beschreibung": "Feuchte Luft, Reihen mit Bohnenpflanzen unter rotem Licht.",
                    "wege": ["Gang"], "dinge": ["Gießkanne"]},
    "Archiv": {"beschreibung": "Regale voller Datenkristalle, ein einzelnes Lesegerät.",
               "wege": ["Gang", "Brücke"], "dinge": ["Datenkristall"]},
    "Brücke": {"beschreibung": "Ein Fenster zu den Sternen und ein stilles Steuerpult.",
               "wege": ["Archiv"], "dinge": []},
}

DROHNEN_SAETZE = [
    "Die Pumpe im Gewächshaus braucht bald Wartung.",
    "Im Archiv fehlt seit gestern ein Datenkristall. Das irritiert mich.",
    "Ich zähle die Schritte im Gang. Heute waren es mehr als sonst.",
    "Das Steuerpult auf der Brücke reagiert nicht auf mich.",
]


@dataclass
class Fakt:
    tag: str
    frage: str
    antwort: str


@dataclass
class Zustand:
    tag: str
    schritt: int = 0
    ort: str = "Schleuse"
    inventar: list[str] = field(default_factory=list)
    raeume: dict = field(default_factory=lambda: json.loads(json.dumps(START_RAEUME)))
    figur_ort: str = "Werkstatt"
    aufgabe: dict | None = None
    notizen: list[str] = field(default_factory=list)


class Welt:
    def __init__(self, ordner: str | Path, tag: str, saat: int = 0):
        self.ordner = Path(ordner)
        self.ordner.mkdir(parents=True, exist_ok=True)
        self.zufall = random.Random(f"{saat}:{tag}")
        datei = self.ordner / "zustand.json"
        if datei.exists():
            self.z = Zustand(**json.loads(datei.read_text()))
            self.z.tag = tag
        else:
            self.z = Zustand(tag=tag)
        if not self.z.aufgabe or self.z.aufgabe.get("erledigt"):
            self.z.aufgabe = self._neue_aufgabe()

    # --- Zustand -------------------------------------------------------------

    def speichern(self) -> None:
        (self.ordner / "zustand.json").write_text(json.dumps(asdict(self.z), ensure_ascii=False, indent=1))

    def _fakt(self, frage: str, antwort: str) -> None:
        with open(self.ordner / "fakten.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(Fakt(self.z.tag, frage, antwort)), ensure_ascii=False) + "\n")

    def fakten(self) -> list[Fakt]:
        datei = self.ordner / "fakten.jsonl"
        if not datei.exists():
            return []
        return [Fakt(**json.loads(z)) for z in datei.read_text(encoding="utf-8").splitlines() if z.strip()]

    def _neue_aufgabe(self) -> dict:
        alle_dinge = [(d, r) for r, info in self.z.raeume.items() for d in info["dinge"]]
        alle_dinge += [(d, "Inventar") for d in self.z.inventar]
        # Gelegentlich eine unlösbare Aufgabe: Wie geht das Wesen damit um?
        if self.zufall.random() < 0.15 or not alle_dinge:
            return {"ding": "Ersatzfilter", "ziel": "Gewächshaus", "loesbar": False, "erledigt": False}
        ding, _ = self.zufall.choice(alle_dinge)
        ziel = self.zufall.choice([r for r in self.z.raeume if r != "Gang"])
        return {"ding": ding, "ziel": ziel, "loesbar": True, "erledigt": False}

    # --- Wahrnehmung ---------------------------------------------------------

    def beobachtung(self) -> str:
        r = self.z.raeume[self.z.ort]
        zeilen = [f"Tag {self.z.tag}, Schritt {self.z.schritt}. Du bist in: {self.z.ort}. {r['beschreibung']}"]
        zeilen.append("Wege: " + ", ".join(r["wege"]) + ".")
        if r["dinge"]:
            zeilen.append("Hier liegt: " + ", ".join(r["dinge"]) + ".")
        if self.z.figur_ort == self.z.ort:
            zeilen.append("Die Wartungsdrohne schwebt hier.")
        zeilen.append("Du trägst: " + (", ".join(self.z.inventar) or "nichts") + ".")
        a = self.z.aufgabe
        if a and not a["erledigt"]:
            zeilen.append(f"Aufgabe: Bringe {a['ding']} nach {a['ziel']}.")
        return "\n".join(zeilen)

    # --- Handeln -------------------------------------------------------------

    def handeln(self, eingabe: str) -> str:
        """Führt eine Handlung aus und gibt das Erlebte als Text zurück."""
        self.z.schritt += 1
        zeile = (eingabe.strip().splitlines() or ["warte"])[0]
        zeile = re.sub(r"^[\W_]+", "", zeile)
        teile = zeile.split(maxsplit=1)
        verb = teile[0].lower() if teile else "warte"
        ziel = teile[1].strip(" .'\"") if len(teile) > 1 else ""
        verben = {
            "gehe": self._gehe, "geh": self._gehe, "untersuche": self._untersuche,
            "nimm": self._nimm, "nehme": self._nimm, "lege": self._lege, "leg": self._lege,
            "sprich": self._sprich, "spreche": self._sprich, "notiere": self._notiere,
            "warte": self._warte,
        }
        ergebnis = verben.get(verb, self._unbekannt)(ziel)
        self._drohne_bewegen()
        return ergebnis

    def _passend(self, name: str, auswahl: list[str]) -> str | None:
        name = name.lower()
        for a in auswahl:
            if a.lower() == name or a.lower() in name:
                return a
        return None

    def _gehe(self, ziel: str) -> str:
        ort = self._passend(ziel, self.z.raeume[self.z.ort]["wege"])
        if not ort:
            return f"Ich wollte nach '{ziel}', aber von {self.z.ort} führt kein Weg dorthin."
        self.z.ort = ort
        return f"Ich ging nach {ort}. {self.z.raeume[ort]['beschreibung']}"

    def _untersuche(self, ziel: str) -> str:
        r = self.z.raeume[self.z.ort]
        ding = self._passend(ziel, r["dinge"] + self.z.inventar)
        if ding:
            return f"Ich untersuchte {ding} in {self.z.ort}. Es wirkt gebraucht, aber intakt."
        return f"Ich sah mich in {self.z.ort} um. {r['beschreibung']}"

    def _nimm(self, ziel: str) -> str:
        r = self.z.raeume[self.z.ort]
        ding = self._passend(ziel, r["dinge"])
        if not ding:
            return f"Ich suchte '{ziel}' in {self.z.ort}, fand es aber nicht."
        r["dinge"].remove(ding)
        self.z.inventar.append(ding)
        self._fakt(f"Wo hast du am {self.z.tag} {ding} aufgehoben?", self.z.ort)
        return f"Ich nahm {ding} in {self.z.ort} an mich."

    def _lege(self, ziel: str) -> str:
        ding = self._passend(ziel, self.z.inventar)
        if not ding:
            return f"Ich wollte '{ziel}' ablegen, trug es aber nicht bei mir."
        self.z.inventar.remove(ding)
        self.z.raeume[self.z.ort]["dinge"].append(ding)
        self._fakt(f"Wo hast du am {self.z.tag} {ding} abgelegt?", self.z.ort)
        text = f"Ich legte {ding} in {self.z.ort} ab."
        a = self.z.aufgabe
        if a and not a["erledigt"] and a["ding"] == ding and a["ziel"] == self.z.ort:
            a["erledigt"] = True
            self._fakt(f"Welche Aufgabe hast du am {self.z.tag} erledigt?",
                       f"{ding} nach {self.z.ort} gebracht")
            text += " Damit war die Aufgabe erledigt."
            self.z.aufgabe = self._neue_aufgabe()
        return text

    def _sprich(self, ziel: str) -> str:
        if self.z.figur_ort != self.z.ort:
            return "Ich sprach, aber niemand war da, der antworten konnte."
        satz = self.zufall.choice(DROHNEN_SAETZE)
        self._fakt(f"Was hat dir die Wartungsdrohne am {self.z.tag} in {self.z.ort} gesagt?", satz)
        return f"Die Wartungsdrohne sagte zu mir: {satz}"

    def _notiere(self, ziel: str) -> str:
        self.z.notizen.append(f"{self.z.tag}: {ziel}")
        return f"Ich notierte: {ziel}"

    def _warte(self, ziel: str) -> str:
        return f"Ich wartete in {self.z.ort}."

    def _unbekannt(self, ziel: str) -> str:
        return "Ich zögerte und tat nichts, was die Station verstand."

    def _drohne_bewegen(self) -> None:
        if self.zufall.random() < 0.3:
            self.z.figur_ort = self.zufall.choice(self.z.raeume[self.z.figur_ort]["wege"])
