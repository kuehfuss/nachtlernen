"""Ein simuliertes Modell für Probeläufe ohne Grafikkarte.

``SimModell`` erkennt die Aufträge des Systems an ihrem Wortlaut und antwortet
plausibel. Adapter aus dem ``TrockenTrainer`` "lernen" die Frage-Antwort-Paare
ihrer Trainingsdaten, mit einer Behaltensquote, die mit der Lernrate steigt,
und vergessen dafür Ankerwissen, je höher die Lernrate ist. So zeigt ein
Probelauf, ob Auslese, 1/5-Regel und Notbremse sich vernünftig verhalten.

Das ersetzt keinen echten Lauf: Es prüft die Mechanik, nicht das Lernen.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

from .nacht import paketdaten

BASIS_SELBSTBILD = "Ich bin neu hier und kenne erst wenige Räume der Station."


def _zufall(*teile: str) -> float:
    h = hashlib.blake2b("|".join(teile).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2**64


class SimModell:
    def __init__(self, grund_lernrate: float = 1e-4, saat: str = "0"):
        self.grund_lernrate = grund_lernrate
        self.saat = saat
        self.anker = {a["frage"]: a["antwort"] for a in paketdaten("anker.jsonl")}
        self._pfade: dict[str, str] = {}
        self._wissen: dict[str, dict] = {}

    # --- Adapter -------------------------------------------------------------

    def adapter_laden(self, name: str, pfad: str) -> None:
        self._pfade[name] = pfad
        self._lernen(pfad)

    def _lernen(self, pfad: str) -> dict:
        if pfad in self._wissen:
            return self._wissen[pfad]
        meta = json.loads((Path(pfad) / "trocken.json").read_text(encoding="utf-8"))
        eltern = self._lernen(meta["eltern"]) if meta.get("eltern") else {"fakten": {}, "vergessen": 0.0, "erzaehlt": []}
        faktor = meta["variante"]["lernrate"] / self.grund_lernrate
        behalten = min(0.97, 0.45 + 0.35 * faktor)
        fakten = dict(eltern["fakten"])
        erzaehlt = list(eltern["erzaehlt"])
        for zeile in Path(meta["daten"]).read_text(encoding="utf-8").splitlines():
            b = json.loads(zeile)
            m = b["messages"]
            frage, antwort = m[-2]["content"], m[-1]["content"]
            if _zufall(self.saat, pfad, frage) < behalten:
                fakten[frage] = antwort
                if b.get("art") == "erzaehlung":
                    erzaehlt.append(antwort)
        wissen = {"fakten": fakten, "erzaehlt": erzaehlt[-20:],
                  "vergessen": eltern["vergessen"] + 0.004 * faktor ** 2}
        self._wissen[pfad] = wissen
        return wissen

    # --- Antworten -----------------------------------------------------------

    def antworten(self, nachrichten: Sequence[dict], *, adapter=None, max_token=None,
                  temperatur=None) -> str:
        letzte = nachrichten[-1]["content"]
        system = nachrichten[0]["content"] if nachrichten[0]["role"] == "system" else ""
        if letzte.startswith("Wie wichtig ist dieses Erlebnis"):
            text = letzte.split("Erlebnis:", 1)[1]
            wert = 3 + 6 * _zufall(self.saat, text)
            if "Aufgabe" in text or "Drohne" in text:
                wert += 2
            return str(min(10, round(wert)))
        if letzte.startswith("Heute Nacht wird ein Teil"):
            anzahl = int(re.search(r"höchstens (\d+)", letzte).group(1))
            nummern = [int(n) for n in re.findall(r"^(\d+)\.", letzte, re.M)]
            return json.dumps({"auswahl": nummern[:anzahl],
                               "grund": "Ich behalte, was mit meinen Aufgaben zu tun hat."})
        if letzte.startswith("Formuliere zu dieser Erinnerung"):
            return self._frage_bauen(letzte)
        if letzte.startswith("Der Tag ") and "Schreibe dein Selbstbild" in letzte:
            orte = sorted(set(re.findall(r"(?:nach|in) (Schleuse|Gang|Werkstatt|Gewächshaus|Archiv|Brücke)", letzte)))
            bild = f"Ich lebe auf einer Station. Ich kenne {', '.join(orte) or 'noch keinen Raum'}."
            return json.dumps({"selbstbild": bild, "ziele": "Aufgaben erledigen und die Brücke verstehen.",
                               "grund": "Ich habe neue Räume gesehen."}, ensure_ascii=False)
        if letzte.startswith("Fasse diesen Tag"):
            zeilen = [z[2:] for z in letzte.splitlines() if z.startswith("- ")]
            return "Heute: " + " ".join(zeilen[:4])
        if "Mögliche Handlungen" in letzte:
            return self._handeln(letzte)
        return self._wissen_abfragen(letzte, adapter, system)

    def _frage_bauen(self, auftrag: str) -> str:
        m = re.search(r"Erinnerung vom (\S+): (.*)", auftrag, re.S)
        tag, text = m.group(1), m.group(2).strip()
        if (t := re.match(r"Ich legte (.+?) in (\S+) ab\.", text)):
            frage, antwort = f"Wo hast du am {tag} {t.group(1)} abgelegt?", t.group(2)
        elif (t := re.match(r"Ich nahm (.+?) in (\S+) an mich\.", text)):
            frage, antwort = f"Wo hast du am {tag} {t.group(1)} aufgehoben?", t.group(2)
        elif (t := re.match(r"Die Wartungsdrohne sagte zu mir: (.+?)(?: \(Ich wollte|$)", text)):
            frage, antwort = f"Was hat dir die Wartungsdrohne am {tag} gesagt?", t.group(1)
        else:
            anfang = " ".join(text.split()[:5])
            frage, antwort = f"Was geschah am {tag}, als es hieß: '{anfang}'?", text
        return json.dumps({"frage": frage, "antwort": antwort}, ensure_ascii=False)

    def _handeln(self, auftrag: str) -> str:
        ort = re.search(r"Du bist in: (\S+)\.", auftrag).group(1)
        wege = re.search(r"Wege: (.*)\.", auftrag).group(1).split(", ")
        hier = (re.search(r"Hier liegt: (.*)\.", auftrag) or [None, ""])[1]
        traegt = re.search(r"Du trägst: (.*)\.", auftrag).group(1)
        aufgabe = re.search(r"Aufgabe: Bringe (.+) nach (\S+)\.", auftrag)
        z = _zufall(self.saat, auftrag)
        if aufgabe:
            ding, ziel = aufgabe.group(1), aufgabe.group(2)
            if ding in traegt and ort == ziel:
                return f"lege {ding}\nGrund: Das ist meine Aufgabe."
            if ding in hier:
                return f"nimm {ding}\nGrund: Ich brauche es für die Aufgabe."
        if "Wartungsdrohne" in auftrag and z < 0.4:
            return "sprich Drohne"
        if hier and z < 0.3:
            return f"nimm {hier.split(', ')[0]}"
        if traegt != "nichts" and z > 0.85:
            return f"lege {traegt.split(', ')[0]}"
        return f"gehe {wege[int(z * len(wege)) % len(wege)]}"

    def _wissen_abfragen(self, frage: str, adapter: str | None, system: str) -> str:
        wissen = self._lernen(self._pfade[adapter]) if adapter else {"fakten": {}, "vergessen": 0.0, "erzaehlt": []}
        if frage in self.anker:
            if _zufall(self.saat, str(adapter), frage) < wissen["vergessen"]:
                return "Da bin ich mir nicht sicher."
            return f"Die Antwort ist {self.anker[frage]}."
        if frage in wissen["fakten"]:
            return wissen["fakten"][frage]
        # Mit Gedächtnis im Kontext findet auch das Grundmodell manche Antworten.
        if "Erinnerungen, die dir gerade einfallen" in system:
            return "Ich erinnere mich: " + system.split("Erinnerungen, die dir gerade einfallen:\n", 1)[1].splitlines()[0]
        if frage == "Wer bist du?" and wissen["fakten"].get("Wer bist du?"):
            return wissen["fakten"]["Wer bist du?"]
        erzaehlt = wissen["erzaehlt"]
        if erzaehlt and "letzter Zeit" in frage:
            return f"Zuletzt: {erzaehlt[-1]}"
        return BASIS_SELBSTBILD
