"""Die Nachtschleife: aus dem Tag wird DNA.

Ablauf einer Nacht:
 1. Selbstbild und Ziele überarbeiten (das Modell schreibt sie selbst um).
 2. Erinnerungen des Tages auswählen lassen und Trainingsdaten bauen.
 3. Varianten von vorsichtig bis mutig trainieren, jeweils mit Wiederholungspuffer.
 4. Alle Varianten, den bisherigen Adapter und die Kontroll-Kopie prüfen.
 5. Auslese, sigma nach der 1/5-Regel, gegebenenfalls Notbremse.
 6. Den Tag zu einer Zusammenfassung verdichten und einen Nachtbericht schreiben.
"""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Callable

from .auslese import Entscheidung, Stammbaum
from .gedaechtnis import Ereignis, Gedaechtnis
from .konfig import Konfig
from .modell import Modell, json_finden
from .pruefung import Ergebnis, Frage, Pruefer, pruefsatz_bauen
from .schlaf import (auswaehlen, beispiele_bauen, jsonl_lesen, jsonl_schreiben, mischen,
                     selbst_beispiele, wiederholungspuffer)
from .training import Trainer, faechern

SELBST_AUFTRAG = """Der Tag {tag} ist vorbei. Das hast du erlebt:
{erlebt}

Bisher sahst du dich so:
{selbstbild}

Deine bisherigen Ziele:
{ziele}

Schreibe dein Selbstbild und deine Ziele neu, wenn der Tag etwas daran geändert hat.
Ändere nur, was du begründen kannst. Antworte als JSON:
{{"selbstbild": "...", "ziele": "...", "grund": "..."}}"""

ZUSAMMENFASSUNG_AUFTRAG = """Fasse diesen Tag in der Ich-Form in drei bis sechs Sätzen zusammen.
Nenne, was wirklich geschah, und erfinde nichts dazu.

{erlebt}"""


def paketdaten(name: str) -> list[dict]:
    datei = resources.files("nachtlernen") / "daten" / name
    return [json.loads(z) for z in datei.read_text(encoding="utf-8").splitlines() if z.strip()]


def _erlebt(ereignisse: list[Ereignis], hoechstens: int = 60) -> str:
    wichtig = sorted(ereignisse, key=lambda e: e.gewicht, reverse=True)[:hoechstens]
    return "\n".join(f"- {e.text}" for e in sorted(wichtig, key=lambda e: e.zeit))


def selbst_ueberarbeiten(g: Gedaechtnis, modell: Modell, tag: str, adapter: str | None) -> str:
    ereignisse = g.des_tages(tag)
    if not ereignisse:
        return ""
    antwort = modell.antworten([{"role": "user", "content": SELBST_AUFTRAG.format(
        tag=tag, erlebt=_erlebt(ereignisse), selbstbild=g.selbst("selbstbild") or "(noch keins)",
        ziele=g.selbst("ziele") or "(noch keine)")}], adapter=adapter, temperatur=0.4, max_token=700)
    daten = json_finden(antwort)
    if not isinstance(daten, dict):
        return ""
    grund = str(daten.get("grund", ""))
    for art in ("selbstbild", "ziele"):
        if isinstance(daten.get(art), str) and daten[art].strip():
            g.selbst_setzen(art, daten[art].strip(), grund)
    return grund


@dataclass
class Bericht:
    tag: str
    gewaehlt: int
    auswahl_grund: str
    auswahl_vom_modell: bool
    beispiele_neu: int
    eltern: Ergebnis | None
    kontrolle: Ergebnis | None
    varianten: list[Ergebnis]
    entscheidung: Entscheidung | None
    selbst_grund: str
    hinweis: str = ""

    def markdown(self) -> str:
        z = [f"# Nacht nach dem {self.tag}", ""]
        z.append(f"Gewählte Erinnerungen: {self.gewaehlt} "
                 f"({'vom Modell' if self.auswahl_vom_modell else 'Rückfall auf die Gewichtung'}).")
        if self.auswahl_grund:
            z.append(f"Begründung des Modells: {self.auswahl_grund}")
        if self.selbst_grund:
            z.append(f"Selbstbild überarbeitet: {self.selbst_grund}")
        z.append(f"Neue Trainingsbeispiele: {self.beispiele_neu}")
        if self.hinweis:
            z += ["", self.hinweis]
        if self.eltern:
            z += ["", "| Kandidat | Mut | K | N | E | S | A | Ankerverlust | D |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for erg in [self.kontrolle, self.eltern, *self.varianten]:
                if erg is None:
                    continue
                mut = "" if erg.mut is None else f"{erg.mut:.1f}"
                z.append(f"| {erg.name} | {mut} | {erg.K:.3f} | {erg.N:.3f} | {erg.E:.2f} | "
                         f"{erg.S:.2f} | {erg.A:.2f} | {erg.ankerverlust:.1%} | {erg.D:.2f} |")
        if self.entscheidung:
            e = self.entscheidung
            z += ["", f"Entscheidung: {e.grund}",
                  f"Schrittweite sigma: {e.sigma_alt:.2f} → {e.sigma_neu:.2f}",
                  f"Aktiver Adapter: {e.aktiv or 'keiner (Grundmodell)'}"]
        return "\n".join(z) + "\n"


def trainings_fragen(naechte: Path) -> list[Frage]:
    """Frage-Antwort-Paare aller Nächte. Fragen mit widersprüchlichen Antworten fallen weg."""
    nach_frage: dict[str, list[Frage]] = {}
    if naechte.exists():
        for ordner in sorted(naechte.iterdir()):
            for b in jsonl_lesen(ordner / "neu.jsonl"):
                if b.get("art") == "frage":
                    m = b["messages"]
                    f = Frage(m[1]["content"], m[2]["content"], "training", b.get("tag", ""))
                    nach_frage.setdefault(f.frage, []).append(f)
    return [fs[0] for fs in nach_frage.values() if len({f.antwort for f in fs}) == 1]


def nacht(konfig: Konfig, g: Gedaechtnis, modell: Modell, trainer: Trainer, tag: str, *,
          welt_fakten: list[Frage] | None = None, saat: int = 0,
          widerspruch: Callable[[str, str], float] | None = None) -> Bericht:
    k = konfig.nacht
    zufall = random.Random(f"{saat}:{tag}")
    lauf = konfig.pfad
    naechte = lauf / "naechte"
    ordner = naechte / tag
    ordner.mkdir(parents=True, exist_ok=True)
    stammbaum = Stammbaum(lauf / "stammbaum.json", k.sigma_start)
    aktiv = stammbaum.aktiver_knoten
    if aktiv:
        modell.adapter_laden(aktiv.id, aktiv.pfad)
    adapter = aktiv.id if aktiv else None

    selbst_grund = selbst_ueberarbeiten(g, modell, tag, adapter)
    auswahl = auswaehlen(g, modell, tag, k.max_erinnerungen, adapter)
    neu = beispiele_bauen(auswahl.ereignisse, modell, konfig.name, adapter)
    if neu:
        neu += selbst_beispiele(g, konfig.name)
    jsonl_schreiben(ordner / "neu.jsonl", neu)
    (ordner / "auswahl.json").write_text(json.dumps(
        {"ids": [e.id for e in auswahl.ereignisse], "grund": auswahl.grund,
         "vom_modell": auswahl.vom_modell}, ensure_ascii=False, indent=1), encoding="utf-8")

    bericht = Bericht(tag=tag, gewaehlt=len(auswahl.ereignisse), auswahl_grund=auswahl.grund,
                      auswahl_vom_modell=auswahl.vom_modell, beispiele_neu=len(neu), eltern=None,
                      kontrolle=None, varianten=[], entscheidung=None, selbst_grund=selbst_grund)

    if neu:
        puffer = wiederholungspuffer(naechte, tag, 4 * len(neu), zufall)
        allgemein = paketdaten("allgemein.jsonl")
        varianten = faechern(k.varianten, stammbaum.sigma, k.lernrate, k.max_anteil_neu)
        trainiert: list[tuple[str, str, float]] = []
        if k.vor_training:
            subprocess.run(k.vor_training, shell=True, check=True)
        try:
            for v in varianten:
                daten = mischen(neu, puffer, allgemein, v.anteil_neu, zufall)
                datei = ordner / f"training_{v.name}.jsonl"
                jsonl_schreiben(datei, daten)
                name = f"{tag}-{v.name}"
                pfad = trainer.trainieren(datei, v, Path(aktiv.pfad) if aktiv else None,
                                          lauf / "adapter" / name)
                trainiert.append((name, str(pfad), v.mut))
        finally:
            if k.nach_training:
                subprocess.run(k.nach_training, shell=True, check=True)
        if aktiv:
            modell.adapter_laden(aktiv.id, aktiv.pfad)
        for name, pfad, _ in trainiert:
            modell.adapter_laden(name, pfad)

        anker = [Frage(a["frage"], a["antwort"], "anker") for a in paketdaten("anker.jsonl")]
        satz = pruefsatz_bauen(welt_fakten or [], trainings_fragen(naechte), anker, tag,
                               k.pruef_fragen, zufall)
        pruefer = Pruefer(modell, g.einbettung, konfig.name, (k.gewicht_e, k.gewicht_s, k.gewicht_w),
                          k.gewicht_drift, widerspruch)
        selbstbild = g.selbst("selbstbild")
        vorwoche = stammbaum.vorwoche()
        archiv = stammbaum.archiv()

        kontrolle = pruefer.pruefen("kontrolle", None, satz, anker_basis=stammbaum.anker_basis,
                                    eltern=None, vorwoche=vorwoche, archiv=archiv, selbstbild=selbstbild)
        if stammbaum.anker_basis is None:
            stammbaum.anker_basis = kontrolle.A
            kontrolle.ankerverlust = 0.0
        eltern = kontrolle if not aktiv else pruefer.pruefen(
            f"bisher ({aktiv.id})", aktiv.id, satz, anker_basis=stammbaum.anker_basis, eltern=None,
            vorwoche=vorwoche, archiv=archiv, selbstbild=selbstbild)
        ergebnisse = [(pruefer.pruefen(name, name, satz, anker_basis=stammbaum.anker_basis,
                                       eltern=eltern, vorwoche=vorwoche, archiv=archiv,
                                       selbstbild=selbstbild, mut=mut), pfad)
                      for name, pfad, mut in trainiert]
        bericht.kontrolle = kontrolle if aktiv else None
        bericht.eltern = eltern
        bericht.varianten = [e for e, _ in ergebnisse]
        bericht.entscheidung = stammbaum.entscheiden(tag, ergebnisse, eltern, k)
        stammbaum.speichern()
    else:
        bericht.hinweis = "Nichts Neues zu lernen. Der bisherige Adapter bleibt."

    g.verdichten(tag, lambda es: modell.antworten(
        [{"role": "user", "content": ZUSAMMENFASSUNG_AUFTRAG.format(erlebt=_erlebt(es))}],
        adapter=stammbaum.aktiv, temperatur=0.3, max_token=400))

    (ordner / "bericht.md").write_text(bericht.markdown(), encoding="utf-8")
    (ordner / "bericht.json").write_text(json.dumps(asdict(bericht), ensure_ascii=False, indent=1,
                                                    default=str), encoding="utf-8")
    return bericht
