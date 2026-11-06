"""Kommandozeile.

  nachtlernen geburt      Lauf anlegen, erstes Selbstbild und Ziele setzen
  nachtlernen sprechen    Gespräch im Terminal (auf dem KI-Rechner)
  nachtlernen welt        Einen Tag in der Textwelt leben
  nachtlernen nacht       Die Nachtschleife für einen Tag ausführen
  nachtlernen status      Stammbaum, sigma und letzte Nächte zeigen
  nachtlernen blindtest   Fragebogen: gewachsenes Modell gegen Kontroll-Kopie
  nachtlernen leitung     KI-Seite der seriellen Textleitung
  nachtlernen gateway     Gateway-Seite der Textleitung (Text von stdin)
  nachtlernen probelauf   Mehrere simulierte Tage und Nächte ohne Grafikkarte

Mit ``--trocken`` laufen alle Befehle mit simuliertem Modell und ohne Training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .auslese import Stammbaum
from .einbettung import erstellen as einbettung_erstellen
from .gedaechtnis import Gedaechtnis
from .konfig import Konfig
from .modell import ServerModell
from .pruefung import SELBST_FRAGEBOGEN, Frage
from .training import PeftTrainer, TrockenTrainer
from .welt import Welt
from .wesen import Wesen, in_welt_leben


def _heute() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def aufbauen(konfig: Konfig, trocken: bool):
    konfig.pfad.mkdir(parents=True, exist_ok=True)
    einbettung = einbettung_erstellen("hash" if trocken else konfig.gedaechtnis.einbettung)
    g = Gedaechtnis(konfig.pfad / "gedaechtnis.sqlite", einbettung, konfig.gedaechtnis.halbwertszeit_tage)
    if trocken:
        from .simulation import SimModell

        modell = SimModell(konfig.nacht.lernrate)
        trainer = TrockenTrainer()
    else:
        m = konfig.modell
        modell = ServerModell(m.url, m.grundmodell, adapter_modus=m.adapter_modus,
                              max_token=m.max_token, temperatur=m.temperatur, zeitlimit_s=m.zeitlimit_s)
        trainer = PeftTrainer(m.gewichte or m.grundmodell, konfig.nacht.lora_rang,
                              konfig.nacht.lora_alpha, konfig.nacht.max_laenge_token)
    return g, modell, trainer


def _wesen(konfig: Konfig, g, modell, kontrolle: bool = False) -> Wesen:
    stammbaum = Stammbaum(konfig.pfad / "stammbaum.json", konfig.nacht.sigma_start)
    aktiv = None if kontrolle else stammbaum.aktiver_knoten
    if aktiv:
        modell.adapter_laden(aktiv.id, aktiv.pfad)
    return Wesen(konfig.name, g, modell, aktiv.id if aktiv else None, konfig.gedaechtnis.abruf_anzahl)


def _welt_fakten(konfig: Konfig) -> list[Frage]:
    welt = Welt(konfig.pfad / "welt", _heute())
    return [Frage(f.frage, f.antwort, "welt", f.tag) for f in welt.fakten()]


# --- Befehle -------------------------------------------------------------------

def cmd_geburt(a, konfig: Konfig) -> None:
    g, _, _ = aufbauen(konfig, a.trocken)
    if g.anzahl() or g.selbst("selbstbild"):
        sys.exit("Dieser Lauf existiert schon. Ein neuer Lauf braucht einen neuen Ordner.")
    g.selbst_setzen("selbstbild", a.selbstbild, "Geburt")
    g.selbst_setzen("ziele", a.ziele, "Geburt")
    (konfig.pfad / "geburt.json").write_text(json.dumps(
        {"zeit": time.time(), "name": konfig.name, "grundmodell": konfig.modell.grundmodell},
        ensure_ascii=False, indent=1))
    print(f"{konfig.name} ist geboren. Ordner: {konfig.pfad}")


def _grundmodell_pruefen(konfig: Konfig) -> None:
    datei = konfig.pfad / "geburt.json"
    if datei.exists():
        geburt = json.loads(datei.read_text())
        if geburt["grundmodell"] != konfig.modell.grundmodell:
            sys.exit(f"Grundmodell geändert ({geburt['grundmodell']} → {konfig.modell.grundmodell}). "
                     "Die Adapter passen dann nicht mehr. Abbruch.")


def cmd_sprechen(a, konfig: Konfig) -> None:
    g, modell, _ = aufbauen(konfig, a.trocken)
    wesen = _wesen(konfig, g, modell, a.kontrolle)
    print(f"Gespräch mit {konfig.name}. Leere Zeile beendet.")
    for zeile in sys.stdin:
        if not zeile.strip():
            break
        print(wesen.sprechen(zeile.strip(), von=a.von, merken=not a.kontrolle))


def cmd_welt(a, konfig: Konfig) -> None:
    g, modell, _ = aufbauen(konfig, a.trocken)
    wesen = _wesen(konfig, g, modell)
    welt = Welt(konfig.pfad / "welt", a.tag or _heute())
    for zeile in in_welt_leben(wesen, welt, a.schritte):
        print(zeile)


def cmd_nacht(a, konfig: Konfig) -> None:
    from .nacht import nacht

    _grundmodell_pruefen(konfig)
    g, modell, trainer = aufbauen(konfig, a.trocken)
    bericht = nacht(konfig, g, modell, trainer, a.tag or _heute(), welt_fakten=_welt_fakten(konfig))
    print(bericht.markdown())


def cmd_status(a, konfig: Konfig) -> None:
    s = Stammbaum(konfig.pfad / "stammbaum.json", konfig.nacht.sigma_start)
    print(f"Aktiver Adapter: {s.aktiv or 'keiner (Grundmodell)'}")
    print(f"Schrittweite sigma: {s.sigma:.2f}")
    print(f"Ahnenreihe: {' ← '.join(s.ahnen(s.aktiv)) or '-'}")
    print(f"Nächte: {len(s.naechte)}, Sieger: {len(s.sieger())}")
    for n in s.naechte[-a.anzahl:]:
        zeichen = " NOTBREMSE" if n.notbremse else (" mutig" if n.mutig_gewann else "")
        print(f"  {n.tag}: K={n.K_aktiv:.3f} aktiv={n.aktiv_nachher}{zeichen}")


def cmd_blindtest(a, konfig: Konfig) -> None:
    """Gleiche Fragen an gewachsenes Modell und Kontroll-Kopie, Reihenfolge verdeckt."""
    g, modell, _ = aufbauen(konfig, a.trocken)
    gewachsen = _wesen(konfig, g, modell)
    kontrolle = _wesen(konfig, g, modell, kontrolle=True)
    if gewachsen.adapter is None:
        sys.exit("Noch kein Adapter: Es gibt nichts zu vergleichen.")
    zufall = random.Random()
    fragen = list(SELBST_FRAGEBOGEN)
    fakten = _welt_fakten(konfig)
    fragen += [f.frage for f in zufall.sample(fakten, min(len(fakten), a.fragen - len(fragen)))]
    bogen, schluessel = ["# Blindtest", ""], {}
    for i, frage in enumerate(fragen[: a.fragen], 1):
        paar = [("gewachsen", gewachsen.sprechen(frage, merken=False)),
                ("kontrolle", kontrolle.sprechen(frage, merken=False))]
        zufall.shuffle(paar)
        schluessel[i] = {"A": paar[0][0], "B": paar[1][0]}
        bogen += [f"## {i}. {frage}", "", f"**A:** {paar[0][1]}", "", f"**B:** {paar[1][1]}", "",
                  "Welche Antwort stammt vom gewachsenen Modell? A / B", ""]
    ordner = konfig.pfad / "blindtests"
    ordner.mkdir(exist_ok=True)
    stempel = datetime.now().strftime("%Y-%m-%d_%H%M")
    (ordner / f"{stempel}_bogen.md").write_text("\n".join(bogen), encoding="utf-8")
    (ordner / f"{stempel}_schluessel.json").write_text(json.dumps(schluessel, indent=1))
    print(f"Bogen: {ordner / f'{stempel}_bogen.md'} (Schlüssel getrennt daneben, erst nach dem Ausfüllen öffnen)")


def cmd_leitung(a, konfig: Konfig) -> None:
    from .leitung import ki_seite, serielle_leitung

    g, modell, _ = aufbauen(konfig, a.trocken)
    wesen = _wesen(konfig, g, modell)
    port = serielle_leitung(a.port, a.baud)
    print(f"Warte auf Nachrichten an {a.port}.")
    ki_seite(port, port, lambda von, text: wesen.sprechen(text, von=von),
             protokoll=konfig.pfad / "leitung.jsonl")


def cmd_gateway(a, konfig: Konfig) -> None:
    from .leitung import gateway_senden, serielle_leitung

    port = serielle_leitung(a.port, a.baud)
    for zeile in sys.stdin:
        if zeile.strip():
            print(gateway_senden(port, port, a.von, zeile.strip(), Path(a.protokoll)))


def cmd_probelauf(a, konfig: Konfig) -> None:
    from .nacht import nacht

    konfig.ordner = a.ziel
    if Path(a.ziel).exists() and any(Path(a.ziel).iterdir()):
        sys.exit(f"{a.ziel} ist nicht leer.")
    g, modell, trainer = aufbauen(konfig, trocken=True)
    g.selbst_setzen("selbstbild", "Ich bin neu und neugierig.", "Geburt")
    g.selbst_setzen("ziele", "Die Station kennenlernen.", "Geburt")
    start = date(2027, 7, 1)
    for t in range(a.tage):
        tag = (start + timedelta(days=t)).isoformat()
        beginn = datetime.fromisoformat(tag).replace(hour=8, tzinfo=timezone.utc).timestamp()
        wesen = _wesen(konfig, g, modell)
        in_welt_leben(wesen, Welt(konfig.pfad / "welt", tag), a.schritte, start=beginn)
        bericht = nacht(konfig, g, modell, trainer, tag, welt_fakten=_welt_fakten(konfig))
        e = bericht.entscheidung
        print(f"{tag}: {bericht.gewaehlt} Erinnerungen gewählt; "
              + (f"{e.grund} sigma {e.sigma_neu:.2f}" if e else bericht.hinweis))
    print(f"\nNachtberichte liegen in {konfig.pfad / 'naechte'}. "
          f"Überblick: nachtlernen --ordner {konfig.pfad} status")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="nachtlernen", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--konfig", help="TOML-Datei (Vorgabe: Werte aus dem Bauplan)")
    p.add_argument("--ordner", help="Arbeitsordner des Laufs (überschreibt die Konfiguration)")
    p.add_argument("--trocken", action="store_true", help="simuliertes Modell, kein Training")
    sub = p.add_subparsers(dest="befehl", required=True)

    s = sub.add_parser("geburt")
    s.add_argument("--selbstbild", required=True)
    s.add_argument("--ziele", required=True)
    s.set_defaults(f=cmd_geburt)

    s = sub.add_parser("sprechen")
    s.add_argument("--von", default="Fabian")
    s.add_argument("--kontrolle", action="store_true", help="als Kontroll-Kopie ohne Adapter, merkt sich nichts")
    s.set_defaults(f=cmd_sprechen)

    s = sub.add_parser("welt")
    s.add_argument("--schritte", type=int, default=100)
    s.add_argument("--tag")
    s.set_defaults(f=cmd_welt)

    s = sub.add_parser("nacht")
    s.add_argument("--tag", help="Tag im Format JJJJ-MM-TT (Vorgabe: heute, UTC)")
    s.set_defaults(f=cmd_nacht)

    s = sub.add_parser("status")
    s.add_argument("--anzahl", type=int, default=10)
    s.set_defaults(f=cmd_status)

    s = sub.add_parser("blindtest")
    s.add_argument("--fragen", type=int, default=30)
    s.set_defaults(f=cmd_blindtest)

    for name, f in (("leitung", cmd_leitung), ("gateway", cmd_gateway)):
        s = sub.add_parser(name)
        s.add_argument("--port", required=True, help="z. B. /dev/ttyUSB0")
        s.add_argument("--baud", type=int, default=115200)
        if name == "gateway":
            s.add_argument("--von", default="Fabian")
            s.add_argument("--protokoll", default="gateway.jsonl")
        s.set_defaults(f=f)

    s = sub.add_parser("probelauf")
    s.add_argument("--tage", type=int, default=14)
    s.add_argument("--schritte", type=int, default=40)
    s.add_argument("--ziel", default="probelauf", help="neuer, leerer Ordner für den Probelauf")
    s.set_defaults(f=cmd_probelauf)

    a = p.parse_args(argv)
    konfig = Konfig.laden(a.konfig)
    if a.ordner and a.befehl != "probelauf":
        konfig.ordner = a.ordner
    a.f(a, konfig)


if __name__ == "__main__":
    main()
