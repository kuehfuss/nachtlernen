"""Auslese, bewegliche Grenze und Stammbaum.

Jede Nacht gewinnt die innovativste Variante, deren Kohärenz nicht unter die
Untergrenze fällt und deren Ankerverlust unter 2 % bleibt. Gewinnt keine,
bleibt der bisherige Adapter.

Die Breite des Fächers (sigma) regelt sich nach Rechenbergs 1/5-Erfolgsregel:
Gewinnt die mutige Hälfte über ein Fenster von Nächten häufiger als in einem
Fünftel der Nächte, wird sigma größer, sonst kleiner.

Notbremse: Wird mehrere Male in Folge ein Sieger übernommen, der im selben
Prüfsatz weniger kohärent ist als sein Vorgänger (erlaubt durch die
Toleranz), summiert sich das zu einem schleichenden Zerfall. Dann kehrt das
System zum Adapter vor dieser Folge zurück und halbiert sigma. Verglichen wird
immer im selben Prüfsatz derselben Nacht, weil die Prüfsätze von Nacht zu
Nacht verschieden schwer sind.

Der Stammbaum hält alle Varianten mit Eltern, Messwerten und Status fest. Er
ist zugleich das Archiv für das Neuheitsmaß und die "DNA" des Wesens.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .konfig import NachtKonfig
from .pruefung import Ergebnis


@dataclass
class Knoten:
    id: str
    eltern: str | None
    tag: str
    pfad: str
    status: str                       # "sieger", "verworfen", "ausgeschieden"
    mut: float | None = None
    werte: dict = field(default_factory=dict)
    antworten: dict = field(default_factory=dict)


@dataclass
class Nacht:
    tag: str
    aktiv_vorher: str | None
    aktiv_nachher: str | None
    K_aktiv: float
    mutig_gewann: bool
    notbremse: bool = False
    sigma: float = 1.0
    # Ein neuer Sieger wurde übernommen, obwohl er im selben Prüfsatz unter dem
    # bisherigen Adapter lag (innerhalb der Toleranz).
    abfall: bool = False


@dataclass
class Entscheidung:
    sieger: Ergebnis | None
    grund: str
    notbremse: bool
    sigma_alt: float
    sigma_neu: float
    aktiv: str | None


class Stammbaum:
    def __init__(self, datei: str | Path, sigma_start: float = 1.0):
        self.datei = Path(datei)
        self.sigma = sigma_start
        self.aktiv: str | None = None
        self.anker_basis: float | None = None
        self.knoten: dict[str, Knoten] = {}
        self.naechte: list[Nacht] = []
        if self.datei.exists():
            roh = json.loads(self.datei.read_text(encoding="utf-8"))
            self.sigma = roh["sigma"]
            self.aktiv = roh["aktiv"]
            self.anker_basis = roh.get("anker_basis")
            self.knoten = {k["id"]: Knoten(**k) for k in roh["knoten"]}
            self.naechte = [Nacht(**n) for n in roh["naechte"]]

    def speichern(self) -> None:
        self.datei.parent.mkdir(parents=True, exist_ok=True)
        roh = {"sigma": self.sigma, "aktiv": self.aktiv, "anker_basis": self.anker_basis,
               "knoten": [asdict(k) for k in self.knoten.values()],
               "naechte": [asdict(n) for n in self.naechte]}
        zwischen = self.datei.with_suffix(".tmp")
        zwischen.write_text(json.dumps(roh, ensure_ascii=False, indent=1), encoding="utf-8")
        zwischen.replace(self.datei)

    # --- Abfragen ------------------------------------------------------------

    @property
    def aktiver_knoten(self) -> Knoten | None:
        return self.knoten.get(self.aktiv) if self.aktiv else None

    def sieger(self) -> list[Knoten]:
        return [k for k in self.knoten.values() if k.status == "sieger"]

    def archiv(self) -> list[dict[str, str]]:
        """Antworten aller bisherigen Sieger, für das Neuheitsmaß."""
        return [k.antworten for k in self.sieger() if k.antworten]

    def vorwoche(self, tage: int = 7) -> dict[str, str] | None:
        """Antworten des Siegers, der vor ``tage`` Nächten aktiv war."""
        if not self.naechte:
            return None
        nacht = self.naechte[-tage] if len(self.naechte) >= tage else self.naechte[0]
        k = self.knoten.get(nacht.aktiv_nachher) if nacht.aktiv_nachher else None
        return k.antworten if k else None

    def ahnen(self, knoten_id: str | None) -> list[str]:
        kette = []
        while knoten_id:
            kette.append(knoten_id)
            knoten_id = self.knoten[knoten_id].eltern
        return kette

    # --- Entscheidung --------------------------------------------------------

    def entscheiden(self, tag: str, ergebnisse: Sequence[tuple[Ergebnis, str]],
                    eltern: Ergebnis, k: NachtKonfig) -> Entscheidung:
        """``ergebnisse``: (Prüfergebnis, Adapterpfad) je Variante dieser Nacht."""
        vorher = self.aktiv
        untergrenze = max(k.kohaerenz_min, eltern.K - k.kohaerenz_toleranz)
        zulaessig = []
        for erg, pfad in ergebnisse:
            if erg.ankerverlust > k.max_ankerverlust:
                status = "ausgeschieden"
            elif erg.K < untergrenze:
                status = "verworfen"
            else:
                status = "verworfen"
                zulaessig.append(erg)
            self.knoten[erg.name] = Knoten(
                id=erg.name, eltern=vorher, tag=tag, pfad=pfad, status=status, mut=erg.mut,
                werte={x: getattr(erg, x) for x in ("E", "S", "W", "A", "ankerverlust", "D", "N", "K")},
                antworten=erg.antworten)

        sieger = max(zulaessig, key=lambda e: (e.N, e.K)) if zulaessig else None
        if sieger:
            self.knoten[sieger.name].status = "sieger"
            self.aktiv = sieger.name
            grund = (f"{sieger.name} ist die innovativste Variante über der Untergrenze "
                     f"K ≥ {untergrenze:.3f}.")
        else:
            grund = (f"Keine Variante erreichte K ≥ {untergrenze:.3f} bei Ankerverlust "
                     f"≤ {k.max_ankerverlust:.0%}. Der bisherige Adapter bleibt.")
        K_aktiv = sieger.K if sieger else eltern.K
        mutig = bool(sieger and sieger.mut is not None and sieger.mut > 0.5)
        sigma_alt = self.sigma
        self.naechte.append(Nacht(tag=tag, aktiv_vorher=vorher, aktiv_nachher=self.aktiv,
                                  K_aktiv=K_aktiv, mutig_gewann=mutig, sigma=self.sigma,
                                  abfall=bool(sieger and sieger.K < eltern.K)))

        notbremse = self._notbremse(k)
        if notbremse:
            grund += (f" Notbremse: {k.notbremse_naechte} Sieger in Folge waren weniger kohärent "
                      f"als ihr Vorgänger, zurück zu {self.aktiv or 'dem Grundmodell'}.")
        elif len(self.naechte) % k.sigma_fenster == 0:
            fenster = self.naechte[-k.sigma_fenster:]
            quote = sum(n.mutig_gewann for n in fenster) / len(fenster)
            self.sigma *= 1.2 if quote > 0.2 else 1 / 1.2
        self.sigma = min(k.sigma_max, max(k.sigma_min, self.sigma))
        self.naechte[-1].aktiv_nachher = self.aktiv
        self.naechte[-1].notbremse = notbremse
        return Entscheidung(sieger=sieger, grund=grund, notbremse=notbremse,
                            sigma_alt=sigma_alt, sigma_neu=self.sigma, aktiv=self.aktiv)

    def _notbremse(self, k: NachtKonfig) -> bool:
        n = k.notbremse_naechte
        if not self.naechte or self.naechte[-1].aktiv_vorher == self.naechte[-1].aktiv_nachher:
            return False
        uebernahmen = []
        for nacht in reversed(self.naechte):
            if nacht.notbremse:
                break
            if nacht.aktiv_vorher != nacht.aktiv_nachher:
                uebernahmen.append(nacht)
            if len(uebernahmen) == n:
                break
        if len(uebernahmen) < n or not all(x.abfall for x in uebernahmen):
            return False
        self.aktiv = uebernahmen[-1].aktiv_vorher
        self.sigma /= 2
        return True
