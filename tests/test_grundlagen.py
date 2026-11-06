import io
import json
import random

import pytest

from nachtlernen.einbettung import HashEinbettung, kosinus
from nachtlernen.gedaechtnis import Gedaechtnis
from nachtlernen.konfig import Konfig
from nachtlernen.leitung import bereinigen, gateway_senden, ki_seite
from nachtlernen.modell import SkriptModell, json_finden, zahl_finden
from nachtlernen.pruefung import richtig
from nachtlernen.schlaf import auswaehlen, mischen
from nachtlernen.welt import Welt

TAG = 1_800_000_000.0  # fester Zeitpunkt (2027-01-15 UTC)


@pytest.fixture
def g(tmp_path):
    gd = Gedaechtnis(tmp_path / "g.sqlite", HashEinbettung())
    yield gd
    gd.schliessen()


def test_einbettung_aehnlich():
    e = HashEinbettung()
    a, b, c = e.einbetten(["Ich ging in die Werkstatt", "Ich ging zur Werkstatt", "Bohnen unter rotem Licht"])
    assert kosinus(a, b) > kosinus(a, c)


def test_gedaechtnis_abruf_und_ueberraschung(g):
    erste = g.merken("Die Drohne sagte, die Pumpe im Gewächshaus sei kaputt.", wichtigkeit=8, zeit=TAG)
    zweite = g.merken("Die Drohne sagte, die Pumpe im Gewächshaus sei kaputt.", wichtigkeit=8, zeit=TAG + 1)
    g.merken("Ich legte den Schraubenschlüssel in die Werkstatt.", wichtigkeit=3, zeit=TAG + 2)
    e1, e2 = g.holen([erste, zweite])
    assert e1.ueberraschung == 1.0
    assert e2.ueberraschung < 0.05  # Wiederholung überrascht nicht
    treffer = g.abrufen("Was ist mit der Pumpe?", anzahl=1, jetzt=TAG + 10)
    assert "Pumpe" in treffer[0].text
    assert g.holen([treffer[0].id])[0].wiederholungen == 1


def test_wichtigste_und_verdichtung(g):
    for i, w in enumerate([2, 9, 5]):
        g.merken(f"Ereignis Nummer {i} mit Gewicht {w}", wichtigkeit=w, ueberraschung=0.5, zeit=TAG + i)
    tag = g.des_tages("2027-01-15")[0].tag
    assert [e.wichtigkeit for e in g.wichtigste_des_tages(tag, 2)] == [9, 5]
    neu = g.verdichten(tag, lambda es: f"{len(es)} Dinge geschahen.")
    assert neu and g.holen([neu])[0].ebene == 1
    assert g.verdichten(tag, lambda es: "nochmal") is None


def test_selbstbild_versioniert(g):
    g.selbst_setzen("selbstbild", "Ich bin neu.", "Geburt")
    g.selbst_setzen("selbstbild", "Ich bin neu.", "keine Änderung")
    g.selbst_setzen("selbstbild", "Ich kenne die Werkstatt.", "Tag 1")
    assert g.selbst("selbstbild") == "Ich kenne die Werkstatt."
    assert [v[2] for v in g.selbst_verlauf("selbstbild")] == ["Geburt", "Tag 1"]
    with pytest.raises(ValueError):
        g.selbst_setzen("traeume", "x")


def test_json_und_zahl_finden():
    assert json_finden('Gern! {"auswahl": [1, 3], "grund": "weil"} Fertig.') == {"auswahl": [1, 3], "grund": "weil"}
    assert json_finden("Ich wähle [2, 4].") == [2, 4]
    assert json_finden("keine Angabe") is None
    assert zahl_finden("Ich würde sagen 7.", 5, 1, 10) == 7
    assert zahl_finden("sehr wichtig", 5, 1, 10) == 5
    assert zahl_finden("42", 5, 1, 10) == 10


def test_richtig():
    assert richtig("Ich habe ihn in der Werkstatt abgelegt.", "Werkstatt")
    assert not richtig("Im Archiv.", "Werkstatt")
    assert richtig("Die Antwort ist 144.", "144")
    assert richtig("Häuser natürlich", "Häuser")


def test_auswahl_vom_modell_und_rueckfall(g):
    for i in range(6):
        g.merken(f"Erlebnis {i} am Morgen", wichtigkeit=i + 1, ueberraschung=0.5, zeit=TAG + i)
    tag = "2027-01-15"
    gut = SkriptModell(lambda n, a: '{"auswahl": [2, 2, 99, 1], "grund": "Das zählt."}')
    a = auswaehlen(g, gut, tag, obergrenze=3)
    assert a.vom_modell and a.grund == "Das zählt."
    assert [e.wichtigkeit for e in a.ereignisse] == [5, 6]  # Duplikat und Unsinn verworfen
    schlecht = SkriptModell(lambda n, a: "Ich kann mich nicht entscheiden.")
    b = auswaehlen(g, schlecht, tag, obergrenze=2)
    assert not b.vom_modell and [e.wichtigkeit for e in b.ereignisse] == [6, 5]


def test_mischen_hoechstens_haelfte_neu():
    neu = [{"messages": [], "herkunft": "neu", "n": i} for i in range(10)]
    alt = [{"messages": [], "herkunft": "neu", "a": i} for i in range(3)]
    allg = [{"messages": [], "art": "allgemein"}]
    for anteil in (0.25, 0.5):
        m = mischen(neu, alt, allg, anteil, random.Random(1))
        assert sum(b["herkunft"] == "neu" for b in m) / len(m) <= anteil + 1e-9
    with pytest.raises(ValueError):
        mischen(neu, [], [], 0.5, random.Random(1))


def test_welt_fakten_und_aufgabe(tmp_path):
    w = Welt(tmp_path, "2027-01-15", saat=3)
    w.z.aufgabe = {"ding": "Helmlampe", "ziel": "Gang", "loesbar": True, "erledigt": False}
    assert "Helmlampe" in w.beobachtung()
    assert "nahm Helmlampe" in w.handeln("nimm Helmlampe")
    assert "kein Weg" in w.handeln("gehe Brücke")
    w.handeln("gehe Gang\nGrund: Aufgabe")
    assert "Aufgabe erledigt" in w.handeln("lege Helmlampe")
    fragen = {f.frage: f.antwort for f in w.fakten()}
    assert fragen["Wo hast du am 2027-01-15 Helmlampe abgelegt?"] == "Gang"
    assert "Unsinn" not in w.handeln("_fakt Unsinn")  # interne Methoden sind nicht erreichbar
    w.speichern()
    w2 = Welt(tmp_path, "2027-01-16", saat=3)
    assert w2.z.ort == "Gang" and w2.z.tag == "2027-01-16"


def test_leitung_hin_und_zurueck(tmp_path):
    rein = io.BytesIO()
    for text in ["Hallo\x07 Wesen", "", "x" * 10]:
        rein.write((json.dumps({"von": "Fabian", "text": text}) + "\n").encode())
    rein.write(b"kein json\n")
    rein.seek(0)
    raus = io.BytesIO()
    erhalten = []
    n = ki_seite(rein, raus, lambda von, t: erhalten.append((von, t)) or f"Echo: {t}",
                 protokoll=tmp_path / "p.jsonl")
    assert n == 2
    assert erhalten[0] == ("Fabian", "Hallo Wesen")
    antworten = [json.loads(z) for z in raus.getvalue().decode().splitlines()]
    assert antworten[0]["antwort"] == "Echo: Hallo Wesen"
    assert antworten[1]["fehler"] and antworten[3]["fehler"]
    assert len((tmp_path / "p.jsonl").read_text().splitlines()) == 4

    gw_raus = io.BytesIO()
    antwort = gateway_senden(io.BytesIO(b'{"antwort": "Guten Abend"}\n'), gw_raus, "Fabian", "Hi")
    assert antwort == "Guten Abend"
    assert json.loads(gw_raus.getvalue()) == {"von": "Fabian", "text": "Hi"}
    assert bereinigen("a\x1b[31mb", 3) == "a[3"


def test_konfig_laden(tmp_path):
    datei = tmp_path / "k.toml"
    datei.write_text('name = "Ilse"\n[nacht]\nvarianten = 5\n[modell]\nurl = "http://127.0.0.1:9000"\n')
    k = Konfig.laden(datei)
    assert k.name == "Ilse" and k.nacht.varianten == 5 and k.nacht.max_anteil_neu == 0.5
    assert k.modell.url.endswith("9000")
    datei.write_text("[nacht]\ntippfehler = 1\n")
    with pytest.raises(ValueError):
        Konfig.laden(datei)
