import json

from nachtlernen.auslese import Stammbaum
from nachtlernen.cli import main
from nachtlernen.konfig import NachtKonfig
from nachtlernen.pruefung import Ergebnis
from nachtlernen.training import faechern


def erg(name, K, N=0.0, verlust=0.0, mut=None):
    return Ergebnis(name=name, E=K, S=1.0, W=1.0, A=1.0, ankerverlust=verlust, D=0.0, N=N, K=K,
                    antworten={"Wer bist du?": name}, mut=mut)


def test_faecher_von_vorsichtig_bis_mutig():
    v = faechern(3, sigma=1.0, lernrate=1e-4, max_anteil_neu=0.5)
    assert [x.mut for x in v] == [0.0, 0.5, 1.0]
    assert v[0].lernrate < v[1].lernrate == 1e-4 < v[2].lernrate
    assert all(x.anteil_neu <= 0.5 for x in v)
    breit = faechern(3, sigma=2.0, lernrate=1e-4, max_anteil_neu=0.5)
    assert breit[2].lernrate / breit[0].lernrate > v[2].lernrate / v[0].lernrate


def test_auslese_innovativste_ueber_untergrenze(tmp_path):
    k = NachtKonfig()
    s = Stammbaum(tmp_path / "s.json")
    eltern = erg("bisher", 0.70)
    e = s.entscheiden("2027-07-01", [
        (erg("v0", 0.72, N=0.1, mut=0.0), "a/v0"),
        (erg("v1", 0.68, N=0.3, mut=0.5), "a/v1"),      # innerhalb der Toleranz, innovativer
        (erg("v2", 0.90, N=0.9, verlust=0.03, mut=1.0), "a/v2"),  # Ankerverlust zu groß
        (erg("v3", 0.50, N=0.8, mut=1.0), "a/v3"),      # unter der Untergrenze
    ], eltern, k)
    assert e.sieger.name == "v1" and s.aktiv == "v1"
    assert s.knoten["v2"].status == "ausgeschieden"
    assert s.knoten["v3"].status == "verworfen"
    assert s.naechte[-1].abfall  # 0.68 < 0.70
    s.speichern()
    wieder = Stammbaum(tmp_path / "s.json")
    assert wieder.aktiv == "v1" and wieder.knoten["v1"].eltern is None


def test_kein_sieger_alter_bleibt(tmp_path):
    s = Stammbaum(tmp_path / "s.json")
    s.aktiv = "alt"
    e = s.entscheiden("2027-07-01", [(erg("v0", 0.1, mut=0.0), "p")], erg("alt", 0.7), NachtKonfig())
    assert e.sieger is None and s.aktiv == "alt"


def test_ein_fuenftel_regel(tmp_path):
    k = NachtKonfig(sigma_fenster=5)
    s = Stammbaum(tmp_path / "s.json", sigma_start=1.0)
    for i in range(5):  # mutig gewinnt jede Nacht
        s.entscheiden(f"2027-07-0{i + 1}", [(erg(f"m{i}", 0.8 + i * 0.01, N=1, mut=1.0), "p")],
                      erg("x", 0.8 + i * 0.01 - 0.005), k)
    assert s.sigma == 1.2
    for i in range(5):  # vorsichtig gewinnt jede Nacht
        s.entscheiden(f"2027-07-1{i}", [(erg(f"v{i}", 0.9 + i * 0.01, N=1, mut=0.0), "p")],
                      erg("x", 0.9 + i * 0.01 - 0.005), k)
    assert abs(s.sigma - 1.0) < 1e-9


def test_notbremse_nach_drei_schwaecheren_siegern(tmp_path):
    k = NachtKonfig(notbremse_naechte=3)
    s = Stammbaum(tmp_path / "s.json", sigma_start=2.0)
    s.entscheiden("2027-07-01", [(erg("a", 0.80, mut=0.0), "p")], erg("basis", 0.70), k)
    for i, name in enumerate(["b", "c"]):
        e = s.entscheiden(f"2027-07-0{i + 2}", [(erg(name, 0.79 - i * 0.01, mut=0.0), "p")],
                          erg("vorher", 0.80 - i * 0.01), k)
        assert not e.notbremse
    # Eine Nacht ohne Übernahme unterbricht die Folge nicht.
    s.entscheiden("2027-07-04", [(erg("schlecht", 0.1, mut=0.0), "p")], erg("c", 0.78), k)
    e = s.entscheiden("2027-07-05", [(erg("d", 0.77, mut=0.0), "p")], erg("c", 0.78), k)
    assert e.notbremse and s.aktiv == "a" and s.sigma == 1.0


def test_probelauf_ende_zu_ende(tmp_path, capsys):
    ziel = tmp_path / "probe"
    main(["probelauf", "--tage", "4", "--schritte", "25", "--ziel", str(ziel)])
    stammbaum = json.loads((ziel / "stammbaum.json").read_text())
    assert len(stammbaum["naechte"]) == 4
    assert stammbaum["anker_basis"] == 1.0
    for tag in ("2027-07-01", "2027-07-04"):
        bericht = (ziel / "naechte" / tag / "bericht.md").read_text()
        assert "Entscheidung:" in bericht
        training = (ziel / "naechte" / tag / "training_v0.jsonl").read_text().splitlines()
        anteil_neu = sum(json.loads(z)["herkunft"] == "neu" for z in training) / len(training)
        assert anteil_neu <= 0.5
    # Selbstbild hat sich durch die Nächte verändert und ist versioniert.
    main(["--ordner", str(ziel), "status"])
    assert "Schrittweite sigma" in capsys.readouterr().out
    # Weiterleben mit dem aktiven Adapter und dann Blindtest gegen die Kontroll-Kopie.
    main(["--trocken", "--ordner", str(ziel), "welt", "--schritte", "3", "--tag", "2027-07-05"])
    if stammbaum["aktiv"]:
        main(["--trocken", "--ordner", str(ziel), "blindtest", "--fragen", "8"])
        boegen = list((ziel / "blindtests").glob("*_bogen.md"))
        assert boegen and boegen[0].read_text().count("**A:**") == 8
