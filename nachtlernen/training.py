"""Nächtliches QLoRA-Training einer Variantenfächer.

Jede Nacht entstehen mehrere Varianten des Adapters, von vorsichtig bis mutig.
Sie unterscheiden sich in Lernrate und Anteil neuer Daten; die Breite des
Fächers ist die Schrittweite sigma (siehe ``auslese``). Jede Variante setzt den
Adapter der Vornacht fort, das Grundmodell bleibt unverändert.

``PeftTrainer`` trainiert wirklich (braucht ``pip install .[training]`` und
eine Grafikkarte) und startet dafür einen eigenen Prozess pro Variante, damit
der Grafikspeicher danach vollständig frei ist. ``TrockenTrainer`` schreibt nur
eine Beschreibung des Adapters und dient Tests und Trockenläufen.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class Variante:
    name: str
    mut: float          # 0 = vorsichtig, 1 = mutig
    lernrate: float
    anteil_neu: float
    epochen: int = 1


def faechern(anzahl: int, sigma: float, lernrate: float, max_anteil_neu: float) -> list[Variante]:
    varianten = []
    for i in range(anzahl):
        mut = 0.5 if anzahl == 1 else i / (anzahl - 1)
        varianten.append(Variante(
            name=f"v{i}",
            mut=mut,
            lernrate=lernrate * 2.0 ** (sigma * (mut - 0.5) * 2.0),
            anteil_neu=max_anteil_neu * (0.5 + 0.5 * mut),
        ))
    return varianten


class Trainer(Protocol):
    def trainieren(self, daten: Path, variante: Variante, eltern: Path | None, ziel: Path) -> Path: ...


class TrockenTrainer:
    def trainieren(self, daten: Path, variante: Variante, eltern: Path | None, ziel: Path) -> Path:
        ziel.mkdir(parents=True, exist_ok=True)
        anzahl = sum(1 for z in daten.read_text(encoding="utf-8").splitlines() if z.strip())
        (ziel / "trocken.json").write_text(json.dumps(
            {"variante": asdict(variante), "eltern": str(eltern) if eltern else None,
             "daten": str(daten), "beispiele": anzahl}, ensure_ascii=False, indent=1))
        return ziel


@dataclass
class PeftTrainer:
    gewichte: str            # lokaler Pfad zum Grundmodell
    rang: int = 32
    alpha: int = 64
    max_laenge: int = 2048
    grafikkarten: str = "1"  # CUDA_VISIBLE_DEVICES, z. B. "1" für Karte 2, "0,1" für beide
    saat: int = 0

    def trainieren(self, daten: Path, variante: Variante, eltern: Path | None, ziel: Path) -> Path:
        ziel.mkdir(parents=True, exist_ok=True)
        auftrag = {"gewichte": self.gewichte, "rang": self.rang, "alpha": self.alpha,
                   "max_laenge": self.max_laenge, "saat": self.saat, "daten": str(daten),
                   "variante": asdict(variante), "eltern": str(eltern) if eltern else None,
                   "ziel": str(ziel)}
        umgebung = dict(os.environ, CUDA_VISIBLE_DEVICES=self.grafikkarten)
        with open(ziel / "training.log", "w") as log:
            subprocess.run([sys.executable, "-m", "nachtlernen.training", json.dumps(auftrag)],
                           check=True, env=umgebung, stdout=log, stderr=subprocess.STDOUT)
        return ziel


def _qlora(auftrag: dict) -> None:
    """Läuft im eigenen Prozess. Getestet gegen transformers 4.4x, peft 0.1x, trl >= 0.20."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    v = auftrag["variante"]
    tok = AutoTokenizer.from_pretrained(auftrag["gewichte"])
    modell = AutoModelForCausalLM.from_pretrained(
        auftrag["gewichte"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    modell = prepare_model_for_kbit_training(modell, use_gradient_checkpointing=True)
    if auftrag["eltern"]:
        modell = PeftModel.from_pretrained(modell, auftrag["eltern"], is_trainable=True)
    else:
        modell = get_peft_model(modell, LoraConfig(
            r=auftrag["rang"], lora_alpha=auftrag["alpha"], lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM"))
    daten = load_dataset("json", data_files=auftrag["daten"], split="train").select_columns(["messages"])
    einstellungen = SFTConfig(
        output_dir=str(Path(auftrag["ziel"]) / "_arbeit"),
        num_train_epochs=v["epochen"],
        learning_rate=v["lernrate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        max_length=auftrag["max_laenge"],
        report_to=[],
        seed=auftrag["saat"],
    )
    SFTTrainer(model=modell, args=einstellungen, train_dataset=daten, processing_class=tok).train()
    modell.save_pretrained(auftrag["ziel"])


if __name__ == "__main__":
    _qlora(json.loads(sys.argv[1]))
