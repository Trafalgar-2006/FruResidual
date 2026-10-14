"""Assemble the deployable static site from generated results.

The site is deliberately static: every number and every sound was
produced by the real simulation ahead of time, so serving it needs no
GPU, no server and no backend. That is why it can go on any static host
and still be working in two years.

Run after make_library.py and make_demo_data.py.

Usage:  python scripts/build_site.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "reports" / "library"
DEMO = ROOT / "reports" / "demo_data.json"
VALIDATION = ROOT / "models" / "male_courting_dt0.2_validation.json"
SITE = ROOT / "site"

NICE_STIMULUS = {
    "courtship_song": ("Courtship song",
                       ("A male's own song, built from its published structure: ~250 Hz "
                        "pulses about 35 ms apart alternating with a ~150 Hz hum. The "
                        "stimulus this circuit evolved to process.")),
    "tone_sequence": ("Tone sequence", "Eight notes, to test whether pitch survives the circuit."),
    "steady_tone": ("Steady tone", "300 Hz held flat — the most predictable input."),
    "rising_sweep": ("Rising sweep", "120 Hz climbing to 620 Hz."),
    "beat": ("Beat", "250 Hz gated at 5 Hz — pure rhythm, no melody."),
}

NICE_CONDITION = {
    ("male", "courting"): ("Courting male", "Ears plus the P1 command. He sings."),
    ("male", "hearing"): ("Male, listening", "Ears only. Read at the brain's output channel."),
    ("male", "hearing_only"): ("Male, not courting", "Ears only, read at the wing. Silent."),
    ("female", "hearing"): ("Female, listening",
                            ("Ears only. She has no song circuit, and FlyWire is brain-only "
                             "with no motor neurons at all.")),
}


def main() -> None:
    if not (LIBRARY / "library.json").exists():
        raise SystemExit("run scripts/make_library.py first")

    lib = json.loads((LIBRARY / "library.json").read_text())
    demo = json.loads(DEMO.read_text()) if DEMO.exists() else {}
    validation = json.loads(VALIDATION.read_text()) if VALIDATION.exists() else {}

    SITE.mkdir(parents=True, exist_ok=True)
    audio_dir = SITE / "audio"
    if audio_dir.exists():
        shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True)

    wavs = 0
    for w in LIBRARY.glob("*.wav"):
        shutil.copy2(w, audio_dir / w.name)
        wavs += 1

    for e in lib["entries"]:
        label, blurb = NICE_CONDITION.get((e["sex"], e["condition"]),
                                          (e["condition"], ""))
        e["label"], e["blurb"] = label, blurb
    for name, meta in lib["stimuli"].items():
        title, blurb = NICE_STIMULUS.get(name, (name, ""))
        meta["title"], meta["blurb"] = title, blurb

    bundle = {
        "library": lib,
        "meta": demo.get("meta", {}),
        "traces": demo.get("traces", {}),
        "stages": demo.get("stages", []),
        "ablation": demo.get("ablation", {}),
        "breakdown": demo.get("breakdown", {}),
        "sex": demo.get("sex", {}),
        "calibration": demo.get("calibration", {}),
        "validation": validation,
    }
    (SITE / "data.js").write_text(
        "window.FLY=" + json.dumps(bundle, separators=(",", ":")) + ";")

    size = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file())
    print(f"site/        {len(list(SITE.rglob('*')))} files, {size/1e6:.1f} MB")
    print(f"  audio/     {wavs} wav files")
    print(f"  data.js    {(SITE / 'data.js').stat().st_size/1024:.0f} KB")
    print(f"  entries    {len(lib['entries'])}")
    if not (SITE / "index.html").exists():
        print("\nindex.html is missing -- it is hand-written and lives in the repo")


if __name__ == "__main__":
    main()
