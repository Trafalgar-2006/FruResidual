# Courtship.exe

**Same command, two outcomes. Which neurons decide whether the fly sings?**

A male fly courts by vibrating one wing to produce a song. The circuit is
known end to end: pheromone input → **pC1/P1** command neurons → **pIP10**
descending neuron → **TN1/dPR1** nerve-cord premotor → **wing motor
neurons**. The *fruitless*/*doublesex* genetic switch builds that circuit,
which is why females given the same input do not sing.

This project runs that pathway as a spiking simulation on the **real male
connectome** — 165,122 neurons, 25.5 million signed synaptic connections —
and then silences specific populations to ask which ones the song depends
on.

The population of interest is the ***fru*/*dsx*-unexplained residual***:
neurons that are sexually dimorphic but carry no *fru*/*dsx* annotation,
characterised in [FruResidual](https://github.com/Trafalgar-2006/FruResidual).
If the genetic switch is the whole story, silencing them should do
nothing.

## Why MaleCNS and not FlyWire

Every published connectome simulation runs on FlyWire, which is
**brain-only**. TN1, dPR1 and the wing motor neurons are all nerve-cord
cells — that half of the song circuit does not exist there. FlyGM, the
current connectome-based motor controller, lists 106 neurons in motor
classes and wires brain output straight to a simulated body. MaleCNS has
the whole CNS: **909 motor and efferent neurons**, and the real
descending → VNC → muscle pathway.

## What the model does and doesn't earn

The connectome reproduces published circuitry it was never told about:
pIP10's strongest single target comes out as dPR1, a song-pathway neuron
that was never searched for. Reverse directions along the pathway measure
exactly 0 synapses.

The synapse gain is **calibrated on specificity**, not assumed — the song
readout responds to the courtship command at **128×** the rate it
responds to size-matched control populations. Turning the gain up gives a
louder signal and a worse experiment: at 0.275 the readout can no longer
tell courtship from noise at all.

Full detail, including what this model cannot support, is in
[docs/methods.md](docs/methods.md).

## Status

| | |
|---|---|
| connectome built + verified | done |
| circuit located + validated | done |
| simulator calibrated | done |
| ablation experiment | running |
| brain-only vs whole-CNS | written, not yet run |
| female (FlyWire) comparison | **blocked** — Zenodo 504 |

## Relationship to FruResidual

FruResidual is finished, published, and **not modified by this project.**
The residual population is defined there, so `scripts/export_residual.py`
imports that package rather than re-implementing
`dimorphic AND NOT has_fru_dsx`. A second copy of that definition would be
free to drift from the validated one, and every result here depends on
these being the same neurons. Current export: **921 residual**, **1,447
explained**, of 2,368 dimorphic in 165,122 traced.

## Data

Raw connectome data is read **read-only** from `C:/FruResidualData/raw/`.
This project writes only to `C:/CourtshipData/`. Both sit outside
OneDrive, which throttles large binary writes.

## Running it

```
conda activate fruresidual

python scripts/export_residual.py     # residual IDs, from FruResidual
python scripts/build_connectome.py    # 165,122 x 165,122 signed sparse
python scripts/verify_connectome.py   # transposition gate -- run this
python scripts/calibrate_gain.py      # gain sweep + specificity check
python scripts/run_ablation.py        # the experiment
python scripts/run_brain_only.py      # does amputating the VNC matter?
python scripts/make_figures.py

python -m pytest tests/ -q
```
