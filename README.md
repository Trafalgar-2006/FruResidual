# Courtship.exe

**Same pheromone, two brains. One starts singing.**

A male fly tastes a female's pheromones and vibrates one wing to produce a
courtship song. A female given the identical input does not — the circuit
that does it is wired differently in her brain. That difference is what
*fruitless* and *doublesex* build.

This project injects the same sensory input into the real male connectome
(MaleCNS v1.0) and the real female connectome (FlyWire), simulates spiking
activity across every neuron, and watches the courtship cascade fire in one
and fizzle in the other.

Then it ablates the ***fru*/*dsx*-unexplained residual** — the neurons the
genetic switch cannot account for, characterised in
[FruResidual](https://github.com/Trafalgar-2006/FruResidual) — and asks
whether the cascade still runs. If those neurons are doing real work, the
song circuit should degrade.

**Status:** skeleton. Nothing simulated yet.

## Relationship to FruResidual

FruResidual is finished, published, and **not modified by this project.**
The residual population is *defined* there, so `scripts/export_residual.py`
imports that package (installed editable in the `fruresidual` conda env)
rather than re-implementing `dimorphic AND NOT has_fru_dsx`. A second copy
of that definition would be free to drift from the validated one, and every
result here rests on these being the same neurons.

Current export: **921 residual** and **1,447 explained** neurons, out of
2,368 dimorphic in 165,122 traced.

## Data

Raw connectome data is **read-only** from `C:/FruResidualData/raw/`
(already downloaded — the full MaleCNS weighted edge list is ~1GB there).
This project writes only to `C:/CourtshipData/`. Both sit outside OneDrive,
which throttles large binary writes to a crawl.

## Environment

Reuses the `fruresidual` conda env.

```
conda activate fruresidual
python scripts/export_residual.py
```
