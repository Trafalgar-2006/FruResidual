# FruResidual

Predicting sex-specific rewiring in the *Drosophila* connectome from the
*fruitless*/*doublesex* genetic switch — and treating the neurons the
switch can't explain as candidate evidence for a second, uncharacterized
driver of brain sex differences.

Full plan, data sources, and background reading: [docs/project_plan.md](docs/project_plan.md).

## Project layout

```
configs/            config.yaml — paths, dataset ids, Phase-0 aggregation rule
data/                raw/ interim/ processed/ — gitignored, fetched via scripts/
docs/                project_plan.md, phaseN_results.md — plan + per-phase writeups
notebooks/           one notebook per phase; thin, calls into src/fruresidual
reports/figures/     small PNG deliverables per phase — tracked in git, unlike data/
scripts/             run_phaseN.py, download_*.py — standalone entry points
src/fruresidual/     the actual package — data clients, features, models, eval, viz
tests/               pytest
```

## Environment setup (native Windows + conda/mamba)

Verified working end-to-end on 2026-09-16 (RTX 3050 Laptop GPU, driver
CUDA 13.3). Steps below are exactly what was run.

1. Install Miniforge (conda + mamba):
   ```
   winget install --id=CondaForge.Miniforge3 -e
   ```
2. Create the base environment (no PyTorch yet — see step 3):
   ```
   mamba env create -f environment.yml
   conda activate fruresidual
   ```
3. Install PyTorch via pip, **not conda** — conda-forge's `pytorch` +
   `pytorch-cuda=12.1` combo resolved to a CPU-only build here despite
   the CUDA pin. Pick the newest `whl/cuXXX` tag on
   [download.pytorch.org](https://download.pytorch.org/whl/torch/) that
   has a version matching (or close to) what `environment.yml` implies;
   `cu130` is what worked here:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu130
   ```
   Verify it worked before moving on:
   ```
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   This must print `True` and your GPU name. If it prints `False`, stop
   and fix this before continuing — everything downstream assumes GPU
   is visible.
4. Install PyTorch Geometric (must come after step 3, since it builds
   against the installed torch):
   ```
   pip install torch_geometric
   ```
5. Install the connectomics-specific clients — kept as separate installs
   so one flaky package can't break the rest; install one at a time:
   ```
   pip install navis
   pip install fafbseg   # pulls in caveclient as a dependency
   pip install banc
   ```
6. Copy `.env.example` to `.env` and fill in your neuPrint / CAVE tokens
   (see comments in that file for where to get them).
7. Sanity check everything at once:
   ```
   python -m pytest
   ```

### Compute tiers

- **Local (this laptop, RTX 3050 4GB):** day-to-day dev — data pulls,
  joins, atlas building, baselines, GNN prototyping via mini-batch
  neighbor sampling.
- **Colab / Kaggle (free tier):** Linux-native fallback, zero install
  friction, bigger VRAM ceiling on demand.
- **College A5000 (24GB):** full GNN hyperparameter sweeps, the
  degree-preserving null-model reruns for significance testing, and
  Phase-4 cross-dataset validation runs.

## Data

Fetched separately from the repo, into `C:/FruResidualData` (see
"Environment setup" note above — kept off OneDrive so large downloads
aren't throttled). Get it with:
```
python scripts/download_male_cns.py   # MaleCNS bulk files, no auth needed
```
The FlyWire `flywire_annotations` table is fetched on first use by
`fruresidual.data.flywire_client.load_flywire_annotations` (also cached
into the same directory).

## Status

**Phase 0 (kill-gate): ALIVE.** Run `python scripts/run_phase0.py` to
reproduce. Full writeup: [docs/phase0_results.md](docs/phase0_results.md).
Headline: 38.9-45.6% of dimorphic/male-specific MaleCNS neurons have no
fru/dsx annotation (robust across both aggregation rules), and that
residual is statistically enriched (Fisher's exact, BH-corrected) in
optic-lobe-intrinsic neurons — essentially all of which lack a fru/dsx
call — plus smaller enrichment in ascending/descending VNC-connector
neurons. Non-trivial and structured, so the project proceeds to Phase 1.

**Phase 1 (atlas): done.** Run `python scripts/run_phase1.py` to
reproduce (needs `NEUPRINT_TOKEN` for a targeted, fast ROI query — not a
full-connectome pull). Full writeup:
[docs/phase1_results.md](docs/phase1_results.md). Figures:
[reports/figures/phase1/](reports/figures/phase1/). Headline: the
residual isn't just "optic lobe" — it's a peripheral-sensorimotor
pattern. Optic-lobe neuropils (ME, LO) are 95-100% fru/dsx-unexplained;
VNC/gnathal-ganglion/abdominal-tract regions sit at 70-76%; central-brain
integrative neuropils (SIP, AVLP, SNP) sit at 14-32%, closely explained.
Same pattern confirmed independently via superclass and soma-neuromere
enrichment. Doesn't yet distinguish real biology from an fru/dsx
annotation-coverage bias toward the central brain (the plan already
flags ~75% of annotation effort landed there) — that's Phase 3's job.

Next: Phase 2, the predictive model — see docs/project_plan.md section (d).
