# FruResidual

Predicting sex-specific rewiring in the *Drosophila* connectome from the
*fruitless*/*doublesex* genetic switch — and treating the neurons the
switch can't explain as candidate evidence for a second, uncharacterized
driver of brain sex differences.

**Read the full result: [docs/paper.md](docs/paper.md)** (paper-shaped:
intro, methods, results, honest limitations, novelty statement). This
README covers setup and reproduction; the plan behind the project is in
[docs/project_plan.md](docs/project_plan.md).

![Headline figure](reports/figures/summary_headline.png)

*The figure shows the four data-driven phases. The fifth line of
evidence below — the independent-annotation cross-check — has no figure
of its own; it lives in
[docs/annotation_crosscheck_results.md](docs/annotation_crosscheck_results.md).*

## Headline result

Five independent lines of evidence, each gating the next:

1. **The residual is real and structured** (Phase 0/1) — 38.9-45.6% of
   dimorphic/male-specific neurons carry no *fru*/*dsx* annotation,
   concentrated sharply in the optic lobes (95-100%) and less sharply in
   the VNC/periphery (70-76%), against a well-explained central brain
   (14-32%).
2. **Structure predicts it about as well as the genetic switch does**
   (Phase 2) — a gradient-boosting model gets R²=0.219 from connectivity
   structure alone; adding *fru*/*dsx* as a feature moves that to
   R²=0.216, within noise. (A GNN on the connectome subgraph was tried
   and is reported as a negative result: under 5-fold CV over the graph it
   does not beat that baseline.)
3. **One clean circuit-level candidate** (Phase 3) — `ascending_neuron`
   yields a degree-adjusted, resolution-robust connectivity cluster
   concentrating 60.7% of its residual neurons (odds ratio 2.68,
   empirical p=0.012 against a 1,000-shuffle label-permutation null)
   plus 25 partner-type motifs that survive Cochran-Mantel-Haenszel
   adjustment for connectivity intensity.
4. **Independently replicated** (Phase 4) — BANC, a second female
   reconstruction from a different research group, confirms the residual
   and the *fru*/*dsx*-explained set as dimorphic at statistically
   indistinguishable rates (87.8% vs. 84.8%, p=0.768) — the residual
   isn't disproportionately made of artifacts.

5. **Not just an annotation gap** (cross-check) — an independent
   annotation source agrees the residual lacks fru/dsx for **97.7%** of
   it; the measured ascertainment-bias rate is 2.3%.

No new gene or confirmed mechanism is claimed. See
[docs/paper.md](docs/paper.md) section 6 for exactly what is and isn't
novel here, and section 5 for every limitation stated plainly.

## Project layout

```
configs/            config.yaml — paths, dataset ids, fru/dsx aggregation rule
data/                raw/ interim/ processed/ — gitignored, fetched via scripts/
docs/                paper.md (start here), project_plan.md, phaseN_results.md
notebooks/           scratch space only — the real entry points are scripts/
reports/figures/     small PNG deliverables per phase — tracked in git, unlike data/
scripts/             run_phaseN.py, download_*.py, run_annotation_crosscheck.py, …
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
   ruff check .
   ```
   Both must be clean. Note that `conda activate fruresidual` really is
   required first — calling the env's `python.exe` by absolute path
   without activating it leaves conda's `Library/bin` off `PATH`, and
   scikit-learn then dies with a Windows delay-load failure
   (`0xc06d007f`) deep inside its BLAS calls rather than a normal
   ImportError.

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
That fetches the MaleCNS bulk files the pipeline needs:
`body-annotations-male-cns-v1.0-minconf-0.5.feather` (per-neuron
`dimorphism` and `fruDsx`) and `body-neurotransmitters-male-cns-v1.0.feather`.

Fetched automatically on first use, into the same directory:

- `flywire_annotations.feather` — by `flywire_client.load_flywire_annotations`
- `banc_888_meta.feather` — by `banc_client.load_banc_meta` (Phase 4)

**Fetched manually** (needs a browser login, so no script can do it):

- `codex_labels.csv.gz` — FlyWire Codex → Download Data → Community
  Labels (Raw), with the dataset selector on **FAFB v783**. Required
  only for `run_annotation_crosscheck.py --source codex`. If you
  download other Codex exports and aren't sure which carries the
  fru/dsx annotations, `python scripts/inspect_codex_download.py` scans
  them and tells you which file and column, ranked.

## Phase-by-phase reproduction log

The narrative result lives in [docs/paper.md](docs/paper.md); this
section is the per-phase build log — how to reproduce each phase
individually, and what was found while building it (including bugs
caught and fixed, not just clean headlines).

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

**Phase 2 (predictive model): done.** Run `python scripts/run_phase2_data.py`
then `python scripts/run_phase2.py` to reproduce (needs both
`NEUPRINT_TOKEN` and `CAVE_TOKEN`; the data step takes ~40 min — it
queries live FlyWire connectivity for 371 matched cell types).
Full writeup: [docs/phase2_results.md](docs/phase2_results.md). Figures:
[reports/figures/phase2/](reports/figures/phase2/). Headline: on a
634-case/634-control population (dimorphic vs. isomorphic, matched to
FlyWire types), a gradient-boosting baseline predicts wiring difference
from structure alone at R²=0.219 — and adding fru/dsx as a feature adds
nothing (R²=0.216, within noise). The GNN is reported as a **negative**
result: under 5-fold CV over the graph repeated across 3 seeds it gives
R²=-0.388 ± 0.665 and beats the gradient-boosting baseline on only 1/15 folds — too little data (1,268
nodes) to train stably. Two corrections landed here on 2026-09-19: the
wiring-difference **target itself was rebuilt** (two partner categories
were expressible by FlyWire but structurally unreachable from MaleCNS,
inflating the signal ~4x — the case/control separation and the ablation
conclusion both survive the fix, but all absolute R² values dropped), and
the ridge baseline was un-mis-specified (one-hot encoding plus in-pipeline
scaling). Per-neuron residual scores are saved for Phase 3.

**Phase 3 (hypothesis): done.** Run `python scripts/run_phase3_data.py`
then `python scripts/run_phase3.py` to reproduce (~1 min total; the
first script fetches and caches the connectivity profiles, skipping
anything already cached). Full writeup:
[docs/phase3_results.md](docs/phase3_results.md). Figures:
[reports/figures/phase3/](reports/figures/phase3/). This phase found and
fixed three real bugs before trusting any result: a regional confound
(`ol_intrinsic` was 100% residual — zero internal variance, making any
comparison against it tautological) fixed by stratifying to superclasses
with a genuine residual/explained mix; a multi-label statistical bug
(partner-type membership isn't single-label like superclass — the first
implementation silently counted edges instead of neurons, corrupting
every p-value); and an unvalidated clustering resolution (a fixed,
never-checked value had silently replaced the sensitivity sweep the code
itself called for — fixed by actually sweeping 5 resolutions per stratum
and only trusting a cluster result robust across a majority of them). A
fourth issue — residual neurons being more highly connected overall in 2
of 3 strata — is now **corrected** (2026-09-19) via Cochran-Mantel-Haenszel
testing within degree quartiles, not just flagged. Headline: the cleanest
result is `ascending_neuron` (25 degree-adjusted partner-type motifs,
down from 68 marginal; a connectivity cluster holding 60.7% of residual
vs. 33% background, odds ratio 2.68, robust across the resolution sweep)
— a genuine candidate second-driver signature. `vnc_intrinsic` largely
dissolved under degree adjustment (306 → 44, 86% was artifact), while
`cb_intrinsic` did not (1,397 → 1,429), so its old degree caveat is
resolved — though at 29% of all tested types it reads as a broad
population-level difference rather than a specific circuit. Phase 1's
annotation dimensions are now re-tested within strata rather than folded
in marginally, and unassigned-lineage placeholders (`putative_primary`,
`TBD`) are excluded. Consolidated shortlist (1,531 candidates, all motif
rows degree-adjusted) saved to
`data_processed/phase3_candidate_shortlist.csv`.

**Phase 4 (cross-dataset validation): done.** Run
`python scripts/run_phase4.py` (needs `banc_888_meta.feather`, fetched
automatically on first run — 57.5MB, no auth). Full writeup:
[docs/phase4_results.md](docs/phase4_results.md). Figure:
[reports/figures/phase4/concordance_bar.png](reports/figures/phase4/).
The plan called for re-deriving Phase 2's wiring-difference metric using
BANC connectivity — not possible: our CAVE token authenticates fine but
this account lacks the separate, non-self-serve "view" permission BANC's
API requires. Adapted instead around what BANC's public bulk metadata
*does* expose with no auth: direct neuron-level MaleCNS cross-matches
plus BANC's own independent dimorphism calls. Matched 17,206 neurons
unambiguously. Headline: BANC confirms 87.2% of MaleCNS's dimorphism
calls (recall; precision is lower at 16.5% because BANC's own survey
flags more candidates overall — a real cross-group sensitivity
difference, not disagreement). The actual finding: residual (fru/dsx-
unexplained) and explained neurons are confirmed at statistically
indistinguishable rates (87.8% vs 84.8%, p=0.768) — the residual isn't
disproportionately made of artifacts, supporting that it reflects real,
independently-replicable biology.

**Annotation cross-check (biology vs. ascertainment bias): ANSWERED.**
Run `python scripts/run_annotation_crosscheck.py --source codex --codex-file codex_labels.csv.gz`.
Full writeup: [docs/annotation_crosscheck_results.md](docs/annotation_crosscheck_results.md).
This settles the project's central open question — is the residual real,
or just neurons nobody annotated? Tested against FlyWire **Codex
community labels**, a genuinely independent source (it calls fru/dsx on
13 neurons MaleCNS called negative, and disagrees in both directions).
Result: the residual is 7.3% Codex-positive against 75.7% for the
explained set (p=3.4e-43). **The measured ascertainment-bias rate is
2.3% — the other 97.7% of the residual is called fru/dsx-negative by
both sources independently.** The 13 exceptions are real known fru
neurons (aSP22, DNp48, DNp67…), which is a check that the method
detects what it should.

A first attempt, against FlyWire's *curated* `fru_dsx` column, gave an
apparently overwhelming result (0% vs 80.2%, p=6.7e-67) that had to be
discarded: that column never fires where MaleCNS didn't, so the
residual's 0% was true by construction. The script detects such nesting
automatically and returns INCONCLUSIVE rather than the biology verdict
the statistics would otherwise support. Both runs are documented.

**scRNA-seq route for the optic-lobe gap: assessed and rejected.** Full
writeup: [docs/scrnaseq_route_assessment.md](docs/scrnaseq_route_assessment.md).
The one region Phase 6 could not speak to is the optic lobe (372 of 576
matched residual neurons). Single-cell RNA-seq looked like the answer;
it is not, for three independent reasons. `fru` cannot be measured by 3'
10x at all (its promoters differ at the 5' end, and only P1 is the
sex-specific transcript the connectome annotations mean — the data shows
`fru` in ~48% of all optic cells, a different quantity entirely). The
`dsx` signal is real and sharp but sits in two **glial** clusters
(`repo`/`wrapper`/`alrm`/`moody`, zero neuronal markers), and glia are
not in the connectome. And the cluster→cell-type mapping needed to join
any of it is not in the public files. Assessed in about half an hour
rather than discovered weeks in.

All four data-driven phases (0-4) are now complete.

**Phase 5 (package): done.** Run `python scripts/make_summary_figure.py`
after Phases 1-4 to regenerate the headline figure at the top of this
README. This phase produced [docs/paper.md](docs/paper.md) (the
paper-shaped writeup), the headline figure, this README restructure, and
a repo-wide audit (dead config values, dead docs, lint, a
data-assumption check against the real files) — see the git history for
the audit commit if you want the details of what was checked.

