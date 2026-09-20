# Biology vs. ascertainment bias: the independent-annotation cross-check

Runs: 2026-09-19 (curated table, inconclusive) and **2026-09-20 (Codex
community labels — this one answers the question)**, via
`scripts/run_annotation_crosscheck.py`.

---

## ANSWER (2026-09-20): the residual is mostly not ascertainment bias

Using FlyWire **Codex community labels** as the second source — a
genuinely independent annotation, unlike the curated table — the test is
admissible and it comes back:

| population | n matched | Codex says fru/dsx+ |
|---|---|---|
| residual (MaleCNS fru/dsx−) | 178 | **13 (7.3%)** |
| explained (MaleCNS fru/dsx+) | 177 | 134 (75.7%) |

(restricted to super_classes where Codex demonstrably annotates fru/dsx;
Fisher p = 3.4e-43. Unrestricted: 13/576 vs 136/189, p = 1.5e-89.)

**Why this one counts and the earlier one did not:**
`independence_diagnostic` reports **PARTIALLY INDEPENDENT** —
`n_malecns_neg_flywire_pos = 13`, i.e. Codex calls fru/dsx on 13 neurons
MaleCNS called negative. It is demonstrably *capable* of catching
MaleCNS's misses. It disagrees in both directions (53 the other way,
8.6% overall). Nothing is true by construction here.

**The measured ascertainment-bias rate is 2.3%.** Of 576 residual
neurons matched to FlyWire, exactly 13 turn out to carry an independent
fru/dsx call that MaleCNS missed. The other **97.7% are called
fru/dsx-negative by both sources independently**.

Those 13 are real and biologically sensible, which is itself a check on
the method — they are known fru neurons, not noise:

| bodyId(s) | type | superclass |
|---|---|---|
| 10090, 10104 | **aSP22** — fru+ courtship descending neuron (McKellar et al. 2019) | descending_neuron |
| 10680, 10823 | DNp48 | descending_neuron |
| 12325, 12435 | DNp67 | descending_neuron |
| 13432 | DNg66 | descending_neuron |
| 10406, 531863 | AOTU012 | cb_intrinsic |
| 13388, 14935 | SMP551 / SLPpm3_H01 | cb_intrinsic |
| 21860, 22040 | LHAV2b2_c | cb_intrinsic |

**The obvious objection, tested:** Codex is the *smaller* annotation set
(1,226 positives vs the curated table's 3,308), so could its low hit-rate
on the residual just be sparseness? No — the explained set is the control
for exactly that, same source and same sparseness, and Codex finds
**72.0%** of it. If sparseness drove the residual's rate, the explained
set would be low too.

**What this does and does not establish.** It rules out the strong
version of the ascertainment-bias explanation: the residual is not simply
"neurons MaleCNS forgot to annotate", because an independent source that
provably can catch such misses catches them for only 2.3% of it. It does
not prove a second genetic driver exists — both sources could share a
blind spot inherited from the same underlying light-microscopy
literature, and neither has any real coverage in the optic lobe
(Codex: 0.01%), where most of the residual lives. Read together with
Phase 3's degree-adjusted `cb_intrinsic` result, the two independent
lines now point the same way.

---

## The 2026-09-19 run, kept because the failure is instructive

This targets the project's central open question, flagged since Phase 1:
is the fru/dsx-unexplained residual **real biology**, or an artifact of
where MaleCNS's annotation effort was spent (~75% of it in the central
brain)? One annotation source cannot tell those apart. The idea here was
to re-check the residual against a second source — FlyWire's own
`fru_dsx` column — on the reasoning that:

- if residual neurons' FlyWire counterparts are often fru/dsx-**positive**,
  MaleCNS simply missed them → **ascertainment bias**;
- if they are negative in **both**, two annotations agree → **biology**.

## The headline numbers look decisive. They are not.

| population | n matched | FlyWire-positive |
|---|---|---|
| residual (MaleCNS fru/dsx−) | 576 | **0 (0.0%)** |
| explained (MaleCNS fru/dsx+) | 189 | 148 (78.3%) |

Restricted to super_classes where FlyWire demonstrably *does* annotate
fru/dsx (≥1% coverage: central, descending, sensory, visual_projection),
which controls the obvious "both datasets share the same blind spot"
objection:

| population | n | FlyWire-positive | expected if it behaved like explained |
|---|---|---|---|
| residual | 182 | **0 (0.0%)** | 146.0 |
| explained | 177 | 142 (80.2%) | — |

Fisher p = 6.7e-67. Within `central` alone — FlyWire's best-covered
stratum at 7.71% — explained is 80.5% positive and residual is 0.0%
(p=1.9e-42).

Taken at face value that is overwhelming support for biology. **It
should not be taken at face value.**

## Why the test does not work: the second annotation is not independent

`docs/project_plan.md` section (b.2) records that flywire_annotations'
`fru_dsx` column was added "via cross-validation against MaleCNS". The
data confirms it, and the giveaway is an asymmetry, not an overall
agreement rate:

| | count |
|---|---|
| MaleCNS+ / FlyWire+ | 148 |
| MaleCNS+ / FlyWire− | 41 |
| MaleCNS− / FlyWire+ | **0** |
| MaleCNS− / FlyWire− | 576 |

FlyWire calls fru/dsx on **zero** neurons that MaleCNS called negative.
Its labels are a strict *subset* of MaleCNS's. So

> P(FlyWire-positive | MaleCNS-negative) = 0 **by construction**

and "the residual is 0% FlyWire-positive" is not a finding — it is a
restatement of how the column was built. The p-value of 6.7e-67 is
measuring the annotation's provenance, not biology. `run_annotation_crosscheck.py`
detects this automatically (`flywire_labels_nested_within_malecns`) and
returns **INCONCLUSIVE** rather than the biology verdict the raw
statistics would otherwise support.

The 41 MaleCNS+/FlyWire− cases show the column is not a *pure* copy —
FlyWire does drop calls MaleCNS makes — so it carries some information
of its own. But information flows one way only, which is exactly the
direction that cannot help here.

## What this rules in and out

- **Ruled out:** this particular test as evidence either way. Reporting
  the 0% vs 80% contrast as independent confirmation would have been a
  serious overclaim, and it is the kind that survives peer review right
  up until someone checks the annotation lineage.
- **Not ruled out:** the biology hypothesis itself. Nothing here argues
  *against* it. The best evidence for it remains Phase 3's: `cb_intrinsic`
  sits in the well-annotated central brain — where "residual =
  under-annotated" is least plausible — and its residual neurons show
  connectivity differences surviving both regional stratification and
  degree adjustment. That argument does not depend on a second
  annotation at all.
- **Resolved 2026-09-20** by the Codex community labels — see the ANSWER
  section at the top. The candidates listed below were the plan at the
  time; the first of them is what worked.

- **What was needed:** a genuinely independent fru/dsx source, one built
  without reference to MaleCNS. Two candidates:
  1. **FlyWire Codex's `gene` field** (`gene==Fruitless`, `gene==Doublesex`).
     Per the plan (section b.5) these exist in Codex, and Codex's README
     explicitly warns its annotations come from a *mix of sources* that
     "likely diverge" from the curated table — for once that divergence
     is the point. Requires a Codex export; not in the cached data.
  2. **The Deutsch/Murthy-lab female fru/dsx catalogue**
     (bioRxiv 2025.06.10.658788): 91 putative Fru/Dsx cell types, ~1,400
     neurons, annotated from light-microscopy expression on the FlyWire
     side. Independent by construction.

  The first of these is what resolved it: the `gene` field is backed by
  Codex **community labels**, which is where the fru/dsx calls actually
  live (there is no gene-expression file on the download page). It
  reported `PARTIALLY INDEPENDENT` and the test became informative,
  exactly as predicted here.

## Reproduce

```
python scripts/run_annotation_crosscheck.py                  # curated table (the run above)
python scripts/run_annotation_crosscheck.py --source codex   # once the Codex export exists
```

Outputs go to `data_processed/`, suffixed per source so the two runs
cannot overwrite each other: `annotation_crosscheck_summary.json` and
`..._summary_codex.json` (plus the matching `_by_super_class.csv`).

## Running it against Codex — the step that would settle this

The ingestion path is built and tested (`load_codex_annotations`,
`tests/test_codex_loader.py`); the only missing piece is the export
itself, which needs a browser login and so cannot be fetched from here.

**Before anything else: check the dataset selector.** Codex hosts five
datasets (FAFB, BANC, MANC, MAOL, **MCNS**). The selector is top-right
and must read **FAFB v783** — the female brain the rest of this project
compares against. Exporting MCNS by mistake would be comparing MaleCNS
against itself; the loader catches it (`test_export_from_the_wrong_dataset_fails_with_a_useful_message`)
but it is the easiest way to waste the trip.

### Which file holds the gene annotations — not documented, so find it

Codex's download page (checked 2026-09-20) has **no file named "gene
expression"**. The `Gene Expression: Fruitless` field visible in the web
UI lives inside one of the annotation exports and which one is not
stated. Rather than guess, download the four small candidates —
**Cell Types** (902 KB), **Classification / Hierarchical Annotations**
(934 KB), **Community Labels (Refined)** (1 MB), **Community Labels
(Raw)** (4.8 MB), ~7.6 MB total — into `C:/FruResidualData/raw/` and run:

```
python scripts/inspect_codex_download.py
```

It scans every table for cells mentioning fruitless/doublesex and
reports which file, which column, and how many, ranked. Then point the
cross-check at the winner with `--codex-file <name>`.

Skip the big ones: Connections (68 MB / 277 MB), Synapse Table (2.7 GB)
and Neuron Skeletons (13 GB) are connectivity and geometry, not
annotations.

### Two dead ends already ruled out, so they are not worth retrying

- The curated table's `fru_dsx` column — nested inside MaleCNS (the
  finding this document is about).
- The curated table's **`synonyms`** column, which carries 207
  `fru-F-XXXXXX` light-microscopy clone identifiers and looked like it
  might be an independent source sitting in already-cached data.
  Checked: all 207 are *also* `fru_dsx`-positive, zero outside it.
  Nested too, so it adds nothing.

### Route A — search export (usually easiest)

1. On **codex.flywire.ai** with **FAFB v783** selected, search
   `gene == Fruitless`.
2. Download the result list. Repeat for `gene == Doublesex` and
   concatenate, or try `gene == Fruitless || gene == Doublesex` in one
   query.
3. Save as `C:/FruResidualData/raw/codex_annotations.csv`.

The file only needs a `root_id` (or `cell_type`) column — it is just the
list of **positives**. The full population, and the `super_class` /
`cell_type` fields, come from the curated table already cached. That is
safe: those are structural annotations, not fru/dsx calls, so they
cannot leak the variable under test.

### Route B — full dump

**Download Data** → the annotation/cell-info export carrying a `gene`
column for every neuron. Same destination path.

### Then

```
python scripts/run_annotation_crosscheck.py --source codex
```

The loader works out which of the two kinds of file it got, so there is
nothing to declare. Column names are auto-detected (`cell_type`/`type`,
`super_class`/`superclass`/`class`, `gene`/`genes`/`markers`,
`root_id`/`id`/`segment_id`) because Codex has renamed export headers
between releases. If nothing matches, it raises with the columns
actually present rather than failing deep inside a groupby. Root ids are
read as strings throughout — parsed as float64 a 64-bit FlyWire id loses
its last digits and would silently match nothing.

**The result to look at first is not the p-value.** It is
`independence_diagnostic.independence_verdict` in the summary JSON:

- `PARTIALLY INDEPENDENT` → Codex calls fru/dsx on at least some neurons
  MaleCNS called negative, so the source carries information of its own
  and the residual comparison below it becomes meaningful. **This is the
  outcome that settles the question**, in whichever direction the
  numbers then point.
- `NOT INDEPENDENT (nested)` → Codex's labels are also a subset of
  MaleCNS's, the same dead end as the curated table, and the remaining
  independent option is the Deutsch/Murthy-lab catalogue
  (bioRxiv 2025.06.10.658788).

Either way the run is informative: one settles the project's central
question, the other rules out the second of three candidate sources.
