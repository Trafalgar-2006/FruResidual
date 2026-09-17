# Project Plan: Predicting the Fly Brain's Sexual Rewiring — and Hunting the Neurons the Genetic Switch Can't Explain

*A solo ML research project on the Drosophila (fruit fly) connectome. Every dataset, number, and access route below has been verified against current primary sources (September 2026).*

---

## a) Plain-language project summary and the one-sentence pitch

**One-sentence pitch:** Using the newly completed male and female fruit-fly connectomes, I train a model to predict each neuron's sex-specific rewiring from the known genetic sex-switch (the *fruitless*/*doublesex* genes) — and treat the neurons the model *fails* to explain as candidate evidence for a second, uncharacterized driver of brain sex differences.

**Summary in plain terms:**
- Fruit flies have two "master switch" genes, *fruitless* (*fru*) and *doublesex* (*dsx*), long known to build the sex differences in the brain and the sexually dimorphic behaviors (courtship, aggression, egg-laying) they control.
- In September 2026, scientists released the first complete wiring diagram (connectome) of a male fly's central nervous system, and compared it neuron-by-neuron to the existing female connectome. They found that a small set of cell types are wired differently between the sexes, and that these mostly — but not perfectly — carry *fru*/*dsx*.
- **"Not perfectly" is the whole opportunity.** Per Berg et al. (*Cell* 2026), only **60.3% of dimorphic neurons** and **90.4% of male-specific neurons** in the central brain are *fru*+/*dsx*+; conversely **39.7% of dimorphic neurons and 9.6% of male-specific neurons are *fru*−/*dsx*−.** The paper itself says this "likely results from incomplete gene expression data… however, other genetic factors or non-cell autonomous mechanisms may shape some dimorphisms."
- My project asks: **can we predict which neurons are rewired, and what is left over when we predict only from the genetic switch?**
- The **residual** — neurons that are sex-different in wiring but that the switch does not explain — is the intended discovery. If those leftover neurons cluster on something shared (a neurotransmitter, a developmental lineage, a wiring motif), that shared thing is a candidate "second driver."
- The project is structured as ONE question at four levels of depth, protected by a fast "fruit-first" kill-check up front so I don't sink weeks into a dead end.

---

## b) What we're working with (datasets) and exactly how to access each

### 1. MaleCNS v1.0 — the male connectome (primary dataset)
- **What it is:** the complete male *Drosophila* central nervous system (brain + ventral nerve cord). Per the Berg et al. *Cell* 2026 abstract: *"This contains 166,700 neurons spanning the brain and nerve cord, fully proofread and annotated, including fruitless/doublesex expression and 11,710 neuron types."* Released as v1.0 on June 8, 2026; paper published in *Cell* on September 3, 2026 (Berg S, Beckett IR, Costa M, Schlegel P, Januszewski M, et al.; doi:10.1016/j.cell.2026.08.015).
- **License:** CC-BY 4.0.
- **Access route A (queries):** neuPrint, dataset string `male-cns:v1.0`, hosted at `neuprint.janelia.org`. **Standard neuPrint token only — no special authorization** (Clio auth is only for write access). Get a token by logging in at neuprint.janelia.org with a Google account and copying it from the Account page.
```python
from neuprint import Client, fetch_neurons, NeuronCriteria as NC
c = Client('neuprint.janelia.org', dataset='male-cns:v1.0', token='YOUR_TOKEN')
c.fetch_version()
neurons_df, roi_counts_df = fetch_neurons(NC(type='.*'))  # all neurons + per-ROI synapse counts
```
- **Access route B (bulk download, no auth):** public Google bucket `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` (also reachable over plain HTTPS at `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/<file>`). Key files:
  - `connectome-weights-male-cns-v1.0-minconf-0.5.feather` — the full segment-to-segment connection graph (edge weights).
  - `body-annotations-male-cns-v1.0-minconf-0.5.feather` (~13–14 MB) — per-neuron annotations (classes, types, sides), excluding neurotransmitter.
  - `body-neurotransmitters-male-cns-v1.0.feather` (~42–43 MB) — aggregate per-neuron neurotransmitter predictions; `tbar-neurotransmitters-male-cns-v1.0.feather` for per-synapse.
- **Access route C (morphology):** `navis` + `navis.interfaces.neuprint` fetches skeletons/meshes as navis objects, e.g. `neu.fetch_skeletons(neu.NeuronCriteria(type="DNge104"))`.
- **Access route D (R/natverse):** the `malecns` package wraps `neuprintr` and exposes extra metadata columns including `flywireType` (the corresponding FlyWire cell type). Default dataset is the public `male-cns:v1.0`.
- **Key metadata fields on neuprint:** hierarchy `superclass → hemilineage → supertype → type`; side from `somaSide`/`rootSide`; hemilineage from `itoleeHl` and `trumanHl`; cross-dataset type matches `flywireType`, `mancType`, `hemibrainType`.

### 2. FlyWire / FAFB — the female brain connectome (primary comparison)
- **What it is:** the complete adult female fly *brain* connectome — **139,255 proofread neurons** and ~54.5 million chemical synapses (Dorkenwald et al., *Nature* 2024), with a companion cell-type catalog of >8,400 types (Schlegel et al., *Nature* 2024). This is the reference the MaleCNS team compared against. Note it is brain-only (no VNC, no retina).
- **Access route A (browse + bulk CSV):** Codex at `codex.flywire.ai`; export static CSVs via Info → Download Data. Codex does **not** offer a general live-query API — use the static downloads.
- **Access route B (programmatic):** `fafbseg` (Python and R) + `CAVEclient`. Get a CAVE token at `https://global.daf-apis.com/auth/api/v1/user/token` (log in with your FlyWire account).
- **Access route C (annotations incl. *fru*/*dsx*):** the `flyconnectome/flywire_annotations` GitHub repo — the authoritative, cross-checked annotation table (`supplemental_files/Supplemental_file1_neuron_annotations.tsv`). Since **v3.0.0** it carries the crucial columns added via cross-validation against MaleCNS: **`dimorphism`** (isomorphic/dimorphic/sex-specific), **`matching_notes`**, **`fru_dsx`**, `synonyms`, and `supertype`. Note the README caution: Codex "presents a mix of annotations from different sources which likely diverge from the systematic and cross-checked annotations presented here" — prefer this repo for rigor.
- **Confirmed:** *fru*/*dsx* gene labels ARE shared in FlyWire Codex (query `gene==Fruitless` or `gene==Doublesex`).

### 3. BANC — the second female dataset (for Level-4 validation)
- **What it is:** the Brain-And-Nerve-Cord connectome, a *different* single adult female fly. Per the BANC project README (accompanying Bates, Phelps, Kim, Yang et al., "Distributed control circuits across a brain-and-cord connectome," *Nature* 2026): *"a reconstruction of the central nervous system of one adult female Drosophila melanogaster — approximately 188,000 neurons and 199 million predicted synapses, spanning the brain, suboesophageal zone, cervical connective, and the entire ventral nerve cord."* Imaged by GridTape serial-section TEM at 4 nm in-plane. Available in Codex and at flywire.ai since November 2025.
- **Why it matters:** it is an independent female reconstruction (different animal, different EM technique). Re-running the residual against BANC tests whether "unexplained" neurons are real biology, not FlyWire reconstruction artifacts.
- **Access route A (Python):** `pip install banc`; save CAVE credentials via `banc.save_cave_credentials("TOKEN")` (writes `~/.cloudvolume/secrets/cave-secret.json`).
- **Access route B (R/natverse):** `bancr`; the compiled per-neuron meta table `banc_888_meta.feather` (~188k rows × 75 columns) includes the cell-type hierarchy, cross-dataset matches (`fafb_*`, `manc_*`, `hemibrain_*`, `malecns_*`, `fanc_*`), neurotransmitter, hemilineage, and neuropil — no CAVE auth needed for that file (`banc_meta_create_cache()`).

### 4. The dimorphism catalogue / male↔female correspondence
- **What it provides:** the MaleCNS "Dimorphism Explorer" (`male-cns.janelia.org/build/dimorphism_overview`) is a browsable catalogue of every dimorphic, male-specific, and female-specific cell TYPE, grouped by terminal type, clone/synonym, brain region, supertype, or hemilineage.
- **Granularity (critical):** the dimorphism classification is a **cell-TYPE-level** property, assigned by cross-matching the male CNS to the female FlyWire brain. It is NOT a per-neuron measurement.
- **How to obtain programmatically:** the type-level correspondence flows through the MaleCNS `flywireType` field (neuprint) on the male side and the `dimorphism` column of the `flywire_annotations` repo on the female side. **There is no single downloadable table of paired male/female body IDs;** correspondence is expressed at the type level plus the paper's neuron counts (see Details). `coconatfly` (natverse, R) is the recommended tool for cross-dataset co-clustering/matching.

### 5. The genetic labels (*fru*/*dsx*) and the extra features for Level 3
- ***fru*/*dsx* in MaleCNS:** annotated **per-neuron** with high/low confidence flags, by co-registering light-microscopy *fru*/*dsx* expression patterns onto the connectome. The paper annotated **4,858 (2,804 high-confidence) *fru*** and **412 (331 high-confidence) *dsx*** neurons in the male CNS (2.9% and 0.2% of neurons); **258 co-express both**; ~75% of annotations are in the central brain, where 10.6% of male neurons are *fru*+ (an upper bound; lower bound 5% using high-confidence only).
- ***fru*/*dsx* in FlyWire:** the `fru_dsx` column in `flywire_annotations` and the `gene` field in Codex.
- **Granularity mismatch (the data-hygiene subtlety, verified):** dimorphism status is per-TYPE; *fru*/*dsx* is per-NEURON with confidence. The join must be done carefully — you cannot naively merge a per-type dimorphism label onto per-neuron *fru*/*dsx* without deciding how to aggregate (e.g. "any high-confidence *fru*+ neuron in the type" vs "majority of neurons in the type"). The Explorer itself warns: *"not necessarily all neurons in a hemilineage are fruitless- or doublesex-positive."*
- **Neurotransmitter predictions:** available for MaleCNS (`body-neurotransmitters-male-cns-v1.0.feather`; classes include acetylcholine, GABA, glutamate, dopamine, serotonin, octopamine, histamine), giving an excitatory/inhibitory sign proxy. Needed for Level-3 residual characterization.
- **Hemilineage annotations:** available in MaleCNS neuprint via `itoleeHl` and `trumanHl` (developmental lineage). Needed for Level-3.

---

## c) What to learn, mapped to gaps
*(Assumes you are strong in PyTorch/ML/CV/transformers but new to connectomics tooling and fly neuroscience.)*

**Gap 1 — connectomics tooling (biggest gap, ~3–4 days):**
- Read the **neuprint-python Quickstart and Query Tutorial** (`fetch_neurons`, `fetch_adjacencies`, `NeuronCriteria`). This is the SQL-of-connectomes; learn it first.
- Skim the **navis neuPrint tutorial** for morphology objects.
- Skim the **fafbseg-py "FlyWire setup"** tutorial (token, CAVEclient) and the **`flywire_annotations` README**.
- Output of this step: a notebook that loads the male connectome as a `scipy.sparse` adjacency matrix keyed by bodyId, plus a per-neuron metadata DataFrame.

**Gap 2 — fly sex-biology background (~2 days, read in this order):**
- **Berg et al. 2026 *Cell*** (the MaleCNS paper) — abstract, Figure 3 (sexual dimorphism), and the *fru*/*dsx* correspondence section (Fig 3M–N). The single most important paper.
- **Rubin et al. 2026 *Current Biology*** ("Networks of sexually dimorphic neurons that regulate social behaviors") — how the field currently reasons about dimorphic circuits (e.g. the pC1/pC1x male-specific network).
- **Deutsch/Murthy-lab "Sexually-dimorphic neurons in the *Drosophila* whole-brain connectome"** (bioRxiv 2025.06.10.658788 / Research Square rs-6881911) — the female *fru*/*dsx* catalogue.
- One classic review on *fru*/*dsx* biology ("Sexual Dimorphism in the Fly Brain" lineage of work) for grounding.

**Gap 3 — graph ML on connectomes (~2–3 days):**
- PyTorch Geometric tutorials on node/edge-level tasks and the **Explainer** module (GNNExplainer / CaptumExplainer) — fast for you since you know PyTorch.
- One connectome link-prediction paper (e.g. NCPNet, bioRxiv 2023.06.29.547138) to calibrate expectations.

---

## d) Phase-by-phase build plan (~8–10 weeks)

### PHASE 0 — Fruit-first kill-gate (2–3 days). **GATE.**
1. Load MaleCNS per-neuron metadata (type, dimorphism status via `flywireType` cross-match, *fru*/*dsx*, hemilineage, neurotransmitter) and the female `flywire_annotations` table.
2. **Resolve the granularity mismatch explicitly:** decide and document the type→neuron aggregation rule for *fru*/*dsx*, and propagate the per-type dimorphism label to neurons carefully.
3. Build the **2×2 cross-tab**: [sex-different: yes/no] × [carries *fru*/*dsx*: yes/no]. Measure the size and internal structure of the "sex-different but NO *fru*/*dsx*" cell.
4. **Gate criterion:** we already know from the *Cell* paper that 39.7% of dimorphic and 9.6% of male-specific neurons are *fru*−/*dsx*−, so the box should be non-trivial. If, after careful joining, that box is fat AND structured (concentrates in specific regions/types) → project is alive. If it collapses to near-empty or pure noise → pivot (see Risks).

### PHASE 1 — LEVEL 1: Atlas of where the unexplained dimorphism concentrates (~1 week).
1. For every neuron, compute a wiring-difference score between sexes (metric defined in Level 2) and an "explained by *fru*/*dsx*" flag.
2. Aggregate the *unexplained* dimorphism by neuropil/ROI, by superclass, and by sensory/behavioral system (vision, olfaction, taste, mechanosensation, motor/descending).
3. Produce an **atlas** — brain-region heatmaps and system-level bar charts, not a single number. Contextualize against the paper's known findings: dimorphism concentrates in higher-order centers, and *"Isomorphic neurons are distributed relatively evenly across 7 layers (mean layer 3.9). By contrast, sex-specific and dimorphic neurons occupied deeper layers (mean at layers 4.3 and 4.5, respectively). The sensory periphery (layers 1–3) contains fewer such neurons: just 2.5% of male-specific versus 18.5% of isomorphic neurons."*
4. **Deliverable:** the "unexplained-dimorphism atlas" figure set.

### PHASE 2 — LEVEL 2: Predict the FORM of rewiring (~2–2.5 weeks). **Main ML model.**
1. Define per-neuron rewiring targets between matched male/female cell types: (a) connections gained, (b) connections lost, (c) reweighted (magnitude of change) — beyond binary dimorphic-yes/no.
2. **Features that DO NOT include the answer:** shared-ground-plan connectivity context (connectivity to isomorphic partners), cell class/superclass, neuropil, morphology summaries — but NOT *fru*/*dsx*, NOT the dimorphism label, NOT cell-type identity used as a direct proxy for the answer.
3. **Baselines first:** logistic regression and gradient-boosting (scikit-learn / XGBoost/LightGBM) on hand-built features. Then a GNN (PyTorch Geometric or DGL) for node/edge-level prediction on the connectome graph.
4. **Feature-ablation design:** train (i) with *fru*/*dsx* included, (ii) without. The performance gap quantifies how much the genetic switch explains; the model WITHOUT the answer, plus the leftover error, defines the residual.
5. **Deliverable:** predictive model + ablation table showing which features carry the signal; per-neuron residual scores.

### PHASE 3 — LEVEL 3: Turn the residual into a hypothesis (~2 weeks). **Deep, higher-risk.**
1. Take the high-residual set (sex-different, not explained by the genetic switch, *fru*/*dsx*-negative).
2. Test for shared structure via **enrichment analysis**: neurotransmitter type, hemilineage (`itoleeHl`/`trumanHl`), and connectivity motifs (shared input/output partners, motif census).
3. **Clustering** (Leiden/hierarchical on connectivity embeddings) + statistical enrichment (hypergeometric / Fisher tests with multiple-testing correction) to ask whether the residual set is enriched for any category above chance.
4. **Deliverable:** a ranked shortlist of candidate "second-driver" signatures with effect sizes and corrected p-values. If nothing clusters, report that honestly (still a result).

### PHASE 4 — LEVEL 4: Cross-dataset validation (~1–1.5 weeks). **Rigor.**
1. Re-derive the female side of the wiring-difference and the residual using **BANC** (independent female) instead of / in addition to FlyWire.
2. Check whether the same neurons/types remain "unexplained" — quantify overlap (Jaccard of residual sets, rank correlation of residual scores).
3. **Deliverable:** a reproducibility figure showing which residual findings survive a second female dataset.

### PHASE 5 — Package (~1 week).
- Clean repo, reproducible environment, paper-shaped writeup, figures, README with the one-sentence pitch.

---

## e) Full tool/stack list
- **Data access:** `neuprint-python`, `navis`, `fafbseg` (+ `CAVEclient`), `banc`/`bancr`, natverse `malecns` (R, optional), `pyarrow` (feather I/O).
- **Graph/compute:** `networkx`, `scipy.sparse`, `numpy`, `pandas`.
- **ML:** `scikit-learn`, `xgboost`/`lightgbm`, `PyTorch`, `PyTorch Geometric` (or `DGL`); `POT` (optimal transport) if comparing connectivity distributions across sexes/datasets.
- **Stats/clustering:** `scipy.stats`, `statsmodels`, `leidenalg`/`igraph`, optionally `scanpy` for enrichment/clustering utilities.
- **Viz:** `matplotlib`, `seaborn`, `navis` 3D.

---

## f) Deliverables framed for resume/reputation
- A **paper-shaped writeup** (intro → methods → results → honest limitations) — the four levels map cleanly to a results section.
- A **clean, documented GitHub repo** with reproducible notebooks and a one-command data-download script.
- **Four headline figures:** the atlas (Level 1), the ablation table + residual model (Level 2), the candidate second-driver shortlist (Level 3), and the cross-dataset validation figure (Level 4).
- **The single sentence** (the pitch above) for a CV/portfolio headline.

---

## g) Honest novelty statement
- **Already published (do NOT claim as mine):**
  - The male connectome and the male↔female comparison itself (Berg et al., *Cell* 2026), including the finding that dimorphism concentrates in higher-order centers and that *fru*/*dsx* correspondence with dimorphism is "high but not perfect" with the specific percentages above.
  - The dimorphic social-behavior networks / pC1 story (Rubin et al., *Current Biology* 2026).
  - The female *fru*/*dsx* catalogue: Deutsch/Murthy-lab (bioRxiv 2025.06.10.658788) *"identify 91 putative Fru or Dsx cell types, comprising ~1400 neurons… All Fru and Dsx labels reported here are shared within FlyWire Codex (codex.flywire.ai; gene==Fruitless or Doublesex)."*
- **Prior hobby/preprint-adjacent work to position against:** at least one open GitHub project (ConnectomeLens) already trains a classifier on the male connectome alone to predict published dimorphism labels — reporting *"AUC-PR 0.759, significantly above a degree-preserving randomized null model (p = 0.002)"* (chance 0.041; null 0.712 ± 0.005). So **"predict dimorphism from wiring" is not itself novel** and must be positioned carefully.
- **Genuinely novel (mine):**
  1. The **generative/predictive framing of the FORM of rewiring** (gain/lose/reweight and magnitude), not binary dimorphic-yes/no.
  2. The **systematic residual-as-discovery pipeline** — deliberately model out the genetic switch, then characterize the leftover set for a candidate second driver.
  3. **Cross-dataset (BANC) validation** of that residual.
  The contribution is a **method + a hypothesis**, not a claim of a confirmed new gene.

---

## h) Risks and failure modes with mitigations
- **Residual is tiny or unstructured (the central risk):** mitigated by the Phase-0 kill-gate — we quantify the "unexplained" box before any modeling. If it's thin, pivot to a descriptive atlas of the known *fru*−/*dsx*− dimorphic neurons (still publishable as a resource).
- **Granularity mismatch corrupts the join:** mitigated by explicitly documenting the type→neuron aggregation rule and running sensitivity analyses under alternative rules (any-positive vs majority vs high-confidence-only).
- **Incomplete *fru*/*dsx* annotation inflates the residual:** the paper flags incomplete gene-expression data as one reason for the mismatch. Mitigation: use high-confidence *fru*/*dsx* only in the primary analysis and report low-confidence as a sensitivity check; frame residuals as "candidates," never confirmed.
- **Reconstruction artifacts masquerade as biology:** mitigated by Level 4 (BANC cross-validation).
- **Cross-dataset alignment is hard (different animals, different EM):** mitigated by working at the cell-type level where cross-matches already exist (`flywireType`, BANC `malecns_*` columns), and reporting overlap statistics rather than requiring perfect neuron-to-neuron identity.
- **GNN overfits / underperforms baselines:** mitigated by baseline-first design and honest ablations; if the GNN doesn't beat gradient boosting, report that.

---

## Caveats and data-accuracy notes
- **The dimorphism counts differ between the preprint and the final paper — report the *Cell* (final) figures and flag the change.** The **bioRxiv preprint** (2025.10.09.680999) reported **7,205 isomorphic, 114 dimorphic, 262 male-specific, 69 female-specific** types (166,691 neurons; 11,691 types). The **final *Cell* paper** (2026) reports **8,069 isomorphic, 138 dimorphic, 289 male-specific, 71 female-specific** types (166,700 neurons; 11,710 types). Both give ~4.8% of male and ~2.4% of female central-brain neurons as sex-specific/dimorphic. Use the *Cell* numbers in any public-facing writeup and note the preprint→paper revision.
- **Neuron-count reconciliation:** the *Cell* headline is 166,700 neurons; the preprint methods text also cites a connectome graph of 166,391 neurons / 25.6M edges. These are the same dataset at slightly different processing snapshots — cite the published 166,700 figure.
- **Male↔female matched-neuron counts (for building targets):** per the *Cell* paper, criteria identified **1,420 male-specific vs 363 female-specific neurons**, and **948 dimorphic neurons in the male matched 813 in the female** — the discrepancy is attributed to biological variability (especially optic-lobe column number: male ~900 vs FAFB ~800), not error. Plan for imperfect 1:1 matching.
- **Do not conflate two similarly named papers.** (1) Berg et al., *Cell* 2026 = the MaleCNS connectome + male↔female comparison (the primary dataset). (2) Deutsch/Murthy-lab bioRxiv 2025.06.10.658788 = a FlyWire-only female *fru*/*dsx* catalogue (91 types / ~1,400 neurons). Their numbers are not interchangeable.
- **Unverified specifics to confirm empirically before coding (flagged honestly):** the exact column name(s) for dimorphism status and *fru*/*dsx* *inside* the MaleCNS neuprint schema / `body-annotations` feather were not confirmable from public docs — the `dimorphism`/`fru_dsx` names are confirmed only for the **FlyWire** `flywire_annotations` repo. Confirm the male-side field names by loading the feather / calling `fetch_neurons()` and inspecting the returned columns. Likewise confirm the exact neurotransmitter column spelling in the male-cns schema.
- **Codex vs curated annotations:** Codex mixes annotation sources that "likely diverge" from the systematic `flywire_annotations` tables — prefer the GitHub repo for the analysis of record, and treat Codex as the convenient browser.