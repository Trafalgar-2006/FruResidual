# Outreach drafts

Two blockers in this project can only be cleared by someone outside it.
Both drafts below are ready to send; fill in the bracketed bits.

Status: **not sent** — update this line when you send them.

---

## 1. Janelia / MaleCNS team — the dimorphism-percentage discrepancy (E6)

**Send to:** the MaleCNS / FlyEM contact for `male-cns:v1.0`. Options, in
order of likely response: the corresponding author on Berg et al. *Cell*
2026; the `neuprint` / male-cns support address on
male-cns.janelia.org; or a GitHub issue on the FlyEM/neuPrint tracker
(public, which is fine — there's nothing sensitive here).

**Subject:** `male-cns:v1.0 — which annotation snapshot produced the Fig 3M–N fru/dsx percentages?`

> Dear Dr. [Berg / corresponding author],
>
> I'm a student working independently with the public MaleCNS v1.0
> release, and I've hit a discrepancy I can't resolve from the public
> data alone. I'd be grateful for a pointer.
>
> Working from `body-annotations-male-cns-v1.0-minconf-0.5.feather`
> (Traced neurons only, n=165,122), I can reproduce several of the
> paper's fru/dsx figures exactly:
>
> - total fru neurons: 4,858 (4,600 fru-only + 258 co-expressing) — matches
> - total dsx neurons: 412 (154 dsx-only + 258 co-expressing) — matches
> - high-confidence counts: 2,804 fru and 331 dsx — both match
> - central-brain male-specific neurons that are fru+/dsx+: **90.4%** — matches
>
> The one figure I cannot reproduce is the corresponding statistic for
> the "sexually dimorphic" category. The paper reports **60.3%** fru+/dsx+;
> restricting to central-brain superclasses (`cb_*`, n=1,553 of 2,368
> dimorphic/male-specific neurons) I get **79.0%** per-neuron, or 67.6%
> if I aggregate to the cell-type level first. Neither reproduces 60.3%,
> which rules out a simple per-neuron vs per-type counting difference as
> the whole explanation.
>
> Since the absolute counts and the male-specific percentage match
> exactly, I suspect the public v1.0 release is a slightly different
> annotation snapshot from the one behind the published figures — the
> preprint and the final paper already differ in total dimorphic type
> counts (114 vs 138), so a third intermediate revision seems plausible.
>
> My questions:
>
> 1. Which annotation revision produced the Figure 3M–N percentages, and
>    is it publicly retrievable? If it is, I'd just re-run against it.
> 2. Is the "sexually dimorphic" percentage computed over neurons or
>    over cell types, and over which region filter exactly? If your
>    denominator differs from `cb_*` superclasses, that alone might
>    explain it.
>
> Separately, a provenance question about a related dataset, if you
> happen to know: the `fru_dsx` column in the `flyconnectome/flywire_annotations`
> table is documented as having been added "via cross-validation against
> MaleCNS". In my copy it never marks a neuron fru/dsx-positive where
> MaleCNS marks it negative — the labels look like a strict subset. Is
> that column derived from the MaleCNS annotations, or independently
> assigned from FlyWire-side expression data? It matters to me because I
> was hoping to use it as an independent check, and if it's derived then
> it can't serve that purpose.
>
> Happy to share my exact query and the reproduction script if useful.
> Thank you for making the dataset public — it's been a genuinely great
> thing to learn on.
>
> Best regards,
> [Your name]
> [affiliation / GitHub link, optional]

---

## 2. BANC / Lee Lab — dataset `view` permission (E7, unblocks E8 and part of E10)

**Send to:** the BANC data-access contact (Lee Lab, or whoever is listed
on the BANC project page / `banc.community`). If there's a formal access
request form, use it and paste the "purpose" paragraph below into it
rather than emailing.

**Subject:** `Request for BANC datastack view permission — academic connectomics project`

> Dear BANC team,
>
> I'd like to request `view` permission for the BANC datastack for my
> CAVE account.
>
> **Account:** [the email your CAVE account is registered under]
> **Current status:** my token authenticates against the BANC datastack
> but returns `403: Missing permission: view for dataset BANC`. The same
> token works for FlyWire, so I believe this is purely the per-dataset
> grant.
>
> **What I'm doing:** an independent (non-lab, self-directed) research
> project on the MaleCNS v1.0 connectome, asking how completely the
> fruitless/doublesex genetic switch explains sexually dimorphic wiring
> — and characterising the neurons it does *not* explain. Everything is
> public-data-only and the code is open.
>
> **Why I need BANC specifically:** I'm using BANC as an independent
> female reconstruction to test whether those "unexplained" neurons
> replicate, rather than being artifacts of the single FlyWire
> comparison. I've already done as much as the public
> `banc_888_meta.feather` table allows — matching 17,206 neurons via
> `malecns_match` and comparing BANC's own `sexually_dimorphic` calls —
> and that part works well. What I can't do without API access is the
> connectivity-level comparison: re-deriving a per-neuron wiring-difference
> score against BANC the way I currently do against FlyWire. BANC is
> also whole-CNS, so it would let me include ascending/descending and
> VNC neurons that are structurally invisible in a brain-only comparison.
>
> **What I'd query:** per-neuron connectivity for a bounded set of
> roughly 1,500–3,000 matched neurons (partner lists and synapse
> weights), not bulk downloads of the full dataset.
>
> I'm happy to comply with any citation, embargo, or acknowledgement
> requirements, and to share what I produce.
>
> Thank you for considering it, and for making the bulk metadata openly
> available — it has already carried a whole phase of this project.
>
> Best regards,
> [Your name]
> [affiliation, or "independent researcher"]
> [GitHub link]

---

## What changes if each one lands

- **Janelia reply** → re-run `scripts/run_phase0.py` against the named
  revision; `docs/phase0_results.md` gets a resolution instead of a
  standing caveat. The provenance answer also decides whether
  `scripts/run_annotation_crosscheck.py` can ever use the FlyWire column.
- **BANC access** → unblocks the connectivity-based Phase 4 the plan
  originally specified, the rank-correlation check (E8), and the
  brain-only limitation in Phase 2 (E10).
  `fruresidual.data.banc_client` and `features/cross_validation.py` are
  already structured to take it.
