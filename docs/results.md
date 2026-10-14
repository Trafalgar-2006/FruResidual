# Results

Run 2026-10-13. Baseline: pC1 stimulated at 150 Hz for 200 ms, wing-motor
(T2) mean rate **13.82 Hz**. 50 degree-matched random ablations per
condition; empirical p is `(k+1)/(n+1)`, so **0.0196 is the floor** at
n=50 and means *no* matched control came close.

## The controls work

| condition | n | song drop | degree-matched null | p |
|---|---|---|---|---|
| pIP10 (positive control) | 2 | **52.5%** | −0.01% | 0.0196 |
| TN1 (positive control) | 35 | **20.6%** | −0.31% | 0.0392 |
| residual: optic lobe only | 490 | **0.0%** | −0.24% | 0.5098 |

**Silencing two neurons halves the song.** pIP10 is the documented
bottleneck of the song pathway and the model reproduces that without
being told — it was found by searching cell-type names, not by tuning.

The third row is the one that makes the rest readable. 490 optic-lobe
residual neurons — more than half the entire residual — change the song
by **exactly nothing**, p=0.51. The method is not "remove neurons, output
falls". Anatomy decides, and the optic lobe is not in this circuit.

## The *fru*/*dsx*-explained set is one neuron type wearing a trenchcoat

| condition | n | song drop | p |
|---|---|---|---|
| explained: descending only | 16 | **91.2%** | 0.0196 |
| explained: descending **minus pIP10** | 14 | **2.3%** | 0.0196 |
| explained: central (cb+desc+asc) | 1,249 | 94.0% | 0.0196 |

Remove pIP10 from the set and the *fru*/*dsx*-explained descending
neurons' contribution collapses from 91.2% to **2.3%**. Essentially the
entire explained-descending effect is two neurons. That is a real
finding, not a nuisance: the genetic switch's grip on this circuit is
extraordinarily concentrated rather than distributed.

(The 1,249-neuron row is reported for completeness but is weak evidence —
at that size a matched null removes a large fraction of the brain too.
Note the stimulated pC1 neurons were excluded from every set; leaving
them in handed this arm a free 100% drop.)

## The residual carries real, measurable signal

| condition | n | song drop | degree-matched null | p |
|---|---|---|---|---|
| residual: all | 915 | **15.4%** | −0.35% | 0.0392 |
| residual: descending only | 89 | **7.2%** | −0.57% | 0.0196 |
| residual: central (cb+desc+asc) | 382 | 6.2% | −0.40% | 0.0588 |
| residual: optic lobe only | 490 | 0.0% | −0.24% | 0.5098 |

Every degree-matched null sits at or slightly **below** zero — silencing
a size- and traffic-matched random set does not weaken the song at all.
So these drops are attributable to *which* neurons were removed, not how
many or how busy.

The headline comparison:

> The 89 *fru*/*dsx*-**unexplained** descending neurons drop the song
> **7.2%**. The 14 *fru*/*dsx*-**explained** descending neurons, once the
> pIP10 bottleneck is set aside, drop it **2.3%**.

Neurons the genetic switch cannot account for are doing non-trivial work
in the circuit the genetic switch is supposed to build.

## The residual pulls in both directions

Breaking the residual down by superclass (same protocol, one ablation per
class) shows the aggregate numbers above were hiding a cancellation:

| superclass | n | song drop | per neuron | null | p |
|---|---|---|---|---|---|
| `ascending_neuron` | 84 | **+13.6%** | **+0.162%** | −3.8% | **0.0392** |
| `descending_neuron` | 89 | **+7.2%** | +0.081% | −2.6% | **0.0392** |
| `cb_motor` | 2 | 0.0% | 0.0% | 0.0% | 1.0000 |
| `ol_intrinsic` | 490 | 0.0% | 0.0% | −2.5% | 0.3529 |
| `visual_centrifugal` | 2 | 0.0% | 0.0% | −0.9% | 0.8627 |
| `sensory_descending` | 6 | 0.0% | 0.0% | −0.1% | 0.9412 |
| `vnc_intrinsic` | 33 | −4.5% | −0.136% | −2.8% | 0.6863 |
| `cb_intrinsic` | 209 | **−23.5%** | −0.113% | −4.0% | 1.0000 |

**Only two classes are significant, and they are the same two:**
`ascending_neuron` and `descending_neuron` — the classes that connect
brain to nerve cord. Nothing else in the residual moves the song.

Silencing the 209 residual `cb_intrinsic` neurons makes the song **23.5%
louder**: they are net inhibitory onto the pathway, so removing them
disinhibits it.

> **Caveat on that p=1.0000.** The empirical test is one-sided for
> *decreases* — it counts how often a matched null drops the song at
> least as much. A negative drop therefore scores p=1.0 **by
> construction**, not because the increase was tested and found wanting.
> The magnitude (−23.5% against a −4.0% null) suggests a real
> disinhibition, but assessing it properly needs the opposite tail. It is
> left untested rather than quietly claimed.

This explains the earlier puzzle: "residual: central" bundles
`cb_intrinsic` + `descending` + `ascending` together and came out at only
6.2%, which reads as a weak effect. It is not weak — it is an excitatory
contribution and an inhibitory one partially cancelling. Reported as an
aggregate alone, that would have been misleading.

The largest effect, both in total (13.6%) and per neuron (0.162%, twice
the descending rate), is `ascending_neuron` — the class carrying nerve-cord
→ brain feedback. **Those neurons do not exist in a brain-only
connectome.** The single most influential part of the residual is in the
half of the nervous system every published connectome simulation leaves
out.

## Does amputating the nerve cord change the brain? No — negative result

The hypothesis was that brain-only connectome models (every published
one) are missing something real, because they delete the 1,846 ascending
neurons carrying nerve-cord → brain feedback. MaleCNS can test it
directly: run the same stimulus on the full network and on a brain-only
version of itself.

| condition | pIP10 | dPR1 | TN1 | wing | brain mean |
|---|---|---|---|---|---|
| whole CNS | 200.0 | 200.0 | 47.3 | 13.8 | 0.467 |
| brain only (22,186 silenced) | 212.5 | 0.0 | 0.0 | 0.0 | 0.473 |
| ascending only (1,846 silenced) | 210.0 | 222.5 | 35.7 | 11.6 | 0.469 |

At face value this looked like a finding: pIP10, the brain's own command
output, fires **6.2% faster** without the nerve cord, and silencing just
the ascending neurons reproduces most of it. **It does not survive its
control.**

```
ascending silenced:   pIP10  +10.0 Hz vs whole CNS
degree-matched null:        -14.7 +/- 35.4 Hz
two-sided empirical p = 0.4118
```

Silencing *any* degree-matched set of 1,846 neurons swings pIP10 by tens
of Hz. A +10 Hz shift is deep inside that noise. **For this readout, the
brain-only abstraction is defensible** — removing the nerve cord destroys
the output (dPR1, TN1 and wing motor all go to zero, which is trivially
true since those cells are gone) but does not measurably change the
brain's own command dynamics.

This is reported because it was the motivating hypothesis for using
MaleCNS at all, and it failed. The case for MaleCNS now rests on the
ablation result above — where the two significant residual classes are
`ascending_neuron` and `descending_neuron` — not on this.

## Male vs female: no difference at the only readout both share

FlyWire (female) is brain-only. pIP10, TN1, dPR1 and every wing motor
neuron are absent, so "song" cannot be measured there at all. The one
stage both datasets contain is the **descending population**, and it is
almost perfectly size-matched: 1,314 male against 1,303 female.

Three confounds had to be removed first, and the raw numbers are
uninterpretable without them:

| | male | female |
|---|---|---|
| neurons | 165,122 (143k after brain-only restriction) | 139,248 |
| edges | 25,563,197 | 15,090,883 |
| synapses | 124,025,046 | 54,490,417 |
| calibrated gain | 0.0298 | 0.0928 |
| pC1 | **156** | **10** |

The gain differs threefold because the datasets were reconstructed by
different groups with different synapse detection. Each network is
bisected to the same population firing rate before anything is compared.

**The stimulus-matched comparison is not usable.** Forcing both sides to
10 pC1 neurons gave female 10.71 Hz against male 0.60 Hz — a 17×
difference that is an artefact, not biology. In that regime pC1 is barely
above a random stimulus of the same size (3.0× female, 1.6× male), so the
script prints a warning and the number should not be read as a finding.
It also deletes the thing being studied: the male's 156 pC1 against the
female's 10 **is** the dimorphism.

**Measured at each sex's natural size, against a random stimulus of that
same size in that same network:**

| | n pC1 | pC1 → descending | random control | ratio |
|---|---|---|---|---|
| male | 156 | 7.298 Hz | 3.271 ± 0.960 Hz | **2.2×** |
| female | 10 | 10.706 Hz | 4.780 ± 1.975 Hz | **2.2×** |

Identical. Once gain, network size and stimulus size are all controlled,
the courtship command drives the descending output equally far above
background in both sexes.

### Why that null is probably the readout's fault

This should not be read as "the sexes are the same". The readout averages
**1,314 descending neurons**, and the dimorphism that matters here is
**pIP10 — 2 neurons**. A two-neuron difference is diluted roughly 650-fold
in that mean, so this measurement could not detect it even if it were
enormous.

And the deeper asymmetry cannot be fixed by a better readout: pIP10,
TN1 and dPR1 **have no female counterpart to compare against**. They are
present in one sex and absent in the other. That is a presence/absence
fact, not a quantity with two values, so the honest output is the bulk
null plus the statement that the bulk is the wrong place to look.

## The input library, and the control that holds across all of it

Five synthesised stimuli × four conditions, each averaged over 8 trials
(`scripts/make_library.py`). Clarity is measured by **split-half
reliability** — average the odd and even trials separately and correlate
them. Anything reproducible across independent runs is signal; anything
not is noise. It needs no knowledge of the "correct" answer, so it works
on arbitrary audio.

| condition | readout | mean rate | reliability |
|---|---|---|---|
| male, courting | wing motor | 23.9–26.5 Hz | +0.323 … **+0.754** |
| female, hearing | descending | 5.1–5.4 Hz | +0.173 … +0.271 |
| male, hearing | descending | 1.75–1.84 Hz | +0.226 … +0.401 |
| male, **hearing only** | wing motor | 0.43–0.46 Hz | **−0.002 … +0.020** |

**The last row is the result worth keeping.** Hearing *without* the
courtship command produces no reproducible signal at the wing on any of
five very different stimuli — the highest reliability across all of them
is +0.020, and one is slightly negative. The same ears with pC1 engaged
reach +0.754. A condition that ought to be silent measures silent, which
is the strongest available evidence that the readout tracks the circuit
rather than the stimulus.

Mean rates barely move between stimuli within a condition (23.9–26.5 Hz
across all five under courting), so the drive level sets the rate and the
stimulus structure lives entirely in the **timing**. That is also why the
density rendering carries pitch while the rate alone does not — and why
the surrogate had to be validated on pitch tracking rather than
correlation (`docs/surrogate.md`).

## What this does not show

- **Not per-neuron dominance.** 7.2% over 89 neurons is 0.08% each;
  2.3% over 14 is 0.17% each. Per neuron the explained set is still
  stronger. The residual's contribution is *collective*.
- **Not a courtship mechanism.** Wing-motor firing rate is a proxy for
  "singing". It says nothing about song structure — the pulse/sine
  patterning the real circuit exists to produce.
- **Not independent of the model.** Every number depends on one gain
  constant, chosen for specificity (128× over control populations). The
  qualitative ordering is what should be trusted, not the percentages.
- **p is floored.** 0.0196 means "beat all 50 matched controls", not a
  precisely estimated tail.
- **Single female comparison missing.** See `docs/methods.md` §5.

## Figures

- `reports/figures/cascade.png` — the song propagating pC1 → pIP10 →
  dPR1/TN1 → wing motor, intact vs pIP10 silenced.
- `reports/figures/ablation.png` — every condition against its
  degree-matched null.
