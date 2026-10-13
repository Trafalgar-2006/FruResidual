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
