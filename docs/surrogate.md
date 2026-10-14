# The surrogate: a negative result

**Verdict: not usable for arbitrary uploads. The site runs on the
precomputed library instead.**

## What it was for

Simulating one second of audio costs about 3.4 seconds of GPU time, so a
website that accepts uploads needs either a GPU per request or something
faster. The idea was to train a small network to imitate the simulation,
run it in the browser, and need no GPU at all.

The approximation target is unusually friendly: every ear neuron receives
the same drive and the readout is one population density, so the system
is a single-input, single-output, causal, time-invariant nonlinear
filter, and the connectome never changes between runs.

## What was built

A dilated causal convolution stack — 39,161 parameters, receptive field
511 steps (102 ms, against a 20 ms membrane time constant), trained on
600 clips totalling 1,200 s of simulated audio spanning tones, chords,
sweeps, amplitude modulation, pulse trains, noise bursts and gaps.

Targets are deliberately single-trial. MSE regression converges on the
conditional mean, so the network learns the trial-averaged response
without ever seeing one, and data generation costs 8× less.

## Two bugs found on the way

**The model was non-causal.** The first version used `GroupNorm`, which
normalises over channels *and time* — so every output depended on the
whole sequence including later steps. The model was reading the future.
Measured as a 0.033 leak backwards in time. Nothing in the loss curve
showed it; only an explicit causality test did. Replaced with
`weight_norm`, which is parameter-level and cannot mix timesteps.

**The evaluation was measuring noise.** Single-trial targets are roughly
0.93 spikes per step across 186 neurons — mostly Poisson. Correlating a
mean-predictor against one noisy draw caps the achievable correlation far
below 1 no matter how good the model is. The library had already measured
single-trial split-half reliability at 0.32–0.75 for this condition: a
single trial barely correlates with itself. Pitch and out-of-distribution
tests were changed to compare against an 8-trial average.

## Results

**Pitch — 5 of 6 tones.**

| played | simulated | surrogate | |
|---|---|---|---|
| 180 Hz | **360** | **180** | no |
| 250 Hz | 250 | 250 | yes |
| 330 Hz | 330 | 330 | yes |
| 440 Hz | 440 | 440 | yes |
| 600 Hz | 600 | 600 | yes |
| 800 Hz | 800 | 800 | yes |

The failure is the informative one. At 180 Hz the simulation doubles to
360 — the rectification artefact documented in `methods.md` — and the
surrogate emits a clean 180. **It is wrong by being nicer than the thing
it imitates.** A surrogate that quietly removes the circuit's own
artefacts is worse than no surrogate, because the output looks more
correct while being less faithful.

**Out-of-distribution — the decisive test.** Shapes absent from every
training category, scored against an 8-trial average:

| stimulus | r | sim peak | surrogate peak |
|---|---|---|---|
| vibrato FM | 0.753 | 312 Hz | 288 Hz |
| square wave | 0.717 | 220 Hz | 220 Hz |
| sawtooth | 0.385 | 220 Hz | 220 Hz |
| inharmonic cluster | 0.356 | 263 Hz | 263 Hz |
| pink noise | 0.248 | 306 Hz | **106 Hz** |
| slow clicks | **−0.034** | 319 Hz | 340 Hz |
| **mean** | **0.404** | | |

It handles tonal, harmonically dense input and **collapses on anything
temporally sparse or noise-like**. Slow clicks are uncorrelated. Pink
noise gets the dominant frequency wrong by threefold. Those are exactly
the shapes real uploads contain.

## Why the in-distribution number is not quoted as a score

Held-out in-distribution correlation reads 0.207 ± 0.143, which is
*lower* than the out-of-distribution mean of 0.404. That comparison is
meaningless: the first is measured against single noisy trials from the
stored dataset, the second against 8-trial averages. Different ground
truths, different scales. The script used to print them side by side and
no longer does.

## What would fix it

More and broader training data, not a bigger model. Every failure is a
stimulus family the network never saw, and train and validation losses
were identical throughout (0.00818 vs 0.00813) — no overfitting, so
capacity was never the constraint. Roughly 10× the data, covering
sparse and noise-like inputs, would be the experiment.

## What ships instead

The precomputed library: 20 entries, 4 MB, served as static files with no
GPU and no server. It is the real simulation rather than an imitation of
it, which is the whole point.

Reproduce with:

```
python scripts/make_surrogate_data.py --clips 600 --condition courting
python scripts/train_surrogate.py --epochs 40
python scripts/validate_surrogate.py
```
