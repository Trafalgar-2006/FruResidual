# Methods

## 1. Building the connectome

Source: `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (MaleCNS
v1.0), a `(body_pre, body_post, weight)` edge list where `weight` is a
synapse count.

**The file is not a connectome.** It is the full segmentation at minimum
confidence 0.5: 151,856,684 edge rows spanning ~13.8 million distinct
body IDs, against 166,691 actual neurons. The excess is unproofread
fragments. Only **16.8%** of rows have both endpoints among the 165,122
traced neurons. Restricting to those gives the real network:

| | |
|---|---|
| neurons (nodes) | 165,122 |
| edges | 25,563,197 |
| synapses | 124,025,046 |
| sparse size | ~307 MB |

No weight threshold is applied. At 307 MB the whole network fits in
memory, so discarding weak edges would lose signal for no benefit.

**Signs** come from each *presynaptic* neuron's transmitter:
acetylcholine excitatory; GABA, glutamate (GluCl) and histamine
inhibitory. 98.1% of traced neurons carry a confident call. Monoamines
(dopamine, octopamine, serotonin — 541 neurons) are neuromodulators
rather than fast transmitters and get sign 0, as do the 2.5% with no
call; together that costs 4.0% of edges. Signing by the *postsynaptic*
cell instead would invert much of the network while leaving every
summary statistic unchanged, so the direction is asserted in tests.

### Transposition gate

`scripts/verify_connectome.py` exists for one failure mode. If `A[i,j]`
meant `j→i`, then node count, edge count, density, degree distribution
and E/I ratio would all be **identical** and every result backwards.
Only biology notices:

| class | out/in | expected |
|---|---|---|
| `sensory_ascending` | 4.00 | ≫1, sends into the CNS |
| `cb_sensory` | 3.73 | ≫1 |
| `vnc_sensory` | 3.63 | ≫1 |
| `cb_motor` | 0.07 | ≪1, receives and drives muscle |
| `vnc_motor` | 0.01 | ≪1 |
| `vnc_efferent` | 0.00 | ≪1 |

## 2. The circuit

The male courtship song pathway, verified present as edges:

```
ORN_DA1 (204, cVA pheromone)
   -> antennal-lobe projection neurons
   -> pC1 / P1 (156, command)          --1,941 syn / 167 pairs-->
   -> pIP10 (2, descending)            --3,475 syn /  53 pairs-->
   -> TN1 (35) + dPR1 (2) (VNC premotor) --6,607 syn / 460 pairs-->
   -> wing motor neurons (186, T2) = SONG
```

Both reverse directions tested measure **0 synapses**. pIP10's strongest
single target is dPR1, a song-pathway neuron that was never searched for
— the matrix recovered it unprompted. `ORN_DA1 → pC1` is 0 direct
synapses, which is correct: that route runs through projection neurons.

`TN1`, `dPR1` and the wing motor neurons are nerve-cord cells. **None of
them exists in a brain-only connectome**, which is why this needs
MaleCNS.

## 3. Simulation

Leaky integrate-and-fire, the scheme Shiu et al. (*Nature* 2024) used on
FlyWire and validated by predicting proboscis extension and leg grooming,
then confirming both in live flies. Each neuron integrates signed
synaptic current, fires at threshold, resets, and is refractory.

Defaults: rest/reset −52 mV, threshold −45 mV, τ_m 20 ms, refractory
2.2 ms, dt 0.5 ms, stimulation 150 Hz. Runs on GPU; one timestep is one
25.5M-nnz sparse mat-vec.

### Calibrating the gain

Edge weights are raw synapse counts (mean 4.7, max 2,591), so the
per-synapse current multiplies them. It is swept, not assumed, and chosen
on **specificity** — the readout must respond to the courtship command
and *not* to size-matched control populations:

| gain | wing to pC1 | wing to best control | ratio |
|---|---|---|---|
| 0.005 | 0.0 Hz | 0.0 Hz | dead |
| **0.020** | **13.8 Hz** | **0.1 Hz** | **128.5** |
| 0.050 | 41.3 Hz | 28.3 Hz | 1.5 |
| 0.275 | 100.5 Hz | 105.9 Hz | 0.9 |

Above 0.05 the network is so excitable that the readout no longer
distinguishes courtship from noise — a *louder* signal would be a worse
experiment. Unstimulated, the network produces exactly 0 spikes, so
there is no spontaneous term masquerading as signal.

## 4. Ablation

Silencing is a per-run mask: the connectome is never modified (asserted
in `tests/test_lif.py`). Protocol: stimulate pC1, read mean wing-motor
rate, silence a population, re-measure.

Two confounds are handled explicitly:

- **Silencing the stimulus.** pC1 is both the command population and
  fru/dsx-positive, so it lands inside the "explained" set and would hand
  that arm a free 100% drop. Stimulated neurons are removed from every
  ablation target.
- **Degree.** Silencing 89 busy neurons removes more drive than 89 quiet
  ones regardless of identity. Every condition is compared against
  degree-matched random ablations of the same size, binned by quantile of
  total outgoing synapses.

Positive controls must break the song (pIP10, TN1); size-matched random
sets must not. Empirical p is `(k+1)/(n+1)`, so it is never exactly 0.

## 5. Known limitations

- **No female comparison yet.** FlyWire connectivity is on Zenodo record
  10676866, which returned HTTP 504 on every attempt. It is also
  brain-only, so the wing-motor readout does not exist there and the
  comparison needs a different one (descending-neuron output).
- **Rates above the stimulus.** pIP10 is 2 neurons receiving convergence
  from 156 stimulated pC1 cells and reaches ~200 Hz, above the 150 Hz
  input. Real neurons would saturate; this model only caps at the
  refractory limit.
- **No synaptic delay or neuromodulation.** Transmission is instantaneous
  between timesteps, and monoamines are set to sign 0.
- **Thresholded behaviour, not song structure.** Wing-motor firing rate
  is a proxy for "singing"; it says nothing about the song's temporal
  pattern, which is what the real circuit is tuned for.
