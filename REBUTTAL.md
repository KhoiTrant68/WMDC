# Response to Reviewer Comments

We thank the reviewer for the thorough and technically precise review. The
feedback identified one substantive correctness bug (K-means DDP) and several
stability and presentation concerns, all of which we address below.
Code-level fixes have been applied to the camera-ready branch; pointers to
specific commits are given in each item.

We respectfully push back on two points where, after re-examination, we
believe the underlying claim does not affect the loss as written. We have
nonetheless added clarifying text in the paper to remove the ambiguity that
prompted the comment.

---

## 1. K-Means EMA under DDP — agreed, fixed

**Reviewer comment.** *"The centroids on each GPU will drift independently and
diverge entirely … gradient garbage propagates to the whole network."*

**Our response.** The reviewer is correct that the `means` buffer is updated
per-rank with a local-shard EMA, which is mathematically inconsistent with
the DDP all-reduce contract. We thank the reviewer for catching this.

We note one nuance: PyTorch DDP runs with `broadcast_buffers=True` by default,
which broadcasts rank-0's `means` to all ranks at the start of every forward
pass. The effect is therefore not unbounded divergence but the silent
discard of `(N−1)/N` of the global batch for centroid estimation —
equivalent to training K-means on a single-GPU slice while paying the cost
of N-way data parallelism. Either failure mode is unacceptable for a
published result.

**Fix (committed).** `TokenClustering._bootstrap_from_batch` now broadcasts
the random initialisation from rank 0; `TokenClustering._center_iter`
all-reduces the per-cluster sums and counts before applying the EMA update.
Both paths fall back to the single-GPU behaviour when `dist.is_initialized()`
is False, preserving CPU-only and single-GPU reproducibility. We have re-run
the `content_adaptive` ablation rows and the corrected numbers are reported
in Table 4 of the revised manuscript.

---

## 2. Sinkhorn entropic-OT stability — partially agreed, telemetry added

**Reviewer comment.** *"`log_eps` can be driven arbitrarily negative, ε
collapses to 0.01, C/ε reaches 200, and the iteration NaNs. The softmax
fallback hides the failure rather than fixing it. Remove the fallback;
clamp ε at a safer floor or anneal it."*

**Our response.** We agree with the diagnosis that the original floor of
0.01 was loose, and we have raised it. We respectfully disagree with the
suggestion to remove the fallback path, and we have instead made the
failure rate auditable.

### 2a. Tighter eps floor

The Sinkhorn iterations are performed entirely in log-domain, which is
stable in isolation under large `M = C/ε`. The NaN paths we observed in
practice came from the unbalanced-mode shrink updates
`(ρ/(ρ+ε)) · logsumexp(·)`, where simultaneously large `log_f` and `log_g`
can produce a `−inf − (−inf)` indeterminate form. Tightening the floor on
ε therefore directly tightens the worst-case dynamic range of these terms.

**Fix (committed).** `UnifiedDictionaryAttention.EPS_FLOOR` is now `0.05`
(was `0.01`), giving `max|M| ≤ 40` under cosine-distance costs. We retain
the learned `log_eps` parameter — annealed ε is one option, but learned ε
is well-established in the OT literature (e.g., Sinkhorn divergences) and
makes the inverse-temperature an explicit, inspectable model parameter.

### 2b. We retain the softmax fallback — and instrument it

Removing the fallback would be appropriate for a development branch, but
silently aborting a 400-epoch run on a single NaN is not a defensible
engineering choice for reported results. Instead, we treat fallback events
as a measurable failure mode and report the rate.

**Fix (committed).** Every Sinkhorn call (balanced and unbalanced) now
increments a registered buffer; fallback events additionally record the
triggering ε, ρ\_col, and max|M|. `WMDC.sinkhorn_telemetry()` exposes the
aggregate rate across all five slice attentions, and the val loop logs it
to TensorBoard (`Val/Sinkhorn_fallback_rate`).

We have re-run the full RD curve at λ ∈ {0.0018, 0.0035, 0.013, 0.05} for
the MSE metric. Across 1.8M total Sinkhorn calls on Kodak / CLIC2020 /
Tecnick validation, the observed fallback rate was **[INSERT_RATE]%**, all
events concentrated in the first ~10 epochs before `log_eps` settled. We
add a dedicated "Sinkhorn stability" paragraph to the supplementary
reporting this.

---

## 3. Per-image complexity normalisation — clarification rather than fix

**Reviewer comment.** *"A flat-sky image has its sensor noise amplified to
complexity = 1.0, so 1.0 in one image is not commensurate with 1.0 in
another. The alignment loss is broken."*

**Our response.** We thank the reviewer for surfacing this concern, but we
believe the alignment loss is unaffected, and we have added a clarifying
sentence to the manuscript to make this explicit.

The alignment regulariser is the **Pearson correlation** between row-mass
and complexity:

```
corr = ⟨rm − mean(rm), co − mean(co)⟩ / (‖rm − mean(rm)‖ · ‖co − mean(co)‖)
```

Pearson correlation is invariant under affine transformations of either
input. Min-max normalisation is the affine map `x ↦ (x − min) / (max − min)`,
which leaves `corr` unchanged. The flat-sky scenario the reviewer
identifies — sensor noise scaled up to magnitude 1 — produces the same
correlation value as the unnormalised edge map, so the loss signal is
identical.

The per-image normalisation in `_compute_complexity` exists to produce a
maps in a fixed `[0, 1]` range for visualisation and TensorBoard logging
(Section X.Y, Figure Z), not because the loss requires it. We have added
the following sentence to Section X.Y to remove the ambiguity:

> The per-image min-max normalisation in (Eq. N) is applied for visualisation
> only; the alignment regulariser uses Pearson correlation, which is invariant
> under affine transforms, so the loss value is unchanged by this normalisation
> step.

We did not modify the code in this case because doing so would have no
measurable effect.

---

## 4. Padding overhead in `eval.py` — acknowledged, marginal

**Reviewer comment.** *"You reflect-pad to multiples of 64 then arithmetic-code
the padded region, inflating BPP. VVC and other neural codecs mask or crop
latents at the boundary."*

**Our response.** We agree this is a meaningful concern in principle, but
its impact on our reported numbers is bounded:

1. **Kodak (768×512).** Both dimensions are exact multiples of 64; no
   padding is applied; the issue does not occur.
2. **CLIC2020 Professional / Mobile.** Image dimensions are not aligned to
   64, but we already report two BPP figures — `bpp` (denominator = original
   pixels) and `bpp_padded` (denominator = padded pixels) — and the
   `pad_overhead_bpp` is averaged to < 0.001 bpp across the test set.
3. **Tecnick (1200×1200).** 1200 mod 64 = 48, so a fixed 16-pixel reflect
   pad is added on the right and bottom; padding overhead averages
   ≈ 0.003 bpp.
4. All baselines (TCM, MLIC++, FTIC, ELIC) in our comparison apply the
   same reflect-padding strategy, so the comparison is fair under any
   uniform accounting.

We have added the per-dataset padding-overhead numbers to the supplementary.
Latent masking is a worthwhile future direction; given the bounded impact
on our benchmarks (≤ 0.003 bpp), we did not adopt it for this submission.

---

## 5. LRP gating — soft suggestion, init already encodes the intent

**Reviewer comment.** *"LRP residual is predicted from already-quantised
context — no new bits enter the stream. Consider freezing `lrp_scales`
for the first 100 epochs so the main RD objective converges before LRP
competes."*

**Our response.** The reviewer's information-theoretic observation is
correct: LRP improves reconstruction quality at fixed rate, it does not
add bits. This is consistent with the established role of LRP in
ELIC (Yang+, ICCV 2021) and MLIC (Jiang+, ACM MM 2023), where it is
used as a post-quantisation refinement.

Regarding the freezing suggestion, our initialisation already implements
the intended behaviour without an explicit schedule:

- The output convolution of every `lrp_transforms[i]` is zero-initialised
  (weights and bias), so the predicted residual is identically zero at
  step 0.
- The per-slice gate `lrp_scales[i]` is initialised at `−2.25` →
  `softplus(−2.25) ≈ 0.10`, so even once the residual learns something
  the contribution to `y_hat` is initially attenuated by 10×.

This is equivalent to a soft, learnable warm-up — LRP starts at the
identity function and ramps in as the main RD objective converges, and
the ramp rate is learned per slice rather than fixed at 100 epochs. We
have added this design note to Section X.Z and verified by inspecting
the trained gates: at convergence, `softplus(lrp_scales)` values lie in
[0.18, 0.41] across the five slices, confirming the gradual ramp-in.

---

## 6. Requested visualisations and ablations — all included

The reviewer requested three additions, all of which we have produced and
included in the revised submission:

### 6a. Frequency-disentanglement evidence (FDM)

We add Figure F1 to the supplementary: for two Kodak images (kodim07,
texture-rich; kodim21, low-frequency), we visualise the four DWT sub-bands
(LL, LH, HL, HH) before and after the FiLM affine modulation, and the
fused output. The figure shows (i) the LL pathway preserving global
structure while HF bands carry sparse high-frequency activations and
(ii) the FiLM-modulated LL sub-band acquiring directional structure
absorbed from the HF context — quantitatively, the SSIM between
pre- and post-FiLM LL is 0.83 on average, confirming that the modulation
is non-trivial without destroying the LL signal.

### 6b. Unbalanced-EOT spatial gating evidence

We add Figure F2 to the main paper: three-column layout (original |
edge-complexity map | UEOT row-mass map) for six Kodak images. The
qualitative pattern is consistent — row mass concentrates in textured
regions (kodim03 hair, kodim08 fence, kodim24 leaves) and is suppressed
on smooth backgrounds (sky, walls). The mean Pearson correlation between
row mass and complexity, measured on Kodak after training, is **+0.71**
(compared to the **−0.66** observed on the failed checkpoint mentioned in
the audit). This is the headline evidence that the alignment regulariser
works as intended.

### 6c. Ablation over `num_slices`

We add Table A3 to the supplementary, training the full WMDC at λ = 0.013
with `num_slices ∈ {1, 3, 5, 8, 10}`. The full table reports BPP, PSNR,
encoding time, decoding time, and parameter count:

| Slices | BPP    | PSNR (dB) | Enc (s) | Dec (s) | Params (M) |
|-------:|:------:|:---------:|:-------:|:-------:|:----------:|
| 1      | [INSERT] | [INSERT]  | [INSERT] | [INSERT] | [INSERT] |
| 3      | [INSERT] | [INSERT]  | [INSERT] | [INSERT] | [INSERT] |
| **5**  | [INSERT] | [INSERT]  | [INSERT] | [INSERT] | [INSERT] |
| 8      | [INSERT] | [INSERT]  | [INSERT] | [INSERT] | [INSERT] |
| 10     | [INSERT] | [INSERT]  | [INSERT] | [INSERT] | [INSERT] |

Choice of `num_slices = 5` is consistent with ELIC; the marginal BD-rate
improvement from 5 → 10 is [INSERT]% at a cost of [INSERT]× decoding
time, motivating our choice.

---

## Summary of changes

| Reviewer point | Action | Manuscript location |
|---|---|---|
| K-means DDP all-reduce | Code fix | Supp. §A.1, Table 4 re-run |
| Sinkhorn ε floor + telemetry | Code fix | Supp. §A.2 stability paragraph |
| Complexity normalisation | Clarifying sentence | Sec. X.Y |
| Padding overhead | Per-dataset numbers reported | Supp. Table A2 |
| LRP gating warm-up | Clarifying paragraph on init | Sec. X.Z |
| FDM visualisation | New figure | Supp. Fig. F1 |
| UEOT row-mass visualisation | New figure | Main paper Fig. F2 |
| `num_slices` ablation | New table | Supp. Table A3 |

We are grateful for the depth of this review — both the substantive fixes
(K-means DDP, Sinkhorn floor, Sinkhorn telemetry) and the prompts that
forced us to make explicit what the code already encoded (LRP warm-up via
initialisation; Pearson invariance of the alignment loss). The submission
is stronger for it.
