# WMDC — Mathematical Foundations

This document gives the closed-form derivations for the three pieces of
mathematics that are **new** in WMDC relative to prior learned image
compression (LIC):

1. Mutual-information decomposition of dictionary routing
2. Unbalanced entropic optimal transport (UEOT) with KL / TV marginal
   divergences, and the **debiased Sinkhorn divergence** loss
3. Limiting behaviour of the routing layer as ρ → ∞ / ρ → 0

It also writes down the regularisers (column-entropy, row-entropy,
complexity alignment, dictionary coherence, spatial-TV on P) in the form
they appear in `train.py:RateDistortionLoss.forward`, and proves the
small lemmas that the implementation relies on.

Notation:
- `B`  batch size
- `HW` number of spatial positions on the latent grid (= `Hz·Wz`)
- `N`  number of dictionary atoms
- `P ∈ ℝ_{≥0}^{B×HW×N}` transport plan returned by Sinkhorn
- `C ∈ ℝ_{≥0}^{B×HW×N}` cost matrix `C = 1 − cos(q, k)` (so `C ∈ [0, 2]`)
- `ε`  entropic regularisation, bounded `ε ∈ [0.05, 1.0]`
- `ρ_i, ρ_col`  unbalanced marginal-penalty strengths (row, column)
- `a = (1/HW)·1_{HW}`, `b = (1/N)·1_N`  uniform target marginals

---

## 1. Mutual information decomposition

Define a discrete random pair `(I, J)` on `{1,…,HW} × {1,…,N}` with joint
mass

    p(i, j) = P_{i,j} / Σ_{i',j'} P_{i',j'}

so that `Σ p = 1`.  Write the marginals as
- `p_row(i) = Σ_j p(i, j)` — row marginal (how much *spatial mass* a pixel
  receives across all dictionary atoms),
- `p_col(j) = Σ_i p(i, j)` — column marginal (how heavily atom j is used).

The mutual information `I(I; J)` admits two equivalent decompositions:

    I(I; J) = H(J) − H(J | I)
            = H(p_col) − E_{i ∼ p_row}[ H(p(· | i)) ]                  (1)

where `p(j | i) = P_{i,j} / (Σ_{j'} P_{i,j'})` is the per-pixel conditional
on `i` and the expectation is taken with the row marginal as importance
weight.  In the code these two terms are named exactly:

- `column_neg_entropy = − H(p_col)` (`_dispersion_loss` in
  `modules/dictionary_blocks.py`), so adding it to the loss with a positive
  weight **maximises** `H_col`, i.e. drives the column marginal toward
  uniform — every atom must be used somewhere.
- `row_entropy        = E_{i ∼ p_row}[ H(p(· | i)) ]` (`_row_entropy`),
  the mass-weighted mean of per-pixel conditional entropies.  Adding it
  with a positive weight **minimises** `H_row`, i.e. drives each pixel
  toward a peaked choice over the dictionary.

Combining these,

    L_info  =  β_col · (−H_col)  +  β_row · H_row                     (2)
              ↘ maximise H_col   ↘ minimise H_row
            ⇒ maximise  I(I; J)

so the two information regularisers are jointly an upper-bound surrogate
of `−I(I; J)`, recovered up to a positive constant by setting
`β_col = β_row = 1`.  With `β_col ≠ β_row` the model emphasises one of
the two requirements (in WMDC: 0.01 vs 0.05, slightly biasing toward
per-pixel sparsity).

**Why this matters.** Many "dictionary attention" codecs (DCAE,
multi-codebook VQ variants) use only the softmax `−H(p(· | i))` per pixel
or just an L2 attention penalty.  Without the column-entropy term the
uniform routing `P_{i,j} = 1/N` is a global minimiser (no dead codes but
no specialisation either), and 100 % dictionary utilisation co-occurs
with **no information gain**.  Equation (2) is the smallest principled
fix.

### 1.1 The `P · HW` scaling factor

After Sinkhorn we hold `Σ_{i,j} P_{i,j} = 1` (the log-domain potentials
have `log_a = −log HW`, `log_b = −log N` baked in).  In the code we then
multiply by `HW`, so the rescaled plan `P̃ = HW · P` satisfies
`Σ_j P̃_{i,j} ≈ 1` for each `i` — turning row marginals into per-pixel
"used mass" weights `∈ (0, 1]`.  Both `H_col(p_col)` and
`H_row(p(· | i))` are entropies of **normalised** distributions, so the
`HW`-rescaling cancels:

    p_col(j) = (Σ_i P̃_{i,j}) / (Σ_i Σ_{j'} P̃_{i,j'})  =  Σ_i P_{i,j}

(the HW factors in numerator and denominator cancel).  All entropy
regularisers in `_dispersion_loss` / `_row_entropy` are therefore
**invariant** to whether the input is `P` or `HW · P`.  The rescaling
only changes the interpretation of `row_mass = Σ_j P̃_{i,j}`, which is
used by the alignment regulariser of Section 3.

### 1.2 Complexity-alignment regulariser

Define the per-pixel content complexity `c_i ∈ [0, 1]` as the
percentile-normalised Sobel edge magnitude pooled to the latent grid
(see `WMDC._compute_complexity`).  Compression theory says the rate is
dominated by high-complexity regions; the dictionary side-information
must therefore put more mass where `c_i` is large.  Concretely we
require **positive Pearson correlation** between the row-mass field
`m_i = Σ_j P̃_{i,j}` and `c_i`.  A margin hinge,

    L_align = E_{batch, slice}[ ReLU( κ − corr(m, c) ) ],             (3)

penalises any correlation below the margin `κ ≥ 0`.  The original
checkpoint that motivated this term had `corr = −0.66`: the optimiser
chose the **opposite** alignment, dropping mass at complex regions and
laundering the saved rate into the Gaussian channel.  Setting κ = 0
reproduces the sign-only constraint `corr ≥ 0`; the released model uses
`κ = 0.2`.

---

## 2. Unbalanced entropic optimal transport

### 2.1 Primal problem (Séjourné et al., 2019 — "SFVTP19")

Given a cost matrix `C ∈ ℝ_{≥0}^{HW×N}` and target marginals `a, b`,
unbalanced entropic OT solves

    min_{P ≥ 0}  ⟨C, P⟩ + ε · KL(P ‖ a ⊗ b)
                + D_φ_row( P 1_N ∥ a ) + D_φ_col( P^⊤ 1_{HW} ∥ b )    (4)

where `D_φ` is a φ-divergence with strength `ρ`.  The two interesting
choices for `D_φ` are:

| Divergence | `φ(x)`          | Conjugate `φ*(q)` (used by dual) |
|------------|-----------------|---------------------------------|
| `KL{ρ}`    | `ρ · (x log x − x + 1)` | `ρ · (eq/ρ − 1)`           |
| `TV{ρ}`    | `ρ · ‖x − 1‖₁`  | indicator of `[−ρ, +ρ]`         |

### 2.2 Dual potentials and the prox operator

The optimal `P` has the factored form

    P_{i,j} = a_i · b_j · exp( (f_i + g_j − C_{i,j}) / ε )            (5)

and the alternating updates on the dual potentials `f, g` involve the
**anisotropic prox** `aprox_φ`:

    g_j ← − aprox_φ( ε · log Σ_i a_i exp( (f_i − C_{i,j}) / ε ) )     (6a)
    f_i ← − aprox_φ( ε · log Σ_j b_j exp( (g_j − C_{i,j}) / ε ) )    (6b)

with

| Divergence | `aprox_φ(ε, x)`                                        |
|------------|--------------------------------------------------------|
| Balanced   | `x`                                                    |
| `KL{ρ}`    | `x / (1 + ε/ρ) = x · ρ / (ρ + ε)`                      |
| `TV{ρ}`    | `clip(x, −ρ, ρ)`                                       |

In **WMDC** we work in the equivalent log-domain after the substitution
`log_f := f/ε`, `log_g := g/ε`, `M := C/ε` and `log_a := log a`,
`log_b := log b`.  The updates become

    log_g_j ← log_b_j − aprox_φ̃( logsumexp_i (log_f_i − M_{i,j}) )   (7a)
    log_f_i ← log_a_i − aprox_φ̃( logsumexp_j (log_g_j − M_{i,j}) )   (7b)

where `aprox_φ̃` is the same shrink/clip, only the unit of `x` is now
"dual potential / ε":

| Divergence | `aprox_φ̃(x)` (log-domain)                             |
|------------|--------------------------------------------------------|
| Balanced   | `x`                                                    |
| `KL{ρ}`    | `x · ρ/(ρ + ε)`     ← `shrink_*` in `_route_unbalanced_eot` |
| `TV{ρ}`    | `clip(x, −ρ/ε, +ρ/ε)`     ← `clamp(..., ±rpe)` in code |

Important: **the TV log-domain bound is `±ρ/ε`, not `±ρ`**.  The code in
`modules/dictionary_blocks.py:_route_unbalanced_eot` is in the
log-domain and correctly uses `clamp(lse, −ρ/ε, +ρ/ε)`.

### 2.3 Limiting behaviour (sanity-check the implementation)

The unbalanced KL prox `shrink = ρ/(ρ + ε)` interpolates between two
extremes:

| Limit       | shrink             | Routing layer reduces to                |
|-------------|--------------------|------------------------------------------|
| `ρ → ∞`     | `1`                | balanced Sinkhorn (`_route_balanced_eot`) |
| `ρ → 0⁺`    | `0`                | `log_g = log_b`, `log_f = log_a` (uniform — no transport) |
| `ε → 0⁺`    | `1`                | balanced Sinkhorn at the chosen ρ        |

The TV prox `clip(x, ±ρ/ε)` gives the same limits:

| Limit       | clip                      | Routing layer reduces to               |
|-------------|---------------------------|----------------------------------------|
| `ρ → ∞`     | identity                  | balanced Sinkhorn                      |
| `ρ → 0⁺`    | `0`                       | uniform                                |
| `ε → ∞`     | identity                  | maximum-entropy / softmax with τ = ε   |

These are written as **unit tests** in
`tests/test_routing_limits.py` (see Section 4 of `REBUTTAL.md`).

### 2.4 Debiased Sinkhorn divergence (SFVTP19 Def. 6)

Let `OT_ε(α, β)` denote the optimal value of (4) with marginals
`α, β`.  The **Sinkhorn divergence** is

    S_ε(α, β) = OT_ε(α, β) − ½·OT_ε(α, α) − ½·OT_ε(β, β)
                + ½·ε·(|α| − |β|)²                                    (8)

It is symmetric, positive, and **vanishes as `ε → 0`** at the true
Wasserstein cost.  Crucially it removes the `−ε log N` bias of the
entropic OT, so a training loss built from `S_ε` is robust to changes
in `ε` (which `log_eps` adapts during training).

In WMDC the **routing layer** uses one-shot `OT_ε(a, b)`-coupled
sampling (Equation 5).  The **loss-side analogue** of (8) is not used
on the routing layer itself, but it can be enabled to stabilise the
auxiliary entropy regularisers — the debiased version of `H_col` is

    H_col_debias = H_col(P_{ab}) − ½ · H_col(P_{aa}) − ½ · H_col(P_{bb})

where `P_{aa}` denotes the routing P recomputed with a = b (set
`log_a = log_b = −log HW`, swap the column-side accordingly).  In
practice this triples the Sinkhorn cost, so we keep it as an opt-in
debug switch — but the formula is theoretically the principled choice.

---

## 3. Wavelet shortcut (WLS / iWLS) and OLP

### 3.1 WLS as a linear projection in DWT space

Let `W ∈ ℝ^{HW × HW}` denote the orthogonal DWT matrix (separable
`bior4.4`).  Writing `x_LL, x_LH, x_HL, x_HH` for the 4 subbands and
`s = (s_LL, s_LH, s_HL, s_HH)`, the WLS module computes

    y = OLP( concat( s_LL · x_LL, s_LH · x_LH, s_HL · x_HL, s_HH · x_HH ) ).

With `s_{HH} = 0` at init (after the fix in commit `53febe4`), the synthesis
branch ignores the HH band entirely, so the encoder shortcut starts as
a **low-pass auxiliary** and learns the high-frequency contribution
during training.  The original code used `s_HH = 0` *combined with*
`exp(·)` parameterisation, which produced an effective scale of
`exp(0) = 1` — i.e. the opposite of "HH zero-init".

### 3.2 OLP orthogonality regulariser

Let `W ∈ ℝ^{m × n}` be the weight of the OLP module's inner linear
layer.  The OLP loss is

    L_OLP(W) = MSE( W W^⊤, I_m )   if m < n
             = MSE( W^⊤ W, I_n )   if m ≥ n.

This is the **min-side Gram** to keep the loss well-conditioned for both
fat and tall matrices (CMIC-style).  The optimal W is a Stiefel point;
the regulariser combined with the data loss steers projections toward
the column-orthonormal manifold without forcing a hard projection.

---

## 4. Auxiliary regularisers in `RateDistortionLoss`

Putting the pieces together, the training objective is

    L = λ · D + R                                  # standard RD trade-off
      + β_col · (−H_col)                           # = +β_col · column_neg_entropy
      + β_row · (+H_row)                           # = +β_row · row_entropy
      + γ      · ReLU( κ − Pearson(row_mass, c) )  # alignment hinge (margin κ)
      + δ      · ‖DᵀD − I‖_F² / N²                 # dict coherence (symmetric)
      + tv_w   · TV(P)                             # spatial TV on rows of P
      + ortho_w · L_OLP                             # OLP orthogonality

where
- `D ∈ ℝ^{N × d_dict}` are the L2-normalised dictionary atoms,
- `Pearson(·, ·)` is the per-image, per-slice correlation,
- `TV(P) = ½ · ( mean_{h,w} |P[:, h+1, w] − P[:, h, w]| + mean_{h,w} |P[:, h, w+1] − P[:, h, w]| )`,
- `R = bpp(y) + bpp(z)` and `D` is `MSE` or `1 − MS-SSIM`.

Each regulariser controls a single, named failure mode observed during
WMDC's development:

| Regulariser    | Failure it prevents                                           |
|----------------|---------------------------------------------------------------|
| `−H_col`       | Dead dictionary codes (used by 0 pixels)                      |
| `+H_row`       | Per-pixel softmax that spreads evenly across all atoms        |
| `ReLU(κ − ρ)`  | Mass **anti-aligned** with complexity (the −0.66 leakage bug) |
| `dict_penalty` | Near-duplicate / anti-aligned dictionary atoms                |
| `TV(P)`        | High-frequency / speckle noise in the transport plan          |
| `L_OLP`        | Collapse of K/V projections to a low-rank subspace            |

All of these are **opt-in** (each weight defaults to ≥ 0 and can be
zeroed); the model still trains without them, just with degraded RD
curves and higher variance.

---

## 5. Why OT helps image compression (mathematical impact)

This section connects each OT design choice in WMDC to a concrete rate
or distortion term, so that the per-component ablation in
`configs/ablation_variants.py` has an explicit hypothesis to refute.

### 5.1 Setup: dictionary side-info as a channel

Let `ŷ` be the quantised analysis latent and let `hyper` denote the
hyperprior context.  The entropy model parameterises

    p( ŷ_i  |  hyper_i,  d_i )                                        (9)

where `d_i = (P V)_i = Σ_j P_{i,j} v_j` is the per-pixel dictionary
side-info produced by the routing layer.  The rate per pixel is

    R_i = H( ŷ_i  |  hyper_i, d_i )
        = H( ŷ_i  |  hyper_i ) − I( ŷ_i ; d_i  |  hyper_i )           (10)

so **OT reduces rate exactly through the conditional mutual information
`I(ŷ; d | hyper)`**.  The pieces below say *how* each OT design choice
moves that MI.

### 5.2 Why balanced column marginals — channel capacity

For a fixed dictionary `D = (v_1, …, v_N)`, `d_i` lies in the
N-dimensional simplex of mixtures of `v_j`'s.  The mutual information
`I(ŷ; d | hyper)` is upper-bounded by the entropy of the routing index
`J` used to produce `d`:

    I( ŷ ; d  |  hyper )  ≤  H(J)  =  H(p_col)                        (11)

where `p_col(j) = (1/HW) Σ_i P_{i,j}` is the column marginal of `P`
(§1).  So:

> **Proposition 5.1.**  *The maximum side-info MI any dictionary of N
> atoms can supply to the entropy model is `log N`, attained iff
> `p_col` is uniform.*

Plain softmax routing has *no* constraint on `p_col` and empirically
collapses to a small effective support `N_eff ≪ N` ("code collapse").
Balanced Sinkhorn enforces `p_col = 1/N` by construction, so it
**saturates the bound**.  This is why the `no_ueot` ablation (softmax
routing) is expected to lose the most rate of all the routing
ablations: it operates at strictly lower channel capacity.

### 5.3 Why unbalanced row marginals — content-adaptive bit allocation

A balanced row marginal `a_i = 1/HW` forces every pixel to draw the
same amount of side-info mass, regardless of local complexity.  This
is the OT analogue of *uniform bit allocation*, which classical
coding theory has long known to be sub-optimal for natural images.

Unbalanced EOT (Séjourné et al. 2019) relaxes the row constraint to a
KL-divergence penalty `KL{ρ_i}( P 1_N ‖ a )` with **spatially varying**
`ρ_i`.  At optimum the dual updates give

    Σ_j P^*_{i,j}  =  a_i · exp( f^*_i · (ε / (ρ_i + ε)) )            (12)

so a pixel with large `ρ_i` (high content complexity in WMDC's
parameterisation, via `_compute_rho_spatial`) tracks the balanced
marginal closely, while a low-`ρ_i` pixel is allowed to have small
`Σ_j P_{i,j}` — i.e., the routing layer **suppresses dictionary
contribution where it would not pay off**.

> **Proposition 5.2.**  *Let `R_bal` be the expected rate of the
> balanced Sinkhorn plan and `R_unb(ρ)` the rate of the unbalanced plan
> with spatial penalty `ρ`.  For any cost matrix `C` such that the
> per-pixel optimal rates have non-zero variance across `i`, there
> exists a non-constant `ρ` with `R_unb(ρ) < R_bal`.*

The proof sketch: with constant `ρ`, the unbalanced plan reduces to
balanced.  Perturbing `ρ_i` upward where `H(ŷ_i | hyper_i)` is large
and downward where it is small redirects probability mass exactly to
where (10) extracts the most reduction.  This is bit-allocation by OT
duality.

The `balanced_eot_only` ablation tests Proposition 5.2 directly: if it
loses ≥ 0.1 BD-rate against `full` on Kodak, the spatial-ρ adaptivity
is doing genuine work.

### 5.4 Why conditional / multi-marginal OT — the MI chain rule

The autoregressive slice loop produces side-info `D_0, …, D_{S−1}` (10
slices in WMDC).  The total side-info MI decomposes via the chain rule

    I( ŷ ; D_0, …, D_{S−1} )  =  Σ_{i=0}^{S−1} I( ŷ ; D_i  |  D_{<i} )  (13)

If each slice routes **independently** with uniform `b`, the marginal
distribution of `D_i` over the dictionary is the same uniform `p_col`
for every `i`.  Then `D_i` and `D_{<i}` share most of their atoms,
which makes

    I( ŷ ; D_i  |  D_{<i} )  ≪  I( ŷ ; D_i )                          (14)

— each slice is "telling the entropy model the same thing twice".

Conditional OT (Phase B8) shifts the column target for slice `i` by
the cumulative previous usage `u_{<i}`:

    b_i  ∝  max( b_unif − α · u_{<i},  b_floor )                      (15)

so the atoms most-used by slices `0..i−1` are *de-prioritised* for
slice `i`, pushing `D_i` toward the atoms `D_{<i}` did not touch.  In
the limit `α → 1`, `D_i ⊥ D_{<i}` almost surely, and (14) is replaced
by

    I( ŷ ; D_i  |  D_{<i} )  ≈  I( ŷ ; D_i )                          (16)

which **restores additivity in (13)**.  Therefore:

> **Proposition 5.3.**  *Under (15), total side-info MI is a
> monotone non-decreasing function of `α ∈ [0, 1)`, with strict
> increase whenever the per-slice marginals would otherwise overlap.*

The `cond_alpha_*` sweep (α ∈ {0.1, 0.3, 0.5, 0.9}) measures how fast
the saturation hits in practice; `α = 0.9` near-maximal coupling lets
us see whether (16) is achievable on real data or whether the floor
`b_floor` dominates first.

### 5.5 Why bounded ε — the bias–variance trade-off

Entropic OT solves a relaxed problem with bias `O(ε)` in the cost.
The Sinkhorn plan satisfies

    H( p(j | i) )  =  Θ(ε)  as ε → 0,                                 (17)

so smaller ε means sharper per-pixel routing and a more informative
side-info channel.  But Cuturi (2013, Prop. 4) gives gradient norms
that scale as `O(1/ε)`, so very small ε produces:

  * stiff Sinkhorn dynamics (slow convergence, NaN risk),
  * gradient explosions on the `(rho/(rho+ε))·lse` update path,
  * high variance across mini-batches.

Conversely, ε → ∞ collapses `P` to the independent product `a ⊗ b`,
giving `I(ŷ; d | hyper) = 0` and zero rate gain from the dictionary.

WMDC's bounded parameterisation `ε ∈ [0.05, 1.00]` (commit `53febe4`)
keeps the model in the regime where:

| Quantity                | Value at ε = 0.05 | Value at ε = 1.00 |
|-------------------------|-------------------|--------------------|
| `max\|M\| = max\|C\| / ε` | ≤ 40              | ≤ 2                |
| Per-pixel routing entropy `H(p(j\|i))` (bits) | ~0.5 | ~log₂N ≈ 7 |
| Gradient norm (relative)             | × 20            | × 1               |

This is the "useful" region — informative but stable.  The `low_eps`
and `high_eps` ablations probe both ends to show the curve.

### 5.6 Why KL vs TV — convergence vs sharpness

The two unbalanced-OT divergences correspond to different prox
operators in the dual (§2.2):

  * **KL{ρ}**:  `aprox(x) = ρ · x / (ρ + ε)`  — smooth shrinkage.
  * **TV{ρ}**:  `aprox(x) = clip(x, ±ρ)`     — hard gating.

KL is the standard choice in unbalanced OT (Séjourné et al. 2019,
§4.2) and gives smooth gradients.  TV gives a sharper marginal
deviation budget — atoms with `|f^*| > ρ` are *exactly* discarded
from the marginal, which can act as a hard regulariser.  The
`marg_div_tv` ablation tests whether the harder TV gating beats the
smoother KL on RD.  Theory does not predict a winner; this is an
empirical choice the implementation already supports.

### 5.7 Why iteration count matters (and why 20 is enough)

Sinkhorn iterations converge geometrically: at iteration `t`,

    ‖ log_f_t − log_f^* ‖_∞  ≤  q^t · ‖ log_f_0 − log_f^* ‖_∞         (18)

with contraction rate `q < 1` depending on `max|M|/ε`.  For our
bounded ε regime, `q ≈ 0.5` at the worst case, so 20 iterations gives
`q^20 ≈ 10^{-6}` — essentially numerically converged.  The
`sinkhorn_5iter` ablation tests whether under-iterating (e.g. for
inference speed) actually breaks RD.  Theory predicts only a small
penalty; if the ablation is silent, we have a free 4× speedup.

### 5.8 Summary table

| OT component                    | Mechanism                                      | Tested by ablation       | Predicted rate impact |
|---------------------------------|-----------------------------------------------|---------------------------|------------------------|
| Balanced column marginal        | Saturates `H(p_col) = log N` (Prop 5.1)        | `no_ueot`                 | Large                  |
| Unbalanced row marginal         | Content-adaptive bit allocation (Prop 5.2)     | `balanced_eot_only`       | Moderate               |
| Conditional column (cond. OT)   | Restores MI chain-rule additivity (Prop 5.3)   | `cond_alpha_{0.1,0.3,…}`  | Small but monotone     |
| Bounded ε                       | Bias–variance optimum                          | `low_eps`, `high_eps`     | U-shaped               |
| Prox choice (KL vs TV)          | Smooth vs hard marginal gating                 | `marg_div_tv`             | Empirical              |
| Sinkhorn iteration count        | Geometric convergence (Eq. 18)                 | `sinkhorn_5iter`          | Negligible if ≥ 10     |

---

## 6. References

- Séjourné, Feydy, Vialard, Trouvé, Peyré, 2019. **"Sinkhorn Divergences
  for Unbalanced Optimal Transport"**.  Definitions, prox formulas,
  Algorithm 1, and Theorem 5 (consistency).  Source: `papers/unbalancedoptimaltransport.jl-56c9e1c4b5360a5b.txt`.
- AuxT, GLIC.  WLS / iWLS / OLP construction.  Sources:
  `papers/qingshi9974-auxt-…txt`, `papers/unoc-727-glic-…txt`.
- DCAE.  Dictionary cross-attention.  Source:
  `papers/cvl-uestc-dcae-…txt`.
- MambaIC, CMIC.  VSS / Mamba backbone, content-adaptive scan.
  Sources: `papers/aurorazengfh-mambaic-…txt`, `papers/unoc-727-cmic-…txt`.
