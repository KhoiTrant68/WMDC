# Pre-Submission Checklist (WMDC → CVPR)

Status of reviewer-driven changes and remaining work. Tick boxes as you go.

## ✅ Done (code committed)

- [x] **K-means DDP fix** — `modules/content_adaptive_blocks.py`
  - `_bootstrap_from_batch` broadcasts from rank 0
  - `_center_iter` all-reduces sum and count, uses `index_add_` (no Python
    loop), applies one global EMA update per rank
  - Falls back to single-GPU path when `dist.is_initialized()` is False
- [x] **Sinkhorn telemetry** — `modules/dictionary_blocks.py`
  - `EPS_FLOOR` raised from 0.01 → 0.05 (max|M| ≤ 40 under cosine distance)
  - Per-attention buffers: `_sinkhorn_calls`, `_sinkhorn_fallbacks`,
    `_last_eps_at_fallback`, `_last_rho_col_at_fallback`,
    `_last_max_M_at_fallback`
  - `sinkhorn_telemetry()` method on both attention and model
  - Val loop logs aggregate rate to TensorBoard + log file
- [x] **Rebuttal letter** — `REBUTTAL.md`
- [x] **SOTA comparison framework** — `analyze/compare_against_sota.py` +
  `analyze/sota_kodak_anchors.json`

## 🔴 Critical (blocks submission)

The reviewer feedback is engineering-level. The submission verdict will be
decided by **RD numbers on Kodak / CLIC2020 / Tecnick**, not by code quality.
Until you have these, the CVPR question is undecided.

### 1. Reproduce Kodak RD curve (6 λ points) — required for any compression paper

```bash
# Train 6 lambdas on OpenImages (you have these checkpoints already from
# the real ~400 epoch runs; if not, this is ~6 × 400-epoch jobs)
for lam in 0.0018 0.0035 0.0067 0.013 0.025 0.0483; do
  accelerate launch train.py -d /path/to/openimages \
    --save_path checkpoints --lambda $lam --metric mse \
    --routing-mode unbalanced_eot --epochs 400
done

# Evaluate on Kodak with the real codec (not estimated bpp)
python analyze/compute_bd_rate.py evaluate \
  --variant-name wmdc_full \
  --checkpoints checkpoints/lambda_0.0018_mse/checkpoint_best.pth.tar \
                checkpoints/lambda_0.0035_mse/checkpoint_best.pth.tar \
                checkpoints/lambda_0.0067_mse/checkpoint_best.pth.tar \
                checkpoints/lambda_0.013_mse/checkpoint_best.pth.tar \
                checkpoints/lambda_0.025_mse/checkpoint_best.pth.tar \
                checkpoints/lambda_0.0483_mse/checkpoint_best.pth.tar \
  --dataset /path/to/kodak \
  --output rd_results/wmdc_full_kodak.json \
  --cuda

# Sanity-check against published SOTA
python analyze/compare_against_sota.py \
  --wmdc-json rd_results/wmdc_full_kodak.json \
  --anchors analyze/sota_kodak_anchors.json \
  --output rd_results/sota_comparison_kodak.json \
  --vs-anchor mlic_plus
```

Decision rules from the tool's CVPR-call output:
- **BD-rate vs MLIC++ < 0%** → CVPR submit, story = "first OT-based routing
  that beats MLIC++"
- **BD-rate vs VTM < −13%** → CVPR submit, comparable to SOTA
- **BD-rate vs VTM < −10%** → CVPR risky; ECCV/DCC realistic
- **BD-rate vs VTM < −7%** → ECCV/DCC/TIP target
- **BD-rate vs VTM ≥ −7%** → re-scope the paper before submission

### 2. CLIC2020 + Tecnick numbers

Same pipeline. CLIC2020 has Professional and Mobile subsets — report both
separately. Tecnick is 1200×1200 (24 RGB images).

### 3. Re-run `content_adaptive` ablation row

The K-means DDP fix changes the effective batch for centroid estimation
from 1/N to N/N. The `content_adaptive` ablation row in Table 4 must be
re-trained on the fixed code. Single λ at 0.013 is sufficient for the
ablation row.

## 🟡 Important (strengthens paper, fillable in supp.)

### 4. Sinkhorn stability paragraph

After completing the Kodak runs, fill in the `[INSERT_RATE]` placeholder
in `REBUTTAL.md` §2.2 from the val log:

```bash
grep "Sinkhorn fb:" logs/train_*.log | tail -20
```

Quote the final-epoch fallback rate. Expected value: < 0.5%.

### 5. Visualisation figures (reviewer 6a–6b)

- **Fig. F1 (FDM)**: pick kodim07 + kodim21. Run
  `analyze/visualize_attention.py` or `analyze/visualize_latents.py` to
  extract LL pre/post-FiLM features. 4×3 grid (LL/LH/HL/HH × image × pre/post).
- **Fig. F2 (UEOT row-mass)**: 6 Kodak images × 3 columns (image |
  complexity from `_compute_complexity` | row_mass from `last_row_mass`).
  Use `analyze/visualize_rho_heatmap.py` as starting point.

### 6. `num_slices` ablation

Quick — 5 short training jobs at λ = 0.013 only:

```bash
for ns in 1 3 5 8 10; do
  # Need to expose --num-slices in train.py first (currently hardcoded in main())
  accelerate launch train.py -d /path/to/openimages \
    --lambda 0.013 --num-slices $ns --epochs 100 \
    --save_path checkpoints_slices/ns_$ns
done
```

Note: `train.py:936` currently hardcodes `num_slices=5`. Add a CLI flag
before running this ablation.

## 🟢 Nice to have

### 7. Encoding/decoding latency table

Reviewers always ask. From `eval.py` output, you already have `enc_time`
and `dec_time` per image. Aggregate per model variant and produce a
parameters/FLOPs/latency table for the supplementary. Use
`analyze/profile_model.py` for FLOPs.

### 8. Document rebuttal numbers

Once Kodak runs land, edit `REBUTTAL.md` §2.2 and §6 to replace every
`[INSERT_*]` placeholder with real numbers.

## Verification before submitting

```bash
# 1. Sinkhorn fallback rate is reasonable
grep "Sinkhorn fb:" logs/train_*.log | tail -3
# Expect < 1% in late epochs

# 2. K-means fix doesn't NaN
python -c "
from modules.content_adaptive_blocks import TokenClustering
import torch
tc = TokenClustering(cluster_num=8, feature_dim=64)
x = torch.randn(2, 100, 64)
tc.train()
for _ in range(5):
    a = tc(x)
    assert not torch.isnan(tc.means).any()
print('K-means: OK')
"

# 3. BD-rate comparison runs end-to-end
python analyze/compare_against_sota.py \
  --wmdc-json rd_results/wmdc_full_kodak.json \
  --anchors analyze/sota_kodak_anchors.json \
  --output /tmp/test_comparison.json
```

## Honest CVPR probability table

Given (a) reviewer fixes done and (b) assumed Kodak BD-rate vs MLIC++:

| BD-rate vs MLIC++ | CVPR accept probability | Realistic venue |
|:---:|:---:|:---:|
| < −2% (clearly beats) | 50–65% | CVPR / ICCV |
| 0% to −2% (comparable) | 25–40% | CVPR risky / ECCV likely |
| 0% to +3% (slightly worse) | 10–20% | ECCV / DCC / TIP |
| > +3% (clearly worse) | < 5% | DCC / TIP / scope change |

This is the question Kodak numbers will answer. Everything else is in
service of that result.
