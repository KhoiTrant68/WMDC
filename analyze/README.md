# WMDC — Experiment Pipeline

This directory contains the analysis and ablation tooling used to
generate the numbers and figures in the paper. Numbers in `XX.XX`
format in the LaTeX file are placeholders that will be replaced by
the JSON outputs of these scripts.

## Required additions to the project

Several scripts assume `train.py` and `models/WMDC.py` accept new CLI
flags. **You must wire these into the trainer before running the
component-removal ablation (Table 4).** The following are the only
strict requirements:

| Flag | Where | Effect |
|---|---|---|
| `--backbone {fdm,cnn,swin,ss2d,fdm_reversed}` | `WMDC.__init__` | Selects the FDM-replacement block from `ablation_models/backbone_variants.py`. Default `fdm` (production). |
| `--use-dense-concat` | `train.py` + slice loop in `WMDC.forward` | When set, replace the stateful memory loop with classic channel-autoregressive concatenation `[Φ, ŷ_<k]`. |
| `--memory-init {bootstrap,zero}` | `WMDC.init_memory` | When `zero`, initialise `M_1` with zeros (bypass the bootstrap conv). |

The other ablation flags (`--routing-mode`, `--disp-weight`,
`--dict-penalty-weight`) already exist in the trainer.

## Required Python packages

```
pip install bjontegaard       # BD-rate computation (compute_bd_rate.py)
pip install fvcore            # GFLOPs counter (preferred)
pip install thop              # Fallback GFLOPs counter
pip install scikit-learn      # t-SNE + KMeans for dictionary plots
pip install matplotlib pillow torchvision   # already standard
```

## Pipeline overview

```
                    train (cluster / single node)
                            │
                ┌───────────┼─────────────┐
                ▼           ▼             ▼
          full WMDC   no_ueot, …    no_dict_penalty
        (6 lambdas)   (3 lambdas    (3 lambdas
                       each)         each)
                            │
                            ▼
              compute_bd_rate.py evaluate
                  (one RD JSON per variant)
                            │
        ┌──────┬────────────┼───────────────┬──────────┐
        ▼      ▼            ▼               ▼          ▼
   benchmark  measure   verify_consistency  visualize  ablation_dictionary_utils
   _backbone  _vram                         _*         (entropy diagnostics)
        │      │                            │          │
        ▼      ▼                            ▼          ▼
   *.json   vram.json                  *.pdf       util.json
        │      │                                       │
        └──────┴────────┬─────────────┬────────────────┘
                        ▼             ▼
              compute_bd_rate.py    aggregate_to_latex.py
              bd-rate              (table fragments)
                        │             │
                        ▼             ▼
              bd_rate_table.json  table*.tex (ready for paper)
```

## Step-by-step

### 1. Train the variants (one-time, ~150 GPU-days for full ablation)

```bash
# Edit DATA_DIR, KODAK_DIR, CHECKPOINT_ROOT in scripts/run_ablation.sh
./scripts/run_ablation.sh dry-run        # inspect commands
./scripts/run_ablation.sh launch         # submit to cluster
```

### 2. Evaluate each variant on Kodak (one RD curve per variant)

```bash
./scripts/run_ablation.sh evaluate
# → produces rd_results/<variant>.json for each variant
# → produces bd_rate_table.json (Table 4 input)
```

### 3. Run diagnostic measurements

#### Backbone latency + GFLOPs (Table 2)

```bash
python analyze/benchmark_backbone.py \
    --backbones fdm cnn swin ss2d fdm_reversed \
    --resolution 768 512 \
    --output backbone_benchmark.json \
    --cuda
```

#### Slice-context VRAM (Table 3)

```bash
python analyze/measure_vram.py \
    --variants stateful dense \
    --resolutions 256 512 768 \
    --mode train \
    --output vram_results.json \
    --cuda
```

#### Per-image bit allocation (Table 3)

```bash
# Run for both stateful and dense checkpoints
python analyze/analysis_bit_allocation.py \
    --checkpoint ckpts/full/lam_0.013/best.pth.tar \
    --image      /data/kodak/kodim01.png \
    -o bit_alloc_stateful.json
```

#### Dictionary utilisation entropy (Table 1)

```bash
python analyze/ablation_dictionary_utils.py \
    --checkpoint-with-disp ckpts/full/lam_0.013/best.pth.tar \
    --checkpoint-no-disp   ckpts/no_disp_bonus/lam_0.013/best.pth.tar \
    --dataset              /data/kodak \
    --json-out             dictionary_utilization.json \
    --cuda
```

#### Sinkhorn convergence (paper Section 3.3)

```bash
python analyze/ablation_sinkhorn_convergence.py \
    --checkpoint ckpts/full/lam_0.013/best.pth.tar \
    --image      /data/kodak/kodim05.png \
    --max-iters  30 \
    --eps-sweep  0.05 0.1 0.2 0.5 \
    --json-out   sinkhorn_convergence.json \
    -o           sinkhorn_convergence.pdf \
    --cuda
```

#### Train/inference consistency (paper Section 3.4)

```bash
python analyze/verify_train_inference_consistency.py \
    --checkpoints ckpts/full/lam_0.013/best.pth.tar \
                  ckpts/full/lam_0.025/best.pth.tar \
    --dataset     /data/kodak \
    --output      consistency_report.json \
    --plot        consistency_histograms.pdf \
    --cuda
```

### 4. Visualisations (figures)

```bash
# ρ heatmap — supports the spatial-adaptive ρ claim
python analyze/visualize_rho_heatmap.py \
    --checkpoint ckpts/full/lam_0.013/best.pth.tar \
    --image-dir  /data/kodak \
    --num-images 4 \
    -o rho_heatmap.pdf \
    --cuda

# Dictionary diversity — supports dict_penalty claim
python analyze/visualize_dictionary_diversity.py \
    --checkpoint-with    ckpts/full/lam_0.013/best.pth.tar \
    --checkpoint-without ckpts/no_dict_penalty/lam_0.013/best.pth.tar \
    --image              /data/kodak/kodim05.png \
    -o dictionary_diversity.pdf \
    --cuda

# Failure cases — supports limitations subsection / rebuttal
python analyze/visualize_failure_cases.py \
    --checkpoint    ckpts/full/lam_0.013/best.pth.tar \
    --dataset       /data/kodak \
    --reference-rd  anchors/vtm_kodak.json \
    --top-k         5 \
    -o              failure_cases.pdf \
    --cuda
```

### 5. Aggregate JSONs into LaTeX table fragments

```bash
mkdir -p tables

# Table 1 — routing
python analyze/aggregate_to_latex.py table1-routing \
    --bd-rate-json   bd_rate_routing.json \
    --entropy-json   dictionary_utilization.json \
    --output-tex     tables/table1_routing.tex

# Table 2 — backbone
python analyze/aggregate_to_latex.py table2-backbone \
    --benchmark-json backbone_benchmark.json \
    --rd-jsons       rd_results/fdm.json rd_results/cnn.json \
                     rd_results/swin.json rd_results/ss2d.json \
                     rd_results/fdm_reversed.json \
    --bpp-target     0.4 \
    --output-tex     tables/table2_backbone.tex

# Table 3 — context
python analyze/aggregate_to_latex.py table3-context \
    --bit-allocation-jsons bit_alloc_stateful.json bit_alloc_dense.json \
    --vram-json            vram_results.json \
    --bd-rate-json         bd_rate_context.json \
    --output-tex           tables/table3_context.tex

# Table 4 — components
python analyze/aggregate_to_latex.py table4-components \
    --bd-rate-json bd_rate_table.json \
    --output-tex   tables/table4_components.tex
```

In the paper LaTeX, replace each table body with `\input{tables/tableX_*.tex}`.

## Compute estimate (with chosen scope)

With 6 components × 3 λ + 3 routing modes × 3 λ + main RD curve at 6 λ:

| Block | Jobs | GPU-days (8-GPU node, 3-day train) |
|---|---|---|
| Main RD curve (6 λ) | 6 | 18 |
| Component ablation (6 variants × 3 λ) | 18 | 54 |
| Routing ablation (3 modes × 3 λ; reuses softmax/balanced from above) | 6 new | 18 |
| Backbone ablation (4 variants × 1 λ; mid only) | 4 | 12 |
| **Total** | **34** | **~102 GPU-days** |

On a single 8-GPU node this is roughly **13 days wall-clock**. On 4 such
nodes in parallel, **~3.5 days**.

## Bug fixes shipped in this drop

The following bugs in the original `analyze/` scripts were silently
broken by the model refactor (h_trunk + h_scale_head/h_mean_head) and
the ModuleList rename (`eot_attentions`):

1. `ablation_dictionary_utils.py` — registered hook on the non-existent
   `model.eot_attention` (singular). Fixed by hooking each entry of
   `model.eot_attentions` (the actual ModuleList).
2. `ablation_dictionary_utils.py` — column-marginal "entropy" was
   computed on a renormalised distribution that didn't match the
   training-time loss. The fixed version reports BOTH the
   loss-aligned and the renormalised entropies so the paper number is
   unambiguous.
3. `ablation_sinkhorn_convergence.py` — called `model.h_scale_s` /
   `model.h_mean_s` (removed during refactor) and assumed
   `model.hyper_to_dict()` returned a single tensor (it now returns a
   tuple `(dt, dict_penalty)`). Fixed by routing through
   `model._hyper_decode()` and unpacking the tuple.

The fixed versions are drop-in replacements for the originals.
