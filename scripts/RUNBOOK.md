# WMDC — Runbook

Toàn bộ lệnh để reproduce thí nghiệm trong paper.  
Thay thế các path sau theo môi trường của bạn:

| Biến | Kaggle | Ý nghĩa |
|------|--------|---------|
| `$REPO` | `/kaggle/working/WMDC` | Thư mục gốc repo |
| `$TRAIN_DATA` | `/kaggle/input/datasets/tranjohan/data-1000-test/dataset_1000_test` | Dataset train/val |
| `$KODAK` | `/kaggle/input/datasets/khitrnminh/kodak-test` | Kodak test set |
| `$CKPT_ROOT` | `/kaggle/working/WMDC/checkpoints` | Thư mục lưu checkpoint |

---

## 1. Setup môi trường

```bash
pip install compressai accelerate einops timm pytorch-msssim flops-profiler bjontegaard
pip install tensorboard==2.14 protobuf==4.25.3

# Build selective_scan CUDA kernel (chạy 1 lần)
cd $REPO/vmamba && pip install . && cd $REPO

# Kiểm tra wavelet module
python -m modules.wavelet_blocks
```

---

## 2. Training — Routing mode ablation (Table 1)

```bash
cd $REPO

for MODE in unbalanced_eot balanced_eot softmax; do
    accelerate launch train.py \
        -d $TRAIN_DATA \
        --save_path $CKPT_ROOT/routing/$MODE \
        --lambda 0.0036 \
        --routing-mode $MODE \
        --epochs 400 --batch-size 8 \
        --lr-milestones 360 380 --last-epochs-with-ste 20
done
```

**Resume từ checkpoint:**

```bash
accelerate launch train.py \
    -d $TRAIN_DATA \
    --lambda 0.0036 \
    --save_path $CKPT_ROOT/routing/unbalanced_eot \
    --checkpoint $CKPT_ROOT/routing/unbalanced_eot/lambda_0.0036_mse/checkpoint_latest.pth.tar
```

---

## 3. Training — Component-removal ablation (Table 4)

23 variants × 3 λ = 69 jobs. Xem `configs/ablation_variants.py` để biết
ý nghĩa từng variant (nguồn duy nhất của truth — RUNBOOK chỉ minh hoạ).
Các variant `OT-specific` (5.x) phủ THEORY.md §5 và là phần chính của
phụ lục OT trong rebuttal.

```bash
cd $REPO

train_variant() {
    local VARIANT=$1 LAM=$2 EXTRA=$3
    accelerate launch train.py \
        -d $TRAIN_DATA \
        --save_path $CKPT_ROOT/ablation/$VARIANT/lam_$LAM \
        --lambda $LAM --metric mse \
        --epochs 400 --batch-size 8 \
        --lr-milestones 360 380 --last-epochs-with-ste 20 \
        $EXTRA
}

for LAM in 0.0036 0.013 0.0483; do
    # ── Original 5 (Table 4) ─────────────────────────────────────────
    train_variant full                  $LAM ""
    train_variant no_ueot               $LAM "--routing-mode softmax"
    train_variant no_fdm                $LAM "--backbone ss2d"
    train_variant no_stateful_mem       $LAM "--use-dense-concat"
    train_variant no_bootstrap_M1       $LAM "--memory-init zero"

    # ── Routing-loss ablations ──────────────────────────────────────
    train_variant no_disp_bonus         $LAM "--column-entropy-weight 0.0 --row-entropy-weight 0.0 --alignment-weight 0.0"
    train_variant no_col_entropy        $LAM "--column-entropy-weight 0.0"
    train_variant no_row_entropy        $LAM "--row-entropy-weight 0.0"
    train_variant no_alignment          $LAM "--alignment-weight 0.0"
    train_variant no_alignment_margin   $LAM "--alignment-margin 0.0"
    train_variant no_dict_penalty       $LAM "--dict-penalty-weight 0.0"

    # ── CMIC components ─────────────────────────────────────────────
    train_variant no_wls                $LAM ""                                  # strip --use-wls-shortcut via config.sh sentinel
    train_variant no_olp                $LAM "--ortho-weight 0.0"
    train_variant no_ste_y              $LAM "--last-epochs-with-ste 0"

    # ── Phase B8: multi-marginal OT (positive ablation) ─────────────
    train_variant full_cond_marg        $LAM "--use-conditional-marginals --cond-alpha 0.5"
    train_variant full_cond_marg_strong $LAM "--use-conditional-marginals --cond-alpha 0.9"

    # ── OT-specific ablations (THEORY.md §5) ────────────────────────
    train_variant balanced_eot_only     $LAM "--routing-mode balanced_eot"             # §5.3 Prop 5.2
    train_variant marg_div_tv           $LAM "--marginal-div tv"                       # §5.6 KL vs TV
    train_variant low_eps               $LAM "--ot-eps 0.06"                           # §5.5 sharp routing
    train_variant high_eps              $LAM "--ot-eps 0.5"                            # §5.5 smooth routing
    train_variant sinkhorn_5iter        $LAM "--sinkhorn-iters 5"                      # §5.7 convergence
    train_variant cond_alpha_0p1        $LAM "--use-conditional-marginals --cond-alpha 0.1"  # §5.4 α sweep
    train_variant cond_alpha_0p3        $LAM "--use-conditional-marginals --cond-alpha 0.3"  # §5.4 α sweep
done
```

> `full` (without `--use-conditional-marginals`) **is** the no-cond-marg
> baseline — the `full_cond_marg*` variants are the positive ablation
> measured against it.  There is no separate `no_cond_marg` variant.

> Trên SLURM: dùng `./scripts/run_ablation.sh launch` sau khi chỉnh `DATA_DIR`, `CHECKPOINT_ROOT`.

---

## 4. Evaluation trên Kodak

```bash
cd $REPO

for MODE in unbalanced_eot balanced_eot softmax; do
    python eval.py \
        --dataset $KODAK \
        --checkpoint $CKPT_ROOT/routing/$MODE/lambda_0.0036_mse/checkpoint_best.pth.tar \
        --output results/routing/$MODE \
        --routing-mode $MODE \
        --cuda \
        --measure-dict-util
done
```

---

## 5. Visualizations

### 5.1 Attention maps

```bash
cd $REPO

for MODE in unbalanced_eot balanced_eot softmax; do
    CK=$CKPT_ROOT/routing/$MODE/lambda_0.0036_mse/checkpoint_best.pth.tar

    python analyze/visualize_attention.py \
        --img_dir $KODAK --checkpoint $CK --routing-mode $MODE \
        --mode top_tokens --slice 4 --top_k 4 --cuda \
        --output results/viz/attention_top_tokens_$MODE.pdf

    python analyze/visualize_attention.py \
        --img_dir $KODAK --checkpoint $CK --routing-mode $MODE \
        --mode slice_evolution --target_token 42 --cuda \
        --output results/viz/attention_slice_evolution_$MODE.pdf

    python analyze/visualize_attention.py \
        --img_dir $KODAK --checkpoint $CK --routing-mode $MODE \
        --mode spatial_gating --cuda \
        --output results/viz/attention_spatial_gating_$MODE.pdf

    python analyze/visualize_attention.py \
        --img_dir $KODAK --checkpoint $CK --routing-mode $MODE \
        --mode entropy_map --cuda \
        --output results/viz/attention_entropy_map_$MODE.pdf
done
```

### 5.2 Patch-level reconstruction

```bash
cd $REPO

for MODE in unbalanced_eot balanced_eot softmax; do
    python analyze/visualize_patches.py \
        -i $KODAK/kodim01.png \
        -c $CKPT_ROOT/routing/$MODE/lambda_0.0036_mse/checkpoint_best.pth.tar \
        --routing-mode $MODE --cuda \
        -o results/viz/patches_$MODE.pdf
done
```

### 5.3 Latent sparsity

```bash
cd $REPO

for MODE in unbalanced_eot balanced_eot softmax; do
    python analyze/visualize_latents.py \
        --image $KODAK/kodim01.png \
        --checkpoint $CKPT_ROOT/routing/$MODE/lambda_0.0036_mse/checkpoint_best.pth.tar \
        --routing-mode $MODE --cuda \
        --output results/viz/latents_$MODE.pdf
done
```

### 5.4 Sinkhorn convergence (Section 3.3)

```bash
cd $REPO

python analyze/ablation_sinkhorn_convergence.py \
    --checkpoint $CKPT_ROOT/routing/unbalanced_eot/lambda_0.0036_mse/checkpoint_best.pth.tar \
    --image $KODAK/kodim04.png \
    --routing-mode unbalanced_eot \
    --max-iters 30 \
    --eps-sweep 0.05 0.1 0.2 0.5 \
    --cuda \
    -o results/viz/sinkhorn_convergence.pdf \
    --json-out results/viz/sinkhorn_convergence.json
```

### 5.5 Dictionary utilization — WITH vs. WITHOUT dispersion bonus

```bash
cd $REPO

python analyze/ablation_dictionary_utils.py \
    --checkpoint-with-disp $CKPT_ROOT/ablation/full/lam_0.013/lambda_0.013_mse/checkpoint_best.pth.tar \
    --checkpoint-no-disp   $CKPT_ROOT/ablation/no_disp_bonus/lam_0.013/lambda_0.013_mse/checkpoint_best.pth.tar \
    --dataset $KODAK \
    --routing-mode unbalanced_eot \
    --cuda \
    -o results/viz/dictionary_utilization.pdf
```

### 5.6 ρ heatmap

```bash
cd $REPO

python analyze/visualize_rho_heatmap.py \
    --image $KODAK/kodim01.png \
    --checkpoint $CKPT_ROOT/routing/unbalanced_eot/lambda_0.0036_mse/checkpoint_best.pth.tar \
    --routing-mode unbalanced_eot --cuda \
    --output results/viz/rho_heatmap.pdf
```

### 5.7 Failure cases

```bash
cd $REPO

python analyze/visualize_failure_cases.py \
    --img_dir $KODAK \
    --checkpoint $CKPT_ROOT/routing/unbalanced_eot/lambda_0.0036_mse/checkpoint_best.pth.tar \
    --routing-mode unbalanced_eot --cuda \
    --output results/viz/failure_cases.pdf
```

---

## 6. BD-rate (Tables 2, 3, 4)

### 6.1 Evaluate ablation variants → RD JSON

```bash
cd $REPO

for VARIANT in full no_ueot no_fdm no_stateful_mem no_bootstrap_M1 \
               no_disp_bonus no_col_entropy no_row_entropy no_alignment no_alignment_margin \
               no_dict_penalty no_wls no_olp no_ste_y \
               full_cond_marg full_cond_marg_strong \
               balanced_eot_only marg_div_tv low_eps high_eps sinkhorn_5iter \
               cond_alpha_0p1 cond_alpha_0p3; do
    CKPTS=""
    for LAM in 0.0036 0.013 0.0483; do
        CKPTS="$CKPTS $CKPT_ROOT/ablation/$VARIANT/lam_$LAM/lambda_${LAM}_mse/checkpoint_best.pth.tar"
    done

    python analyze/compute_bd_rate.py evaluate \
        --variant-name $VARIANT \
        --checkpoints $CKPTS \
        --dataset $KODAK \
        --output results/bd_rate/${VARIANT}.json \
        --routing-mode unbalanced_eot \
        --cuda
done
```

### 6.2 Tính BD-rate vs. "full" anchor

```bash
cd $REPO

python analyze/compute_bd_rate.py bd-rate \
    --anchor-json results/bd_rate/full.json \
    --variant-jsons \
        results/bd_rate/no_ueot.json \
        results/bd_rate/no_fdm.json \
        results/bd_rate/no_stateful_mem.json \
        results/bd_rate/no_bootstrap_M1.json \
        results/bd_rate/no_disp_bonus.json \
        results/bd_rate/no_dict_penalty.json \
    --output results/bd_rate/bd_rate_table.json \
    --method akima
```

### 6.3 Xuất LaTeX table

```bash
cd $REPO
python analyze/aggregate_to_latex.py \
    --bd-rate-json results/bd_rate/bd_rate_table.json
```

---

## 7. Benchmark VRAM & FLOPs

```bash
cd $REPO

python analyze/measure_vram.py \
    --checkpoint $CKPT_ROOT/routing/unbalanced_eot/lambda_0.0036_mse/checkpoint_best.pth.tar \
    --routing-mode unbalanced_eot --cuda

python analyze/benchmark_backbone.py \
    --routing-mode unbalanced_eot --cuda
```

---

## 8. Lưu kết quả

```bash
cd $REPO

mkdir -p result_save
cp -r results/viz/*.pdf result_save/ 2>/dev/null || true
cp -r results/routing   result_save/
cp -r results/bd_rate   result_save/

zip -r result_save.zip result_save/
echo "Done: result_save.zip"
```

---

## Ghi chú

- `--routing-mode` dùng dấu gạch ngang; argparse cũng chấp nhận `--routing_mode`.
- `accelerate launch` cho multi-GPU; đơn GPU thì dùng `python` trực tiếp.
- Warning `selective_scan_cuda` là bình thường nếu chưa build kernel — model tự fallback.
- `bjontegaard` cần cài thêm để chạy Section 6: `pip install bjontegaard`.
