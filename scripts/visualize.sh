#!/usr/bin/env bash
# scripts/visualize.sh
# ====================
# Generate all paper figures. Each subcommand corresponds to one figure group.
#
# Usage:
#   bash scripts/visualize.sh all                  # run everything
#   bash scripts/visualize.sh attention            # Fig: attention maps
#   bash scripts/visualize.sh patches              # Fig: patch reconstruction
#   bash scripts/visualize.sh latents              # Fig: latent sparsity
#   bash scripts/visualize.sh sinkhorn             # Fig: Sinkhorn convergence
#   bash scripts/visualize.sh dict_util            # Fig: dictionary utilization
#   bash scripts/visualize.sh rho                  # Fig: ρ heatmap
#   bash scripts/visualize.sh failures             # Fig: failure cases

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

LAMBDA=0.0036
VIZ_DIR="$RESULTS_DIR/viz"
REF_IMG="$KODAK_DIR/kodim01.png"
CONV_IMG="$KODAK_DIR/kodim04.png"   # used for sinkhorn (textured image)

ckpt_routing() { echo "$CHECKPOINT_ROOT/routing/$1/lambda_${LAMBDA}_mse/checkpoint_best.pth.tar"; }

# ── attention maps ───────────────────────────────────────────────────
run_attention() {
    echo ""
    echo "=== Attention maps ==="
    cd "$REPO"
    mkdir -p "$VIZ_DIR"

    for mode in "${ROUTING_MODES[@]}"; do
        local ck; ck="$(ckpt_routing "$mode")"
        [ -f "$ck" ] || { echo "[skip] missing $ck"; continue; }
        echo "-- $mode"

        $PYTHON analyze/visualize_attention.py \
            --img_dir "$KODAK_DIR" --checkpoint "$ck" --routing-mode "$mode" \
            --mode top_tokens --slice 4 --top_k 4 --cuda \
            --output "$VIZ_DIR/attention_top_tokens_${mode}.pdf"

        $PYTHON analyze/visualize_attention.py \
            --img_dir "$KODAK_DIR" --checkpoint "$ck" --routing-mode "$mode" \
            --mode slice_evolution --target_token 42 --cuda \
            --output "$VIZ_DIR/attention_slice_evolution_${mode}.pdf"

        $PYTHON analyze/visualize_attention.py \
            --img_dir "$KODAK_DIR" --checkpoint "$ck" --routing-mode "$mode" \
            --mode spatial_gating --cuda \
            --output "$VIZ_DIR/attention_spatial_gating_${mode}.pdf"

        $PYTHON analyze/visualize_attention.py \
            --img_dir "$KODAK_DIR" --checkpoint "$ck" --routing-mode "$mode" \
            --mode entropy_map --cuda \
            --output "$VIZ_DIR/attention_entropy_map_${mode}.pdf"
    done
    echo "[done] attention maps → $VIZ_DIR"
}

# ── patch reconstruction ─────────────────────────────────────────────
run_patches() {
    echo ""
    echo "=== Patch-level reconstruction ==="
    cd "$REPO"
    mkdir -p "$VIZ_DIR"

    for mode in "${ROUTING_MODES[@]}"; do
        local ck; ck="$(ckpt_routing "$mode")"
        [ -f "$ck" ] || { echo "[skip] missing $ck"; continue; }
        echo "-- $mode"

        $PYTHON analyze/visualize_patches.py \
            -i "$REF_IMG" -c "$ck" \
            --routing-mode "$mode" --cuda \
            -o "$VIZ_DIR/patches_${mode}.pdf"
    done
    echo "[done] patches → $VIZ_DIR"
}

# ── latent sparsity ──────────────────────────────────────────────────
run_latents() {
    echo ""
    echo "=== Latent sparsity ==="
    cd "$REPO"
    mkdir -p "$VIZ_DIR"

    for mode in "${ROUTING_MODES[@]}"; do
        local ck; ck="$(ckpt_routing "$mode")"
        [ -f "$ck" ] || { echo "[skip] missing $ck"; continue; }
        echo "-- $mode"

        $PYTHON analyze/visualize_latents.py \
            --image "$REF_IMG" --checkpoint "$ck" \
            --routing-mode "$mode" --cuda \
            --output "$VIZ_DIR/latents_${mode}.pdf"
    done
    echo "[done] latents → $VIZ_DIR"
}

# ── Sinkhorn convergence ─────────────────────────────────────────────
run_sinkhorn() {
    echo ""
    echo "=== Sinkhorn convergence ==="
    cd "$REPO"
    mkdir -p "$VIZ_DIR"

    local ck; ck="$(ckpt_routing "unbalanced_eot")"
    [ -f "$ck" ] || { echo "[skip] missing $ck"; return; }

    $PYTHON analyze/ablation_sinkhorn_convergence.py \
        --checkpoint "$ck" \
        --image "$CONV_IMG" \
        --routing-mode unbalanced_eot \
        --max-iters 30 \
        --eps-sweep 0.05 0.1 0.2 0.5 \
        --cuda \
        -o "$VIZ_DIR/sinkhorn_convergence.pdf" \
        --json-out "$VIZ_DIR/sinkhorn_convergence.json"

    echo "[done] sinkhorn → $VIZ_DIR/sinkhorn_convergence.pdf"
}

# ── dictionary utilization ───────────────────────────────────────────
run_dict_util() {
    echo ""
    echo "=== Dictionary utilization (WITH vs. WITHOUT dispersion bonus) ==="
    cd "$REPO"
    mkdir -p "$VIZ_DIR"

    local LAM=0.0036
    local ck_with; ck_with="$(ckpt_path "ablation/full" "$LAM")"
    local ck_no;   ck_no="$(ckpt_path "ablation/no_disp_bonus" "$LAM")"

    if [ ! -f "$ck_with" ] || [ ! -f "$ck_no" ]; then
        echo "[skip] one or both ablation checkpoints missing:"
        echo "       with-disp : $ck_with"
        echo "       no-disp   : $ck_no"
        return
    fi

    $PYTHON analyze/ablation_dictionary_utils.py \
        --checkpoint-with-disp "$ck_with" \
        --checkpoint-no-disp   "$ck_no" \
        --dataset "$KODAK_DIR" \
        --routing-mode unbalanced_eot \
        --cuda \
        -o "$VIZ_DIR/dictionary_utilization.pdf" \
        --json-out "$VIZ_DIR/dictionary_utilization.json"

    echo "[done] dict util → $VIZ_DIR/dictionary_utilization.pdf"
}

# ── ρ heatmap ────────────────────────────────────────────────────────
run_rho() {
    echo ""
    echo "=== ρ heatmap ==="
    cd "$REPO"
    mkdir -p "$VIZ_DIR"

    local ck; ck="$(ckpt_routing "unbalanced_eot")"
    [ -f "$ck" ] || { echo "[skip] missing $ck"; return; }

    $PYTHON analyze/visualize_rho_heatmap.py \
        --image-dir "$KODAK_DIR" \
        --checkpoint "$ck" \
        --routing-mode unbalanced_eot --cuda \
        --output "$VIZ_DIR/rho_heatmap.pdf"

    echo "[done] rho heatmap → $VIZ_DIR/rho_heatmap.pdf"
}

# ── failure cases ────────────────────────────────────────────────────
run_failures() {
    echo ""
    echo "=== Failure cases ==="
    cd "$REPO"
    mkdir -p "$VIZ_DIR"

    local ck; ck="$(ckpt_routing "unbalanced_eot")"
    [ -f "$ck" ] || { echo "[skip] missing $ck"; return; }

    $PYTHON analyze/visualize_failure_cases.py \
        --dataset "$KODAK_DIR" \
        --checkpoint "$ck" \
        --routing-mode unbalanced_eot --cuda \
        --output "$VIZ_DIR/failure_cases.pdf"

    echo "[done] failure cases → $VIZ_DIR/failure_cases.pdf"
}

# ── UEOT bitrate-leakage audit ───────────────────────────────────────
run_audit() {
    echo ""
    echo "=== UEOT rate-vs-gating audit (#9 reviewer concern) ==="
    cd "$REPO"
    mkdir -p "$VIZ_DIR"

    local ck; ck="$(ckpt_routing "unbalanced_eot")"
    [ -f "$ck" ] || { echo "[skip] missing $ck"; return; }

    $PYTHON analyze/audit_rate_vs_gating.py \
        --checkpoint "$ck" \
        --dataset "$KODAK_DIR" \
        --routing-mode unbalanced_eot \
        --output-dir "$VIZ_DIR/audit_gating" \
        --cuda

    echo "[done] audit → $VIZ_DIR/audit_gating/"
}

# ── dispatcher ───────────────────────────────────────────────────────
CMDS=("$@")
[ ${#CMDS[@]} -eq 0 ] && CMDS=(all)

for cmd in "${CMDS[@]}"; do
    case "$cmd" in
        attention)  run_attention ;;
        patches)    run_patches ;;
        latents)    run_latents ;;
        sinkhorn)   run_sinkhorn ;;
        dict_util)  run_dict_util ;;
        rho)        run_rho ;;
        failures)   run_failures ;;
        audit)      run_audit ;;
        all)
            run_attention
            run_patches
            run_latents
            run_sinkhorn
            run_dict_util
            run_rho
            run_failures
            run_audit
            ;;
        *)
            echo "Unknown subcommand: $cmd" >&2
            echo "Usage: $0 {all|attention|patches|latents|sinkhorn|dict_util|rho|failures|audit}"
            exit 1
            ;;
    esac
done

echo ""
echo "=== Visualization done. Figures in: $VIZ_DIR ==="
