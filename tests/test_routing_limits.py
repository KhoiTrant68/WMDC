"""
Sanity tests for the routing-mode limits documented in THEORY.md §2.3
and the `log_b_override` plumbing introduced in Phase B8
(conditional / multi-marginal OT).

Each test instantiates a tiny `UnifiedDictionaryAttention` and verifies a
single property:

  1. ρ → ∞ collapses unbalanced EOT onto balanced Sinkhorn (KL prox).
  2. `log_b_override` actually shapes the column marginal of the plan.
  3. The bounded-eps parameterisation keeps eps ∈ [0.05, 1.0] under all
     log_eps values (regression test for the sigmoid switch).

Run with:   pytest tests/test_routing_limits.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make the repo root importable when pytest is invoked from anywhere.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from modules.dictionary_blocks import UnifiedDictionaryAttention


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_attn(routing_mode: str = "unbalanced_eot", iters: int = 80,
               dict_num: int = 8) -> UnifiedDictionaryAttention:
    """Tiny attention module — fits on CPU, converges in <1 s."""
    attn = UnifiedDictionaryAttention(
        input_dim=4,
        output_dim=4,
        dict_num=dict_num,
        dict_dim=4,
        ot_eps=0.1,
        iters=iters,
        routing_mode=routing_mode,
        marginal_div="kl",
    )
    attn.eval()  # use full `iters` (no grad/no-grad split)
    return attn


def _random_cost(B: int = 2, HW: int = 16, N: int = 8,
                 seed: int = 0) -> torch.Tensor:
    """Cosine-style cost matrix in [0, 2]."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, HW, N, generator=g) * 2.0


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

def test_unbalanced_rho_large_recovers_balanced():
    """
    THEORY.md §2.3, KL prox: as ρ → ∞ the shrink `ρ/(ρ+ε)` → 1, so the
    unbalanced update becomes identical to the balanced one.  We pin both
    row- and column-rho to ~20 (well above ε ≈ 0.14) and verify the two
    plans agree element-wise.
    """
    C = _random_cost()
    B, HW, N = C.shape

    bal = _make_attn(routing_mode="balanced_eot")
    P_bal = bal._route_balanced_eot(C)

    unb = _make_attn(routing_mode="unbalanced_eot")
    unb.log_rho_col.data.fill_(20.0)            # softplus(20)+0.01 ≈ 20
    rho_flat = torch.full((B, HW), 1e3)         # row-rho ≫ ε
    P_unb = unb._route_unbalanced_eot(C, rho_flat)

    max_diff = (P_bal - P_unb).abs().max().item()
    assert max_diff < 5e-3, f"unbalanced @ large ρ deviates by {max_diff:.2e}"


def test_log_b_override_shapes_column_marginal():
    """
    Regression test for the multi-marginal-OT plumbing.  When we pass a
    non-uniform target `b` via `log_b_override`, the column marginal of the
    returned plan should match `b` (this is exactly the marginal constraint
    balanced Sinkhorn enforces).
    """
    C = _random_cost(seed=2)
    B, HW, N = C.shape
    assert N % 2 == 0, "test assumes even N"

    bal = _make_attn(routing_mode="balanced_eot")

    # Non-uniform target: first half of atoms gets 90 % of column mass.
    b = torch.empty(B, 1, N)
    b[:, :, : N // 2] = 0.9 / (N // 2)
    b[:, :, N // 2 :] = 0.1 / (N // 2)
    log_b = b.log()

    P = bal._route_balanced_eot(C, log_b_override=log_b)
    col_marginal = P.sum(dim=1)                 # (B, N)
    expected = b.squeeze(1)                     # (B, N)

    max_diff = (col_marginal - expected).abs().max().item()
    assert max_diff < 1e-3, (
        f"col marginal {col_marginal[0].tolist()} != target "
        f"{expected[0].tolist()} (max diff {max_diff:.2e})"
    )


def test_eps_is_bounded():
    """
    Regression test for the sigmoid parameterisation of ε
    (commit 53febe4).  No matter how far the optimiser drives `log_eps`,
    ε must remain in [EPS_FLOOR, EPS_FLOOR + EPS_RANGE] = [0.05, 1.0].
    """
    attn = _make_attn()
    for v in (-1e6, -10.0, 0.0, 10.0, 1e6):
        attn.log_eps.data.fill_(v)
        eps = float(attn._eps())
        assert 0.05 - 1e-6 <= eps <= 1.0 + 1e-6, (
            f"log_eps={v} produced eps={eps} outside [0.05, 1.0]"
        )
