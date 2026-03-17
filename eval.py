"""
cp_evaluation.py  –  Evaluate conformal prediction across scoring functions.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from CPWrapper import CPWrapper
from scoring_functions import oneminussoftmax, APS  # add more as needed
SCORING_FUNCTIONS = {
    "oneminussoftmax": oneminussoftmax,
    "APS": APS,
}
ALPHA = 0.1


def evaluate(model, calib_dataset, test_dataset, num_classes, device="cpu"):
    for fn_name, fn in SCORING_FUNCTIONS.items():
        print(f"\n{fn_name}")

        wrapper = CPWrapper(model, alpha=ALPHA, scoring_fn=fn, device=device)
        wrapper.fit(calib_dataset)

        pred_sets, true_labels = [], []
        model.eval()
        with torch.no_grad():
            loader = test_dataset if isinstance(test_dataset, DataLoader) else DataLoader(test_dataset, batch_size=256)
            for x, y in loader:
                pred_sets.extend(wrapper.predict(x))
                true_labels.extend(y.numpy().reshape(-1).tolist())

        sizes     = np.array([len(s) for s in pred_sets])
        in_set    = np.array([y in s for s, y in zip(pred_sets, true_labels)])
        n         = len(true_labels)
        small_cap = max(1, int(np.ceil(0.25 * num_classes)))

        # ── Global metrics ────────────────────────────────────────────────
        coverage = in_set.mean()
        status   = "PASS" if coverage >= 1 - ALPHA else "FAIL "
        print(f"  Coverage       : {coverage:.4f}  (target ≥ {1-ALPHA:.2f})  {status}")
        print(f"  Avg set size   : {sizes.mean():.3f}")
        print(f"  Singleton rate : {(sizes == 1).mean():.3f}")
        print(f"  Small-set rate : {(sizes <= small_cap).mean():.3f}  (|C| ≤ {small_cap} = 25% of {num_classes})")

        # ── Per-class coverage ────────────────────────────────────────────
        print(f"\n  Per-class coverage:")
        for k in range(num_classes):
            mask = np.array([y == k for y in true_labels])
            cov  = in_set[mask].mean() if mask.sum() > 0 else float("nan")
            flag = "" if np.isnan(cov) or cov >= 1 - ALPHA else "  ← LOW"
            print(f"    class {k}: {cov:.4f}{flag}")

        # ── Set-size distribution ─────────────────────────────────────────
        print(f"\n  Set-size distribution:")
        for size in sorted(set(sizes)):
            count = int((sizes == size).sum())
            print(f"    |C|={size}: {count:>5} ({100*count/n:.1f}%)")


