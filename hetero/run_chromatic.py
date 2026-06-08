"""Stage-2: does chromatic (kernel) persistence beat controls on classification?

First the synthetic positive-control (chromatic MUST win by construction), then real
hetero datasets (Task 11). Controls: none / chromatic / achromatic / shuffled / random.

run_synthetic uses EXACT chromatic labels (not the neural engine) to first prove the
SIGNAL exists and is separable; the neural-engine path is exercised on real data."""
from __future__ import annotations
import argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hetero.synth_chromatic import make_synth_chromatic
from hetero.chromatic_labels import compute_chromatic_0dim
from node_ph_features import PI_RES
from sg2dgm import PersistenceImager as pimg_mod

_LEGS = ('image', 'kernel', 'cokernel')


def _leg_pi(lab, leg, imager):
    pts = lab[leg]
    return (np.asarray(imager.transform(pts)).reshape(-1) if pts.shape[0]
            else np.zeros(PI_RES * PI_RES))


def run_synthetic(n_per_class=200, seed=0):
    """Logistic-regression AUC of chromatic (3-leg) vs achromatic (ordinary) vs
    kernel-only on the synthetic positive control. Expect chromatic/kernel ~1.0,
    achromatic ~0.5 (same-type topology identical across classes)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    imager = pimg_mod.PersistenceImager(resolution=PI_RES)
    X = {'chromatic': [], 'kernel_only': [], 'achromatic': []}
    yv = []
    rng = np.random.RandomState(seed)
    for _ in range(n_per_class):
        for lab in (0, 1):
            gd = make_synth_chromatic(label=lab, seed=int(rng.randint(1 << 30)))
            # fv ~ U[0,1); global cap max_f=1.0 so essential classes born at the data
            # max are still visible (per-ego max would drop top-born kernels).
            L = compute_chromatic_0dim(gd['fv'], gd['ei'], gd['is_L'], max_f=1.0)
            X['chromatic'].append(np.concatenate([_leg_pi(L, lg, imager) for lg in _LEGS]))
            X['kernel_only'].append(_leg_pi(L, 'kernel', imager))
            X['achromatic'].append(_leg_pi(L, 'ordinary', imager))
            yv.append(lab)
    yv = np.array(yv)
    print(f"=== synthetic positive control (n={2 * n_per_class}) ===")
    aucs = {}
    for name, Xl in X.items():
        Xl = np.array(Xl)
        if np.allclose(Xl.std(0).sum(), 0):
            aucs[name] = 0.5
            print(f"  {name:12s} AUC = 0.500  (degenerate/constant feature)")
            continue
        sc = cross_val_score(LogisticRegression(max_iter=2000), Xl, yv, cv=5, scoring='roc_auc')
        aucs[name] = float(sc.mean())
        print(f"  {name:12s} AUC = {sc.mean():.3f} +- {sc.std():.3f}")
    ok = aucs['kernel_only'] > aucs['achromatic'] + 0.15
    print(f"GATE chromatic/kernel >> achromatic: {'PASS' if ok else 'FAIL'} "
          f"(kernel {aucs['kernel_only']:.3f} vs achromatic {aucs['achromatic']:.3f})")
    return aucs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic')
    ap.add_argument('--n_per_class', type=int, default=200)
    a = ap.parse_args()
    if a.mode == 'synthetic':
        run_synthetic(n_per_class=a.n_per_class)
    else:
        raise SystemExit("real-data path: Task 11 (run_real)")
