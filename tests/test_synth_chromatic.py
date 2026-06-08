"""The synthetic positive control's two classes must differ in chromatic kernel."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hetero.synth_chromatic import make_synth_chromatic
from hetero.chromatic_labels import compute_chromatic_0dim


def _lab(gd):  # global cap max_f=1.0 (fv ~ U[0,1)) so top-born essentials stay visible
    return compute_chromatic_0dim(gd['fv'], gd['ei'], gd['is_L'], max_f=1.0)


def test_kernel_count_differs_by_class():
    # label 0 -> 2 clusters bridged -> 1 essential kernel ; label 1 -> 3 -> 2 essential
    for seed in range(5):
        k0 = _lab(make_synth_chromatic(0, seed))['kernel']
        k1 = _lab(make_synth_chromatic(1, seed))['kernel']
        n0 = int((np.abs(k0[:, 1] - 1.0) < 1e-9).sum()) if k0.shape[0] else 0
        n1 = int((np.abs(k1[:, 1] - 1.0) < 1e-9).sum()) if k1.shape[0] else 0
        assert (n0, n1) == (1, 2), (seed, n0, n1)


def test_ordinary_blind_to_class():
    # full graph K is connected in both classes -> exactly one essential ordinary bar.
    for lab in (0, 1):
        gd = make_synth_chromatic(lab, seed=2)
        o = _lab(gd)['ordinary']
        ess = int((np.abs(o[:, 1] - 1.0) < 1e-9).sum()) if o.shape[0] else 0
        assert ess == 1, (lab, ess)
