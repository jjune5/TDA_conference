"""§14 diagnostic for Variant B (multi-scale HKS-filtration vicinity PI).

Question: does B's per-edge PI COLLAPSE at test (test_pos ~= test_neg ~= 0),
inheriting the membership artifact like exact vicinity-PI, or does it carry
genuine test signal?  Prediction: B collapses (it is vicinity-based; the
candidate edge is removed from the training graph at test).

For each dataset we segment the cache rows into
[train_pos | train_neg | val_pos | val_neg | test_pos | test_neg]
(the layout written by loaddatas.compute_persistence_image, recovered via the
canonical 0.05/0.1 seed-1234 split) and report, per source (exact-PI and B):
  - mean per-edge L1 mass + nonzero %  for train_pos / test_pos / test_neg
  - artifact verdict: test_pos collapsed vs train_pos
                      (nz<5%  OR  L1<0.1% of train_pos  OR  ratio<20%)
  - DISCRIMINATION: a feature-ONLY logistic-regression 5-fold CV AUC on the
    test edges (test_pos vs test_neg) — the strong "is there genuine test
    signal in the feature alone" test.  ~0.5 == collapse/no signal.

Outputs results/diffusion_feat_B/diag_artifact.{csv,txt} (+ JSON raw).
"""
from __future__ import annotations
import os
import sys
import json
import argparse

_REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(_REPO)
sys.path.insert(0, _REPO)

import numpy as np
import loaddatas as lds

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score


def segment_bounds(name):
    """Canonical [tp|tn|vp|vn|tep|ten] bounds via the 0.05/0.1 seed-1234 split."""
    ds = lds.loaddatas(name)
    data = ds[0]
    tr, trf, va, vaf, te, tef = lds.get_edges_split(
        data, val_prop=0.05, test_prop=0.1)
    counts = [len(tr), len(trf), len(va), len(vaf), len(te), len(tef)]
    names = ['train_pos', 'train_neg', 'val_pos', 'val_neg', 'test_pos', 'test_neg']
    bounds, c = {}, 0
    for n, k in zip(names, counts):
        bounds[n] = (c, c + k)
        c += k
    return bounds, c


def seg_stats(pi, lo, hi):
    rows = np.abs(pi[lo:hi]).sum(1)
    return float(rows.mean()), float((rows > 1e-6).mean()), hi - lo


def feature_cv_auc(feat_pos, feat_neg, seed=0):
    """5-fold stratified CV AUC of a logistic regression that sees ONLY the
    per-edge feature, separating test_pos from test_neg. ~0.5 == no signal."""
    X = np.vstack([feat_pos, feat_neg]).astype(np.float64)
    y = np.concatenate([np.ones(len(feat_pos)), np.zeros(len(feat_neg))])
    # If the feature is all-zero (full collapse) there is nothing to fit.
    if not np.any(np.abs(X) > 1e-12):
        return 0.5, 'all-zero'
    n_min = int(min(y.sum(), len(y) - y.sum()))
    if n_min < 5:
        return float('nan'), 'too-few'
    k = min(5, n_min)
    clf = make_pipeline(StandardScaler(with_mean=True),
                        LogisticRegression(max_iter=2000, C=1.0))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    try:
        scores = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc')
        return float(np.mean(scores)), 'ok'
    except Exception as e:
        return float('nan'), f'err:{type(e).__name__}'


def find_cache(subdir_candidates, name):
    for sub in subdir_candidates:
        for cand in (f'{sub}/{name}.npy', f'{sub}/{name.lower()}.npy'):
            if os.path.exists(cand):
                return cand
    return None


def analyze(name, b_dir):
    bounds, total = segment_bounds(name)
    rec = {'name': name, 'expected_rows': total, 'segments': bounds, 'sources': {}}
    sources = {
        'exact': ['data/TLCGNN'],
        'B_hks_multi': [b_dir],
    }
    for tag, subs in sources.items():
        cache = find_cache(subs, name)
        if cache is None:
            rec['sources'][tag] = {'status': 'cache missing', 'searched': subs}
            continue
        pi = np.load(cache)
        s = {'cache': cache, 'rows': int(pi.shape[0]), 'dim': int(pi.shape[1])}
        if pi.shape[0] != total:
            s['status'] = f'ROW MISMATCH got {pi.shape[0]} want {total}'
            rec['sources'][tag] = s
            continue
        seg = {}
        for name_s, (lo, hi) in bounds.items():
            m, nz, n = seg_stats(pi, lo, hi)
            seg[name_s] = {'mean_L1': m, 'nonzero': nz, 'n': n}
        s['per_segment'] = seg
        # artifact verdict (test_pos collapse vs train_pos)
        trp = seg['train_pos']; tep = seg['test_pos']; ten = seg['test_neg']
        collapsed = (tep['nonzero'] < 0.05
                     or tep['mean_L1'] < 0.001 * max(trp['mean_L1'], 1e-12)
                     or (trp['mean_L1'] > 0 and tep['mean_L1'] / trp['mean_L1'] < 0.20))
        s['test_collapsed_vs_train'] = bool(collapsed)
        # discrimination: feature-only CV AUC on test edges
        tp_lo, tp_hi = bounds['test_pos']; tn_lo, tn_hi = bounds['test_neg']
        auc, status = feature_cv_auc(pi[tp_lo:tp_hi], pi[tn_lo:tn_hi])
        s['test_feature_cv_auc'] = auc
        s['test_feature_cv_status'] = status
        # also train-edge feature CV AUC (sanity: feature DOES separate in-graph).
        # Cap both classes to <=5000 rows so the CV stays fast even under heavy
        # CPU load (this is only a sanity number, not the headline metric).
        trp_lo, trp_hi = bounds['train_pos']; trn_lo, trn_hi = bounds['train_neg']
        rng = np.random.RandomState(0)
        pos = pi[trp_lo:trp_hi]
        neg = pi[trn_lo:trn_hi]
        CAP = 5000
        if len(pos) > CAP:
            pos = pos[rng.choice(len(pos), CAP, replace=False)]
        if len(neg) > CAP:
            neg = neg[rng.choice(len(neg), CAP, replace=False)]
        tr_auc, tr_status = feature_cv_auc(pos, neg)
        s['train_feature_cv_auc'] = tr_auc
        s['train_feature_cv_status'] = tr_status
        rec['sources'][tag] = s
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['Cora', 'Chameleon'])
    ap.add_argument('--b_dir', default='data/HKS_MULTI_TLCGNN_0.1x1x10')
    ap.add_argument('--out_dir', default='results/diffusion_feat_B')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = [analyze(n, args.b_dir) for n in args.datasets]

    # JSON raw
    with open(os.path.join(args.out_dir, 'diag_artifact_raw.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # CSV + TXT
    csv_lines = ['dataset,source,dim,trp_L1,trp_nz,tep_L1,tep_nz,ten_L1,ten_nz,'
                 'test_collapsed,test_feat_cv_auc,train_feat_cv_auc']
    txt = []
    txt.append('=' * 100)
    txt.append('§14 DIAGNOSTIC — Variant B (multi-scale HKS-filtration vicinity PI)')
    txt.append('Does B collapse at test (inherits membership artifact) like exact vicinity-PI?')
    txt.append('=' * 100)
    txt.append('')
    hdr = (f"{'dataset':10} {'source':12} {'dim':>4} | "
           f"{'trp L1/nz':>16} | {'tep L1/nz':>16} {'ten L1/nz':>16} | "
           f"{'collapse':>8} {'test_cvAUC':>10} {'train_cvAUC':>11}")
    txt.append(hdr)
    txt.append('-' * len(hdr))
    for rec in results:
        for tag in ('exact', 'B_hks_multi'):
            s = rec['sources'].get(tag, {})
            if 'per_segment' not in s:
                txt.append(f"{rec['name']:10} {tag:12} {'--':>4} | {str(s.get('status', s.get('status', 'missing'))):>16}")
                csv_lines.append(f"{rec['name']},{tag},NA,,,,,,,{s.get('status','missing')},,")
                continue
            ps = s['per_segment']
            def f(k):
                d = ps[k]
                return f"{d['mean_L1']:.4f}/{100*d['nonzero']:.0f}%"
            col = 'YES' if s['test_collapsed_vs_train'] else 'no'
            tcv = s['test_feature_cv_auc']
            trcv = s['train_feature_cv_auc']
            txt.append(f"{rec['name']:10} {tag:12} {s['dim']:>4} | "
                       f"{f('train_pos'):>16} | {f('test_pos'):>16} {f('test_neg'):>16} | "
                       f"{col:>8} {tcv:>10.3f} {trcv:>11.3f}")
            csv_lines.append(
                f"{rec['name']},{tag},{s['dim']},"
                f"{ps['train_pos']['mean_L1']:.6f},{ps['train_pos']['nonzero']:.4f},"
                f"{ps['test_pos']['mean_L1']:.6f},{ps['test_pos']['nonzero']:.4f},"
                f"{ps['test_neg']['mean_L1']:.6f},{ps['test_neg']['nonzero']:.4f},"
                f"{int(s['test_collapsed_vs_train'])},{tcv:.6f},{trcv:.6f}")
        txt.append('')

    txt.append('LEGEND:')
    txt.append('  trp/tep/ten = train_pos / test_pos / test_neg segments')
    txt.append('  L1 = mean per-edge PI L1-mass ; nz = % edges with nonzero PI')
    txt.append('  collapse = test_pos collapsed vs train_pos (nz<5% OR L1<0.1%*trp OR ratio<20%)')
    txt.append('  test_cvAUC  = feature-ONLY 5-fold CV AUC, test_pos vs test_neg (~0.5 = no test signal)')
    txt.append('  train_cvAUC = same on train edges (sanity: vicinity feature DOES separate in-graph edges)')
    txt.append('')
    txt.append('VERDICT (per dataset):')
    for rec in results:
        s = rec['sources'].get('B_hks_multi', {})
        if 'per_segment' not in s:
            txt.append(f"  {rec['name']}: B cache unavailable ({s.get('status','missing')})")
            continue
        col = s['test_collapsed_vs_train']
        tcv = s['test_feature_cv_auc']
        e = rec['sources'].get('exact', {})
        e_tcv = e.get('test_feature_cv_auc', float('nan'))
        verdict = ('COLLAPSES at test (inherits §14 membership artifact)'
                   if col or (not np.isnan(tcv) and tcv < 0.6)
                   else 'does NOT collapse (carries test signal)')
        txt.append(f"  {rec['name']}: B {verdict} | "
                   f"B test_cvAUC={tcv:.3f}, exact test_cvAUC={e_tcv:.3f}, "
                   f"B train_cvAUC={s['train_feature_cv_auc']:.3f}")

    out_txt = '\n'.join(txt)
    print(out_txt)
    with open(os.path.join(args.out_dir, 'diag_artifact.txt'), 'w') as f:
        f.write(out_txt + '\n')
    with open(os.path.join(args.out_dir, 'diag_artifact.csv'), 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')
    print(f"\n[diag] wrote {args.out_dir}/diag_artifact.{{txt,csv,json}}")


if __name__ == '__main__':
    main()
