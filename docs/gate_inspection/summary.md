# Adaptive Gating — Gate value inspection

200 epochs single-trial training, gate values measured on val edges. Seed=0, lr=0.005, wd=0.

| Dataset | Domain | mean gate | std | min | max | Interpretation |
|---|---|---|---|---|---|---|
| Photo | Homo Amazon | 1.000 | 0.000 | 0.997 | 1.000 | PI on (gate > 0.5) — saturated |
| Chameleon | Hetero wiki | 1.000 | 0.000 | 0.998 | 1.000 | PI on (gate > 0.5) — saturated |
| Texas | Hetero web | 1.000 | 0.001 | 0.997 | 1.000 | PI on (gate > 0.5) — saturated |
| ChChMiner | Drug DDI | 1.000 | 0.010 | 0.596 | 1.000 | PI on (gate > 0.5) — saturated |

## Key observation: no discrimination

All four datasets converge to a gate near 1.0 on val edges (every mean ≥ 0.9996, every std ≤ 0.01).
The gate does **not** reflect domain heterophily as hypothesized — heterophilic datasets
(Chameleon/Texas) and the drug-DDI dataset (ChChMiner) do **not** push the gate toward 0 to
suppress PI; instead all datasets push the gate to saturation, keeping PI on. Only ChChMiner shows
any spread at all (min 0.596), but its mean is still ~1.0.

Possible explanations to follow up on:
- BCE loss on edge probabilities rewards using the extra PI signal regardless of homo/hetero,
  so the gate has no incentive to gate PI off.
- Gating net input features (`_edge_features_for_gate`, in_dim=3) may not separate
  homo/hetero edges well at this scale.
- 200 epochs at lr=0.005 may simply drive the sigmoid into saturation early; behaviour at the
  paper's 2000-epoch / 50-trial setting could differ but the trend (mean → 1.0) is consistent.

Practical implication for the 50-trial sweep (Task 2.3): expect gated-PI scores to closely track
the always-on PI baseline rather than to adapt per-domain.
