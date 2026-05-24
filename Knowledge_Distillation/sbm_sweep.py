# Knowledge_Distillation/sbm_sweep.py
"""Submit a 5x5 SBM sweep to SLURM.

Density axis:    p_in + p_out ∈ {0.05, 0.10, 0.20, 0.30, 0.50}
Heterophily ax:  p_out / (p_in + p_out) ∈ {0.10, 0.30, 0.50, 0.70, 0.90}
Fixed: N=500 nodes, K=5 blocks.

Each (density, heterophily) point gets 3 SLURM jobs:
- TLC-GNN exact PI
- PDGNN approx PI (using existing trained checkpoint)
- No PI
"""
from __future__ import annotations
import os
import subprocess

DENSITIES = [0.05, 0.10, 0.20, 0.30, 0.50]
HETEROPHILIES = [0.10, 0.30, 0.50, 0.70, 0.90]
N, K = 500, 5
TRIALS = 50

STATE_FILE = '/tmp/sbm_sweep_state.txt'  # resume across QOS-limited runs


def density_hetero_to_p(d: float, h: float) -> tuple[float, float]:
    """density=p_in+p_out, hetero=p_out/(p_in+p_out). Returns (p_in, p_out)."""
    p_out = d * h
    p_in = d - p_out
    return round(p_in, 4), round(p_out, 4)


def submit_one(d_name: str, tag: str, pi_source: str = 'dionysus',
               no_pi: bool = False):
    script = f"/tmp/sbm_{d_name}_{tag}.sh"
    extra = ""
    if no_pi:
        extra += " --no_pi"
    if pi_source == 'pdgnn':
        extra += " --pi_source pdgnn"
    with open(script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.err
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/TLC-GNN
python pipelines.py --datasets {d_name} --trials {TRIALS} --tag {tag}{extra}
""")
    os.chmod(script, 0o755)
    jid = subprocess.check_output(
        ["sbatch", "--parsable", f"--job-name=sbm-{d_name}-{tag}", script]
    ).decode().strip()
    return jid


def _load_done() -> set:
    if not os.path.exists(STATE_FILE):
        return set()
    done = set()
    with open(STATE_FILE) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                done.add((parts[0], parts[1]))
    return done


def _append_done(name: str, tag: str, jid: str):
    with open(STATE_FILE, 'a') as f:
        f.write(f"{name}\t{tag}\t{jid}\n")


if __name__ == '__main__':
    done = _load_done()
    submitted_now = []
    skipped = 0
    halted = False
    for d in DENSITIES:
        if halted:
            break
        for h in HETEROPHILIES:
            if halted:
                break
            p_in, p_out = density_hetero_to_p(d, h)
            name = f"SBM_{N}_{K}_{p_in}_{p_out}"
            # 3 jobs per cell
            for tag, kw in [('sbmTLCGNN', {}), ('sbmPDGNN', {'pi_source': 'pdgnn'}),
                             ('sbmNoPI', {'no_pi': True})]:
                if (name, tag) in done:
                    skipped += 1
                    continue
                try:
                    jid = submit_one(name, tag, **kw)
                except subprocess.CalledProcessError as e:
                    print(f"sbatch failed for {name} {tag}: {e}. Stopping (likely QOS limit). Re-run later to resume.")
                    halted = True
                    break
                submitted_now.append((name, tag, jid))
                _append_done(name, tag, jid)
                print(f"{name} {tag} → {jid}")
    print(f"\nSubmitted this run: {len(submitted_now)} | skipped (already done): {skipped} | total done: {len(done) + len(submitted_now)} / 75")
