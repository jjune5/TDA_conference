# Knowledge_Distillation/pi_shuffle_exp.py
"""Create a row-shuffled copy of a PI cache (destroys edge<->PI correspondence)
for the signal-vs-regularizer control experiment."""
import os, sys, argparse
import numpy as np

def make_shuffled(name, src_dir='data/TLCGNN', dst_dir='data/SHUFFLED', seed=1234):
    os.makedirs(dst_dir, exist_ok=True)
    src = f'{src_dir}/{name}.npy'
    pi = np.load(src)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(pi))
    np.save(f'{dst_dir}/{name}.npy', pi[perm])
    print(f'shuffled {src} ({pi.shape}) -> {dst_dir}/{name}.npy')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    args = p.parse_args()
    make_shuffled(args.name)
