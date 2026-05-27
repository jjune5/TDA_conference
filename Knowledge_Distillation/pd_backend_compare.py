# Knowledge_Distillation/pd_backend_compare.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import networkx as nx
import gudhi
from sg2dgm import PersistenceImager as pimg_mod
from Knowledge_Distillation.mol_data import load_tudataset, graph_to_pi

_IMAGER = pimg_mod.PersistenceImager(resolution=5)

def _degree_filt(g):
    g = nx.convert_node_labels_to_integers(g)
    deg = np.array([d for _, d in sorted(g.degree(), key=lambda x: x[0])], dtype=float)
    return deg / (deg.max() + 1e-10), g

def gudhi_pi(g):
    filt, g = _degree_filt(g)
    st = gudhi.SimplexTree()
    for v in g.nodes():
        st.insert([int(v)], filtration=float(filt[v]))
    for u, v in g.edges():
        st.insert([int(u), int(v)], filtration=float(max(filt[u], filt[v])))
    st.compute_persistence()
    pairs = []
    for dim, (b, d) in st.persistence():
        if d != float('inf'):
            pairs.append([b, d])
    pd = np.array(pairs, dtype=float) if pairs else np.empty((0, 2))
    if pd.size == 0:
        return np.zeros(25)
    return _IMAGER.transform(pd).reshape(-1)

def main():
    graphs, _ = load_tudataset('MUTAG')
    mses, accel_nz, gudhi_nz = [], 0, 0
    for g in graphs[:30]:
        a = graph_to_pi(g)
        b = gudhi_pi(g)
        mses.append(float(((a - b) ** 2).mean()))
        accel_nz += int(a.any()); gudhi_nz += int(b.any())
    print(f'accelerated_PD vs GUDHI on 30 MUTAG graphs:')
    print(f'  mean per-graph PI MSE: {np.mean(mses):.6f}')
    print(f'  accel nonzero: {accel_nz}/30, gudhi nonzero: {gudhi_nz}/30')
    os.makedirs('scores', exist_ok=True)
    with open('scores/pd_backend_compare.txt', 'w') as f:
        f.write(f'accel_vs_gudhi_MUTAG30 mean_PI_MSE {np.mean(mses):.6f} '
                f'accel_nz {accel_nz} gudhi_nz {gudhi_nz}\n')
    print('saved scores/pd_backend_compare.txt')

if __name__ == '__main__':
    main()
