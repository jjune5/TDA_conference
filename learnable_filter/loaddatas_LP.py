"""Shim that mirrors the missing `learnable_filter.loaddatas_LP` module.

The original PDGNN code references `lds.loaddatas(d_loader, d_name)` from this
package. We delegate to the top-level `loaddatas.py` (which accepts a single
`d_name` argument) and ignore `d_loader` (the loader name is implied by d_name).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loaddatas as _ld


def loaddatas(d_loader=None, d_name=None):
    # Backwards-compatible call:  loaddatas('Planetoid', 'Cora') or loaddatas('Cora')
    if d_name is None:
        d_name = d_loader
    return _ld.loaddatas(d_name)


def get_edges_split(*args, **kwargs):
    return _ld.get_edges_split(*args, **kwargs)
