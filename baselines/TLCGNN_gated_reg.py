# baselines/TLCGNN_gated_reg.py
"""Gating variant with an L1 sparsity penalty pushing gates toward 0. Goal: break
the gate=1 saturation seen in the plain gating model so gates can learn to
suppress PI on heterophilic graphs.

The penalty (mean gate over the last decode call) is stored in
.last_gate_penalty; the training loop adds GATE_REG_LAMBDA * penalty to the loss.
"""
from __future__ import annotations
import os
import numpy as np
import torch
from baselines.TLCGNN_gated import gated_decode, Net as GatedNet

# Sparsity strength; overridable via env for the λ-sweep experiment.
GATE_REG_LAMBDA = float(os.environ.get('TLCGNN_GATE_LAMBDA', '0.1'))


class Net(GatedNet):
    """Same as TLCGNN_gated.Net but records mean-gate penalty for the training loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_gate_penalty = torch.tensor(0.0)

    def decode(self, data, emb, type='train'):
        if type == 'train':
            edges_pos = data.total_edges[:data.train_pos]
            index = np.random.randint(0, data.train_neg, data.train_pos)
            edges_neg = data.total_edges[data.train_pos:data.train_pos + data.train_neg][index]
            total_edges = np.concatenate((edges_pos, edges_neg))
            edges_y = torch.cat((data.total_edges_y[:data.train_pos],
                                  data.total_edges_y[data.train_pos:data.train_pos + data.train_neg][index]))
            PI = np.concatenate(
                (self.PI[:data.train_pos], self.PI[data.train_pos:data.train_pos + data.train_neg][index]))
        elif type == 'val':
            total_edges = data.total_edges[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
            edges_y = data.total_edges_y[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
            PI = self.PI[data.train_pos+data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
        else:
            total_edges = data.total_edges[data.train_pos+data.train_neg+data.val_pos+data.val_neg:]
            edges_y = data.total_edges_y[data.train_pos+data.train_neg+data.val_pos+data.val_neg:]
            PI = self.PI[data.train_pos+data.train_neg+data.val_pos+data.val_neg:]

        emb = emb.renorm(2, 0, 1)
        new_x = torch.tensor(PI.reshape((len(total_edges), -1)), dtype=torch.float, device=emb.device)
        emb_in = emb[total_edges[:, 0]]
        emb_out = emb[total_edges[:, 1]]
        sqdist = (emb_in - emb_out).pow(2)
        edge_feats = self._edge_features_for_gate(total_edges, emb_in, emb_out)
        gates = self.gate_net(edge_feats)
        self.last_gate_penalty = gates.mean()
        feats = gated_decode(sqdist, new_x, gates)
        feats = self.leakyrelu(self.linear_1(feats))
        feats = torch.abs(self.linear(feats)).reshape(-1)
        feats = torch.clamp(feats, min=0, max=40)
        prob = 1. / (torch.exp((feats - 2.0) / 1.0) + 1.0)
        return prob, edges_y.float()


def call(data, name, num_features, num_classes, data_cnt, use_pi: bool = True):
    from baselines.TLCGNN_gated import call as gated_call
    model, data = gated_call(data, name, num_features, num_classes, data_cnt, use_pi=use_pi)
    reg_model = Net(data, num_features, num_classes, PI=model.PI,
                    clustering=model.clustering).to(data.x.device)
    return reg_model, data
