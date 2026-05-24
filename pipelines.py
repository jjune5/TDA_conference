import os
import sys
import copy
import argparse
import torch
import loaddatas as lds
import torch.nn.functional as F
import numpy as np
from baselines import TLCGNN as TLCGNN
from baselines import TLCGNN_gated as TLCGNN_gated
from sklearn.metrics import roc_auc_score,average_precision_score
from torch.nn.init import xavier_normal_ as xavier

def train():
    model.train()
    optimizer.zero_grad()
    emb = model.encode(data)
    x, y = model.decode(data, emb)
    loss = F.binary_cross_entropy(x,y)
    loss.backward()
    optimizer.step()
    return x

def test():
    model.eval()
    accs = []
    emb = model.encode(data)
    for type in ["val", "test"]:
        pred,y = model.decode(data,emb,type=type)
        pred,y = pred.cpu(),y.cpu()
        if type == "val":
            accs.append(F.binary_cross_entropy(pred, y))
            pred = pred.data.numpy()
            roc = roc_auc_score(y, pred)
            accs.append(roc)
            acc = average_precision_score(y,pred)
            accs.append(acc)
        else:
            pred = pred.data.numpy()
            roc = roc_auc_score(y, pred)
            accs.append(roc)
            acc = average_precision_score(y, pred)
            accs.append(acc)
    return accs

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        xavier(m.weight)
        if not m.bias is None:
            torch.nn.init.constant_(m.bias, 0)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


_parser = argparse.ArgumentParser()
_parser.add_argument('--datasets', nargs='+', default=["Photo", "PubMed", "Computers"],
                     help='datasets to run, choose from Photo, PubMed, Computers, Cora, Citeseer, PPI')
_parser.add_argument('--trials', type=int, default=50, help='number of trials per dataset')
_parser.add_argument('--dropout', type=float, default=0.5, help='dropout for the model (0.8 for Cora/Citeseer)')
_parser.add_argument('--tag', type=str, default='', help='suffix tag for score filenames')
_parser.add_argument('--no_pi', action='store_true', help='ablation: disable persistence-image features (zeros instead)')
_parser.add_argument('--pi_source', choices=['dionysus', 'pdgnn'], default='dionysus',
                     help='source of PI cache (dionysus=TLC-GNN exact, pdgnn=neural approx)')
_parser.add_argument('--use_gating', action='store_true',
                     help='use adaptive PI gating (baselines.TLCGNN_gated.Net)')
_args = _parser.parse_args()
os.environ['TLCGNN_PI_SOURCE'] = _args.pi_source

d_names = _args.datasets
if 'PPI' in d_names and len(d_names) == 1:
    times = range(min(_args.trials, 20))
else:
    times = range(_args.trials)


wait_total= 200
total_epochs = 2000


if _args.use_gating:
    pipelines=['TLCGNN_gated']
else:
    pipelines=['TLCGNN']
_pkey = pipelines[0]
pipeline_acc={_pkey:[i for i in times]}
pipeline_acc_sum={_pkey:0}
pipeline_roc={_pkey:[i for i in times]}
pipeline_roc_sum={_pkey:0}
pipeline_acc_same={_pkey:[i for i in times]}
pipeline_acc_same_sum={_pkey:0}
pipeline_roc_same={_pkey:[i for i in times]}
pipeline_roc_same_sum={_pkey:0}
pipeline_acc_diff={_pkey:[i for i in times]}
pipeline_acc_diff_sum={_pkey:0}
pipeline_roc_diff={_pkey:[i for i in times]}
pipeline_roc_diff_sum={_pkey:0}

os.makedirs("./scores", exist_ok=True)

for d_name in d_names:
    f2 = open('scores/pipe_benchmark_' + d_name + '_LP_scores' + _args.tag + '.txt', 'w+')
    f2.write('{0:7} {1:7}\n'.format(d_name, _pkey))
    f2.flush()
    dataset = lds.loaddatas(d_name)
    for data_cnt in times:
        for Conv_method in pipelines:
            if d_name in ['Rand_nnodes_github1000', 'PPI']:
                data = copy.deepcopy(dataset[data_cnt])
            else:
                data = copy.deepcopy(dataset[0])
            if d_name in ['Rand_nnodes_github1000']:
                data.x = data.x[:, :10]
            #data.x = torch.ones(data.x.size())
            index = [i for i in range(len(data.y))]
            if d_name != "PPI":
                model, data = locals()[Conv_method].call(data, dataset.name, data.x.size(1), dataset.num_classes,
                                                     data_cnt, use_pi=not _args.no_pi)
            else:
                model, data = locals()[Conv_method].call(data, 'PPI', data.x.size(1), dataset.num_classes,
                                                         data_cnt, use_pi=not _args.no_pi)
            model.apply(weights_init)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0)
            best_val_acc = test_acc_same = test_acc_diff = test_acc = 0.0
            best_val_roc = test_roc_same = test_roc_diff = test_roc = 0.0
            best_val_loss = np.inf
            # train and val/test
            wait_step = 0

            # train and test
            for epoch in range(1, total_epochs + 1):
                pred = train()
                val_loss, val_roc, val_acc, tmp_test_roc, tmp_test_acc = test()
                if val_roc >= best_val_roc:
                    test_acc = tmp_test_acc
                    test_roc = tmp_test_roc
                    best_val_acc = val_acc
                    best_val_roc = val_roc
                    best_val_loss = val_loss
                    wait_step = 0
                else:
                    wait_step += 1
                    if wait_step == wait_total:
                        print('Early stop! Min loss: ', best_val_loss, ', Max accuracy: ', best_val_acc,
                              ', Max roc: ', best_val_roc)
                        break
            del model
            del data
            # print result

            pipeline_acc[Conv_method][data_cnt] = test_acc
            pipeline_roc[Conv_method][data_cnt] = test_roc

            log = 'Epoch: ' + str(
                total_epochs) + ', dataset name: ' + d_name + ', Method: ' + Conv_method + ' Test pr: {:.4f}, roc: {:.4f} \n'
            print((log.format(pipeline_acc[Conv_method][data_cnt], pipeline_roc[Conv_method][data_cnt])))
            #print(pred)

            f2.write('{}, {:.4f}, {:.4f}\n'.format(data_cnt, pipeline_acc[Conv_method][data_cnt],
                                                     pipeline_roc[Conv_method][data_cnt],))
            f2.flush()
    f2.write('{0:4} {1:4f}\n'.format('std', np.std(pipeline_acc[Conv_method])))
    f2.write('{0:4} {1:4f}\n'.format('mean', np.mean(pipeline_acc[Conv_method])))
    f2.write('{0:4} {1:4f}\n'.format('std', np.std(pipeline_roc[Conv_method])))
    f2.write('{0:4} {1:4f}\n'.format('mean', np.mean(pipeline_roc[Conv_method])))
    f2.close()
