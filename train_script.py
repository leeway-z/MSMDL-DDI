import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
from datetime import datetime
import time
import argparse

import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
import csv
import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataLoader, TOTAL_ATOM_FEATS

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=TOTAL_ATOM_FEATS, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=256, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=86, help='num of interaction types')#86,963,1317
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')#0.0005
parser.add_argument('--n_epochs', type=int, default=100, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=256, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--num_layers', type=int, default=3, help='num of layers')

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])

args = parser.parse_args()
n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
rel_total = args.rel_total
lr = args.lr
n_epochs = args.n_epochs
kge_dim = args.kge_dim
batch_size = args.batch_size
num_layers = args.num_layers

weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
torch.cuda.set_device(1)
device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'

print(args)
############################################################

###### Dataset
# df_ddi_train = pd.read_csv('data/ddi_training.csv')
# df_ddi_val = pd.read_csv('data/ddi_validation.csv')
# df_ddi_test = pd.read_csv('data/ddi_test.csv')

df_ddi_train = pd.read_csv('data/ddi_training.csv')
df_ddi_val = pd.read_csv('data/ddi_validation.csv')
df_ddi_test = pd.read_csv('data/ddi_test.csv')

# df_ddi_train = pd.read_csv('data/twosides_training.csv')
# df_ddi_val = pd.read_csv('data/twosides_validation.csv')
# df_ddi_test = pd.read_csv('data/twosides_test.csv')
# df_test = df_ddi_test.groupby('type')
# for group_name, group_data in df_test:
#     print(f"Processing data for type: {group_name}")
#     print(group_data)

train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
val_tup = [(h, t, r) for h, t, r in zip(df_ddi_val['d1'], df_ddi_val['d2'], df_ddi_val['type'])]
test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False,data_save=True)
test_data = DrugDataset(test_tup, disjoint_split=False)

print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size * 3)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size * 3)


def do_compute(batch, device, training=True):
    '''
        *batch: (pos_tri, neg_tri)
        *pos/neg_tri: (batch_h, batch_t, batch_r)
    '''
    probas_pred, ground_truth = [], []
    pos_tri, neg_tri = batch

    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    p_score = model(pos_tri)
    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)

    return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(np.int_)

    acc = metrics.accuracy_score(target, pred)
    auc_roc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)

    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    auc_prc = metrics.auc(r, p)
    ap = metrics.average_precision_score(target, probas_pred)
    return acc, auc_roc, auc_prc, f1_score, ap

csv_file_name = 'data/positive_samples5.csv'
def train(model, train_data_loader, val_data_loader, loss_fn, optimizer, n_epochs, device, scheduler=None):
    print('Starting training at', datetime.today())
    for i in range(1, n_epochs + 1):
        start = time.time()
        train_loss = 0
        train_loss_pos = 0
        train_loss_neg = 0
        val_loss = 0
        val_loss_pos = 0
        val_loss_neg = 0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []

        for batch in train_data_loader:
            model.train()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(p_score)
        train_loss /= len(train_data)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_auc_roc, train_auc_prc, train_f1_score, train_ap = do_compute_metrics(train_probas_pred,
                                                                                                   train_ground_truth)
            val_probas_pred_truth = []
           #case study
            for batch in val_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device)
                val_probas_pred.append(probas_pred)
                val_ground_truth.append(ground_truth)
                loss, loss_p, loss_n = loss_fn(p_score, n_score)
                val_loss += loss.item() * len(p_score)


                pos_probas_pred = probas_pred[:len(p_score)]
                val_probas_pred_truth.append(pos_probas_pred)




            val_loss /= len(val_data)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_probas_pred_truth = np.concatenate(val_probas_pred_truth)
            positive_samples_data = []
            val_ground_truth = np.concatenate(val_ground_truth)
            val_acc, val_auc_roc, val_auc_prc, val_f1_score, val_ap = do_compute_metrics(val_probas_pred,
                                                                                         val_ground_truth)

        if scheduler:
            # print('scheduling')
            scheduler.step()
        # print(f'Epoch: {i} ({time.time() - start:.4f}s),val_loss: {val_loss:.4f},')
        print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f},'
        f' train_acc: {train_acc:.4f}, val_acc:{val_acc:.4f}')
        print(f'\t\ttrain_roc: {train_auc_roc:.4f}, val_roc: {val_auc_roc:.4f}, train_auprc: {train_auc_prc:.4f}, val_auprc: {val_auc_prc:.4f}, train_f1: {train_f1_score:.4f}, val_f1:　{val_f1_score:.4f},train_ap:{train_ap:.4f},val_ap:{val_ap:.4f}')
        
    print('End at', datetime.today())



model = models.MSMDL_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, num_layers, weight_conv='WeightConv1',
                       multi_channel='False')
loss = custom_loss.SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
# print(model.SMG)
model.to(device=device)

# if __name__ == '__main__':
train(model, train_data_loader, val_data_loader, loss, optimizer, n_epochs, device, scheduler)

