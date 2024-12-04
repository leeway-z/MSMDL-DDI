import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch.nn import Linear, Sequential, ReLU, Sigmoid
from torch_geometric.nn import (global_add_pool, LayerNorm, JumpingKnowledge, SAGPooling,)
from conv.sparse_conv import SparseConv
from conv.weight_conv import WeightConv1, WeightConv2
from torch_geometric.nn import GATConv
from layers import (
    IntraGraphAttention,
    InterGraphAttention,
    CoAttentionLayer,
    RESCAL
)


class MSMDL_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, num_layers, weight_conv, multi_channel):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.num_layers = num_layers
        self.weight_conv = weight_conv
        self.multi_channel = multi_channel
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()

        self.initial_norm = LayerNorm(self.in_features)
        self.SMG = SMG(self.in_features, self.num_layers, self.hidd_dim, self.weight_conv, self.multi_channel)

        self.blocks = []
        self.net_norms = ModuleList()

        for i in range(num_layers):
            block = SMG(in_features, num_layers, hidd_dim, weight_conv, multi_channel)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(hidd_dim))
            in_features = hidd_dim

        self.Drug_x_max_pool = nn.MaxPool1d(self.num_layers)
        self.Drug_y_max_pool = nn.MaxPool1d(self.num_layers)
        self.attention_layer = nn.Linear(self.num_layers, self.num_layers)
        self.drug_x_attention_layer = nn.Linear(self.num_layers, self.num_layers)
        self.drug_y_attention_layer = nn.Linear(self.num_layers, self.num_layers)

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, triples):
        h_data, t_data, rels, b_graph = triples

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        repr_h = []
        repr_t = []
        for i, block in enumerate(self.blocks):
            out = block(h_data, t_data, b_graph)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]  # 1024,128

            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))
            # torch.cuda.empty_cache()

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        kge_heads = repr_h
        kge_tails = repr_t

        attentions = self.co_attention(kge_heads, kge_tails)

        # attentions = None
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)

        return scores


class SMG(torch.nn.Module):
    def __init__(self,
                 in_features,
                 num_layers,
                 hidden,
                 weight_conv='WeightConv1',
                 multi_channel='False'):
        super(SMG, self).__init__()
        self.lin0 = Linear(in_features, hidden)
        self.convs = torch.nn.ModuleList()
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, hidden)
        self.readout = SAGPooling(hidden, min_score=-1)
        self.intraAtt = IntraGraphAttention(hidden)
        self.interAtt = InterGraphAttention(hidden)
        for i in range(num_layers):
            self.convs.append(SparseConv(hidden, hidden))

        self.masks = torch.nn.ModuleList()
        if multi_channel == 'True':
            out_channel = hidden
        else:
            out_channel = 1
        if weight_conv != 'WeightConv2':
            for i in range(num_layers):
                self.masks.append(WeightConv1(hidden, hidden, out_channel))
        else:
            for i in range(num_layers):
                self.masks.append(WeightConv2(Sequential(
                    Linear(hidden * 2, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, out_channel),
                    Sigmoid()
                )))

    def reset_parameters(self):
        self.lin0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for mask in self.masks:
            mask.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, h_data, t_data, b_graph):

        h_data.x = self.lin0(h_data.x)
        t_data.x = self.lin0(t_data.x)
        mask_val_h = None
        mask_val_t = None
        for i, conv in enumerate(self.convs):
            mask = self.masks[i]
            mask_val_h = mask(h_data.x, h_data.edge_index, mask_val_h)
            h_data.x = F.relu(conv(h_data.x, h_data.edge_index, mask_val_h))
            mask_val_t = mask(t_data.x, t_data.edge_index, mask_val_t)
            t_data.x = F.relu(conv(t_data.x, t_data.edge_index, mask_val_t))
        # for i, conv in enumerate(self.convs):
        #     h_data.x = F.relu(conv(h_data.x, h_data.edge_index))
        #     t_data.x = F.relu(conv(t_data.x, t_data.edge_index))
        #intra+inter
        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)

        h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)

        h_rep = torch.cat([h_intraRep, h_interRep], 1)
        t_rep = torch.cat([t_intraRep, t_interRep], 1)


        # h_rep = h_intraRep
        # t_rep = t_intraRep
        h_data.x = h_rep
        t_data.x = t_rep

        # readout
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores = self.readout(h_data.x, h_data.edge_index, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores = self.readout(t_data.x, t_data.edge_index, batch=t_data.batch)

        h_emb = global_add_pool(h_att_x, h_att_batch)
        t_emb = global_add_pool(t_att_x, t_att_batch)

        return h_data, t_data, h_emb, t_emb

    def __repr__(self):
        return self.__class__.__name__


class GAT(nn.Module):
    def __init__(self,
                 in_features,
                 num_layers,
                 hidden,
                 weight_conv='WeightConv1',
                 multi_channel='False'):
        super().__init__()
        self.lin0 = Linear(in_features, hidden)
        self.convs = GATConv(hidden, hidden)
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, hidden)
        self.readout = SAGPooling(hidden, min_score=-1)
        self.intraAtt = IntraGraphAttention(hidden)
        self.interAtt = InterGraphAttention(hidden)
        # for i in range(num_layers):
        #     self.convs.append(GATConv(hidden, hidden))

    def reset_parameters(self):
        self.lin0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, h_data, t_data, b_graph):

        h_data.x = self.lin0(h_data.x)
        t_data.x = self.lin0(t_data.x)


        h_data.x = F.relu(self.convs(h_data.x, h_data.edge_index))
        t_data.x = F.relu(self.convs(t_data.x, t_data.edge_index))
        # intra+inter
        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)

        h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)

        h_rep = torch.cat([h_intraRep, h_interRep], 1)
        t_rep = torch.cat([t_intraRep, t_interRep], 1)

        # h_rep = h_intraRep
        # t_rep = t_intraRep
        h_data.x = h_rep
        t_data.x = t_rep

        # readout
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores = self.readout(h_data.x,
                                                                                                   h_data.edge_index,
                                                                                                   batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores = self.readout(t_data.x,
                                                                                                   t_data.edge_index,
                                                                                                   batch=t_data.batch)

        h_emb = global_add_pool(h_att_x, h_att_batch)
        t_emb = global_add_pool(t_att_x, t_att_batch)

        return h_data, t_data, h_emb, t_emb

    def __repr__(self):
        return self.__class__.__name__
