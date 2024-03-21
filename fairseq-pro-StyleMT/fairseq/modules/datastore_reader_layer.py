# Coded by: Yuzhuang Xu


import torch
import torch.nn as nn

import numpy as np
import math
import random
import time

from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout


class ProjectLayer(nn.Module):

    def __init__(self, embed_dim, hidden_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(self.embed_dim, hidden_size, bias=True)
        self.nonlinearity = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, self.embed_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.nonlinearity(x)
        x = self.fc2(x)
        return x


class DatastoreReaderLayer(nn.Module):

    def __init__(self, args, embed_dim, layer_num, database_class, enable_tune=False, enable_gate=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.enable_tune = enable_tune
        self.enable_gate = enable_gate
        self.layer_num = layer_num
        self.database_class = database_class
        if self.enable_tune:
            self.proj_wq = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            self.proj_wk = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            self.proj_wv = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
            nn.init.xavier_uniform_(self.proj_wq.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.proj_wk.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.proj_wv.weight, gain=1 / math.sqrt(2))
        if self.enable_gate:
            self.proj_wg1 = nn.Linear(self.embed_dim * 2, self.embed_dim, bias=True)
            self.proj_wg2 = nn.Linear(self.embed_dim, 1, bias=True)
        path = args.load_datastore_from + "/" + database_class + "." + str(layer_num) + ".npz"
        
        self.k = torch.from_numpy(np.load(path)["keys"]).to("cuda").half()
        self.v = torch.from_numpy(np.load(path)["values"]).to("cuda").half()
        self.p_inter_dropout = getattr(args, "p_inter_dropout", 0.0)
        self.p_intra_dropout = getattr(args, "p_intra_dropout", 0.0)
        self.inter_dropout_module = FairseqDropout(1.0, module_name=self.__class__.__name__)
        self.intra_dropout_module = FairseqDropout(self.p_intra_dropout, module_name=self.__class__.__name__)

    def forward(self, q, prev_layer_output=None, inter_dropout=False, intra_dropout=False):
        """
        for monolingual datastore, prev_layer_output = q is self-attention output in decoder
        for bilingual datastore, prev_layer_output is cross_attention output, q is encoder output
        """

        if prev_layer_output is None:
            prev_layer_output = q.clone()
        prev_layer_output = prev_layer_output.transpose(0, 1)
        q = q.transpose(0, 1)

        if self.enable_tune:
            _q = self.proj_wq(q)
            _k = self.proj_wk(self.k)
            _v = self.proj_wv(self.v)
        else:
            _q = q
            _k = self.k
            _v = self.v
        _q *= self.embed_dim ** -0.5

        T = 0.5

        attn_weights = torch.matmul(_q, _k.transpose(0, 1))
        attn_weights_float = utils.softmax(attn_weights / T, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        if intra_dropout is True:
            attn_weights = self.intra_dropout_module(attn_weights)
        if inter_dropout is True:
            inter_drop_p = random.random()
            if inter_drop_p <= self.p_inter_dropout:
                attn_weights = self.inter_dropout_module(attn_weights)
        attn = torch.matmul(attn_weights, _v)

        if self.enable_gate:
            attn_prev = torch.cat((attn, prev_layer_output), 2)
            mid_res1 = self.proj_wg1(attn_prev)
            mid_res2 = nn.ReLU()(mid_res1)
            mid_res3 = self.proj_wg2(mid_res2)
            sigma = nn.Sigmoid()(mid_res3)
            res = torch.mul(attn, sigma) + torch.mul(prev_layer_output, 1 - sigma)
            return res.transpose(0, 1)
        else:
            res = (1.0 * attn + 1.0 * prev_layer_output)
            return res.transpose(0, 1)
