# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

class Lightgcn_price(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(Lightgcn_price, self).__init__(config, dataset)

        # --- dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # --- hyperparams
        self.latent_dim = config["embedding_size"] if "embedding_size" in config else 64
        self.n_layers = config["n_layers"] if "n_layers" in config else 3
        self.reg_weight = config["reg_weight"] if "reg_weight" in config else 1e-4
        self.require_pow = config["require_pow"] if "require_pow" in config else False

        # --- fusion config
        self.fusion_type = config["fusion_type"] if "fusion_type" in config else "concat_mlp"
        self.bucket_vocab_size = config["bucket_vocab_size"] if "bucket_vocab_size" in config else None
        self.bucket_emb_size = config["bucket_emb_size"] if "bucket_emb_size" in config else self.latent_dim
        self.mlp_hidden_sizes = config["mlp_hidden_sizes"] if "mlp_hidden_sizes" in config else None
        self.gate_hidden_size = config["gate_hidden_size"] if "gate_hidden_size" in config else self.latent_dim
        self.bucket_reg = config["bucket_reg"] if "bucket_reg" in config else True
        self.item_bucket_index_path = config["item_bucket_index_path"] if "item_bucket_index_path" in config else None

        # --- basic embeddings
        self.user_embedding = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.1)

        item_bucket_tensor = self._load_item_bucket_indices(dataset)
        max_id_in_data = int(item_bucket_tensor.max().item())
        config_vocab_size= config["bucket_vocab_size"] if "bucket_vocab_size" in config else 0
        self.bucket_vocab_size= max(config_vocab_size, max_id_in_data+1)
        print(f"DEBUG: Config Size: {config_vocab_size} | Max Data ID: {max_id_in_data} | Final Embedding Size: {self.bucket_vocab_size}")
        self.bucket_embedding = nn.Embedding(self.bucket_vocab_size, self.bucket_emb_size)

        # --- fusion modules
        if self.fusion_type == "concat_mlp":
            if self.mlp_hidden_sizes is None:
                self.mlp_hidden_sizes = [self.latent_dim + self.bucket_emb_size, self.latent_dim]
            self.concat_mlp = self.build_mlp(
                self.latent_dim + self.bucket_emb_size, self.mlp_hidden_sizes, final_dim=self.latent_dim
            )
            

        elif self.fusion_type == "gated":
            self.gate_linear = nn.Linear(self.latent_dim + self.bucket_emb_size, self.latent_dim)
            self.bucket_proj = (
                nn.Linear(self.bucket_emb_size, self.latent_dim)
                if self.bucket_emb_size != self.latent_dim
                else None
            )

        elif self.fusion_type == "weighted":
            self.w_item = nn.Parameter(torch.tensor(0.9))
            self.w_bucket = nn.Parameter(torch.tensor(0.1))
            self.bucket_proj = (
                nn.Linear(self.bucket_emb_size, self.latent_dim)
                if self.bucket_emb_size != self.latent_dim
                else None
            )

        elif self.fusion_type == "add":
            self.bucket_proj = (
                nn.Linear(self.bucket_emb_size, self.latent_dim)
                if self.bucket_emb_size != self.latent_dim
                else None
            )
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        # --- loss
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.restore_user_e = None
        self.restore_item_e = None

        # --- Buffers (These move to GPU automatically with model.to(device))
        self.register_buffer("norm_adj_matrix", self.get_norm_adj_mat())
        item_bucket_tensor = self._load_item_bucket_indices(dataset)
        self.register_buffer("item_bucket_indices", item_bucket_tensor)

        # init
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = np.array(A.sum(axis=1)).flatten()
        diag = np.power(sumArr + 1e-7, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(i, data, torch.Size(L.shape))

    def build_mlp(self, input_dim, hidden_sizes, final_dim=None):
        layers = []
        prev = input_dim
        for h in hidden_sizes[:-1]:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h))
            prev = h
        last_h = hidden_sizes[-1]
        layers.append(nn.Linear(prev, last_h))
        if final_dim is not None and last_h != final_dim:
            layers.append(nn.ReLU())
            layers.append(nn.Linear(last_h, final_dim))
        return nn.Sequential(*layers)

    def _load_item_bucket_indices(self, dataset):
        try:
            if 'price_bucket' in dataset.field2type:
                item_feat= dataset.get_item_feature()
                if item_feat is not None and 'price_bucket' in item_feat:
                    print("Success: Loaded price_bucket from RecBole dataset.")
                    bucket_tensor= item_feat['price_bucket']
                    return bucket_tensor.to(torch.long).cpu()
        except Exception as e:
            print(f"Info: Could not load 'price_bucket' via dataset API ({e}). Trying .npy...")

        if self.item_bucket_index_path is not None:
            path = os.path.expanduser(self.item_bucket_index_path)
            if os.path.exists(path):
                arr = np.load(path)
                
                padding = np.zeros(1, dtype=arr.dtype)
                arr = np.concatenate([padding, arr])
                
                if len(arr) > self.n_items:
                    arr = arr[:self.n_items]
                elif len(arr) < self.n_items:
                    tail_pad = np.zeros(self.n_items - len(arr), dtype=arr.dtype)
                    arr = np.concatenate([arr, tail_pad])
                
                print(f"Loaded price_bucket from .npy: {path}")
                return torch.LongTensor(arr)

        print("WARNING: Falling back to ZERO price buckets. (Check your .item file or .npy path)")
        return torch.zeros(self.n_items, dtype=torch.long)
        

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        bucket_idxs = self.item_bucket_indices
        bucket_embs = self.bucket_embedding(bucket_idxs)

        if hasattr(self, 'bucket_proj') and self.bucket_proj is not None:
            bucket_proj = self.bucket_proj(bucket_embs)
        else:
            bucket_proj = bucket_embs

        if self.fusion_type == 'add':
            fused_item = item_embeddings + bucket_proj
        elif self.fusion_type == 'concat_mlp':
            concat = torch.cat([item_embeddings, bucket_embs], dim=1)
            fused_item = self.concat_mlp(concat)
        elif self.fusion_type == 'gated':
            concat = torch.cat([item_embeddings, bucket_embs], dim=1)
            gate = torch.sigmoid(self.gate_linear(concat))
            if self.bucket_proj is not None:
                bucket_for_comb = bucket_proj
            else:
                bucket_for_comb = bucket_embs
            fused_item = gate * item_embeddings + (1 - gate) * bucket_for_comb
        elif self.fusion_type == 'weighted':
            wi = F.relu(self.w_item)
            wb = F.relu(self.w_bucket)
            s = wi + wb + 1e-8
            wi = wi / s
            wb = wb / s
            fused_item = wi * item_embeddings + wb * bucket_proj
        else:
            fused_item = item_embeddings

        ego_embeddings = torch.cat([user_embeddings, fused_item], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        
        embeddings_list = [all_embeddings]
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
            
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        same_mask = (pos_item == neg_item)
        if same_mask.any():
            print(f"DEBUG: Found {same_mask.sum().item()} samples where neg_item == pos_item in batch!")
        if pos_item.max().item() >= self.n_items or neg_item.max().item() >= self.n_items:
            print("DEBUG: pos/neg item index out of range!", pos_item.max().item(), neg_item.max().item())
        if hasattr(self, 'item_bucket_indices'):
            if self.item_bucket_indices.max().item() >= self.bucket_vocab_size:
                print("DEBUG: item_bucket_indices contains id >= bucket_vocab_size!",self.item_bucket_indices.max().item(), self.bucket_vocab_size)

        user_all_embeddings, item_all_embeddings = self.forward()
        
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]
        u_embeddings = F.normalize(user_all_embeddings[user], dim=1)
        pos_embeddings = F.normalize(item_all_embeddings[pos_item], dim=1)
        neg_embeddings = F.normalize(item_all_embeddings[neg_item], dim=1)

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)

        if self.bucket_reg:
            pos_bucket_idx = self.item_bucket_indices[pos_item]
            neg_bucket_idx = self.item_bucket_indices[neg_item]
            
            pos_bucket_emb = self.bucket_embedding(pos_bucket_idx)
            neg_bucket_emb = self.bucket_embedding(neg_bucket_idx)
            
            reg_loss += (pos_bucket_emb.norm(p=2, dim=1).pow(2).mean() + neg_bucket_emb.norm(p=2, dim=1).pow(2).mean()) / 2.0

        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = F.normalize(user_all_embeddings[user], dim=1)
        i_embeddings = F.normalize(item_all_embeddings[item], dim=1)
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        
        u_embeddings = F.normalize(self.restore_user_e[user], dim=1)
        i_embeddings = F.normalize(self.restore_item_e, dim=1) 
        scores = torch.matmul(u_embeddings,i_embeddings.transpose(0, 1))
        return scores.view(-1)
