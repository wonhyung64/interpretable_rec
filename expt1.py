#%%
import os
import sys
import json
import torch
import random
import argparse
import subprocess
import numpy as np
import torch.nn as nn
import scipy.sparse as sps
from datetime import datetime
from sklearn.metrics import roc_auc_score
from module.utils import set_device, set_seed
from module.model import ConceptMF
from module.metric import ndcg_func, recall_func, ap_func

try:
    import wandb
except: 
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


#%%
parser = argparse.ArgumentParser()
parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=16384)
parser.add_argument("--dataset-name", type=str, default="ml-latest-small")
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[1,3,5,7,10])
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--depth", type=int, default=0)
parser.add_argument("--key-dir", type=str, default="./")
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

#%%
args.expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(args.random_seed)
args.device = set_device()
# args.data_dir = "/gpfs/home1/wonhyung64/Github/interpretable_rec/data"

wandb_login = False
try:
    wandb_login = wandb.login(key = open(f"{args.key_dir}/wandb_key.txt", 'r').readline())
except:
    pass
if wandb_login:
    args.expt_name = f"concept_mf_{args.expt_num}"
    wandb_var = wandb.init(project="interpretable_rec", config=vars(args))
    wandb.run.name = args.expt_name

train_file = os.path.join(args.data_dir, args.dataset_name, "train.npy")
valid_file = os.path.join(args.data_dir, args.dataset_name, "val.npy")
test_file = os.path.join(args.data_dir, args.dataset_name, "test.npy")

x_train = np.load(train_file)
x_valid = np.load(valid_file)
x_test = np.load(test_file)

x_train, y_train = x_train[:,:-2], x_train[:,-2]
x_valid, y_valid = x_valid[:,:-2], x_valid[:,-2]
x_test, y_test = x_test[:,:-2], x_test[:,-2]

num_users = x_train[:,0].max() + 1
num_items = x_train[:,1].max() + 1
print(f"# user: {num_users}, # item: {num_items}")


#%%

import random
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class PairwiseSampleDataset(Dataset):
    def __init__(self, x, y):
        """
        x: (N, 2) -> [user_id, item_id]
        y: (N,)   -> 1/0
        """
        self.user_pos = {}
        self.user_neg = {}
        self.pos_samples = []

        user_pos = defaultdict(list)
        user_neg = defaultdict(list)

        for (u, i), label in zip(x, y):
            if label == 1:
                user_pos[int(u)].append(int(i))
            else:
                user_neg[int(u)].append(int(i))

        # pairwise 가능한 user만 남김
        valid_users = []
        for u in user_pos:
            if len(user_pos[u]) > 0 and len(user_neg[u]) > 0:
                valid_users.append(u)

        self.user_pos = {u: user_pos[u] for u in valid_users}
        self.user_neg = {u: user_neg[u] for u in valid_users}

        # positive anchor 목록
        for u in valid_users:
            for pos_item in self.user_pos[u]:
                self.pos_samples.append((u, pos_item))

    def neg_sampling(self):
        self.neg_samples = []
        for u, pos_item in self.pos_samples:
            self.neg_samples.append(random.choice(self.user_neg[u]))

    def __len__(self):
        return len(self.pos_samples)

dataset = PairwiseSampleDataset(x_train, y_train)

num_samples = dataset.__len__()
all_idxs = np.arange(num_samples)
total_batch = num_samples // args.batch_size + 1

#%%
import torch
import torch.nn as nn


class ConceptMF(nn.Module):
    def __init__(self, num_users:int, num_items:int, embedding_k:int, tag2items:dict):
        super(ConceptMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        sorted_tag_ids = sorted(tag2items.keys())
        self.num_tags = len(sorted_tag_ids)
        self.user_embedding = nn.Embedding(self.num_users, self.num_tags)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)

        rows, cols, vals = [], [], []
        for new_tag_idx, tag_id in enumerate(sorted_tag_ids):
            item_ids = tag2items[tag_id]
            if len(item_ids) == 0:
                continue
            w = 1.0 / len(item_ids)   # 평균을 위한 weight
            rows.extend([new_tag_idx] * len(item_ids))
            cols.extend(item_ids)
            vals.extend([w] * len(item_ids))
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        concept_mat = torch.sparse_coo_tensor(
            indices,
            values,
            size=(self.num_tags, self.num_items)
        ).coalesce()
        self.register_buffer("concept_mat", concept_mat) # 학습 파라미터가 아니라 고정 텐서

    def get_concept_vectors(self): # [num_tags, num_items] @ [num_items, embedding_k] -> [num_tags, embedding_k]
        return torch.sparse.mm(self.concept_mat, self.item_embedding.weight)

    def forward(self, samples, neg_item):
        anchor_user = samples[:, 0]
        pos_item = samples[:, 1]
        concept_vectors = self.get_concept_vectors()
        user_embed = self.user_embedding(anchor_user)
        pos_item_embed = self.item_embedding(pos_item)
        neg_item_embed = self.item_embedding(neg_item)
        pos_concept_sim = pos_item_embed @ concept_vectors.T
        neg_concept_sim = neg_item_embed @ concept_vectors.T
        return user_embed, pos_concept_sim, neg_concept_sim


    def predict(self, x):
        user_idx = x[:, 0]
        item_idx = x[:, 1]

        concept_vectors = self.get_concept_vectors()
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)

        item_concept_sim = item_embed @ concept_vectors.T
        out = (user_embed * item_concept_sim).sum(dim=-1)

        return out, user_embed, item_concept_sim




with open(f"{args.data_dir}/{args.dataset_name}/tagid2movies.json", "r", encoding="utf-8") as f:
    tag2items = json.load(f)
tag2items = {int(k): v for k, v in tag2items.items()}

model = ConceptMF(num_users, num_items, args.embedding_k, tag2items)
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


#%%
for epoch in range(1, args.num_epochs+1):
    dataset.neg_sampling()
    np.random.shuffle(all_idxs)
    model.train()
    epoch_loss = 0.

    for idx in range(total_batch):
        selected_idx = list(range(args.batch_size*idx, (idx+1)*args.batch_size))
        samples = torch.LongTensor(dataset.pos_samples[args.batch_size*idx : (idx+1)*args.batch_size]).to(args.device)
        neg_item = torch.LongTensor(dataset.neg_samples[args.batch_size*idx : (idx+1)*args.batch_size]).to(args.device)

        user_embed, pos_concept_sim, neg_concept_sim = model(samples, neg_item)
        z = ((pos_concept_sim - neg_concept_sim) * user_embed).sum(dim=-1, keepdim=True)
        loss = nn.functional.softplus(-z).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_loss/total_batch:.4f}")


    if epoch % args.evaluate_interval == 0:

        model.eval()
        x_valid_tensor = torch.LongTensor(x_valid).to(args.device)
        pred_, _, __ = model.predict(x_valid_tensor)
        pred = pred_.flatten().cpu().detach().numpy()

        ndcg_res = ndcg_func(pred, x_valid, y_valid, args.top_k_list)
        ndcg_dict: dict = {}
        for top_k in args.top_k_list:
            ndcg_dict[f"valid_ndcg_{top_k}"] = np.mean(ndcg_res[f"ndcg_{top_k}"])

        recall_res = recall_func(pred, x_valid, y_valid, args.top_k_list)
        recall_dict: dict = {}
        for top_k in args.top_k_list:
            recall_dict[f"valid_recall_{top_k}"] = np.mean(recall_res[f"recall_{top_k}"])

        ap_res = ap_func(pred, x_valid, y_valid, args.top_k_list)
        ap_dict: dict = {}
        for top_k in args.top_k_list:
            ap_dict[f"valid_ap_{top_k}"] = np.mean(ap_res[f"ap_{top_k}"])

        valid_auc = roc_auc_score(y_valid, pred)

        print(f"valid_NDCG: {ndcg_dict}")
        print(f"valid_Recall: {recall_dict}")
        print(f"valid_AP: {ap_dict}")
        print(f"valid_AUC: {valid_auc}")

        model.eval()
        x_test_tensor = torch.LongTensor(x_test).to(args.device)
        pred_, _, __ = model.predict(x_test_tensor)
        pred = pred_.flatten().cpu().detach().numpy()

        ndcg_res = ndcg_func(pred, x_test, y_test, args.top_k_list)
        for top_k in args.top_k_list:
            ndcg_dict[f"test_ndcg_{top_k}"] = np.mean(ndcg_res[f"ndcg_{top_k}"])

        recall_res = recall_func(pred, x_test, y_test, args.top_k_list)
        for top_k in args.top_k_list:
            recall_dict[f"test_recall_{top_k}"] = np.mean(recall_res[f"recall_{top_k}"])

        ap_res = ap_func(pred, x_test, y_test, args.top_k_list)
        for top_k in args.top_k_list:
            ap_dict[f"test_ap_{top_k}"] = np.mean(ap_res[f"ap_{top_k}"])

        test_auc = roc_auc_score(y_test, pred)


        print(f"NDCG: {ndcg_dict}")
        print(f"Recall: {recall_dict}")
        print(f"AP: {ap_dict}")
        print(f"AUC: {test_auc}")


        if wandb_login:
            wandb_var.log(ndcg_dict)
            wandb_var.log(recall_dict)
            wandb_var.log(ap_dict)
            wandb_var.log({"valid_auc": valid_auc})
            wandb_var.log({"test_auc": test_auc})


if wandb_login:
    wandb.finish()
