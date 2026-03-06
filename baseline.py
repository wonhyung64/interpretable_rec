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

try:
    import wandb
except: 
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


class MF(nn.Module):
    """
    Matrix Factorization
    """
    def __init__(self, num_users:int, num_items:int, embedding_k:int):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        out = torch.sum(user_embed.mul(item_embed), 1).unsqueeze(-1)
        return out, user_embed, item_embed


class NCF(nn.Module):
    """
    Neural Collaborative Filtering
    """
    def __init__(self, num_users:int, num_items:int, embedding_k:int, depth:int=0):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.depth = depth
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        layers_y1 = [nn.Linear(self.embedding_k*2, self.embedding_k), nn.ReLU()]
        for _ in range(self.depth):
            layers_y1.append(nn.Linear(self.embedding_k, self.embedding_k))
            layers_y1.append(nn.ReLU())
        layers_y1.append(nn.Linear(self.embedding_k, 1, bias=False))
        self.y1 = nn.Sequential(*layers_y1)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)
        out = self.y1(z_embed)
        return out, user_embed, item_embed




#%%
parser = argparse.ArgumentParser()
parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=8192)
parser.add_argument("--dataset-name", type=str, default="ml-latest-small")
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[1,3,5,7,10])
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--base-model", type=str, default="ncf")
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

wandb_login = False
try:
    wandb_login = wandb.login(key = open(f"{args.data_dir}/wandb_key.txt", 'r').readline())
except:
    pass
if wandb_login:
    args.expt_name = f"{args.base_model}_{expt_num}"
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
num_samples = len(y_train)
print(f"# user: {num_users}, # item: {num_items}, # samples: {num_samples}")

with open(f"{args.data_dir}/{args.dataset_name}/tagid2movies.json", "r", encoding="utf-8") as f:
    tag2items = json.load(f)
tag2items = {int(k): v for k, v in tag2items.items()}

all_idxs = np.arange(num_samples)
total_batch = num_samples // args.batch_size + 1


#%%
if args.base_model == "ncf":
    model = NCF(num_users, num_items, args.embedding_k, depth=args.depth)
elif args.base_model == "mf":
    model = MF(num_users, num_items, args.embedding_k)

model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_fcn = torch.nn.BCEWithLogitsLoss()


#%%
for epoch in range(1, args.num_epochs+1):
    np.random.shuffle(all_idxs)
    model.train()
    epoch_loss = 0.

    for idx in range(total_batch):
        selected_idx = all_idxs[args.batch_size*idx : (idx+1)*args.batch_size]
        sub_x = torch.LongTensor(x_train[selected_idx]).to(args.device)
        sub_y = torch.FloatTensor(y_train[selected_idx]).to(args.device)

        pred, _, __ = model(sub_x)
        loss = loss_fcn(pred, sub_y.unsqueeze(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_loss/total_batch:.4f}")


    if epoch % args.evaluate_interval == 0:
        model.eval()
        x_test_tensor = torch.LongTensor(x_test).to(args.device)
        pred_, _, __ = model(x_test_tensor)
        pred = pred_.flatten().cpu().detach().numpy()


        ndcg_res = ndcg_func(pred, x_test, y_test, args.top_k_list)
        ndcg_dict: dict = {}
        for top_k in args.top_k_list:
            ndcg_dict[f"ndcg_{top_k}"] = np.mean(ndcg_res[f"ndcg_{top_k}"])

        recall_res = recall_func(pred, x_test, y_test, args.top_k_list)
        recall_dict: dict = {}
        for top_k in args.top_k_list:
            recall_dict[f"recall_{top_k}"] = np.mean(recall_res[f"recall_{top_k}"])

        ap_res = ap_func(pred, x_test, y_test, args.top_k_list)
        ap_dict: dict = {}
        for top_k in args.top_k_list:
            ap_dict[f"ap_{top_k}"] = np.mean(ap_res[f"ap_{top_k}"])

        auc = roc_auc_score(y_test, pred)


        print(f"NDCG: {ndcg_dict}")
        print(f"Recall: {recall_dict}")
        print(f"AP: {ap_dict}")
        print(f"AUC: {auc}")


        if wandb_login:
            wandb_var.log(ndcg_dict)
            wandb_var.log(recall_dict)
            wandb_var.log(ap_dict)
            wandb_var.log({"auc": auc})


if wandb_login:
    wandb.finish()
