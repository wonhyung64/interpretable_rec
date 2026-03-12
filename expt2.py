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

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PairwiseSampleDataset(Dataset):
    def __init__(self, x_train, y_train):

        user_positem_dict = defaultdict(list)
        user_negitem_dict = defaultdict(list)
        for (user, item), label in zip(x_train, y_train):
            user = int(user); item = int(item)
            if label == 1:
                user_positem_dict[user].append(item)
            else:
                user_negitem_dict[user].append(item)

        # pairwise 가능한 user만
        valid_user_list = [user for user in user_positem_dict if len(user_positem_dict[user]) > 0 and len(user_negitem_dict[user]) > 0]

        # user id 압축: 0..M-1
        self.user_sample_dict = {user: sample for sample, user in enumerate(valid_user_list)}
        self.posuser_list = np.array(valid_user_list, dtype=np.int64)

        # pos_samples를 (user_idx, pos_item) 배열로 저장
        possample_list = []
        positem_list = []
        for user in valid_user_list:
            sample = self.user_sample_dict[user]
            for positem in user_positem_dict[user]:
                possample_list.append(sample)
                positem_list.append(positem)
        self.possample_arr = np.array(possample_list, dtype=np.int32)
        self.positem_arr = np.array(positem_list, dtype=np.int64)

        # neg를 flat array + offsets/lengths로 저장 (핵심)
        offsets = np.zeros(len(valid_user_list), dtype=np.int64)
        lengths = np.zeros(len(valid_user_list), dtype=np.int32)

        flat_neg = []
        cur = 0
        for sample, user in enumerate(valid_user_list):
            negs = user_negitem_dict[user]
            offsets[sample] = cur
            lengths[sample] = len(negs)
            flat_neg.extend(negs)
            cur += len(negs)

        self.neg_offsets = offsets
        self.neg_lengths = lengths
        self.flat_neg = np.array(flat_neg, dtype=np.int64)

        # numpy RNG (worker마다 다르게 seed 주면 더 좋음)
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.possample_arr)

    def __getitem__(self, idx):
        possample = self.possample_arr[idx]
        positem = self.positem_arr[idx]

        off = self.neg_offsets[possample]
        L = self.neg_lengths[possample]
        j = off + self.rng.integers(L)   # 0..L-1
        negitem = self.flat_neg[j]

        user = self.posuser_list[possample]  # 원래 user id로 복원
        return int(user), int(positem), int(negitem)


def collate_triplets(batch):
    """
    batch: [(u, pos, neg), ...]
    -> torch tensor로 묶어서 반환
    """
    u, p, n = zip(*batch)
    return (
        torch.tensor(u, dtype=torch.long),
        torch.tensor(p, dtype=torch.long),
        torch.tensor(n, dtype=torch.long),
    )


def worker_init_fn(worker_id):
    """
    DataLoader 멀티워커에서 각 워커가 서로 다른 RNG를 쓰도록 시드 설정.
    """
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    # torch initial seed 기반으로 워커마다 다른 seed
    seed = (torch.initial_seed() + worker_id) % (2**32)
    ds.rng = np.random.default_rng(seed)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConceptMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_k: int, tag2items: dict,
                 sparse: bool = True, normalize_centroid: bool = False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.k = embedding_k

        sorted_tag_ids = sorted(tag2items.keys())
        self.num_tags = len(sorted_tag_ids)
        self.normalize_centroid = normalize_centroid

        # ✅ user/item은 k차원 (작게)
        self.user_embedding = nn.Embedding(num_users, self.k, sparse=sparse)
        self.item_embedding = nn.Embedding(num_items, self.k, sparse=sparse)

        # tag-item sparse matrix (centroid 계산할 때만 사용)
        rows, cols, vals = [], [], []
        for new_tag_idx, tag_id in enumerate(sorted_tag_ids):
            item_ids = tag2items[tag_id]
            if len(item_ids) == 0:
                continue
            w = 1.0 / len(item_ids)  # 평균 weight
            rows.extend([new_tag_idx] * len(item_ids))
            cols.extend(item_ids)
            vals.extend([w] * len(item_ids))

        indices = torch.tensor([rows, cols], dtype=torch.long)
        values  = torch.tensor(vals, dtype=torch.float32)
        concept_mat = torch.sparse_coo_tensor(
            indices, values, size=(self.num_tags, self.num_items)
        ).coalesce()
        self.register_buffer("concept_mat", concept_mat)

        # ✅ cache buffers (model.to(device) 시 같이 이동)
        self.register_buffer("concept_vectors_cache", torch.empty(self.num_tags, self.k))
        self.register_buffer("gram_cache", torch.empty(self.k, self.k))
        self.cache_ready = False

    @torch.no_grad()
    def refresh_cache(self):
        """
        C = concept_mat @ item_embedding.weight  (T,k)
        G = C^T C  (k,k)
        둘 다 no_grad로 갱신해서 매 step 역전파 비용 제거.
        """
        C = torch.sparse.mm(self.concept_mat, self.item_embedding.weight)  # (T,k)
        if self.normalize_centroid:
            C = F.normalize(C, dim=-1)

        G = C.t().matmul(C)  # (k,k)

        self.concept_vectors_cache.copy_(C)
        self.gram_cache.copy_(G)
        self.cache_ready = True

    def forward_z(self, users, pos_items, neg_items):
        """
        빠른 z 계산:
          z = u^T G (v_pos - v_neg)
        return z: (B,1)
        """
        assert self.cache_ready, "call model.refresh_cache() first"

        u = self.user_embedding(users)              # (B,k)
        vp = self.item_embedding(pos_items)         # (B,k)
        vn = self.item_embedding(neg_items)         # (B,k)

        dv = vp - vn                                # (B,k)
        uG = u.matmul(self.gram_cache)              # (B,k)
        z = (uG * dv).sum(dim=-1, keepdim=True)     # (B,1)
        return z

    def forward(self, samples, neg_item):
        """
        기존 학습 코드랑 맞추기 위해 forward에서 z만 반환하도록 제공.
        """
        users = samples[:, 0]
        pos_items = samples[:, 1]
        return self.forward_z(users, pos_items, neg_item)

    @torch.no_grad()
    def predict(self, x):
        """
        out = u^T G v  (B,)
        (빠른 스코어만)
        """
        assert self.cache_ready, "call model.refresh_cache() first"
        users = x[:, 0]
        items = x[:, 1]

        u = self.user_embedding(users)          # (B,k)
        v = self.item_embedding(items)          # (B,k)
        uG = u.matmul(self.gram_cache)          # (B,k)
        out = (uG * v).sum(dim=-1)              # (B,)
        return out

    @torch.no_grad()
    def explain_topk(self, user_id: int, item_id: int, topk: int = 20):
        """
        필요할 때만 (u,i) 1개에 대해 태그별 기여도 계산:
          contrib_t = (u·c_t) * (v·c_t)
        """
        assert self.cache_ready, "call model.refresh_cache() first"
        device = self.item_embedding.weight.device

        u = self.user_embedding.weight[user_id].to(device)  # (k,)
        v = self.item_embedding.weight[item_id].to(device)  # (k,)
        C = self.concept_vectors_cache                       # (T,k)

        uc = C.matmul(u)    # (T,)
        vc = C.matmul(v)    # (T,)
        contrib = uc * vc   # (T,)

        vals, idx = torch.topk(contrib, topk)
        return idx.cpu(), vals.cpu()
#%%
parser = argparse.ArgumentParser()
parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--centroid-refresh-every", type=int, default=200,
                    help="몇 step마다 centroid/gram cache를 갱신할지 (0이면 epoch마다 1번)")
parser.add_argument("--centroid-normalize", action="store_true",
                    help="centroid vector를 L2 normalize 할지 여부 (권장)")
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=16384)
parser.add_argument("--dataset-name", type=str, default="ml-32m")
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

with open(f"{args.data_dir}/{args.dataset_name}/tagid2movies.json", "r", encoding="utf-8") as f:
    tag2items = json.load(f)
tag2items = {int(k): v for k, v in tag2items.items()}

num_users = int(max(x_train[:,0].max(), x_valid[:,0].max(), x_test[:,0].max())) + 1

train_max_item = int(x_train[:,1].max())
valid_max_item = int(x_valid[:,1].max())
test_max_item = int(x_test[:,1].max())
tag_max_item = max((max(v) for v in tag2items.values() if len(v) > 0), default=-1)

num_items = max(train_max_item, valid_max_item, test_max_item, tag_max_item) + 1
print(f"# user: {num_users}, # item: {num_items}")


#%%
dataset = PairwiseSampleDataset(x_train, y_train)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,           # pos anchor 섞기
    num_workers=32,          # CPU 여유 있으면 4~8 권장
    pin_memory=True,        # GPU 학습이면 추천
    worker_init_fn=worker_init_fn,
    collate_fn=collate_triplets,
    persistent_workers=True # 에폭 사이 워커 유지(파이썬 오버헤드 감소)
)


num_samples = dataset.__len__()
all_idxs = np.arange(num_samples)
total_batch = num_samples // args.batch_size + 1


#%%
model = ConceptMF(num_users, num_items, args.embedding_k, tag2items,
                  sparse=False, normalize_centroid=args.centroid_normalize).to(args.device)

# ✅ sparse=True면 SparseAdam 추천 (속도/메모리 유리)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


#%%
global_step = 0

for epoch in range(1, args.num_epochs + 1):
    model.train()
    epoch_loss = 0.

    # epoch 시작 시 1회 갱신 (refresh_every==0일 때 특히 중요)
    if (epoch == 1) or (args.centroid_refresh_every == 0):
        model.refresh_cache()

    for i, (u, pos, neg) in enumerate(loader):
        u = u.to(args.device, non_blocking=True)
        pos = pos.to(args.device, non_blocking=True)
        neg_item = neg.to(args.device, non_blocking=True)

        # step 주기 갱신
        if args.centroid_refresh_every > 0 and (global_step % args.centroid_refresh_every == 0):
            model.refresh_cache()

        samples = torch.stack([u, pos], -1)

        # ✅ z를 직접 받음: (B,1)
        z = model(samples, neg_item)

        loss = nn.functional.softplus(-z).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        global_step += 1

    print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_loss/total_batch:.4f}")

    if epoch % args.evaluate_interval == 0:

        model.eval()
        model.refresh_cache()
        x_valid_tensor = torch.LongTensor(x_valid).to(args.device)
        pred_ = model.predict(x_valid_tensor)
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
        pred_ = model.predict(x_test_tensor)
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
