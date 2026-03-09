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

    def forward(self, x):
        user_idx = x[:, 0]
        item_idx = x[:, 1]

        concept_vectors = self.get_concept_vectors()
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)

        item_concept_sim = item_embed @ concept_vectors.T
        out = (user_embed * item_concept_sim).sum(dim=-1)

        return out, user_embed, item_concept_sim


    def predict(self, x):
        user_idx = x[:, 0]
        item_idx = x[:, 1]

        concept_vectors = self.get_concept_vectors()
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)

        item_concept_sim = item_embed @ concept_vectors.T
        out = (user_embed * item_concept_sim).sum(dim=-1)

        return out, user_embed, item_concept_sim
