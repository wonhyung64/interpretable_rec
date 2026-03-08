#%%
import re
import json
import unicodedata
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter
from itertools import combinations
from collections import defaultdict


def normalize_tag(tag):
    if pd.isna(tag):
        return None
    
    tag = str(tag)
    tag = unicodedata.normalize("NFKC", tag)
    tag = tag.lower()
    tag = tag.replace("–", "-").replace("—", "-").replace("−", "-")
    tag = tag.strip()
    tag = re.sub(r"\s+", " ", tag)
    if tag == "":
        return None

    return tag


#%% LOADING
data_dir = "./data/ml-latest-small"
data_dir = "./data/ml-32m"
link_df = pd.read_csv(f"{data_dir}/links.csv")
movie_df = pd.read_csv(f"{data_dir}/movies.csv")
rating_df = pd.read_csv(f"{data_dir}/ratings.csv")
tag_df = pd.read_csv(f"{data_dir}/tags.csv")


#%% PREPROCESSING
movie_df["movieId"] -= 1
rating_df["userId"] -= 1
rating_df["movieId"] -= 1

tag_df["tag_norm"] = tag_df["tag"].apply(normalize_tag)
tag_df = tag_df[tag_df["tag_norm"].notna()].copy()
movie_df["tags"] = ""

all_tag = {}
grouped_tags = tag_df.groupby("movieId")["tag_norm"].apply(
    lambda x: sorted(set(x.tolist()))
)

for movie_id, item_tags in tqdm(grouped_tags.items(), total=len(grouped_tags)):
    for tag in item_tags:
        all_tag[tag] = all_tag.get(tag, 0) + 1

    movie_df.loc[movie_df["movieId"] == movie_id, "tags"] = ",".join(item_tags)

sorted_all_tag = dict(sorted(all_tag.items(), key=lambda x: x[1], reverse=True))

with open(f"{data_dir}/tag_count.json", "w", encoding="utf-8") as f:
    json.dump(sorted_all_tag, f, ensure_ascii=False, indent=2)

#%%
tag2id = {tag: idx for idx, tag in enumerate(all_tag.keys())}
id2tag = {idx: tag for tag, idx in tag2id.items()}

with open(f"{data_dir}/tag2id.json", "w", encoding="utf-8") as f:
    json.dump(tag2id, f, ensure_ascii=False, indent=2)

with open(f"{data_dir}/id2tag.json", "w", encoding="utf-8") as f:
    json.dump(id2tag, f, ensure_ascii=False, indent=2)

#%%
tagid_to_movies = defaultdict(set)

for _, row in movie_df[["movieId", "tags"]].iterrows():
    movie_id = row["movieId"]
    tags = row["tags"]
    
    if pd.isna(tags) or tags == "":
        continue
    
    for tag in tags.split(","):
        tag = tag.strip()
        if tag and tag in tag2id:
            tagid_to_movies[tag2id[tag]].add(movie_id)

tagid_to_movies = {tag_id: sorted(list(movie_ids)) for tag_id, movie_ids in tagid_to_movies.items()}

with open(f"{data_dir}/tagid2movies.json", "w", encoding="utf-8") as f:
    json.dump(tagid_to_movies, f, ensure_ascii=False)

#%%
pair_counter = Counter()

for tags in movie_df["tags"].dropna():
    tag_list = [t.strip() for t in str(tags).split(",") if t.strip()]
    tag_list = sorted(set(tag_list))
    for pair in combinations(tag_list, 2):
        pair_counter[pair] += 1

pair_df = pd.DataFrame(
    [(a, b, cnt) for (a, b), cnt in pair_counter.items()],
    columns=["tag1", "tag2", "count"]
).sort_values("count", ascending=False).reset_index(drop=True)

pair_df.to_csv(f"{data_dir}/pair_count.csv")


#%%

df = rating_df.sort_values(["timestamp", "userId", "movieId"]).reset_index(drop=True).copy()

# binary implicit feedback
df.loc[df["rating"] < 3, "rating"] = 0
df.loc[df["rating"] >= 3, "rating"] = 1

df.columns = ["userId", "movieId", "interaction", "timestamp"]
df["interaction"] = df["interaction"].astype(int)

n_total = len(df)
train_end = int(n_total * 0.8)
val_end = int(n_total * 0.9)

train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
test_df  = df.iloc[val_end:].copy()

# remove cold users/items from val/test
train_users = set(train_df["userId"].unique())
train_items = set(train_df["movieId"].unique())

val_df = val_df[
    val_df["userId"].isin(train_users) &
    val_df["movieId"].isin(train_items)
].copy()

# test also should only contain users/items seen in train (+ optionally val)
seen_users = train_users | set(val_df["userId"].unique())
seen_items = train_items | set(val_df["movieId"].unique())

test_df = test_df[
    test_df["userId"].isin(seen_users) &
    test_df["movieId"].isin(seen_items)
].copy()

# sort back by user-time if your downstream code expects per-user chronological order
train_df = train_df.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
val_df   = val_df.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
test_df  = test_df.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)

print(f"train: {len(train_df)}")
print(f"val:   {len(val_df)}")
print(f"test:  {len(test_df)}")

print(f"#users train/val/test: {train_df['userId'].nunique()} / {val_df['userId'].nunique()} / {test_df['userId'].nunique()}")
print(f"#items train/val/test: {train_df['movieId'].nunique()} / {val_df['movieId'].nunique()} / {test_df['movieId'].nunique()}")

np.save(f"{data_dir}/train.npy", train_df[["userId", "movieId", "interaction", "timestamp"]].to_numpy(), allow_pickle=True)
np.save(f"{data_dir}/val.npy", val_df[["userId", "movieId", "interaction", "timestamp"]].to_numpy(), allow_pickle=True)
np.save(f"{data_dir}/test.npy", test_df[["userId", "movieId", "interaction", "timestamp"]].to_numpy(), allow_pickle=True)


# %%
