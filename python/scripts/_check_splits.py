import pandas as pd
CLASS = {0:"QUIET", 1:"B", 2:"C", 3:"M", 4:"X"}
for split in ["train", "val", "test"]:
    df = pd.read_parquet(f"data/dataset/{split}.parquet")
    dist = df["label"].value_counts().sort_index()
    print(f"=== {split}.parquet ({len(df)} rows) ===")
    for k, v in dist.items():
        name = CLASS.get(k, f"cls_{k}")
        print(f"  {k} ({name}): {v}")
    print()
