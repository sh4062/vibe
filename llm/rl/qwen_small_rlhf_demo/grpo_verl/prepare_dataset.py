import json
import os

import pandas as pd


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(script_dir, "data", "tiny_math.jsonl")
    train_path = os.path.join(script_dir, "data", "train.parquet")
    val_path = os.path.join(script_dir, "data", "val.parquet")

    rows = []
    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append(
                {
                    "prompt": item["prompt"],
                    "ground_truth": item["ground_truth"],
                    "data_source": "tiny_math",
                    "extra_info": {"difficulty": "toy"},
                }
            )

    df = pd.DataFrame(rows)
    df.to_parquet(train_path, index=False)
    df.to_parquet(val_path, index=False)
    print(f"Saved {len(df)} rows to {train_path}")
    print(f"Saved {len(df)} rows to {val_path}")


if __name__ == "__main__":
    main()
