import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def main(input_path: str, output_path: str):
    train = pd.read_parquet(input_path)

    top = (train
           .groupby("item_id", as_index=False)["user_id"]
           .count()
           .sort_values("user_id", ascending=False)
           )
    top10 = top[:10]["item_id"].to_list()

    users_for_rec = pd.DataFrame({"user_id": train["user_id"].unique()})

    print("Compute recommendations. It may take a few minutes.")
    users_for_rec['recs'] = [top10 for _ in range(len(users_for_rec))]
    print("Recommendations computed.")

    result = users_for_rec.explode("recs")

    print(f"Save result to file {output_path}.")
    result.to_csv(output_path, index=False)
    print("File saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommender arguments.")
    parser.add_argument("--input_path", type=str, required=True, help="Input path to train parquet file")
    parser.add_argument("--output_path", type=str, required=True, help="Output path to csv with recommendations")

    args = parser.parse_args()
    main(args.input_path, args.output_path)