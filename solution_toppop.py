import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


@dataclass
class InteractionData:
    """Container for the interaction matrix and mapping metadata."""

    user_item_matrix: csr_matrix
    item_user_matrix: csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    popularity_item_codes: List[int]


def compute_interaction_weights(df: pd.DataFrame) -> pd.Series:
    """Return a numeric weight for every interaction row."""

    timespent = df["timespent"].fillna(0).astype(np.float32)
    max_timespent = max(timespent.max(), 1.0)

    like = df["like"].fillna(False).astype(np.int8)
    share = df["share"].fillna(False).astype(np.int8)
    bookmark = df["bookmark"].fillna(False).astype(np.int8)
    click_on_author = df["click_on_author"].fillna(False).astype(np.int8)
    open_comments = df["open_comments"].fillna(False).astype(np.int8)
    dislike = df["dislike"].fillna(False).astype(np.int8)

    weight = (
        1.0
        + 0.5 * like
        + 0.3 * share
        + 0.3 * bookmark
        + 0.2 * click_on_author
        + 0.2 * open_comments
        + 0.1 * (timespent / max_timespent)
        - 0.4 * dislike
    )

    return weight.clip(lower=0.05).astype(np.float32)


def prepare_sparse_data(df: pd.DataFrame) -> InteractionData:
    """Build sparse matrices and metadata required for model training."""

    weights = compute_interaction_weights(df)
    aggregated = (
        df.assign(weight=weights)
        .groupby(["user_id", "item_id"], as_index=False)["weight"]
        .sum()
    )

    user_codes, user_ids = pd.factorize(aggregated["user_id"])
    item_codes, item_ids = pd.factorize(aggregated["item_id"])

    user_ids = np.asarray(user_ids)
    item_ids = np.asarray(item_ids)

    interaction_matrix = csr_matrix(
        (
            aggregated["weight"].astype(np.float32).to_numpy(),
            (user_codes, item_codes),
        ),
        shape=(len(user_ids), len(item_ids)),
    )

    item_user_matrix = interaction_matrix.T.tocsr()

    # Popularity-based fallback for cold recommendations
    popularity = (
        aggregated.groupby("item_id")["weight"]
        .sum()
        .sort_values(ascending=False)
    )
    item_id_to_code = {item_id: idx for idx, item_id in enumerate(item_ids)}
    popularity_item_codes = [
        item_id_to_code[item_id]
        for item_id in popularity.index
        if item_id in item_id_to_code
    ]

    return InteractionData(
        user_item_matrix=interaction_matrix,
        item_user_matrix=item_user_matrix,
        user_ids=user_ids,
        item_ids=item_ids,
        popularity_item_codes=popularity_item_codes,
    )


def generate_recommendations(
    model: AlternatingLeastSquares,
    data: InteractionData,
    top_k: int,
) -> List[Tuple[int, int]]:
    """Return a list of (user_id, item_id) recommendation pairs."""

    results: List[Tuple[int, int]] = []
    user_matrix = data.user_item_matrix

    for user_idx, user_id in enumerate(data.user_ids):
        user_history = user_matrix[user_idx]
        rec_item_ids, _ = model.recommend(
            userid=user_idx,
            user_items=user_history,
            N=top_k,
            filter_already_liked_items=True,
        )
        recommended_codes = list(rec_item_ids)

        if len(recommended_codes) < top_k:
            seen_codes = set(user_history.indices)
            needed = top_k - len(recommended_codes)
            for code in data.popularity_item_codes:
                if code in seen_codes or code in recommended_codes:
                    continue
                recommended_codes.append(code)
                needed -= 1
                if needed == 0:
                    break

        for code in recommended_codes[:top_k]:
            results.append((int(user_id), int(data.item_ids[code])))

    return results


def main(args: argparse.Namespace) -> None:
    print("Load train data...")
    interactions = pd.read_parquet(args.input_path)
    print(f"Loaded {len(interactions):,} interactions.")

    print("Prepare sparse matrices...")
    data = prepare_sparse_data(interactions)
    print(
        "Matrix built: %d users, %d items, %.2fM non-zero values"
        % (
            data.user_item_matrix.shape[0],
            data.user_item_matrix.shape[1],
            data.user_item_matrix.nnz / 1e6,
        )
    )

    print("Train ALS model...")
    model = AlternatingLeastSquares(
        factors=args.factors,
        iterations=args.iterations,
        regularization=args.regularization,
        random_state=args.random_seed,
    )
    model.fit(data.user_item_matrix)
    print("Model trained.")

    print("Generate recommendations...")
    rec_pairs = generate_recommendations(model, data, top_k=args.top_k)
    print("Recommendations computed.")

    result_df = pd.DataFrame(rec_pairs, columns=["user_id", "recs"])
    print(f"Save result to file {args.output_path}...")
    result_df.to_csv(args.output_path, index=False)
    print("File saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALS recommender for VK clips.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to train parquet file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save csv with recommendations")
    parser.add_argument("--top_k", type=int, default=10, help="Number of recommendations per user")
    parser.add_argument("--factors", type=int, default=64, help="Latent factors for ALS")
    parser.add_argument("--iterations", type=int, default=20, help="Number of ALS iterations")
    parser.add_argument(
        "--regularization", type=float, default=0.1, help="ALS regularization strength"
    )
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for ALS")

    main(parser.parse_args())
