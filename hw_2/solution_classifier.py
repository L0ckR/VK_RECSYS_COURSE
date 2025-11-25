import argparse
import gc
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd


def compute_feedback_weight(df: pd.DataFrame) -> pd.Series:
    """Compute an implicit feedback weight from user actions."""
    timespent_term = df["timespent"].astype(np.float32) / 60.0
    positive_actions = (
        3.0 * df["like"].astype(np.float32)
        + 3.5 * df["share"].astype(np.float32)
        + 2.5 * df["bookmark"].astype(np.float32)
        + 1.5 * df["click_on_author"].astype(np.float32)
        + 1.0 * df["open_comments"].astype(np.float32)
    )
    negative_actions = 2.0 * df["dislike"].astype(np.float32)
    weight = timespent_term + positive_actions - negative_actions
    return weight.clip(lower=0.0)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-6)


def build_user_profiles(
    interactions: pd.DataFrame,
    item_vectors: np.ndarray,
    item_id_to_index: Dict[int, int],
) -> Dict[int, np.ndarray]:
    """Create normalized user embeddings using weighted item histories."""
    profiles: Dict[int, np.ndarray] = {}
    interactions = interactions.assign(
        item_idx=interactions["item_id"].map(item_id_to_index)
    ).dropna(subset=["item_idx"])
    interactions["item_idx"] = interactions["item_idx"].astype(np.int32)

    for user_id, group in interactions.groupby("user_id"):
        weights = group["feedback_weight"].to_numpy(dtype=np.float32)
        idxs = group["item_idx"].to_numpy(dtype=np.int32)
        if weights.size == 0:
            continue

        user_vec = (item_vectors[idxs] * weights[:, None]).sum(axis=0)
        denom = weights.sum()
        if denom <= 0:
            continue
        user_vec = user_vec / denom
        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec = user_vec / norm
        profiles[int(user_id)] = user_vec.astype(np.float32)

    return profiles


def recommend_for_user(
    user_id: int,
    profiles: Dict[int, np.ndarray],
    item_vectors: np.ndarray,
    popularity: np.ndarray,
    item_ids: np.ndarray,
    seen_items: Dict[int, Set[int]],
    top_k: int,
    content_weight: float,
    popularity_weight: float,
) -> List[Tuple[int, int]]:
    """Return top-k recommendations for a single user."""
    if user_id in profiles:
        content_scores = item_vectors @ profiles[user_id]
        scores = content_weight * content_scores + popularity_weight * popularity
    else:
        scores = popularity.copy()

    seen = seen_items.get(user_id)
    if seen:
        mask = np.isin(item_ids, list(seen))
        scores = scores.copy()
        scores[mask] = -1e9

    if not np.isfinite(scores).any():
        scores = popularity.copy()

    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [(user_id, int(item_ids[i])) for i in top_idx]


def main(
    input_path_interactions: str,
    input_path_users: str,
    input_path_items: str,
    input_path_embeddings: str,
    output_path: str,
) -> None:
    print("Load data.")
    interactions = pd.read_parquet(input_path_interactions)
    user_metadata = pd.read_parquet(input_path_users)
    item_embeddings = pd.read_parquet(input_path_embeddings)

    print("Prepare item representations.")
    item_embeddings = item_embeddings.sort_values("item_id").reset_index(drop=True)
    item_ids = item_embeddings["item_id"].to_numpy()
    item_vectors = np.stack(
        item_embeddings["embedding"].apply(
            lambda x: np.array(x, dtype=np.float32)
        ).values
    )
    item_vectors = normalize_rows(item_vectors)
    item_id_to_index = {int(item_id): idx for idx, item_id in enumerate(item_ids)}

    print("Compute interaction weights and popularity.")
    interactions["feedback_weight"] = compute_feedback_weight(interactions)
    popularity = (
        interactions.groupby("item_id")["feedback_weight"].sum().reindex(item_ids)
    ).fillna(0.0)
    popularity = popularity.to_numpy(dtype=np.float32)
    if popularity.max() > 0:
        popularity = popularity / popularity.max()

    seen_items = interactions.groupby("user_id")["item_id"].apply(set).to_dict()

    print("Build user profiles.")
    positive_interactions = interactions.loc[
        interactions["feedback_weight"] > 0, ["user_id", "item_id", "feedback_weight"]
    ]
    user_profiles = build_user_profiles(
        positive_interactions, item_vectors, item_id_to_index
    )
    del positive_interactions
    gc.collect()

    target_users: Iterable[int] = user_metadata["user_id"].unique().tolist()
    top_k = 20
    content_weight = 0.82
    popularity_weight = 0.18

    print("Generate recommendations.")
    recommendations: List[Tuple[int, int]] = []
    for uid in target_users:
        recommendations.extend(
            recommend_for_user(
                int(uid),
                user_profiles,
                item_vectors,
                popularity,
                item_ids,
                seen_items,
                top_k=top_k,
                content_weight=content_weight,
                popularity_weight=popularity_weight,
            )
        )

    result = pd.DataFrame(recommendations, columns=["user_id", "recs"])
    print(f"Save result to file {output_path}.")
    result.to_csv(output_path, index=False)
    print("File saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommender arguments.")
    parser.add_argument(
        "--input_path_interactions",
        type=str,
        required=True,
        help="Input path to train parquet file",
    )
    parser.add_argument(
        "--input_path_users",
        type=str,
        required=True,
        help="Input path to users demographic info parquet file",
    )
    parser.add_argument(
        "--input_path_items",
        type=str,
        required=True,
        help="Input path to items attributive info parquet file",
    )
    parser.add_argument(
        "--input_path_embeddings",
        type=str,
        required=True,
        help="Input path to items embeddings parquet file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path to csv with recommendations",
    )

    args = parser.parse_args()
    main(
        args.input_path_interactions,
        args.input_path_users,
        args.input_path_items,
        args.input_path_embeddings,
        args.output_path,
    )
