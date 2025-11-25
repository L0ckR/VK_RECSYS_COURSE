import pandas as pd
import lightgbm as lgb
import argparse
import gc
from sklearn.metrics.pairwise import cosine_similarity

def main(input_path_interactions: str,
         input_path_users: str,
         input_path_items: str,
         input_path_embeddings: str,
         output_path: str):
    # Загрузка данных
    print("Load data.")
    train_df = pd.read_parquet(input_path_interactions)
    user_metadata = pd.read_parquet(input_path_users)
    item_metadata = pd.read_parquet(input_path_items)
    item_embeddings = pd.read_parquet(input_path_embeddings)

    # Подготовка данных
    item_metadata = item_metadata.merge(item_embeddings, on=["item_id"], how="left")
    train_users = pd.DataFrame({"user_id": train_df["user_id"].unique()})
    train_items = pd.DataFrame({"item_id": train_df["item_id"].unique()})
    train_df["rating"] = train_df["timespent"] > 5

    # ---- Content-based features ----
    print("Prepare content-based features.")
    # Create item similarity matrix using cosine similarity of embeddings
    item_similarity = cosine_similarity(item_embeddings["embedding"].tolist())

    # Prepare train dataset with content-based features
    features = ["age", "gender", "author_id", "duration"]
    train = (train_df
             .merge(user_metadata, on=["user_id"], how="inner")
             .merge(item_metadata, on=["item_id"], how="inner")
             )[["user_id", "item_id", "age", "gender", "author_id", "duration", "rating"]]

    lgb_train = lgb.Dataset(train[features], train["rating"], categorical_feature=["age", "gender", "author_id"])

    # ---- Train LightGBM classifier ----
    print("Train classifier.")
    params = {
        'objective': 'binary',  # For binary classification
        'metric': 'auc',  # Evaluation metric
        'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
        'num_leaves': 31,  # Max number of leaves in one tree
        'learning_rate': 0.05,  # Step size shrinkage
        'feature_fraction': 0.9,  # Fraction of features considered at each split
        'verbose': -1  # Suppress verbose output during training
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,  # Number of boosting rounds (iterations)
                    )

    # ---- Hybrid Recommendations ----
    print("Prepare hybrid recommendations dataframe.")
    top = (train_df
           .groupby("item_id", as_index=False)["user_id"]
           .count()
           .sort_values("user_id", ascending=False)
           )
    sample_items = train_items.sample(frac=0.25)[["item_id"]]
    
    prediction_df = (pd
                     .merge(train_users, sample_items, how="cross")
                     .merge(user_metadata[["user_id", "age", "gender"]], on=["user_id"], how="inner")
                     .merge(item_metadata[["item_id", "author_id", "duration"]], on=["item_id"], how="inner")
                     .merge(train_df[["user_id", "item_id", "rating"]], on=["user_id", "item_id"], how="left")
                     .query("rating.isnull()", engine="python")
                     )

    print("Compute hybrid predictions.")
    y_pred_content = gbm.predict(prediction_df[features], num_iteration=gbm.best_iteration)

    # Adding content-based prediction (using cosine similarity)
    prediction_df["content_based_score"] = prediction_df["item_id"].apply(lambda x: item_similarity[x].mean())

    # Hybrid score combining LightGBM prediction and content-based score (you can adjust the weight)
    prediction_df["final_score"] = 0.7 * prediction_df["content_based_score"] + 0.3 * prediction_df["prediction"]

    # ---- Top-N Recommendations ----
    prediction_df["item_score"] = prediction_df.apply(lambda r: (r.item_id, r.final_score), axis=1)

    prediction_by_user = (prediction_df
                          .groupby("user_id", as_index=False)
                          .agg(scores_list=("item_score", list))
                          .reset_index()
                          )

    del prediction_df
    gc.collect()

    prediction_by_user["recs"] = prediction_by_user.apply(
        lambda r: [i[0] for i in sorted(r.scores_list, key=lambda x: x[1], reverse=True)[:10]], axis=1)

    result = prediction_by_user.explode("recs")

    print(f"Save result to file {output_path}.")
    result.to_csv(output_path, index=False)
    print("File saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommender arguments.")
    parser.add_argument("--input_path_interactions", type=str, required=True, help="Input path to train parquet file")
    parser.add_argument("--input_path_users", type=str, required=True, help="Input path to users demographic info parquet file")
    parser.add_argument("--input_path_items", type=str, required=True, help="Input path to items attributive info parquet file")
    parser.add_argument("--input_path_embeddings", type=str, required=True, help="Input path to items embeddings parquet file")
    parser.add_argument("--output_path", type=str, required=True, help="Output path to csv with recommendations")

    args = parser.parse_args()
    main(args.input_path_interactions,
         args.input_path_users,
         args.input_path_items,
         args.input_path_embeddings,
         args.output_path)
