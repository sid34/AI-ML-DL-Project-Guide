"""
evaluate.py
-----------
Step 4 of the pipeline: TEST / VALIDATE the algorithm and MEASURE
how good its predictions are.

We use TWO kinds of metrics, because recommender systems are evaluated
differently from a normal classifier:

1) RATING-PREDICTION ACCURACY (how close were the predicted numbers?)
   - RMSE (Root Mean Squared Error): penalizes big mistakes more
   - MAE  (Mean Absolute Error): average size of the mistake

2) RECOMMENDATION QUALITY (did we recommend the RIGHT movies?)
   - Precision@K: of the top-K movies we recommended, how many did
     the user actually like?
   - Recall@K: of all the movies the user actually liked (in test data),
     how many did we manage to place in our top-K recommendations?
"""

import numpy as np
import pandas as pd


def rmse_mae(model, test_df):
    """
    Predict a rating for every (user, movie) pair in the TEST set,
    then compare to the real rating the user actually gave.
    """
    errors = []
    for _, row in test_df.iterrows():
        pred = model.predict(row["user_id"], row["movie_id"])
        actual = row["rating"]
        errors.append(pred - actual)

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    return rmse, mae


def precision_recall_at_k(model, train_df, test_df, movies_df, k=5, relevance_threshold=4):
    """
    For every user who has test ratings, check whether the movies WE
    would recommend (top-k, from movies not seen in training) match
    movies that user genuinely liked in the held-out test set
    (rating >= relevance_threshold).
    """
    precisions, recalls = [], []

    test_users = test_df["user_id"].unique()

    for user_id in test_users:
        if user_id not in model.user_item_matrix.index:
            continue  # user had too few ratings to even enter training matrix

        # Ground truth: movies this user actually liked, from the TEST set
        user_test = test_df[test_df["user_id"] == user_id]
        liked_movies = set(user_test[user_test["rating"] >= relevance_threshold]["movie_id"])

        if len(liked_movies) == 0:
            continue  # nothing relevant to evaluate against for this user

        # What would our model recommend?
        recs = model.recommend(user_id, movies_df, top_n=k)
        # Map back from title to movie_id for comparison
        recommended_ids = set(
            movies_df[movies_df["title"].isin(recs["title"])]["movie_id"]
        )

        hits = recommended_ids & liked_movies
        precisions.append(len(hits) / k)
        recalls.append(len(hits) / len(liked_movies))

    return float(np.mean(precisions)), float(np.mean(recalls))


def print_evaluation_report(model, train_df, test_df, movies_df, k=5):
    print("===== MODEL EVALUATION REPORT =====")
    rmse, mae = rmse_mae(model, test_df)
    print(f"RMSE (rating prediction error) : {rmse:.3f}  (lower is better, scale 1-5)")
    print(f"MAE  (rating prediction error) : {mae:.3f}  (lower is better, scale 1-5)")

    precision, recall = precision_recall_at_k(model, train_df, test_df, movies_df, k=k)
    print(f"Precision@{k} (top-{k} recs that were actually liked) : {precision:.3f}")
    print(f"Recall@{k}    (liked movies captured in top-{k})       : {recall:.3f}")
    print("====================================")
    return {"rmse": rmse, "mae": mae, "precision_at_k": precision, "recall_at_k": recall}
