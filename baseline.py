"""
baseline.py
-----------
A "baseline" model, as recommended in Phase 3 (Step 10) of the project
guide: ALWAYS build the simplest possible model first, so you have
something to compare your real algorithm against.

POPULARITY-BASED RECOMMENDER:
    Recommend the same movies to everyone -- whichever movies have the
    highest AVERAGE rating (with a minimum number of ratings, to avoid
    a movie with a single 5-star rating looking "best").

This ignores individual taste completely, so a good collaborative
filtering model should beat it on Precision@K / Recall@K.
"""

import pandas as pd


class PopularityRecommender:
    def __init__(self, min_ratings=3):
        self.min_ratings = min_ratings
        self.ranked_movies = None

    def fit(self, train_ratings_df):
        stats = train_ratings_df.groupby("movie_id")["rating"].agg(["mean", "count"])
        stats = stats[stats["count"] >= self.min_ratings]
        self.ranked_movies = stats.sort_values("mean", ascending=False)
        self.global_mean = train_ratings_df["rating"].mean()
        return self

    def predict(self, user_id, movie_id):
        """Same predicted rating for every user -- the movie's average."""
        if movie_id in self.ranked_movies.index:
            return float(self.ranked_movies.loc[movie_id, "mean"])
        return float(self.global_mean)

    def recommend(self, user_id, movies_df, top_n=5, seen_movie_ids=None):
        seen_movie_ids = seen_movie_ids or set()
        top = self.ranked_movies[~self.ranked_movies.index.isin(seen_movie_ids)].head(top_n)
        result = top.reset_index().rename(columns={"mean": "predicted_rating"})
        result = result.merge(movies_df[["movie_id", "title", "genre"]], on="movie_id")
        return result[["title", "genre", "predicted_rating"]]
