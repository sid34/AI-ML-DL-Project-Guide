"""
collaborative_filtering.py
---------------------------
Step 3 of the pipeline: THE ALGORITHM ITSELF.

We implement USER-BASED COLLABORATIVE FILTERING from scratch, so
students can see every step instead of calling one library function.

CORE IDEA:
    "Users who agreed in the past will agree in the future."
    To predict how much user U will like movie M:
      1. Find users most SIMILAR to U (based on movies they rated the same way)
      2. Look at how those similar users rated movie M
      3. Take a similarity-WEIGHTED AVERAGE of their ratings -> that's our prediction

ALGORITHM DESIGN (the steps below map 1:1 to the code):
    fit()      -> Step A: build the user-item matrix from training data
                  Step B: compute user-user similarity matrix (cosine similarity)
    predict()  -> Step C: for a given (user, movie), predict the rating
    recommend()-> Step D: predict for ALL unrated movies, return the top-N
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    """A simple, transparent user-based collaborative filtering recommender."""

    def __init__(self, k_neighbors=10):
        """
        k_neighbors : how many "most similar users" to consider when
                      predicting a rating. This is the main hyperparameter
                      of the algorithm (similar to 'k' in k-NN).
        """
        self.k_neighbors = k_neighbors
        self.user_item_matrix = None      # rows=users, cols=movies
        self.user_similarity = None       # user x user similarity scores
        self.user_means = None            # average rating per user (for bias correction)

    # ------------------------------------------------------------
    # STEP A + B: TRAINING ("fit" the algorithm to the data)
    # ------------------------------------------------------------
    def fit(self, train_ratings_df):
        """
        Train the recommender:
          A) Build the user-item ratings matrix from the training ratings
          B) Compute cosine similarity between every pair of users
        """
        # A) Build matrix (rows=user, cols=movie), missing values = NaN
        self.user_item_matrix = train_ratings_df.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        )

        # Mean-center each user's ratings (removes the bias that some
        # users always rate high/low -- makes similarity more meaningful)
        self.user_means = self.user_item_matrix.mean(axis=1)
        matrix_filled = self.user_item_matrix.sub(self.user_means, axis=0).fillna(0)

        # B) Cosine similarity between every pair of users
        sim = cosine_similarity(matrix_filled.values)
        self.user_similarity = pd.DataFrame(
            sim,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        return self  # allows chaining, e.g. model = UserBasedCF().fit(train_df)

    # ------------------------------------------------------------
    # STEP C: PREDICTION for a single (user, movie) pair
    # ------------------------------------------------------------
    def predict(self, user_id, movie_id):
        """
        Predict the rating user_id would give to movie_id, using a
        similarity-weighted average of ratings from the k most similar
        users who HAVE rated that movie.
        """
        # Cold-start guards: unseen user or movie -> fall back to global average
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.stack().mean()
        if movie_id not in self.user_item_matrix.columns:
            return self.user_means.get(user_id, self.user_item_matrix.stack().mean())

        # Ratings every user gave to this movie
        movie_ratings = self.user_item_matrix[movie_id]
        raters = movie_ratings.dropna().index  # users who rated this movie
        raters = raters[raters != user_id]

        if len(raters) == 0:
            return self.user_means.get(user_id, self.user_item_matrix.stack().mean())

        # Similarity of our target user to each of those raters
        sims = self.user_similarity.loc[user_id, raters]

        # Keep only the top-k most similar (and only positive similarity)
        sims = sims[sims > 0].sort_values(ascending=False).head(self.k_neighbors)

        if len(sims) == 0:
            return self.user_means.get(user_id, self.user_item_matrix.stack().mean())

        neighbor_ratings = movie_ratings.loc[sims.index]
        neighbor_bias = neighbor_ratings - self.user_means.loc[sims.index]

        # Weighted average of the *bias-adjusted* neighbor ratings,
        # then add the target user's own average rating back
        weighted_sum = (sims.values * neighbor_bias.values).sum()
        sim_sum = np.abs(sims.values).sum()
        predicted = self.user_means.loc[user_id] + (weighted_sum / sim_sum)

        return float(np.clip(predicted, 1, 5))  # ratings must stay in [1,5]

    # ------------------------------------------------------------
    # STEP D: RECOMMENDATION -- rank all unseen movies for a user
    # ------------------------------------------------------------
    def recommend(self, user_id, movies_df, top_n=5):
        """Return the top_n movies (title + predicted rating) for a user."""
        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"User {user_id} not found in training data.")

        rated_movies = self.user_item_matrix.loc[user_id].dropna().index
        all_movies = self.user_item_matrix.columns
        unrated_movies = [m for m in all_movies if m not in rated_movies]

        predictions = [(m, self.predict(user_id, m)) for m in unrated_movies]
        predictions.sort(key=lambda x: x[1], reverse=True)
        top = predictions[:top_n]

        result = pd.DataFrame(top, columns=["movie_id", "predicted_rating"])
        result = result.merge(movies_df[["movie_id", "title", "genre"]], on="movie_id")
        return result[["title", "genre", "predicted_rating"]]
