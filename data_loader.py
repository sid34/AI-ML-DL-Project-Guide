"""
data_loader.py
--------------
Step 1 of the pipeline: LOAD and PREPROCESS the raw data into a
user-item ratings MATRIX, which is the standard input format for
collaborative filtering.

Why a matrix?
    Rows    = users
    Columns = movies
    Cell    = rating given by that user to that movie (NaN if not rated)

This is the "design" step where raw (user_id, movie_id, rating) rows
get reshaped into the structure the algorithm actually needs.
"""

import pandas as pd
import numpy as np


def load_raw_data(ratings_path, movies_path):
    """Load the two CSV files into pandas DataFrames."""
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies


def build_user_item_matrix(ratings_df):
    """
    Convert the long-format ratings table:
        user_id, movie_id, rating
    into a wide-format USER-ITEM MATRIX:
        rows = users, columns = movies, values = ratings (NaN = unrated)
    """
    matrix = ratings_df.pivot_table(
        index="user_id", columns="movie_id", values="rating"
    )
    return matrix


def basic_stats(ratings_df, movies_df):
    """Print simple exploratory stats -- always inspect your data first."""
    n_users = ratings_df["user_id"].nunique()
    n_movies = movies_df["movie_id"].nunique()
    n_ratings = len(ratings_df)
    possible = n_users * n_movies
    sparsity = 100 * (1 - n_ratings / possible)

    print("----- Dataset summary -----")
    print(f"Users        : {n_users}")
    print(f"Movies       : {n_movies}")
    print(f"Ratings      : {n_ratings}")
    print(f"Matrix cells : {possible}")
    print(f"Sparsity     : {sparsity:.2f}%  "
          f"(this is normal -- real-world rating data is 95-99% empty)")
    print(f"Rating range : {ratings_df['rating'].min()} - {ratings_df['rating'].max()}")
    print(f"Avg rating   : {ratings_df['rating'].mean():.2f}")


if __name__ == "__main__":
    ratings, movies = load_raw_data("data/ratings.csv", "data/movies.csv")
    basic_stats(ratings, movies)
    matrix = build_user_item_matrix(ratings)
    print("\nUser-item matrix shape:", matrix.shape)
    print(matrix.iloc[:5, :6])  # peek at a small corner of the matrix
