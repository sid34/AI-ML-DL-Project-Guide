"""
train_test_split.py
--------------------
Step 2 of the pipeline: SPLIT the ratings into a TRAIN set and a TEST set.

Why splitting is trickier for recommender systems than for a normal
classifier: we can't just split rows randomly and forget about it,
because we need EVERY user to still have some ratings left in the
training matrix (so the algorithm has something to learn their taste
from), while a portion of their ratings is hidden away for testing.

Strategy used here ("leave-k-out per user"):
    For each user, hide ~20% of their ratings as TEST data.
    The rest stays as TRAIN data.
This mimics the real task: "predict the ratings a user hasn't given yet."
"""

import numpy as np
import pandas as pd


def train_test_split_per_user(ratings_df, test_fraction=0.2, min_ratings_to_split=5,
                               random_state=42):
    """
    Split ratings into train/test sets, ensuring every user keeps at
    least a few ratings in the training set.

    Parameters
    ----------
    ratings_df : DataFrame with columns [user_id, movie_id, rating, timestamp]
    test_fraction : fraction of each user's ratings to hold out for testing
    min_ratings_to_split : users with fewer ratings than this are kept
                            ENTIRELY in the training set (too little data to test)
    """
    rng = np.random.RandomState(random_state)
    train_rows, test_rows = [], []

    for user_id, group in ratings_df.groupby("user_id"):
        group = group.sample(frac=1, random_state=random_state)  # shuffle
        n = len(group)

        if n < min_ratings_to_split:
            train_rows.append(group)
            continue

        n_test = max(1, int(n * test_fraction))
        test_rows.append(group.iloc[:n_test])
        train_rows.append(group.iloc[n_test:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


if __name__ == "__main__":
    ratings = pd.read_csv("data/ratings.csv")
    train_df, test_df = train_test_split_per_user(ratings)
    print(f"Total ratings : {len(ratings)}")
    print(f"Train ratings : {len(train_df)}  ({100*len(train_df)/len(ratings):.1f}%)")
    print(f"Test ratings  : {len(test_df)}  ({100*len(test_df)/len(ratings):.1f}%)")
