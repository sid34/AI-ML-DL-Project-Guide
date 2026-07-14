"""
generate_dataset.py
--------------------
Creates a small, realistic, SYNTHETIC movie-ratings dataset so that
students can run the whole project offline without downloading anything.

Why synthetic and not a downloaded dataset (like MovieLens)?
1. No internet / login required -> works on any laptop instantly.
2. We control the size -> small enough to understand by eye.
3. We deliberately build in "taste patterns" (genre preferences) so that
   collaborative filtering has real signal to discover -- exactly what
   happens with real-world data, just at teaching scale.

Output files (written to this folder):
    movies.csv   -> movie_id, title, genre
    ratings.csv  -> user_id, movie_id, rating (1-5), timestamp
"""

import numpy as np
import pandas as pd
import os

# Reproducibility: always generates the SAME dataset every run
np.random.seed(42)

# ----------------------------------------------------------------
# STEP 1: Define movies and their genres
# ----------------------------------------------------------------
GENRES = ["Action", "Comedy", "Drama", "SciFi", "Romance"]

MOVIE_TITLES = [
    "Iron Sky", "Laugh Track", "Silent Tears", "Star Voyage", "Love in Paris",
    "Battle Zone", "Office Chaos", "The Long Goodbye", "Galaxy Run", "Two Hearts",
    "Rogue Squad", "Sitcom Life", "Mother's Grief", "Time Machine", "First Kiss",
    "War Front", "Stand-Up Nights", "The Divorce", "Alien Contact", "Wedding Bells",
    "Explosive Chase", "Comedy Central", "Broken Family", "Mars Colony", "Summer Romance",
    "Special Ops", "The Prank", "Losing Hope", "Robot Uprising", "Second Chance Love",
    "Tank Battalion", "Funny Business", "The Funeral", "Interstellar Drift", "Blind Date",
    "Combat Elite", "Class Clown", "Grief Counselor", "Space Pirates", "Love Letters",
    "Night Raid", "The Roast", "Family Ties", "Wormhole", "Valentine Surprise",
]

n_movies = len(MOVIE_TITLES)

# Assign each movie a "genre profile": how much it belongs to each genre (0-1)
# A movie can lean into more than one genre a little, but has ONE dominant genre.
movie_genre_matrix = np.random.dirichlet(alpha=[0.5] * len(GENRES), size=n_movies)

movies_df = pd.DataFrame({
    "movie_id": range(1, n_movies + 1),
    "title": MOVIE_TITLES,
    "genre": [GENRES[i] for i in movie_genre_matrix.argmax(axis=1)]
})

# ----------------------------------------------------------------
# STEP 2: Define users and their genre PREFERENCES (hidden taste)
# ----------------------------------------------------------------
n_users = 60
user_genre_pref = np.random.dirichlet(alpha=[0.5] * len(GENRES), size=n_users)

# ----------------------------------------------------------------
# STEP 3: Simulate ratings = (user taste . movie genre) + noise
# ----------------------------------------------------------------
rows = []
timestamp = 1_600_000_000  # arbitrary starting unix time

for u in range(n_users):
    # Each user rates a random subset of movies (simulate real sparse data)
    n_rated = np.random.randint(8, 25)  # each user rates 8-24 movies
    rated_movies = np.random.choice(range(n_movies), size=n_rated, replace=False)

    for m in rated_movies:
        # Base affinity score from dot product of taste vectors (0 to ~0.8)
        affinity = np.dot(user_genre_pref[u], movie_genre_matrix[m])
        # sqrt-scale spreads affinity more evenly across the 1-5 rating
        # range (raw dot products cluster near 0, which would make almost
        # every rating low) + add human-like noise on top
        rating = 1 + 4 * np.sqrt(affinity) + np.random.normal(0, 0.4)
        rating = int(np.clip(round(rating), 1, 5))

        rows.append({
            "user_id": u + 1,
            "movie_id": m + 1,
            "rating": rating,
            "timestamp": timestamp + np.random.randint(0, 10_000_000)
        })

ratings_df = pd.DataFrame(rows)

# ----------------------------------------------------------------
# STEP 4: Save to CSV
# ----------------------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__))
movies_df.to_csv(os.path.join(here, "movies.csv"), index=False)
ratings_df.to_csv(os.path.join(here, "ratings.csv"), index=False)

print(f"Generated {len(movies_df)} movies and {len(ratings_df)} ratings "
      f"from {n_users} users.")
print("Saved to: movies.csv, ratings.csv")
