# 🎬 Movie Recommendation System — Collaborative Filtering
### A complete, working example for B.Tech (AI) first-year students

This repository is a **fully working, end-to-end AI project** built to show — in real code —
exactly how an AI/ML algorithm is **designed, trained, tested, validated, and measured.**

It solves one problem: *recommend movies to a user based on what similar users liked*,
using **User-Based Collaborative Filtering**, implemented from scratch (not just a
one-line library call) so every step is visible and explainable in a viva.

No downloads, logins, or GPUs required — the dataset is generated locally in seconds.

---

## 1. The Problem

> **Objective:** Recommend movies a user is likely to enjoy, based on the ratings of
> other users with similar taste.
> **Input:** A table of `(user_id, movie_id, rating)`.
> **Output:** A ranked list of movies the user hasn't seen yet, each with a predicted rating.
> **Success criteria:** Lower prediction error (RMSE/MAE) and higher Precision@K / Recall@K
> than a simple "recommend the most popular movies to everyone" baseline.

This is the classic **collaborative filtering** approach used by real systems like
Netflix and Amazon (in simplified form).

---

## 2. Project Structure

```
movie-recommendation-system/
├── data/
│   ├── generate_dataset.py         # creates the synthetic dataset (run once)
│   ├── movies.csv                  # movie_id, title, genre
│   └── ratings.csv                 # user_id, movie_id, rating, timestamp
├── src/
│   ├── data_loader.py              # Step 1: load + explore data
│   ├── train_test_split.py         # Step 2: split ratings per user
│   ├── collaborative_filtering.py  # Step 3: THE ALGORITHM (from scratch)
│   ├── baseline.py                 # Simple baseline model for comparison
│   └── evaluate.py                 # Step 4: RMSE, MAE, Precision@K, Recall@K
├── notebooks/
│   └── movie_recommendation_walkthrough.ipynb   # same pipeline, explained cell-by-cell
├── main.py                         # runs the full pipeline end-to-end
├── compare_models.py               # baseline vs main model comparison table
├── requirements.txt
└── README.md
```

---

## 3. How to Run

```bash
# 1. Clone this repo
git clone <your-github-repo-url>
cd movie-recommendation-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset (only needed once — files are already included, but you
#    can regenerate them, or change n_users / n_movies inside the script)
python data/generate_dataset.py

# 4. Run the full pipeline
python main.py

# 5. Compare baseline vs main model
python compare_models.py

# 6. OR open the guided notebook for a step-by-step explanation
jupyter notebook notebooks/movie_recommendation_walkthrough.ipynb
```

---

## 4. How the Dataset Was Built

Instead of downloading an external dataset, `data/generate_dataset.py` **simulates**
one that behaves like real rating data:

- Each **movie** has a genre profile (e.g., mostly Action, a little SciFi).
- Each **user** has a hidden taste profile (e.g., loves Comedy, dislikes Drama).
- A rating = how well the user's taste matches the movie's genre, **plus random noise**
  (because real people aren't perfectly predictable).
- Each user only rates a random subset of movies — just like real life, where nobody
  has watched everything (this is called **sparsity**).

This means the data has **real, learnable patterns**, exactly like real-world rating
data, but small enough to inspect by eye and understand completely.

---

## 5. How the Algorithm Works (Design)

**User-Based Collaborative Filtering** — core idea: *"users who agreed in the past
will agree in the future."*

| Step | What happens | Where in code |
|---|---|---|
| A. Build matrix | Reshape `(user, movie, rating)` rows into a user × movie matrix | `collaborative_filtering.py → fit()` |
| B. Similarity | Compute cosine similarity between every pair of users (after removing each user's personal rating bias) | `fit()` |
| C. Predict | For a target (user, movie), take a similarity-weighted average of ratings from the `k` most similar users who rated that movie | `predict()` |
| D. Recommend | Predict for every movie the user hasn't rated, sort, return the top N | `recommend()` |

The number of neighbors `k` is the main **hyperparameter** — try changing
`UserBasedCF(k_neighbors=10)` to `5` or `20` and see how results change.

---

## 6. How It's Trained

Training = calling `.fit(train_df)`. Unlike a neural network, there's no gradient
descent here — "training" means **building the user-item matrix and computing
similarities purely from the training portion of the data.** The test portion is
never shown to the model during this step.

```python
model = UserBasedCF(k_neighbors=10)
model.fit(train_df)
```

---

## 7. How It's Tested & Validated

We use a **leave-k-out per-user split** (`src/train_test_split.py`): for every user,
~20% of their ratings are hidden as the test set, and the model is trained only on the
remaining 80%. This mimics the real task — predicting ratings a user hasn't given yet.

```python
train_df, test_df = train_test_split_per_user(ratings_df, test_fraction=0.2)
```

The model **never sees test ratings during training** — this is what makes the
evaluation meaningful (avoiding "cheating" by testing on data it already memorized).

---

## 8. How Output Is Measured

Two families of metrics, both computed in `src/evaluate.py`:

| Metric | Question it answers | Good value |
|---|---|---|
| **RMSE** | How far off were the predicted rating *numbers*, on average (penalizes big misses)? | Lower is better |
| **MAE** | Same idea, but treats all errors equally | Lower is better |
| **Precision@K** | Of the top-K movies we recommended, how many did the user actually like? | Higher is better |
| **Recall@K** | Of all the movies the user actually liked, how many did we catch in our top-K? | Higher is better |

We also compare against a **baseline** (`src/baseline.py`) that just recommends the
overall most popular movies to everyone, ignoring individual taste. A good
collaborative filtering model should beat this baseline — that comparison is exactly
what `compare_models.py` prints out.

Example output:
```
Model                                  RMSE     MAE
Baseline (Popularity)                 0.775   0.632
Collaborative Filtering (main)        0.752   0.609
```

---

## 9. What to Change to Make This Your Own Project

- Swap the synthetic dataset for a real one (e.g., MovieLens) — the pipeline code
  doesn't need to change, only the CSV files.
- Try **item-based** collaborative filtering (swap the roles of users and movies).
- Try **matrix factorization** (SVD) using the `scikit-surprise` library.
- Add a **content-based** layer using movie genres, and build a hybrid recommender.
- Wrap `model.recommend()` in a simple **Streamlit** app for a live demo.

---

## 10. Suggested GitHub Repo Setup

```bash
git init
git add .
git commit -m "Initial commit: movie recommendation system (collaborative filtering)"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

Make sure `requirements.txt`, `README.md`, and this folder structure stay intact —
they're what makes the repo understandable to anyone (including your evaluator)
who opens it for the first time.
