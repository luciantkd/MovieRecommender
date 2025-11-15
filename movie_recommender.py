"""
Movie Recommender System using the MovieLens 100k dataset.

This script:
1. Compares different recommendation algorithms (user-based, item-based, SVD).
2. Tunes SVD hyperparameters using GridSearchCV.
3. Generates top-N movie recommendations with real movie titles.

The comments assume you are a beginner, so they explain both Python
and the machine learning ideas step by step.
"""

import os
import pandas as pd  # Pandas is a library for working with tables of data (like Excel sheets)

# surprise is a library made specifically for recommender systems
from surprise import Dataset, KNNWithMeans, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy


# ============================
# Phase 1 & 2: Core utilities
# ============================

def load_data(test_size: float = 0.25):
    """
    Load the MovieLens 100k dataset and split it into train and test sets.

    - "train" data is used to teach the model.
    - "test" data is used to check how well the model learned.

    test_size=0.25 means:
      25% of the data will be used for testing,
      75% will be used for training.
    """
    # Surprise can automatically download and load the 'ml-100k' dataset
    data = Dataset.load_builtin("ml-100k")

    # Split the data into training and testing sets
    trainset, testset = train_test_split(data, test_size=test_size)

    return trainset, testset


def user_based_recommender(trainset, testset, sim_name: str = "cosine"):
    """
    Build and evaluate a user-based KNNWithMeans recommender.

    "User-based" means:
      We compare users to other users and assume that similar users like similar movies.

    sim_name is the similarity metric we use to compare users:
      - 'cosine'
      - 'msd' (mean squared difference)
      - 'pearson'
      - 'pearson_baseline'
    """
    sim_options = {
        "name": sim_name,   # which similarity formula to use
        "user_based": True  # True means user-user collaborative filtering
    }

    # Create the KNNWithMeans algorithm with the chosen similarity options
    algo = KNNWithMeans(sim_options=sim_options)

    # Train the algorithm on the training set
    algo.fit(trainset)

    # Test it: the algorithm will predict ratings for the user-item pairs in the test set
    predictions = algo.test(testset)

    # Calculate the RMSE (Root Mean Squared Error) to measure accuracy
    rmse = accuracy.rmse(predictions, verbose=False)

    # Return both the trained model and its error
    return algo, rmse


def item_based_recommender(trainset, testset, sim_name: str = "cosine"):
    """
    Build and evaluate an item-based KNNWithMeans recommender.

    "Item-based" means:
      We compare movies to other movies and assume that similar movies
      will be liked by the same users.

    sim_name is the similarity metric (same options as above).
    """
    sim_options = {
        "name": sim_name,
        "user_based": False,  # False means item-item collaborative filtering
    }

    # Create the algorithm
    algo = KNNWithMeans(sim_options=sim_options)

    # Train it
    algo.fit(trainset)

    # Evaluate it on the test data
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)

    return algo, rmse


def svd_recommender(trainset, testset):
    """
    Build and evaluate a model-based recommender using SVD (Singular Value Decomposition).

    SVD is a matrix factorisation technique.
    Instead of directly comparing users and movies, it learns hidden patterns
    (called "latent factors") that explain user preferences.
    """
    # Create the SVD model with default parameters
    algo = SVD()

    # Train it
    algo.fit(trainset)

    # Evaluate it
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)

    return algo, rmse


def compare_algorithms():
    """
    Phase 2 – Compare different recommendation approaches:

    We compare:
      - User-based vs Item-based KNNWithMeans
      - Different similarity metrics: cosine, msd, pearson
      - KNNWithMeans (memory-based) vs SVD (model-based)

    The function prints out the RMSE for each approach,
    then returns the best model found.
    """
    # Get train and test sets
    trainset, testset = load_data()

    # The similarity metrics we want to test
    sim_metrics = ["cosine", "msd", "pearson"]

    results = []               # store (algorithm, type, sim, rmse)
    best_rmse = float("inf")   # start with a very large error
    best_model = None          # will hold the best-performing model
    best_model_label = None    # description of best model

    print("=== Phase 2: KNNWithMeans - User-based vs Item-based with different similarities ===\n")

    # Try user-based and item-based for each similarity metric
    for sim in sim_metrics:
        # 1) User-based collaborative filtering
        user_algo, user_rmse = user_based_recommender(trainset, testset, sim_name=sim)
        print(f"[USER-BASED]   sim={sim:<7} -> RMSE = {user_rmse:.4f}")
        results.append(("KNNWithMeans", "user", sim, user_rmse))

        # Check if this is the best model so far
        if user_rmse < best_rmse:
            best_rmse = user_rmse
            best_model = user_algo
            best_model_label = f"KNNWithMeans (user-based, sim={sim})"

        # 2) Item-based collaborative filtering
        item_algo, item_rmse = item_based_recommender(trainset, testset, sim_name=sim)
        print(f"[ITEM-BASED]   sim={sim:<7} -> RMSE = {item_rmse:.4f}")
        results.append(("KNNWithMeans", "item", sim, item_rmse))

        # Check if this is now the best model
        if item_rmse < best_rmse:
            best_rmse = item_rmse
            best_model = item_algo
            best_model_label = f"KNNWithMeans (item-based, sim={sim})"

    # Now also try SVD
    print("\n=== SVD (Model-based Matrix Factorisation) ===\n")

    svd_algo, svd_rmse = svd_recommender(trainset, testset)
    print(f"[SVD]                       -> RMSE = {svd_rmse:.4f}")
    results.append(("SVD", "n/a", "n/a", svd_rmse))

    if svd_rmse < best_rmse:
        best_rmse = svd_rmse
        best_model = svd_algo
        best_model_label = "SVD (model-based)"

    # Print a nice summary
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS (Phase 2)")
    print("=" * 60)
    for algo_name, kind, sim, rmse in results:
        print(f"{algo_name:12} | {kind:5} | sim={sim:7} | RMSE = {rmse:.4f}")

    print("\nBest model overall (Phase 2):")
    print(f"  {best_model_label} with RMSE = {best_rmse:.4f}")

    # Return:
    # - the best model
    # - its label (for logging)
    # - the trainset it was trained on
    return best_model, best_model_label, trainset


def demo_single_prediction(algo, trainset, user_id: str = "196", item_id: str = "302"):
    """
    Use the best Phase 2 model to make a single prediction for a user–item pair.

    This is like asking:
      "What rating would user 196 probably give to movie 302?"
    """
    # The Surprise library uses string IDs for users and items
    prediction = algo.predict(user_id, item_id)

    # prediction.est is the estimated rating (a float)
    print(
        f"\nExample prediction with best Phase 2 model:"
        f"\n  User {user_id} on item {item_id} -> estimated rating = {prediction.est:.3f}"
    )


# ============================
# Phase 3: Hyperparameter tuning
# ============================
def tune_svd_with_gridsearch():
    """
    Phase 3 – Hyperparameter tuning for SVD using GridSearchCV.

    Hyperparameters are "settings" for the algorithm, for example:
      - How many training epochs?
      - What learning rate?
      - How strong is regularisation?

    This function:
      1. Uses GridSearchCV to search for the best hyperparameters.
      2. Trains a new SVD model using the best settings on the full dataset.
      3. Returns the trained SVD model so we can use it later for recommendations.
    """
    print("\n\n=== Phase 3: Hyperparameter Tuning for SVD (GridSearchCV) ===\n")

    # For cross-validation, we pass the Dataset object directly (not split yet)
    data = Dataset.load_builtin("ml-100k")

    # Define the grid of parameters to try
    param_grid = {
        "n_epochs": [5, 10],      # number of times we go through the training data
        "lr_all": [0.002, 0.005], # learning rate: how big the training steps are
        "reg_all": [0.4, 0.6],    # regularisation: how much we penalise large weights (to avoid overfitting)
    }

    # GridSearchCV will train many SVD models with different parameter combinations
    gs = GridSearchCV(
        SVD,
        param_grid,
        measures=["rmse", "mae"],  # evaluate with RMSE and MAE
        cv=3,                       # 3-fold cross-validation
    )

    # Run the search
    gs.fit(data)

    # Print the best RMSE and the parameters that gave that result
    print(f"Best RMSE from GridSearchCV: {gs.best_score['rmse']:.4f}")
    print("Best parameters for RMSE:")
    print(gs.best_params["rmse"])

    # Extract best parameters from the search
    best_params = gs.best_params["rmse"]

    # Now train a final SVD model using ALL the data (no test split)
    full_data = Dataset.load_builtin("ml-100k")
    full_trainset = full_data.build_full_trainset()

    # Create a new SVD model with the best hyperparameters
    best_svd = SVD(
        n_epochs=best_params["n_epochs"],
        lr_all=best_params["lr_all"],
        reg_all=best_params["reg_all"],
    )

    # Fit on the full training set (this will be the model we actually use)
    best_svd.fit(full_trainset)

    # Return:
    # - the best SVD model
    # - the trainset it was trained on (if we need it later)
    # - the GridSearch object (in case we want to inspect it)
    return best_svd, full_trainset, gs


# ============================
# Phase 4: Top-N recommendations
# ============================

def load_movie_titles(file_path: str = "u.item"):
    """
    Load MovieLens movie titles from u.item using pandas.

    The u.item file comes with the MovieLens 100k dataset.
    Each line in u.item looks like:
      movie_id | title | release_date | ...

    We only need:
      - movie_id
      - title

    So this function returns a dictionary:
      { "movie_id_as_string": "Movie Title", ... }
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find {file_path}. "
            "Download the MovieLens 100k dataset and place 'u.item' in this folder."
        )

    df = pd.read_csv(
        file_path,
        sep="|",             # fields are separated by |
        header=None,         # there is no header row
        encoding="latin-1",  # encoding that works for these files
        usecols=[0, 1],      # we only want column 0 (id) and 1 (title)
        names=["movie_id", "title"],
    )

    # Convert movie_id to string so it matches Surprise's raw IDs
    df["movie_id"] = df["movie_id"].astype(str)

    # Build a dictionary: {movie_id: title}
    return dict(zip(df["movie_id"], df["title"]))


def load_ratings(file_path: str = "u.data"):
    """
    Load MovieLens ratings from u.data using pandas.

    Each line in u.data looks like:
      user_id  movie_id  rating  timestamp

    We return a pandas DataFrame with columns:
      user_id, movie_id, rating, timestamp
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find {file_path}. "
            "Download the MovieLens 100k dataset and place 'u.data' in this folder."
        )

    cols = ["user_id", "movie_id", "rating", "timestamp"]

    df = pd.read_csv(
        file_path,
        sep="\t",         # fields are separated by a tab character
        header=None,      # no header line in the file
        names=cols,
        engine="python",  # use Python engine to be safe with separators
    )

    return df


def get_top_n_recommendations(
    algo,
    ratings_df: pd.DataFrame,
    movie_titles: dict,
    user_id_raw: str,
    n: int = 10,
):
    """
    Phase 4 – For a given user, return top-N movie recommendations.

    We want:
      - Movies the user has NOT rated yet.
      - Sorted by highest predicted rating.

    Parameters:
      algo: a trained Surprise model (e.g. tuned SVD).
      ratings_df: DataFrame with all (user, movie, rating).
      movie_titles: dict {movie_id: title}
      user_id_raw: user ID as a string (e.g. "196")
      n: how many movies to recommend.
    """
    # Convert user_id to int because our ratings_df stores it as an integer
    user_id_int = int(user_id_raw)

    # 1. Find all movie_ids this user has already rated
    rated_movie_ids = set(
        ratings_df.loc[ratings_df["user_id"] == user_id_int, "movie_id"].astype(str)
    )

    # 2. Get the set of all movies we know about
    all_movie_ids = set(movie_titles.keys())

    # 3. Candidate movies are those the user has NOT rated yet
    candidate_ids = all_movie_ids - rated_movie_ids

    predictions = []

    # 4. Predict rating for each candidate movie
    for movie_id in candidate_ids:
        # algo.predict(user, item) returns a Prediction object
        pred = algo.predict(user_id_raw, movie_id)
        predictions.append(pred)

    # 5. Sort predictions by estimated rating (higher first)
    predictions.sort(key=lambda p: p.est, reverse=True)

    # 6. Get the top N predictions
    top_n = predictions[:n]

    # 7. Build a simple list of tuples: (movie_id, title, predicted_rating)
    results = []
    for pred in top_n:
        movie_id = pred.iid   # iid = item id (movie id)
        title = movie_titles.get(movie_id, "<Unknown title>")
        results.append((movie_id, title, pred.est))

    return results


def demo_top_n_recommendations(
    algo,
    ratings_df: pd.DataFrame,
    movie_titles: dict,
    user_id_raw: str = "196",
    n: int = 10,
):
    """
    Print top-N recommendations for a given user, using a trained Surprise algorithm.
    """
    top_n = get_top_n_recommendations(
        algo=algo,
        ratings_df=ratings_df,
        movie_titles=movie_titles,
        user_id_raw=user_id_raw,
        n=n,
    )

    print(f"\n=== Phase 4: Top {n} recommendations for user {user_id_raw} ===\n")
    for rank, (movie_id, title, est_rating) in enumerate(top_n, start=1):
        print(f"{rank:2d}. {title} (movie_id={movie_id}) – predicted rating: {est_rating:.3f}")


# ============================
# Main orchestration
# ============================

if __name__ == "__main__":
    """
    This block runs only when you execute this file directly, like:

        python movie_recommender.py

    It will:
      - Run Phase 2 (compare algorithms)
      - Run Phase 3 (tune SVD)
      - Run Phase 4 (print top-N recommendations)
    """

    # ----- Phase 2 – Compare baseline algorithms -----
    best_model, best_label, trainset = compare_algorithms()
    demo_single_prediction(best_model, trainset, user_id="196", item_id="302")

    # ----- Phase 3 – Tune SVD hyperparameters and fit best SVD on full data -----
    best_svd_model, svd_trainset, gs = tune_svd_with_gridsearch()

    # ----- Phase 4 – Load movie metadata and ratings -----
    movie_titles = load_movie_titles("u.item")
    ratings_df = load_ratings("u.data")

    # Use the tuned SVD model to recommend movies for user 196
    demo_top_n_recommendations(
        algo=best_svd_model,
        ratings_df=ratings_df,
        movie_titles=movie_titles,
        user_id_raw="196",
        n=10,
    )
