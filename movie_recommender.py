import os
import pandas as pd

from surprise import Dataset, KNNWithMeans, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy


# ============================
# Phase 1 & 2: Core utilities
# ============================

def load_data(test_size: float = 0.25):
    """
    Load the MovieLens 100k dataset and split into train/test.
    Used for Phase 2 comparisons.
    """
    data = Dataset.load_builtin("ml-100k")
    trainset, testset = train_test_split(data, test_size=test_size)
    return trainset, testset


def user_based_recommender(trainset, testset, sim_name: str = "cosine"):
    """
    Build and evaluate a user-based KNNWithMeans recommender.
    sim_name can be: 'cosine', 'msd', 'pearson', 'pearson_baseline'.
    """
    sim_options = {
        "name": sim_name,
        "user_based": True,  # user-user CF
    }

    algo = KNNWithMeans(sim_options=sim_options)
    algo.fit(trainset)

    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    return algo, rmse


def item_based_recommender(trainset, testset, sim_name: str = "cosine"):
    """
    Build and evaluate an item-based KNNWithMeans recommender.
    sim_name can be: 'cosine', 'msd', 'pearson', 'pearson_baseline'.
    """
    sim_options = {
        "name": sim_name,
        "user_based": False,  # item-item CF
    }

    algo = KNNWithMeans(sim_options=sim_options)
    algo.fit(trainset)

    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    return algo, rmse


def svd_recommender(trainset, testset):
    """
    Build and evaluate a model-based recommender using vanilla SVD.
    """
    algo = SVD()
    algo.fit(trainset)

    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    return algo, rmse


def compare_algorithms():
    """
    Phase 2 – Compare:
      - User-based vs Item-based KNNWithMeans
      - Similarities: cosine, msd, pearson
      - KNNWithMeans vs SVD
    """
    trainset, testset = load_data()

    sim_metrics = ["cosine", "msd", "pearson"]

    results = []
    best_rmse = float("inf")
    best_model = None
    best_model_label = None

    print("=== Phase 2: KNNWithMeans - User-based vs Item-based with different similarities ===\n")

    for sim in sim_metrics:
        # User-based
        user_algo, user_rmse = user_based_recommender(trainset, testset, sim_name=sim)
        print(f"[USER-BASED]   sim={sim:<7} -> RMSE = {user_rmse:.4f}")
        results.append(("KNNWithMeans", "user", sim, user_rmse))

        if user_rmse < best_rmse:
            best_rmse = user_rmse
            best_model = user_algo
            best_model_label = f"KNNWithMeans (user-based, sim={sim})"

        # Item-based
        item_algo, item_rmse = item_based_recommender(trainset, testset, sim_name=sim)
        print(f"[ITEM-BASED]   sim={sim:<7} -> RMSE = {item_rmse:.4f}")
        results.append(("KNNWithMeans", "item", sim, item_rmse))

        if item_rmse < best_rmse:
            best_rmse = item_rmse
            best_model = item_algo
            best_model_label = f"KNNWithMeans (item-based, sim={sim})"

    print("\n=== SVD (Model-based Matrix Factorisation) ===\n")

    svd_algo, svd_rmse = svd_recommender(trainset, testset)
    print(f"[SVD]                       -> RMSE = {svd_rmse:.4f}")
    results.append(("SVD", "n/a", "n/a", svd_rmse))

    if svd_rmse < best_rmse:
        best_rmse = svd_rmse
        best_model = svd_algo
        best_model_label = "SVD (model-based)"

    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS (Phase 2)")
    print("=" * 60)
    for algo_name, kind, sim, rmse in results:
        print(f"{algo_name:12} | {kind:5} | sim={sim:7} | RMSE = {rmse:.4f}")

    print("\nBest model overall (Phase 2):")
    print(f"  {best_model_label} with RMSE = {best_rmse:.4f}")

    return best_model, best_model_label, trainset


def demo_single_prediction(algo, trainset, user_id: str = "196", item_id: str = "302"):
    """
    Use the best Phase 2 model to make a single prediction for a user–item pair.
    """
    prediction = algo.predict(user_id, item_id)
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

    1. Use GridSearchCV to find best hyperparameters.
    2. Refit a fresh SVD model with those hyperparameters
       on the full MovieLens 100k trainset, so we can use
       it later for predictions and recommendations.
    """
    print("\n\n=== Phase 3: Hyperparameter Tuning for SVD (GridSearchCV) ===\n")

    # For CV (GridSearch) we pass the Dataset object directly
    data = Dataset.load_builtin("ml-100k")

    param_grid = {
        "n_epochs": [5, 10],
        "lr_all": [0.002, 0.005],
        "reg_all": [0.4, 0.6],
    }

    gs = GridSearchCV(
        SVD,
        param_grid,
        measures=["rmse", "mae"],
        cv=3,
    )

    gs.fit(data)

    print(f"Best RMSE from GridSearchCV: {gs.best_score['rmse']:.4f}")
    print("Best parameters for RMSE:")
    print(gs.best_params["rmse"])

    # Extract best parameters for RMSE
    best_params = gs.best_params["rmse"]

    # Now build a full trainset (all MovieLens 100k data)
    full_data = Dataset.load_builtin("ml-100k")
    full_trainset = full_data.build_full_trainset()

    # Create a fresh SVD model with the best params and fit it on the full trainset
    best_svd = SVD(
        n_epochs=best_params["n_epochs"],
        lr_all=best_params["lr_all"],
        reg_all=best_params["reg_all"],
    )

    best_svd.fit(full_trainset)

    # Return the trained model and the trainset if needed later
    return best_svd, full_trainset, gs



# ============================
# Phase 4: Top-N recommendations
# ============================

def load_movie_titles(file_path: str = "u.item"):
    """
    Load MovieLens movie titles from u.item using pandas.

    Expects the MovieLens 100k file 'u.item' to be present in the project folder.
    Format: movie_id | title | release_date | ... (pipe-separated).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find {file_path}. "
            "Download the MovieLens 100k dataset and place 'u.item' in this folder."
        )

    df = pd.read_csv(
        file_path,
        sep="|",
        header=None,
        encoding="latin-1",
        usecols=[0, 1],
        names=["movie_id", "title"],
    )
    df["movie_id"] = df["movie_id"].astype(str)
    return dict(zip(df["movie_id"], df["title"]))


def load_ratings(file_path: str = "u.data"):
    """
    Load MovieLens ratings from u.data using pandas.

    Format: user_id, movie_id, rating, timestamp (tab-separated).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find {file_path}. "
            "Download the MovieLens 100k dataset and place 'u.data' in this folder."
        )

    cols = ["user_id", "movie_id", "rating", "timestamp"]
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=cols,
        engine="python",
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
    Phase 4 – For a given user, return top-N movie recommendations
    (movies they haven't rated yet), with predicted ratings.

    Works with any Surprise algorithm trained on MovieLens (e.g. tuned SVD).
    """
    user_id_int = int(user_id_raw)

    # Movies this user has already rated
    rated_movie_ids = set(
        ratings_df.loc[ratings_df["user_id"] == user_id_int, "movie_id"].astype(str)
    )

    # All candidate movies = all movies minus those already rated
    all_movie_ids = set(movie_titles.keys())
    candidate_ids = all_movie_ids - rated_movie_ids

    predictions = []
    for movie_id in candidate_ids:
        # Surprise uses raw ids as strings
        pred = algo.predict(user_id_raw, movie_id)
        predictions.append(pred)

    # Sort by estimated rating, descending
    predictions.sort(key=lambda p: p.est, reverse=True)

    # Take top-N
    top_n = predictions[:n]

    results = []
    for pred in top_n:
        movie_id = pred.iid
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
    Print top-N recommendations for a given user, using a trained Surprise algo.
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
    # Phase 2 – Compare baseline algorithms
    best_model, best_label, trainset = compare_algorithms()
    demo_single_prediction(best_model, trainset, user_id="196", item_id="302")

    # Phase 3 – Tune SVD hyperparameters and fit best SVD on full data
    best_svd_model, svd_trainset, gs = tune_svd_with_gridsearch()

    # Phase 4 – Make it useful: Top-N recommendations with real titles
    movie_titles = load_movie_titles("u.item")
    ratings_df = load_ratings("u.data")

    # Use tuned SVD to recommend movies for user 196
    demo_top_n_recommendations(
        algo=best_svd_model,
        ratings_df=ratings_df,
        movie_titles=movie_titles,
        user_id_raw="196",
        n=10,
    )

