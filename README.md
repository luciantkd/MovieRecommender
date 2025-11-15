# 🎬 Movie Recommender System (Collaborative Filtering + SVD)

A complete end-to-end **movie recommendation system** built using the **MovieLens 100k dataset** and the **Surprise** machine learning library.
This project compares multiple recommendation algorithms, tunes hyperparameters, and generates **Top-N personalised movie recommendations** with real movie titles.

---

## 🚀 Features

### ✅ **1. Algorithm Comparison (Phase 2)**

Compares:

* **User-based** Collaborative Filtering
* **Item-based** Collaborative Filtering
* Similarity metrics:

  * `cosine`
  * `msd`
  * `pearson`
* **Model-based** recommendation using **SVD**

### ✅ **2. Hyperparameter Tuning (Phase 3)**

Uses `GridSearchCV` to optimise:

* `n_epochs`
* `lr_all`
* `reg_all`

### ✅ **3. Top-N Movie Recommendations (Phase 4)**

Generates a personalised ranked list of movies a user has **not** rated yet:

* Shows predicted ratings using the tuned SVD model
* Includes real movie titles from `u.item`

### Example Output

```
=== Top 10 recommendations for user 196 ===

 1. Close Shave, A (1995) – predicted rating: 4.252
 2. Wrong Trousers, The (1993) – predicted rating: 4.220
 3. Schindler's List (1993) – predicted rating: 4.214
 4. Shawshank Redemption, The (1994) – predicted rating: 4.190
 5. Casablanca (1942) – predicted rating: 4.158
 6. Usual Suspects, The (1995) – predicted rating: 4.150
 7. Wallace & Gromit: Best of Aardman (1996) – predicted rating: 4.139
 8. Rear Window (1954) – predicted rating: 4.120
 9. 12 Angry Men (1957) – predicted rating: 4.117
10. Star Wars (1977) – predicted rating: 4.115
```

---

## 📦 Project Structure

```
MovieRecommender/
│
├── movie_recommender.py     # Main project script (all phases)
├── requirements.txt         # Python dependencies
├── u.data                   # Ratings from MovieLens 100k
├── u.item                   # Movie titles from MovieLens 100k
├── venv/                    # Virtual environment (ignored in git)
└── README.md                # This file
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/luciantkd/MovieRecommender.git
cd MovieRecommender
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Download MovieLens 100k data

Download from:
[https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/)

Extract and copy these files into your project folder:

* `u.data`
* `u.item`

---

## ▶️ Run the project

```bash
python movie_recommender.py
```

You’ll see:

* Phase 2: Algorithm comparison
* Phase 3: Hyperparameter tuning
* Phase 4: Top-N recommendations for a chosen user

---

## 🧠 How It Works

### 📍 1. Load & preprocess data

Using Surprise’s built-in MovieLens loader + Pandas for title lookup.

### 📍 2. Collaborative Filtering (Memory-based)

Using `KNNWithMeans` with:

* Centered cosine
* MSD
* Pearson correlation

User–user and item–item similarities tested.

### 📍 3. Matrix Factorisation (Model-based)

Using Singular Value Decomposition (SVD) — the classic Netflix Prize technique.

### 📍 4. Hyperparameter tuning

Evaluated via 3-fold cross-validation.

### 📍 5. Generate recommendations

Predict ratings for all unseen movies → sort → top-N.

---

## 📈 Results Summary

| Algorithm    | Type        | Similarity | RMSE                             |
| ------------ | ----------- | ---------- | -------------------------------- |
| KNNWithMeans | Item-based  | MSD        | ⭐ **0.9367** (best memory-based) |
| SVD          | Model-based | –          | 0.9380                           |

---


## 🏆 Why This Project Matters

This repository demonstrates:

* Understanding of **collaborative filtering**
* Use of **matrix factorisation**
* Practical ML workflows:

  * train/test split
  * model evaluation
  * hyperparameter tuning
* Real, usable **recommendation system logic**
* Clean, modular Python code

Perfect for your portfolio and interviews — especially for data roles, AI roles, or ML engineering internships.

---

## 👤 Author

**Lucian Procopciuc**
GitHub: [https://github.com/luciantkd](https://github.com/luciantkd)

---
