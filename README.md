# Predicting Fourth-Down Conversions in the NFL

Machine learning models that estimate the probability of converting a fourth down, built for CSCI 4380 (Data Mining) at the University of Georgia, Fall 2025.

Fourth downs are some of the highest-leverage calls in football, and analytics has long argued that coaches punt too often. This project trains three classifiers on nine seasons of NFL play-by-play data and compares a simple model-based "go for it" recommendation against what coaches actually chose.

## Results

Held-out test set (25% of plays, `random_state=42`):

| Model | Accuracy | ROC-AUC | Recall (success) |
|---|---|---|---|
| Logistic regression | 0.906 | 0.873 | 0.20 |
| Random forest | 0.911 | 0.898 | 0.29 |
| LightGBM (class-balanced) | 0.856 | **0.899** | **0.77** |

- The tree ensembles beat logistic regression on AUC, consistent with the nonlinear relationship between distance, field position and game state.
- LightGBM with balanced class weights gives up some accuracy to catch far more successful conversions (recall 0.77 vs. 0.20–0.29).
- Yards to go and field position are the most important features.
- The notebook also includes ROC curves, calibration curves and confusion matrices for all three models.

## Data

`data/fourth_down_2016_2024.csv` — 37,489 fourth-down plays from the 2016–2024 regular seasons, pulled from [nflfastR](https://www.nflfastr.com/) play-by-play data (via `nflreadr`). 21 columns including down, distance, field position, clock, score differential, EPA, win probability, play type and first-down outcome.

## Features

| Feature | Description |
|---|---|
| `yardline` | Field position from the offense's point of view |
| `ydstogo`, `log_ydstogo` | Yards needed for a first down (raw and log-transformed) |
| `score_diff` | Offense score minus defense score |
| `time_remaining` | Seconds left in the game |
| `off_epa_per_play` | Rolling 50-play mean EPA for the offense |
| `def_epa_allowed` | Rolling 50-play mean EPA allowed by the defense |

## Repository structure

```
├── data/
│   └── fourth_down_2016_2024.csv
├── notebooks/
│   └── fourth_down_model.ipynb   # cleaning, features, models, ROC/calibration
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/senaitpirani/nfl-fourth-down.git
cd nfl-fourth-down
pip install -r requirements.txt
jupyter notebook notebooks/fourth_down_model.ipynb
```

## Limitations and next steps

- **Target definition.** `success` is the `first_down` flag across *all* fourth-down plays, including punts and field goals, so only ~10% of rows are positive. Restricting training to actual run/pass attempts (where the conversion rate is about 53%) would make the model estimate conversion probability directly rather than partly learning which situations teams go for it in.
- **Rolling EPA features** are computed over the fourth-down rows in file order and include the current play's EPA; a cleaner version would sort by game and play and shift by one so each feature uses only prior plays.
- **Expected-value comparison** uses the realized EPA of each play and a constant punt value (−1.2), so the go/no-go recommendation is a rough illustration rather than a full decision model.
- **Random split.** A season-based split (train on 2016–2022, test on 2023–2024) would better reflect how a model would be used.
- **Model naming.** The ROC plot legend labels the LightGBM curve "XGBoost"; the model trained is LightGBM.
- No player- or formation-level data (e.g., QB sneaks, mobile quarterbacks), which drive many of the misclassifications.

## Tools

Python, pandas, NumPy, scikit-learn, LightGBM, matplotlib, seaborn, Jupyter
