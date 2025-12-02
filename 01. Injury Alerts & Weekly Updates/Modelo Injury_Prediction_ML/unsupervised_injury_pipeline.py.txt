#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsupervised injury risk pipeline for Defensive Backs Weekly Status Dataset (2020)
Python 3.11 • pandas • numpy • scikit-learn • openpyxl • matplotlib

What this script does:
1) Load the Excel dataset (read-only/streaming friendly) with openpyxl/pandas.
2) Select numeric, objective features for modeling (explicitly listed below).
3) Clean & standardize:
   - Convert decimal commas -> dots
   - Coerce to numeric
   - Median imputation
   - Per-player z-score normalization
4) Train Isolation Forest (unsupervised) and compute injury_risk_score in [0, 1].
5) Flag top-5% as high_risk_flag.
6) Calibrate with proxy events:
   - Pre-gap (missing next consecutive week)
   - Pre-injury-flip (previous_injury: none -> non-none next week)
7) Save scored dataset and calibration summaries.
8) Plot risk distribution (optional).

USAGE:
    python unsupervised_injury_pipeline.py /path/to/DBs_Weekly_Status_Dataset_2020.xlsx

Outputs (in same directory as input by default):
    - DBs_Weekly_Status_with_injury_risk.csv
    - unsupervised_risk_calibration_summary.csv
    - unsupervised_risk_calibration_per_player.csv
    - risk_histogram.png

NOTE:
- You can adjust feature list, thresholds, or IsolationForest params below.
"""

import sys
import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------

# Features used for modeling (numeric, objective)
FEATURES: List[str] = [
    "games_played",
    "solo_tackle",
    "assisted_tackles",
    "sack",
    "qb_hit",
    "interception",
    "fumble_forced",
    "total_snaps_played",
    "defense_snaps",
    "defense_snaps_%",
    "team_defense_snaps",
    "special_teams_snaps",
    "special_teams_snap_rate",
    "seasons_played",
    "index_defensive_effort",
    "fatigue_index",
]

ID_COLS: List[str] = ["player_id", "player_name", "team", "season", "season_type", "week"]

# Isolation Forest parameters
IF_PARAMS = dict(
    n_estimators=200,
    contamination=0.05,   # top ~5% anomalous
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)

# -----------------------------
# Helpers
# -----------------------------

def decimal_commas_to_dots(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            )
    return df

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def per_player_zscore(df: pd.DataFrame, group_key: str, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(group_key, group_keys=False)
    for c in cols:
        mu = g[c].transform("mean")
        sd = g[c].transform("std").replace(0, np.nan)
        out[c] = ((out[c] - mu) / sd).fillna(0.0)
    return out

def minmax_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)

# -----------------------------
# Load data
# -----------------------------

def load_excel_safely(xlsx_path: str) -> pd.DataFrame:
    """
    Try fast pandas read first. If it fails/times out in other contexts, fall back
    to openpyxl streaming. Here we use pandas directly (engine=openpyxl).
    """
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    return df

# -----------------------------
# Calibration utilities
# -----------------------------

def build_next_week_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds next_week and next_prev_injury for each (player_id, season) ordered by week.
    """
    req = ["player_id", "season", "week", "previous_injury"]
    for c in req:
        if c not in df.columns:
            df[c] = np.nan

    df = df.copy()
    df["previous_injury"] = df["previous_injury"].astype(str).str.strip().str.lower()
    df["previous_injury"] = df["previous_injury"].replace({"nan": "none", "": "none"})

    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    def _shift(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("week").reset_index(drop=True)
        g["next_week"] = g["week"].shift(-1)
        g["next_prev_injury"] = g["previous_injury"].shift(-1)
        return g

    df = df.groupby(["player_id", "season"], group_keys=False).apply(_shift)
    return df

def calibrate_proxies(scored: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Compute calibration summary and per-player snapshot:
    - Pre-gap: next_week > week + 1
    - Pre-injury-flip: prev_injury none -> non-none
    """
    df = scored.copy()
    # Define proxies
    df["gap_next_week"] = np.where(
        df["next_week"].notna() & (df["next_week"] > df["week"] + 1),
        1, 0
    ).astype(int)

    df["injury_flip_next_week"] = np.where(
        (df["previous_injury"] == "none") &
        df["next_prev_injury"].notna() &
        (df["next_prev_injury"] != "none"),
        1, 0
    ).astype(int)

    def _lift(a, b):
        return float(a) / float(b) if (b and b != 0) else np.nan

    summary_rows = []
    for label, mask in [
        ("Pre-gap (future gap exists)", df["gap_next_week"] == 1),
        ("Pre-injury-flip (none→non-none next week)", df["injury_flip_next_week"] == 1),
    ]:
        grp = df[mask]
        ctr = df[~mask]

        mean_event = grp["injury_risk_score"].mean()
        mean_non = ctr["injury_risk_score"].mean()
        hr_event = grp["high_risk_flag"].mean()
        hr_non = ctr["high_risk_flag"].mean()

        summary_rows.append({
            "Proxy event": label,
            "Rows (event)": int(mask.sum()),
            "Rows (non-event)": int((~mask).sum()),
            "Mean risk (event)": round(float(mean_event), 4) if not np.isnan(mean_event) else np.nan,
            "Mean risk (non-event)": round(float(mean_non), 4) if not np.isnan(mean_non) else np.nan,
            "Risk lift (event / non)": round(_lift(mean_event, mean_non), 3) if mean_non else np.nan,
            "High-risk rate (event)": round(float(hr_event), 4) if not np.isnan(hr_event) else np.nan,
            "High-risk rate (non)": round(float(hr_non), 4) if not np.isnan(hr_non) else np.nan,
            "High-risk lift (event / non)": round(_lift(hr_event, hr_non), 3) if hr_non else np.nan
        })

    summary = pd.DataFrame(summary_rows)

    per_player = (df.groupby("player_id")
                    .apply(lambda g: pd.Series({
                        "rows": len(g),
                        "pre_gap_rows": int((g["gap_next_week"] == 1).sum()),
                        "pre_flip_rows": int((g["injury_flip_next_week"] == 1).sum()),
                        "% high-risk overall": float(g["high_risk_flag"].mean()),
                        "% high-risk before gap": float(g.loc[g["gap_next_week"] == 1, "high_risk_flag"].mean()) if (g["gap_next_week"] == 1).any() else np.nan,
                        "% high-risk before flip": float(g.loc[g["injury_flip_next_week"] == 1, "high_risk_flag"].mean()) if (g["injury_flip_next_week"] == 1).any() else np.nan,
                    }))
                    .reset_index())

    return summary, per_player

# -----------------------------
# Main
# -----------------------------

def main(xlsx_path: str):
    out_dir = str(Path(xlsx_path).parent)

    # 1) Load
    df = load_excel_safely(xlsx_path)

    # 2) Keep only ID + FEATURES + previous_injury for calibration
    needed_cols = list(dict.fromkeys(ID_COLS + FEATURES + ["previous_injury"]))
    existing = [c for c in needed_cols if c in df.columns]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        print("WARNING: Missing columns:", missing)

    df = df[existing].copy()

    # 3) Clean numeric formatting & coerce
    df = decimal_commas_to_dots(df, FEATURES)
    df = coerce_numeric(df, FEATURES + ["season", "week"])

    # 4) Impute missing numeric values with medians
    for c in FEATURES:
        if c in df.columns:
            med = df[c].median(skipna=True)
            df[c] = df[c].fillna(med)

    # 5) Per-player z-score normalization (only on FEATURES)
    z = per_player_zscore(df[ID_COLS + FEATURES].copy(), "player_id", FEATURES)

    # 6) Isolation Forest training on z-scored features
    X = z[FEATURES].values
    iso = IsolationForest(**IF_PARAMS)
    iso.fit(X)

    # 7) Score -> injury_risk_score in [0,1]
    decision = iso.decision_function(X)  # larger = more normal
    risk = -decision                     # invert
    risk_01 = minmax_01(risk)
    df["injury_risk_score"] = np.round(risk_01, 4)

    # 8) High-risk flag: top 5%
    thr = float(pd.Series(risk_01).quantile(0.95))
    df["high_risk_flag"] = (risk_01 >= thr).astype(int)

    # 9) Build next-week fields for calibration
    df_cal = build_next_week_fields(df.copy())

    # 10) Calibration summaries
    summary, per_player = calibrate_proxies(df_cal)

    # 11) Save outputs
    out_scored = os.path.join(out_dir, "DBs_Weekly_Status_with_injury_risk.csv")
    out_summary = os.path.join(out_dir, "unsupervised_risk_calibration_summary.csv")
    out_per_player = os.path.join(out_dir, "unsupervised_risk_calibration_per_player.csv")
    df.to_csv(out_scored, index=False)
    summary.to_csv(out_summary, index=False)
    per_player.to_csv(out_per_player, index=False)

    print("Saved:")
    print(" -", out_scored)
    print(" -", out_summary)
    print(" -", out_per_player)

    # 12) Optional: risk histogram plot
    try:
        plt.figure(figsize=(7, 4))
        plt.hist(df["injury_risk_score"].values, bins=30)
        plt.xlabel("injury_risk_score")
        plt.ylabel("count")
        plt.title("Distribution of Unsupervised Injury Risk Scores")
        plot_path = os.path.join(out_dir, "risk_histogram.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        print(" -", plot_path)
    except Exception as e:
        print("Plotting failed:", e)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python unsupervised_injury_pipeline.py /path/to/DBs_Weekly_Status_Dataset_2020.xlsx")
        sys.exit(1)
    main(sys.argv[1])
