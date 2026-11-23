"""
Task 1 - Regression: Delivery Time Estimation (UTS Machine Learning)
Reproducible single-command pipeline.

Usage:
    python kode/task_1_regresi.py --train data/regresi_train.csv.gz --test data/regresi_test.csv.gz --output prediksi_regresi.csv
"""

import argparse
import json
from pathlib import Path
import sys
import warnings

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
CAT_COLS = ["layanan", "mitra_kurir", "hub_asal", "kota_asal", "pulau_tujuan", "kota_tujuan"]


def clean_features(df: pd.DataFrame, route_dist_median: pd.Series | None = None, global_dist_median: float | None = None):
    """Feature engineering and data quality cleaning using ONLY pickup-time features."""
    out = df.copy()
    dt = pd.to_datetime(out["tgl_pickup"])
    out["bulan"] = dt.dt.month
    out["hari"] = dt.dt.day
    out["hari_dalam_minggu"] = dt.dt.dayofweek
    out["akhir_pekan"] = (dt.dt.dayofweek >= 5).astype(int)
    out["jam_sin"] = np.sin(2 * np.pi * out["jam_pickup"] / 24.0)
    out["jam_cos"] = np.cos(2 * np.pi * out["jam_pickup"] / 24.0)

    # Fix corrupt jarak_km sentinels (0 and 9999)
    out["jarak_corrupt_flag"] = ((out["jarak_km"] == 0) | (out["jarak_km"] == 9999)).astype(int)
    out.loc[out["jarak_corrupt_flag"] == 1, "jarak_km"] = np.nan

    route_key = out["kota_asal"].astype(str) + "_" + out["kota_tujuan"].astype(str)
    if route_dist_median is None:
        route_dist_median = out.groupby(route_key)["jarak_km"].median()
        global_dist_median = float(out["jarak_km"].median())

    mapped_dist = route_key.map(route_dist_median)
    out["jarak_km"] = out["jarak_km"].fillna(mapped_dist).fillna(global_dist_median)

    # Pickup-time interaction features
    out["berat_missing_flag"] = out["berat_kg"].isna().astype(int)
    out["log_jarak"] = np.log1p(np.clip(out["jarak_km"], 0, None))
    out["log_antrean"] = np.log1p(np.clip(out["antrean_hub_asal"], 0, None))
    out["log_nilai_barang"] = np.log1p(np.clip(out["nilai_barang"], 0, None))
    out["total_hujan"] = out["hujan_asal_mm"] + out["hujan_tujuan_mm"]
    out["densitas_paket"] = out["berat_kg"] / (out["volume_liter"] + 0.1)

    return out, route_dist_median, global_dist_median


def add_smoothed_target_encoding(train_df: pd.DataFrame, target: pd.Series, other_dfs: list[pd.DataFrame], cols: list[str], m: float = 50.0):
    """Out-of-fold / train-only smoothed target encoding for high-cardinality & unseen categories."""
    global_med = float(target.median())
    global_p90 = float(target.quantile(0.90))
    tr_out = train_df.copy()
    others_out = [d.copy() for d in other_dfs]

    for col in cols:
        stats = target.groupby(train_df[col]).agg(
            cnt="count",
            med="median",
            p90=lambda s: float(np.quantile(s, 0.90)),
        )
        smooth_med = (stats["cnt"] * stats["med"] + m * global_med) / (stats["cnt"] + m)
        smooth_p90 = (stats["cnt"] * stats["p90"] + m * global_p90) / (stats["cnt"] + m)

        # Out-of-fold for train to avoid leakage
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        oof_med = np.full(len(train_df), global_med, dtype=float)
        oof_p90 = np.full(len(train_df), global_p90, dtype=float)
        for tr_idx, va_idx in kf.split(train_df):
            s_fold = target.iloc[tr_idx].groupby(train_df[col].iloc[tr_idx]).agg(
                cnt="count", med="median", p90=lambda s: float(np.quantile(s, 0.90))
            )
            sm_m = (s_fold["cnt"] * s_fold["med"] + m * global_med) / (s_fold["cnt"] + m)
            sm_9 = (s_fold["cnt"] * s_fold["p90"] + m * global_p90) / (s_fold["cnt"] + m)
            oof_med[va_idx] = train_df[col].iloc[va_idx].map(sm_m).fillna(global_med).to_numpy()
            oof_p90[va_idx] = train_df[col].iloc[va_idx].map(sm_9).fillna(global_p90).to_numpy()

        tr_out[f"te_med_{col}"] = oof_med
        tr_out[f"te_p90_{col}"] = oof_p90
        for d in others_out:
            d[f"te_med_{col}"] = d[col].map(smooth_med).fillna(global_med).to_numpy()
            d[f"te_p90_{col}"] = d[col].map(smooth_p90).fillna(global_p90).to_numpy()

    return tr_out, others_out


def prepare_lgb_cats(dfs: list[pd.DataFrame], cat_cols: list[str]) -> list[pd.DataFrame]:
    """Align pandas category dtypes across train/val/test cleanly (unseen test categories -> NaN)."""
    out = [d.copy() for d in dfs]
    for col in cat_cols:
        cats = pd.Index(out[0][col].dropna().unique())
        for d in out:
            d[col] = pd.Categorical(d[col], categories=cats)
    return out


def generate_plots(err_layanan: pd.DataFrame, err_pulau: pd.DataFrame, top_shap: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # 1. Top 15 TreeSHAP features
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    top_plot = top_shap.head(15).iloc[::-1]
    ax.barh(top_plot["feature"], top_plot["mean_abs_shap"], color="steelblue")
    ax.set_title("Task 1: Top 15 Fitur Paling Berpengaruh (Mean |TreeSHAP| - Model P50)")
    ax.set_xlabel("Mean |SHAP value| (jam)")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_task1_shap.png", dpi=150)
    plt.close()

    # 2. Error breakdown by layanan & pulau_tujuan
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    el = err_layanan.sort_values("pinball_p50")
    axes[0].barh(el["layanan"], el["pinball_p50"], color="teal", label="Pinball P50")
    axes[0].barh(el["layanan"], el["pinball_p90"], color="coral", alpha=0.55, label="Pinball P90")
    axes[0].set_title("Error Breakdown per Layanan")
    axes[0].set_xlabel("Pinball Loss (jam)")
    axes[0].legend()

    ep = err_pulau.sort_values("pinball_p50")
    axes[1].barh(ep["pulau_tujuan"], ep["pinball_p50"], color="teal", label="Pinball P50")
    axes[1].barh(ep["pulau_tujuan"], ep["pinball_p90"], color="coral", alpha=0.55, label="Pinball P90")
    axes[1].set_title("Error Breakdown per Pulau Tujuan")
    axes[1].set_xlabel("Pinball Loss (jam)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_task1_error_breakdown.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Task 1 Regression Pipeline (UTS ML)")
    parser.add_argument("--train", type=str, default="data/regresi_train.csv.gz")
    parser.add_argument("--test", type=str, default="data/regresi_test.csv.gz")
    parser.add_argument("--output", type=str, default="prediksi_regresi.csv")
    parser.add_argument("--artifacts-dir", type=str, default="kode/outputs_task1")
    args = parser.parse_args()

    print("=" * 80)
    print("TASK 1 - DELIVERY TIME ESTIMATION (REPRODUCIBLE PIPELINE)")
    print("=" * 80)
    print(f"Python: {sys.version.split()[0]} | numpy: {np.__version__} | pandas: {pd.__version__} "
          f"| scikit-learn: {sklearn.__version__} | lightgbm: {lgb.__version__} | seed: {RANDOM_SEED}")

    df_train_raw = pd.read_csv(args.train)
    df_test_raw = pd.read_csv(args.test)
    df_train_raw["tgl_pickup_dt"] = pd.to_datetime(df_train_raw["tgl_pickup"])

    # ------------------------------------------------------------------
    # [1] DATA AUDIT: UNIT ERROR & LEAKAGE & CORRUPT DISTANCE
    # ------------------------------------------------------------------
    n_unit_err = int((df_train_raw["durasi_jam"] > 720).sum())
    n_dist_err = int(((df_train_raw["jarak_km"] == 0) | (df_train_raw["jarak_km"] == 9999)).sum())
    n_unseen_kurir = int((~df_test_raw["mitra_kurir"].isin(df_train_raw["mitra_kurir"].unique())).sum())
    print(f"\n[1] Data Audit:")
    print(f"    Train: {df_train_raw.shape} | Test: {df_test_raw.shape}")
    print(f"    Target unit error (durasi_jam > 720 -> recorded in minutes, 60x): {n_unit_err} rows ({n_unit_err/len(df_train_raw)*100:.3f}%)")
    print(f"    Corrupt jarak_km (0 or 9999): {n_dist_err} rows in train")
    print(f"    Unseen mitra_kurir in test (K13): {n_unseen_kurir} rows ({n_unseen_kurir/len(df_test_raw)*100:.2f}%)")

    # Fix target unit error ONLY on training data (keep raw validation target unmodified for honest comparison, also report clean val)
    df_train = df_train_raw.copy()
    df_train["durasi_jam_clean"] = np.where(df_train["durasi_jam"] > 720, df_train["durasi_jam"] / 60.0, df_train["durasi_jam"])

    # Time-based split: Train Jan-Jun 2025 | Val Jul-Aug 2025 (Jul = calibration, Aug = holdout evaluation)
    mask_tr = df_train["tgl_pickup_dt"] < "2025-07-01"
    mask_va = ~mask_tr
    tr_raw = df_train.loc[mask_tr].copy()
    va_raw = df_train.loc[mask_va].copy()

    tr_fe, route_med, glob_med = clean_features(tr_raw)
    va_fe, _, _ = clean_features(va_raw, route_med, glob_med)

    drop_base = ["id_paket", "tgl_pickup", "tgl_pickup_dt", "jumlah_scan", "durasi_jam", "durasi_jam_clean"]
    te_cols = ["kota_tujuan", "kota_asal", "mitra_kurir"]
    tr_fe, [va_fe] = add_smoothed_target_encoding(tr_fe, tr_fe["durasi_jam_clean"], [va_fe], te_cols)

    feat_cols = [c for c in tr_fe.columns if c not in drop_base]
    X_tr, X_va = prepare_lgb_cats([tr_fe[feat_cols], va_fe[feat_cols]], CAT_COLS)
    y_tr_clean = tr_fe["durasi_jam_clean"].to_numpy()
    y_va_raw = va_fe["durasi_jam"].to_numpy()
    y_va_clean = va_fe["durasi_jam_clean"].to_numpy()

    # ------------------------------------------------------------------
    # [2] MEASURE IMPACT OF LEAKAGE (jumlah_scan) & UNIT ERROR & RANDOM CV
    # ------------------------------------------------------------------
    print("\n[2] Auditing Leakage Impact & Time-Based vs Random CV:")
    # Leakage experiment: WITH jumlah_scan
    feat_leak = feat_cols + ["jumlah_scan"]
    X_tr_l, X_va_l = prepare_lgb_cats([tr_fe[feat_leak], va_fe[feat_leak]], CAT_COLS)
    m_leak = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=400, learning_rate=0.05, num_leaves=63, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1)
    m_leak.fit(X_tr_l, y_tr_clean)
    loss_leak_p50 = mean_pinball_loss(y_va_clean, m_leak.predict(X_va_l), alpha=0.5)

    # Random CV (3-fold on 60k subsample for speed) vs Time-Based Split
    full_fe, r_full, g_full = clean_features(df_train)
    full_fe, _ = add_smoothed_target_encoding(full_fe, full_fe["durasi_jam_clean"], [], te_cols)
    [X_full_all] = prepare_lgb_cats([full_fe[feat_cols]], CAT_COLS)
    y_full_clean = full_fe["durasi_jam_clean"].to_numpy()
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    cv_losses = []
    for f_tr, f_va in kf.split(X_full_all):
        m_cv = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=400, learning_rate=0.05, num_leaves=63, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1)
        m_cv.fit(X_full_all.iloc[f_tr], y_full_clean[f_tr])
        cv_losses.append(mean_pinball_loss(y_full_clean[f_va], m_cv.predict(X_full_all.iloc[f_va]), alpha=0.5))
    random_cv_p50 = float(np.mean(cv_losses))

    # ------------------------------------------------------------------
    # [3] MODEL COMPARISON (Baseline, Ridge Log-Target, HistGBR, Tuned LightGBM)
    # ------------------------------------------------------------------
    print("\n[3] Comparing Baseline, Ridge (Log-Target), HistGradientBoosting, and Tuned LightGBM:")
    results = []

    # 3a. Simple Baseline (global median & P90)
    b_p50 = np.full(len(y_va_clean), np.quantile(y_tr_clean, 0.50))
    b_p90 = np.full(len(y_va_clean), np.quantile(y_tr_clean, 0.90))
    results.append({
        "model": "Baseline (Global Quantile)",
        "pinball_p50_clean": mean_pinball_loss(y_va_clean, b_p50, alpha=0.5),
        "pinball_p90_clean": mean_pinball_loss(y_va_clean, b_p90, alpha=0.9),
        "coverage_p90_clean": float(np.mean(y_va_clean <= b_p90)),
        "pinball_p50_raw": mean_pinball_loss(y_va_raw, b_p50, alpha=0.5),
    })

    # 3b. Linear Model with log(durasi_jam) target transformation (Ridge)
    num_only = [c for c in feat_cols if c not in CAT_COLS]
    low_card_cats = ["layanan", "hub_asal", "pulau_tujuan", "mitra_kurir"]
    pre_lin = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_only),
        ("cat", OneHotEncoder(handle_unknown="ignore"), low_card_cats),
    ])
    ridge = Pipeline([("pre", pre_lin), ("reg", Ridge(alpha=10.0, random_state=RANDOM_SEED))])
    ridge.fit(tr_fe[num_only + low_card_cats], np.log(y_tr_clean))
    log_pred_va = ridge.predict(va_fe[num_only + low_card_cats])
    log_resid_tr = np.log(y_tr_clean) - ridge.predict(tr_fe[num_only + low_card_cats])
    r_p50 = np.exp(log_pred_va)
    r_p90 = np.exp(log_pred_va + np.quantile(log_resid_tr, 0.90))
    results.append({
        "model": "Ridge (Log-Target)",
        "pinball_p50_clean": mean_pinball_loss(y_va_clean, r_p50, alpha=0.5),
        "pinball_p90_clean": mean_pinball_loss(y_va_clean, r_p90, alpha=0.9),
        "coverage_p90_clean": float(np.mean(y_va_clean <= r_p90)),
        "pinball_p50_raw": mean_pinball_loss(y_va_raw, r_p50, alpha=0.5),
    })

    # 3c. HistGradientBoostingRegressor (Quantile)
    hgb_p50 = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=300, learning_rate=0.05, max_leaf_nodes=63, random_state=RANDOM_SEED)
    hgb_p90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, max_iter=300, learning_rate=0.05, max_leaf_nodes=63, random_state=RANDOM_SEED)
    hgb_p50.fit(tr_fe[num_only], y_tr_clean)
    hgb_p90.fit(tr_fe[num_only], y_tr_clean)
    hp50 = hgb_p50.predict(va_fe[num_only])
    hp90 = np.maximum(hgb_p90.predict(va_fe[num_only]), hp50)
    results.append({
        "model": "HistGBR (Quantile)",
        "pinball_p50_clean": mean_pinball_loss(y_va_clean, hp50, alpha=0.5),
        "pinball_p90_clean": mean_pinball_loss(y_va_clean, hp90, alpha=0.9),
        "coverage_p90_clean": float(np.mean(y_va_clean <= hp90)),
        "pinball_p50_raw": mean_pinball_loss(y_va_raw, hp50, alpha=0.5),
    })

    # 3d. LightGBM Quantile Tuning inside time-based validation
    best_cfg_p50, best_loss_p50, best_m50, best_p50_va = None, 1e9, None, None
    for cfg in [
        {"num_leaves": 31, "min_child_samples": 40, "colsample_bytree": 0.8},
        {"num_leaves": 63, "min_child_samples": 60, "colsample_bytree": 0.8},
        {"num_leaves": 95, "min_child_samples": 100, "colsample_bytree": 0.75},
    ]:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=0.5, n_estimators=1200, learning_rate=0.04,
            subsample=0.8, subsample_freq=1, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1, **cfg
        )
        m.fit(X_tr, y_tr_clean, eval_set=[(X_va, y_va_clean)], callbacks=[lgb.early_stopping(60, verbose=False)])
        pred = m.predict(X_va)
        loss = mean_pinball_loss(y_va_clean, pred, alpha=0.5)
        if loss < best_loss_p50:
            best_loss_p50, best_cfg_p50, best_m50, best_p50_va = loss, cfg, m, pred

    best_cfg_p90, best_loss_p90, best_m90, best_p90_va = None, 1e9, None, None
    for cfg in [
        {"num_leaves": 31, "min_child_samples": 60, "colsample_bytree": 0.8},
        {"num_leaves": 63, "min_child_samples": 100, "colsample_bytree": 0.8},
    ]:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=0.9, n_estimators=1000, learning_rate=0.04,
            subsample=0.8, subsample_freq=1, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1, **cfg
        )
        m.fit(X_tr, y_tr_clean, eval_set=[(X_va, y_va_clean)], callbacks=[lgb.early_stopping(60, verbose=False)])
        pred = np.maximum(m.predict(X_va), best_p50_va)
        loss = mean_pinball_loss(y_va_clean, pred, alpha=0.9)
        if loss < best_loss_p90:
            best_loss_p90, best_cfg_p90, best_m90, best_p90_va = loss, cfg, m, pred

    results.append({
        "model": f"LightGBM Tuned (leaves={best_cfg_p50['num_leaves']}/{best_cfg_p90['num_leaves']})",
        "pinball_p50_clean": best_loss_p50,
        "pinball_p90_clean": best_loss_p90,
        "coverage_p90_clean": float(np.mean(y_va_clean <= best_p90_va)),
        "pinball_p50_raw": mean_pinball_loss(y_va_raw, best_p50_va, alpha=0.5),
    })

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    print(f"\n    Leakage check (WITH jumlah_scan) Pinball P50: {loss_leak_p50:.4f} vs Clean: {best_loss_p50:.4f} (artificial drop: -{(best_loss_p50-loss_leak_p50)/best_loss_p50*100:.1f}%)")
    print(f"    Random 3-Fold CV Pinball P50: {random_cv_p50:.4f} vs Time-Based Val: {best_loss_p50:.4f}")

    # ------------------------------------------------------------------
    # [4] CONFORMAL CALIBRATION FOR P90 (Calibrate on July, Evaluate on August)
    # ------------------------------------------------------------------
    mask_jul = (va_fe["tgl_pickup_dt"].dt.month == 7).to_numpy()
    mask_aug = (va_fe["tgl_pickup_dt"].dt.month == 8).to_numpy()
    resid_jul = y_va_clean[mask_jul] - best_p90_va[mask_jul]
    n_cal = len(resid_jul)
    rank = int(np.ceil((n_cal + 1) * 0.90))
    delta_p90 = float(np.sort(resid_jul)[min(rank - 1, n_cal - 1)])

    cov_aug_before = float(np.mean(y_va_clean[mask_aug] <= best_p90_va[mask_aug]))
    loss_aug_before = float(mean_pinball_loss(y_va_clean[mask_aug], best_p90_va[mask_aug], alpha=0.9))
    p90_aug_cal = np.maximum(best_p90_va[mask_aug] + delta_p90, best_p50_va[mask_aug])
    cov_aug_after = float(np.mean(y_va_clean[mask_aug] <= p90_aug_cal))
    loss_aug_after = float(mean_pinball_loss(y_va_clean[mask_aug], p90_aug_cal, alpha=0.9))

    print(f"\n[4] Conformal Calibration (Calibrated on July n={n_cal:,}, Evaluated on August n={int(mask_aug.sum()):,}):")
    print(f"    Delta P90: {delta_p90:+.4f} hours")
    print(f"    August BEFORE -> Coverage: {cov_aug_before*100:.2f}%, Pinball P90: {loss_aug_before:.4f}")
    print(f"    August AFTER  -> Coverage: {cov_aug_after*100:.2f}%, Pinball P90: {loss_aug_after:.4f}")

    # ------------------------------------------------------------------
    # [5] ERROR BREAKDOWN & TREESHAP INTERPRETABILITY
    # ------------------------------------------------------------------
    va_eval = va_fe[["layanan", "pulau_tujuan", "mitra_kurir"]].copy()
    va_eval["y"] = y_va_clean
    va_eval["p50"] = best_p50_va
    va_eval["p90"] = np.maximum(best_p90_va + delta_p90, best_p50_va)

    def breakdown(col):
        rows = []
        for k, g in va_eval.groupby(col):
            rows.append({
                col: k,
                "n": len(g),
                "mae_p50": float(mean_absolute_error(g["y"], g["p50"])),
                "pinball_p50": float(mean_pinball_loss(g["y"], g["p50"], alpha=0.5)),
                "pinball_p90": float(mean_pinball_loss(g["y"], g["p90"], alpha=0.9)),
                "coverage_p90": float(np.mean(g["y"] <= g["p90"])),
            })
        return pd.DataFrame(rows).sort_values("pinball_p50")

    err_layanan = breakdown("layanan")
    err_pulau = breakdown("pulau_tujuan")
    err_kurir = breakdown("mitra_kurir")

    # TreeSHAP on 3,000 validation rows via LightGBM pred_contrib=True
    X_shap = X_va.sample(n=3000, random_state=RANDOM_SEED)
    shap_vals = best_m50.predict(X_shap, pred_contrib=True)[:, :-1]
    top_shap = pd.DataFrame({
        "feature": feat_cols,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    print("\n[5] Top 10 TreeSHAP Features (P50):")
    print(top_shap.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # [6] FINAL RETRAINING ON FULL JAN-AUG 2025 & PREDICT TEST
    # ------------------------------------------------------------------
    print("\n[6] Retraining Final LightGBM P50 & P90 on Full Training Set (250,000 rows)...")
    full_tr_fe, route_med_f, glob_med_f = clean_features(df_train)
    test_fe, _, _ = clean_features(df_test_raw, route_med_f, glob_med_f)
    full_tr_fe, [test_fe] = add_smoothed_target_encoding(full_tr_fe, full_tr_fe["durasi_jam_clean"], [test_fe], te_cols)
    X_full, X_test = prepare_lgb_cats([full_tr_fe[feat_cols], test_fe[feat_cols]], CAT_COLS)
    y_full = full_tr_fe["durasi_jam_clean"].to_numpy()

    final_m50 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.5, n_estimators=max(best_m50.best_iteration_, 200),
        learning_rate=0.04, subsample=0.8, subsample_freq=1, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1, **best_cfg_p50
    ).fit(X_full, y_full)

    final_m90 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.9, n_estimators=max(best_m90.best_iteration_, 200),
        learning_rate=0.04, subsample=0.8, subsample_freq=1, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1, **best_cfg_p90
    ).fit(X_full, y_full)

    test_p50 = np.clip(final_m50.predict(X_test), 1.0, None)
    test_p90 = np.maximum(final_m90.predict(X_test) + delta_p90, test_p50)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({
        "id_paket": df_test_raw["id_paket"],
        "p50": np.round(test_p50, 4),
        "p90": np.round(test_p90, 4),
    })
    assert len(sub) == len(df_test_raw), "Row count mismatch!"
    assert (sub["p90"] >= sub["p50"]).all(), "Quantile crossing detected!"
    assert sub.isna().sum().sum() == 0, "NaNs detected in submission!"
    sub.to_csv(out_path, index=False)
    print(f"    Saved valid submission: {out_path.resolve()} ({len(sub):,} rows) | p50 med={sub['p50'].median():.2f}h | p90 med={sub['p90'].median():.2f}h")

    # Save artifacts
    art_dir = Path(args.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(art_dir / "model_comparison.csv", index=False)
    err_layanan.to_csv(art_dir / "error_layanan.csv", index=False)
    err_pulau.to_csv(art_dir / "error_pulau.csv", index=False)
    err_kurir.to_csv(art_dir / "error_kurir.csv", index=False)
    top_shap.to_csv(art_dir / "shap_importance.csv", index=False)
    generate_plots(err_layanan, err_pulau, top_shap, art_dir / "figures")

    summary = {
        "python": sys.version.split()[0],
        "lightgbm": lgb.__version__,
        "random_seed": RANDOM_SEED,
        "n_unit_errors_fixed": n_unit_err,
        "n_corrupt_jarak_fixed": n_dist_err,
        "n_unseen_kurir_test": n_unseen_kurir,
        "pinball_p50_with_leakage": round(float(loss_leak_p50), 4),
        "pinball_p50_random_cv": round(float(random_cv_p50), 4),
        "pinball_p50_time_val": round(float(best_loss_p50), 4),
        "pinball_p90_time_val": round(float(best_loss_p90), 4),
        "delta_p90_conformal": round(float(delta_p90), 4),
        "aug_coverage_before": round(float(cov_aug_before), 4),
        "aug_coverage_after": round(float(cov_aug_after), 4),
        "aug_pinball_p90_before": round(float(loss_aug_before), 4),
        "aug_pinball_p90_after": round(float(loss_aug_after), 4),
        "best_iter_p50": int(best_m50.best_iteration_),
        "best_iter_p90": int(best_m90.best_iteration_),
    }
    (art_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE!")


if __name__ == "__main__":
    main()
