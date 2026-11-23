"""
Task 2 - Classification: Fraud Detection (UTS Machine Learning)
Reproducible single-command pipeline.

Usage:
    python kode/task_2_klasifikasi.py --train data/klasifikasi_train.csv.gz --test data/klasifikasi_test.csv.gz --output prediksi_klasifikasi.csv
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
import seaborn as sns
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
REVIEW_COST = 25000.0
CAT_COLS = ["kategori_merchant", "kanal", "tipe_perangkat", "email_domain"]


def compute_cost(y_true: np.ndarray, flags: np.ndarray, nominal: np.ndarray) -> dict:
    missed = (y_true == 1) & (flags == 0)
    fraud_loss = float(nominal[missed].sum())
    review_fee = float(flags.sum() * REVIEW_COST)
    return {
        "n_flagged": int(flags.sum()),
        "n_missed_fraud": int(missed.sum()),
        "n_caught_fraud": int(((y_true == 1) & (flags == 1)).sum()),
        "fraud_loss": fraud_loss,
        "review_fee": review_fee,
        "total_cost": fraud_loss + review_fee,
        "precision": float(precision_score(y_true, flags, zero_division=0)),
        "recall": float(recall_score(y_true, flags, zero_division=0)),
    }


def build_causal_behavioral_features(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build strictly causal (past-only) user & merchant behavioral features without future leakage.
    Combined train+test is sorted by timestamp for unsupervised behavioral aggregates (counts, nominal).
    For supervised merchant historical fraud rate, we enforce a strict 21-day confirmation lag
    using ONLY training labels from >= 21 days earlier.
    """
    tr = df_train.copy()
    te = df_test.copy()
    tr["_is_train"] = 1
    te["_is_train"] = 0
    te["is_fraud"] = np.nan
    tr["_orig_idx"] = np.arange(len(tr))
    te["_orig_idx"] = np.arange(len(te))

    comb = pd.concat([tr, te], ignore_index=True)
    comb["waktu_dt"] = pd.to_datetime(comb["waktu"])
    comb = comb.sort_values(["waktu_dt", "id_transaksi"]).reset_index(drop=True)

    # Time features
    comb["hari_dalam_minggu"] = comb["waktu_dt"].dt.dayofweek
    comb["akhir_pekan"] = (comb["hari_dalam_minggu"] >= 5).astype(int)
    comb["jam_sin"] = np.sin(2 * np.pi * comb["jam"] / 24.0)
    comb["jam_cos"] = np.cos(2 * np.pi * comb["jam"] / 24.0)
    comb["log_nominal"] = np.log1p(np.clip(comb["nominal"], 0, None))
    comb["first_tx_missing_flag"] = comb["menit_sejak_trx_terakhir"].isna().astype(int)

    # Strictly causal expanding user features (shifted by 1 within each id_pengguna)
    g_u = comb.groupby("id_pengguna", sort=False)
    comb["user_prior_tx_cnt"] = g_u.cumcount().astype(float)
    u_cum_nom = g_u["nominal"].cumsum() - comb["nominal"]
    comb["user_prior_mean_nom"] = np.where(comb["user_prior_tx_cnt"] > 0, u_cum_nom / np.maximum(comb["user_prior_tx_cnt"], 1.0), np.nan)
    comb["user_nom_vs_prior"] = comb["nominal"] / (comb["user_prior_mean_nom"] + 1.0)

    # Strictly causal expanding merchant features
    g_m = comb.groupby("id_merchant", sort=False)
    comb["merch_prior_tx_cnt"] = g_m.cumcount().astype(float)
    m_cum_nom = g_m["nominal"].cumsum() - comb["nominal"]
    comb["merch_prior_mean_nom"] = np.where(comb["merch_prior_tx_cnt"] > 0, m_cum_nom / np.maximum(comb["merch_prior_tx_cnt"], 1.0), np.nan)
    comb["merch_nom_vs_prior"] = comb["nominal"] / (comb["merch_prior_mean_nom"] + 1.0)

    # User-Merchant repeat interaction (past-only)
    comb["user_merch_prior_cnt"] = comb.groupby(["id_pengguna", "id_merchant"], sort=False).cumcount().astype(float)
    comb["is_new_merchant_for_user"] = (comb["user_merch_prior_cnt"] == 0).astype(int)

    # Strictly 21-day delayed merchant historical fraud rate (daily cumulative shifted by 21 days)
    comb["tgl"] = comb["waktu_dt"].dt.floor("D")
    tr_daily = (
        comb.loc[comb["_is_train"] == 1]
        .groupby(["id_merchant", "tgl"], as_index=False)
        .agg(day_tx=("is_fraud", "size"), day_fr=("is_fraud", "sum"))
        .sort_values(["id_merchant", "tgl"])
    )
    tr_daily["cum_tx"] = tr_daily.groupby("id_merchant")["day_tx"].cumsum()
    tr_daily["cum_fr"] = tr_daily.groupby("id_merchant")["day_fr"].cumsum()
    # Available date = tgl + 21 days (strictly mature!)
    tr_daily["avail_tgl"] = tr_daily["tgl"] + pd.Timedelta(days=21)
    tr_daily = tr_daily[["id_merchant", "avail_tgl", "cum_tx", "cum_fr"]].sort_values("avail_tgl")

    comb = comb.sort_values("tgl")
    merged = pd.merge_asof(
        comb[["id_merchant", "tgl"]],
        tr_daily,
        left_on="tgl",
        right_on="avail_tgl",
        by="id_merchant",
        direction="backward",
        allow_exact_matches=True,
    )
    prior_rate = 0.007
    m_smooth = 30.0
    c_tx = merged["cum_tx"].fillna(0.0).to_numpy()
    c_fr = merged["cum_fr"].fillna(0.0).to_numpy()
    comb["merch_delayed_fraud_rate"] = (c_fr + m_smooth * prior_rate) / (c_tx + m_smooth)

    # Restore train/test in original order
    tr_out = comb.loc[comb["_is_train"] == 1].sort_values("_orig_idx").drop(columns=["_is_train", "_orig_idx", "tgl"]).reset_index(drop=True)
    te_out = comb.loc[comb["_is_train"] == 0].sort_values("_orig_idx").drop(columns=["_is_train", "_orig_idx", "tgl", "is_fraud"]).reset_index(drop=True)
    return tr_out, te_out


def align_cats(dfs: list[pd.DataFrame], cat_cols: list[str]) -> list[pd.DataFrame]:
    out = [d.copy() for d in dfs]
    for c in cat_cols:
        cats = pd.Index(out[0][c].fillna("__MISSING__").astype(str).unique())
        for d in out:
            d[c] = pd.Categorical(d[c].fillna("__MISSING__").astype(str), categories=cats)
    return out


def generate_plots(weekly_df: pd.DataFrame, shift_kanal: pd.DataFrame, y_eval: np.ndarray, p_raw: np.ndarray, p_platt: np.ndarray, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # 1. Delayed label (right-censoring) weekly fraud rate
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    tail_wk = weekly_df.tail(16).copy()
    x_labels = [str(w).split("/")[0] for w in tail_wk.index]
    colors = ["crimson" if i >= len(tail_wk) - 3 else "steelblue" for i in range(len(tail_wk))]
    ax.bar(range(len(tail_wk)), tail_wk["rate_pct"], color=colors)
    ax.set_xticks(range(len(tail_wk)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Audit Delayed Label (Right-Censoring): Fraud Rate Mingguan (Merah = 3 Minggu Terakhir Belum Matang)")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_task2_delayed_labels.png", dpi=150)
    plt.close()

    # 2. Pattern shift by kanal across months (Jan-Jul + Aug 1-10)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for kanal, grp in shift_kanal.groupby("kanal"):
        ax.plot(grp["bulan"], grp["fraud_rate_pct"], marker="o", label=kanal)
    ax.set_title("Pergeseran Pola Fraud per Kanal Pembayaran (Jan – 10 Agu 2025)")
    ax.set_xlabel("Bulan (2025)")
    ax.set_ylabel("Fraud Rate (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_task2_pattern_shift.png", dpi=150)
    plt.close()

    # 3. Calibration reliability curve
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    prob_true_r, prob_pred_r = calibration_curve(y_eval, p_raw, n_bins=10, strategy="quantile")
    prob_true_p, prob_pred_p = calibration_curve(y_eval, p_platt, n_bins=10, strategy="quantile")
    ax.plot([0, max(prob_pred_r.max(), prob_true_r.max())], [0, max(prob_pred_r.max(), prob_true_r.max())], "k--", label="Perfect Calibration")
    ax.plot(prob_pred_r, prob_true_r, marker="o", label="Raw LightGBM")
    ax.plot(prob_pred_p, prob_true_p, marker="s", label="Platt Scaling")
    ax.set_title("Kurva Kalibrasi Probabilitas (Holdout Matang 1 Jul – 10 Agu)")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Fraud Frequency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_task2_calibration.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Task 2 Fraud Detection Pipeline (UTS ML)")
    parser.add_argument("--train", type=str, default="data/klasifikasi_train.csv.gz")
    parser.add_argument("--test", type=str, default="data/klasifikasi_test.csv.gz")
    parser.add_argument("--output", type=str, default="prediksi_klasifikasi.csv")
    parser.add_argument("--artifacts-dir", type=str, default="kode/outputs_task2")
    args = parser.parse_args()

    print("=" * 80)
    print("TASK 2 - FRAUD DETECTION (REPRODUCIBLE PIPELINE)")
    print("=" * 80)
    print(f"Python: {sys.version.split()[0]} | numpy: {np.__version__} | pandas: {pd.__version__} "
          f"| scikit-learn: {sklearn.__version__} | lightgbm: {lgb.__version__} | seed: {RANDOM_SEED}")

    df_train_raw = pd.read_csv(args.train)
    df_test_raw = pd.read_csv(args.test)

    print(f"\n[1] Building strictly causal user & merchant behavioral features...")
    tr_fe, te_fe = build_causal_behavioral_features(df_train_raw, df_test_raw)

    # Audit weekly delayed-label right-censoring
    tr_fe["week"] = tr_fe["waktu_dt"].dt.to_period("W")
    weekly_df = tr_fe.groupby("week").agg(n=("is_fraud", "size"), fraud=("is_fraud", "sum"), rate=("is_fraud", "mean"))
    weekly_df["rate_pct"] = (weekly_df["rate"] * 100).round(3)
    print("    Last 5 weeks fraud rate (evidence of ~21-day confirmation delay):")
    print(weekly_df.tail(5).to_string())

    # Pattern shift table (Jan - Aug 10 mature data)
    mature_all = tr_fe[tr_fe["waktu_dt"] < "2025-08-11"].copy()
    mature_all["bulan"] = mature_all["waktu_dt"].dt.month
    shift_kanal = (
        mature_all.groupby(["bulan", "kanal"], as_index=False)
        .agg(n=("is_fraud", "size"), fraud=("is_fraud", "sum"))
    )
    shift_kanal["fraud_rate_pct"] = shift_kanal["fraud"] / shift_kanal["n"] * 100

    # Time-based mature splits:
    # Train: Jan 1 - May 31 | Calib/Tune: Jun 1 - Jun 30 | Mature Holdout: Jul 1 - Aug 10 | Censored check: Aug 11 - Aug 31
    m_tr = tr_fe["waktu_dt"] < "2025-06-01"
    m_cal = (tr_fe["waktu_dt"] >= "2025-06-01") & (tr_fe["waktu_dt"] < "2025-07-01")
    m_val = (tr_fe["waktu_dt"] >= "2025-07-01") & (tr_fe["waktu_dt"] < "2025-08-11")
    m_cens = tr_fe["waktu_dt"] >= "2025-08-11"

    drop_cols = ["id_transaksi", "id_pengguna", "id_merchant", "waktu", "waktu_dt", "week", "flag_review_manual", "is_fraud"]
    feat_cols = [c for c in tr_fe.columns if c not in drop_cols]

    X_tr, X_cal, X_val, X_cens = align_cats(
        [tr_fe.loc[m_tr, feat_cols], tr_fe.loc[m_cal, feat_cols], tr_fe.loc[m_val, feat_cols], tr_fe.loc[m_cens, feat_cols]],
        CAT_COLS,
    )
    y_tr = tr_fe.loc[m_tr, "is_fraud"].to_numpy(dtype=int)
    y_cal = tr_fe.loc[m_cal, "is_fraud"].to_numpy(dtype=int)
    y_val = tr_fe.loc[m_val, "is_fraud"].to_numpy(dtype=int)
    y_cens = tr_fe.loc[m_cens, "is_fraud"].to_numpy(dtype=int)
    nom_cal = tr_fe.loc[m_cal, "nominal"].to_numpy(dtype=float)
    nom_val = tr_fe.loc[m_val, "nominal"].to_numpy(dtype=float)

    print(f"\n[2] Mature Time-Based Splits (features={len(feat_cols)}):")
    print(f"    Train (Jan-May): {len(y_tr):,} ({y_tr.mean()*100:.3f}% fraud)")
    print(f"    Calib (Jun)    : {len(y_cal):,} ({y_cal.mean()*100:.3f}% fraud)")
    print(f"    Val (Jul-Aug10): {len(y_val):,} ({y_val.mean()*100:.3f}% fraud) [MATURE]")
    print(f"    Censored (Aug11-31): {len(y_cens):,} ({y_cens.mean()*100:.3f}% fraud) [EXCLUDED FROM TRAINING/VAL]")

    # ------------------------------------------------------------------
    # [3] LEAKAGE IMPACT AUDIT (WITH vs WITHOUT flag_review_manual)
    # ------------------------------------------------------------------
    feat_leak = feat_cols + ["flag_review_manual"]
    X_tr_l, X_cal_l, X_val_l = align_cats(
        [tr_fe.loc[m_tr, feat_leak], tr_fe.loc[m_cal, feat_leak], tr_fe.loc[m_val, feat_leak]], CAT_COLS
    )
    m_leak = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=63, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1)
    m_leak.fit(X_tr_l, y_tr, eval_set=[(X_cal_l, y_cal)], callbacks=[lgb.early_stopping(40, verbose=False)])
    prauc_leak = float(average_precision_score(y_val, m_leak.predict_proba(X_val_l)[:, 1]))

    # ------------------------------------------------------------------
    # [4] COMPARE 3 IMBALANCE APPROACHES + TIME-DECAY ADAPTATION
    # ------------------------------------------------------------------
    print("\n[3] Comparing Imbalance Approaches & Time-Decay Pattern-Shift Adaptation on Mature Holdout (Jul 1 - Aug 10):")
    imb_rows = []

    # Approach 1: Default LightGBM (Logloss) + Bayes Cost Decision
    m_base = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, num_leaves=63, min_child_samples=80,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1
    )
    m_base.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)], callbacks=[lgb.early_stopping(50, verbose=False)])
    p_base_cal = m_base.predict_proba(X_cal)[:, 1]
    p_base_val = m_base.predict_proba(X_val)[:, 1]
    p_base_cens = m_base.predict_proba(X_cens)[:, 1]
    c_base = compute_cost(y_val, (p_base_val * nom_val > REVIEW_COST).astype(int), nom_val)
    imb_rows.append({"approach": "Default LightGBM + Cost Rule", "pr_auc": average_precision_score(y_val, p_base_val), "brier": brier_score_loss(y_val, p_base_val), **c_base})

    # Approach 2: Class Weighting (scale_pos_weight)
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    m_wt = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, num_leaves=63, min_child_samples=80,
        scale_pos_weight=spw, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1
    )
    m_wt.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)], callbacks=[lgb.early_stopping(50, verbose=False)])
    p_wt_val = m_wt.predict_proba(X_val)[:, 1]
    c_wt = compute_cost(y_val, (p_wt_val * nom_val > REVIEW_COST).astype(int), nom_val)
    imb_rows.append({"approach": f"Class Weighting (spw={spw:.1f})", "pr_auc": average_precision_score(y_val, p_wt_val), "brier": brier_score_loss(y_val, p_wt_val), **c_wt})

    # Approach 3: Random Undersampling (1:10 ratio)
    rng = np.random.RandomState(RANDOM_SEED)
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = rng.choice(np.where(y_tr == 0)[0], size=min(len(pos_idx) * 10, (y_tr == 0).sum()), replace=False)
    rus_idx = np.sort(np.concatenate([pos_idx, neg_idx]))
    m_rus = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, num_leaves=63, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1
    )
    m_rus.fit(X_tr.iloc[rus_idx], y_tr[rus_idx], eval_set=[(X_cal, y_cal)], callbacks=[lgb.early_stopping(50, verbose=False)])
    p_rus_val = m_rus.predict_proba(X_val)[:, 1]
    c_rus = compute_cost(y_val, (p_rus_val * nom_val > REVIEW_COST).astype(int), nom_val)
    imb_rows.append({"approach": "Random Undersampling (1:10)", "pr_auc": average_precision_score(y_val, p_rus_val), "brier": brier_score_loss(y_val, p_rus_val), **c_rus})

    # Approach 4: Time-Decay Sample Weighting (Pattern-Shift Adaptation, half-life = 90 days)
    tr_days_ago = (tr_fe.loc[m_tr, "waktu_dt"].max() - tr_fe.loc[m_tr, "waktu_dt"]).dt.total_seconds() / 86400.0
    sw_decay = np.exp(-np.log(2) * tr_days_ago.to_numpy() / 90.0)
    m_dec = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, num_leaves=63, min_child_samples=80,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1
    )
    m_dec.fit(X_tr, y_tr, sample_weight=sw_decay, eval_set=[(X_cal, y_cal)], callbacks=[lgb.early_stopping(50, verbose=False)])
    p_dec_cal = m_dec.predict_proba(X_cal)[:, 1]
    p_dec_val = m_dec.predict_proba(X_val)[:, 1]
    c_dec = compute_cost(y_val, (p_dec_val * nom_val > REVIEW_COST).astype(int), nom_val)
    imb_rows.append({"approach": "Time-Decay Weighted LightGBM", "pr_auc": average_precision_score(y_val, p_dec_val), "brier": brier_score_loss(y_val, p_dec_val), **c_dec})

    df_imb = pd.DataFrame(imb_rows)
    print(df_imb[["approach", "pr_auc", "brier", "n_flagged", "n_missed_fraud", "total_cost", "precision", "recall"]].to_string(index=False))
    print(f"\n    Leakage check (WITH flag_review_manual) PR-AUC: {prauc_leak:.5f} vs Clean: {average_precision_score(y_val, p_base_val):.5f}")
    print(f"    Censored Aug 11-31 PR-AUC (showing why censored labels mislead): {average_precision_score(y_cens, p_base_cens):.5f}")

    # Pick best model between Default and Time-Decay by PR-AUC on mature holdout
    use_decay = average_precision_score(y_val, p_dec_val) >= average_precision_score(y_val, p_base_val)
    best_cal_p = p_dec_cal if use_decay else p_base_cal
    best_val_p = p_dec_val if use_decay else p_base_val
    best_iter = int(m_dec.best_iteration_ if use_decay else m_base.best_iteration_)

    # ------------------------------------------------------------------
    # [5] PROBABILITY CALIBRATION & COST-BASED DECISION COMPARISON
    # ------------------------------------------------------------------
    print("\n[4] Probability Calibration (Fitted on June, Evaluated on Mature Jul 1 - Aug 10):")
    # Platt Scaling (logistic regression on log-odds)
    eps = 1e-7
    logit_cal = np.log(np.clip(best_cal_p, eps, 1 - eps) / (1 - np.clip(best_cal_p, eps, 1 - eps))).reshape(-1, 1)
    logit_val = np.log(np.clip(best_val_p, eps, 1 - eps) / (1 - np.clip(best_val_p, eps, 1 - eps))).reshape(-1, 1)
    platt = LogisticRegression(C=1e6).fit(logit_cal, y_cal)
    p_platt_val = platt.predict_proba(logit_val)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip").fit(best_cal_p, y_cal)
    p_iso_val = iso.predict(best_val_p)

    # Compare Flag None, Flag All, Raw Cost Rule, Platt Cost Rule, Isotonic Cost Rule
    cost_rows = [
        {"strategy": "Flag None", "brier": np.nan, "logloss": np.nan, **compute_cost(y_val, np.zeros(len(y_val), dtype=int), nom_val)},
        {"strategy": "Flag All", "brier": np.nan, "logloss": np.nan, **compute_cost(y_val, np.ones(len(y_val), dtype=int), nom_val)},
        {"strategy": "Raw Prob (p*nom > 25k)", "brier": brier_score_loss(y_val, best_val_p), "logloss": log_loss(y_val, best_val_p), **compute_cost(y_val, (best_val_p * nom_val > REVIEW_COST).astype(int), nom_val)},
        {"strategy": "Platt Scaling (p*nom > 25k)", "brier": brier_score_loss(y_val, p_platt_val), "logloss": log_loss(y_val, p_platt_val), **compute_cost(y_val, (p_platt_val * nom_val > REVIEW_COST).astype(int), nom_val)},
        {"strategy": "Isotonic (p*nom > 25k)", "brier": brier_score_loss(y_val, p_iso_val), "logloss": log_loss(y_val, np.clip(p_iso_val, 1e-6, 1-1e-6)), **compute_cost(y_val, (p_iso_val * nom_val > REVIEW_COST).astype(int), nom_val)},
    ]
    df_cost = pd.DataFrame(cost_rows)
    print(df_cost[["strategy", "brier", "logloss", "n_flagged", "n_missed_fraud", "total_cost", "precision", "recall"]].to_string(index=False))

    # Choose best calibration strategy by lowest total_cost among Raw / Platt / Isotonic
    cand_costs = {
        "raw": cost_rows[2]["total_cost"],
        "platt": cost_rows[3]["total_cost"],
        "isotonic": cost_rows[4]["total_cost"],
    }
    best_cal_name = min(cand_costs, key=cand_costs.get)
    best_val_flags = {
        "raw": (best_val_p * nom_val > REVIEW_COST).astype(int),
        "platt": (p_platt_val * nom_val > REVIEW_COST).astype(int),
        "isotonic": (p_iso_val * nom_val > REVIEW_COST).astype(int),
    }[best_cal_name]
    print(f"    Selected calibration for decision rule: {best_cal_name} (lowest total_cost = Rp {cand_costs[best_cal_name]:,.0f})")

    # ------------------------------------------------------------------
    # [6] ERROR BREAKDOWN BY KANAL & KATEGORI MERCHANT
    # ------------------------------------------------------------------
    val_err_df = tr_fe.loc[m_val, ["kanal", "kategori_merchant", "nominal", "is_fraud"]].copy()
    val_err_df["flag"] = best_val_flags

    def group_err(col):
        rows = []
        for k, g in val_err_df.groupby(col):
            c = compute_cost(g["is_fraud"].to_numpy(dtype=int), g["flag"].to_numpy(dtype=int), g["nominal"].to_numpy(dtype=float))
            rows.append({col: k, "n": len(g), "n_fraud": int(g["is_fraud"].sum()), **c})
        return pd.DataFrame(rows).sort_values("total_cost", ascending=False)

    err_kanal = group_err("kanal")
    err_kat = group_err("kategori_merchant")
    print("\n[5] Error Breakdown by Kanal (Mature Holdout):")
    print(err_kanal[["kanal", "n", "n_fraud", "n_caught_fraud", "n_missed_fraud", "n_flagged", "total_cost", "recall"]].to_string(index=False))

    # ------------------------------------------------------------------
    # [7] FINAL RETRAINING ON MATURE LABELS (JAN 1 - AUG 10, 2025) & PREDICT TEST
    # ------------------------------------------------------------------
    print("\n[6] Retraining Final LightGBM on Mature Training Data (Jan 1 - Aug 10, 2025)...")
    m_mat = tr_fe["waktu_dt"] < "2025-08-11"
    X_mat, X_test = align_cats([tr_fe.loc[m_mat, feat_cols], te_fe[feat_cols]], CAT_COLS)
    y_mat = tr_fe.loc[m_mat, "is_fraud"].to_numpy(dtype=int)

    if use_decay:
        mat_days_ago = (tr_fe.loc[m_mat, "waktu_dt"].max() - tr_fe.loc[m_mat, "waktu_dt"]).dt.total_seconds() / 86400.0
        sw_mat = np.exp(-np.log(2) * mat_days_ago.to_numpy() / 120.0)
    else:
        sw_mat = None

    final_clf = lgb.LGBMClassifier(
        n_estimators=max(best_iter, 250), learning_rate=0.03, num_leaves=63, min_child_samples=80,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1
    )
    final_clf.fit(X_mat, y_mat, sample_weight=sw_mat)
    prob_test_raw = final_clf.predict_proba(X_test)[:, 1]

    if best_cal_name == "platt":
        logit_te = np.log(np.clip(prob_test_raw, eps, 1 - eps) / (1 - np.clip(prob_test_raw, eps, 1 - eps))).reshape(-1, 1)
        prob_test_final = platt.predict_proba(logit_te)[:, 1]
    elif best_cal_name == "isotonic":
        prob_test_final = iso.predict(prob_test_raw)
    else:
        prob_test_final = prob_test_raw

    prob_test_final = np.clip(prob_test_final, 1e-7, 1.0 - 1e-7)
    tandai_test = (prob_test_final * df_test_raw["nominal"].to_numpy(dtype=float) > REVIEW_COST).astype(int)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({
        "id_transaksi": df_test_raw["id_transaksi"],
        "prob": np.round(prob_test_final, 6),
        "tandai": tandai_test,
    })
    assert len(sub) == len(df_test_raw), "Row count mismatch!"
    assert ((sub["prob"] >= 0) & (sub["prob"] <= 1)).all(), "Prob out of [0,1]!"
    assert set(sub["tandai"].unique()).issubset({0, 1}), "Invalid tandai values!"
    assert sub.isna().sum().sum() == 0, "NaNs in submission!"
    sub.to_csv(out_path, index=False)
    print(f"    Saved valid submission: {out_path.resolve()} ({len(sub):,} rows) | flagged={int(tandai_test.sum()):,} ({tandai_test.mean()*100:.2f}%) | mean_prob={prob_test_final.mean():.5f}")

    # Save artifacts
    art_dir = Path(args.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    weekly_df.to_csv(art_dir / "weekly_delayed_labels.csv")
    df_imb.to_csv(art_dir / "imbalance_comparison.csv", index=False)
    df_cost.to_csv(art_dir / "calibration_cost_comparison.csv", index=False)
    err_kanal.to_csv(art_dir / "error_kanal.csv", index=False)
    err_kat.to_csv(art_dir / "error_kategori_merchant.csv", index=False)
    shift_kanal.to_csv(art_dir / "pattern_shift_kanal.csv", index=False)
    generate_plots(weekly_df, shift_kanal, y_val, best_val_p, p_platt_val, art_dir / "figures")

    summary = {
        "python": sys.version.split()[0],
        "lightgbm": lgb.__version__,
        "random_seed": RANDOM_SEED,
        "n_train_mature": int(m_mat.sum()),
        "n_train_censored_excluded": int(m_cens.sum()),
        "pr_auc_with_leakage": round(prauc_leak, 5),
        "pr_auc_clean_mature_val": round(float(average_precision_score(y_val, best_val_p)), 5),
        "pr_auc_censored_aug11_31": round(float(average_precision_score(y_cens, p_base_cens)), 5),
        "used_time_decay": bool(use_decay),
        "selected_calibration": best_cal_name,
        "flag_none_cost": float(cost_rows[0]["total_cost"]),
        "flag_all_cost": float(cost_rows[1]["total_cost"]),
        "best_cost_val": float(cand_costs[best_cal_name]),
        "cost_reduction_vs_none_pct": round((1 - cand_costs[best_cal_name] / cost_rows[0]["total_cost"]) * 100, 2),
        "test_flagged_count": int(tandai_test.sum()),
        "test_flagged_pct": round(float(tandai_test.mean() * 100), 3),
    }
    (art_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE!")


if __name__ == "__main__":
    main()
