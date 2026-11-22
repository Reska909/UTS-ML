"""
Task 3 - Clustering: Customer Segmentation (UTS Machine Learning)
Reproducible single-command pipeline.

Usage:
    python kode/task_3_clustering.py --data data/clustering.csv.gz --output cluster.csv
"""

import argparse
import json
from itertools import combinations
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_SEED = 42


def clr_transform(arr: np.ndarray, delta: float = 1e-4) -> np.ndarray:
    """Multiplicative zero replacement + Centered Log-Ratio (CLR) for compositional data."""
    X = arr.astype(float).copy()
    row_sums = X.sum(axis=1, keepdims=True)
    X = X / np.where(row_sums == 0, 1.0, row_sums)
    zero_mask = X == 0
    n_zeros = zero_mask.sum(axis=1, keepdims=True)
    X = np.where(zero_mask, delta, X * (1.0 - n_zeros * delta))
    log_x = np.log(X)
    return log_x - log_x.mean(axis=1, keepdims=True)


def audit_feature_signal(df: pd.DataFrame, sample_size: int = 30000) -> pd.DataFrame:
    """Compute max |Spearman rho| with all other numeric columns to identify pure noise features."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    samp = df[num_cols].sample(n=min(sample_size, len(df)), random_state=RANDOM_SEED)
    corr = samp.corr(method="spearman").abs()
    arr = corr.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    corr_clean = pd.DataFrame(arr, index=corr.index, columns=corr.columns)
    return pd.DataFrame({
        "max_spearman": corr_clean.max(axis=1),
        "top_partner": corr_clean.idxmax(axis=1),
    }).sort_values("max_spearman")


def detect_anomalies(df: pd.DataFrame) -> np.ndarray:
    """
    Detect bot/scraper/click-farm anomalies using IsolationForest on session & engagement features.
    In our audit, ~1.5% (1,800 rows) exhibit extreme sesi_per_minggu (~141/wk), wishlist (~139),
    klik_iklan_30h (~188), ultra-short durasi_sesi_menit (~0.84 min), and near-zero conversion.
    """
    anom_df = pd.DataFrame({
        "log_sesi": np.log1p(np.clip(df["sesi_per_minggu"].to_numpy(dtype=float), 0, None)),
        "log_wishlist": np.log1p(np.clip(df["wishlist"].to_numpy(dtype=float), 0, None)),
        "log_klik": np.log1p(np.clip(df["klik_iklan_30h"].to_numpy(dtype=float), 0, None)),
        "log_durasi": np.log1p(np.clip(df["durasi_sesi_menit"].to_numpy(dtype=float), 0, None)),
        "log_konversi": np.log1p(np.clip(df["rasio_konversi"].to_numpy(dtype=float), 0, None)),
    })
    X_anom = StandardScaler().fit_transform(anom_df)
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.015,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    preds = iso.fit_predict(X_anom)
    return preds == -1


def build_feature_matrix(df_norm: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Preprocess normal customers:
    - Exclude 18 pure noise columns (n01-n15, rating_beri_rata2, kota_tier, jumlah_alamat)
    - Exclude 2 redundant 30-day columns (belanja_30h r=0.97, frekuensi_30h r=0.87)
    - Exclude 3 anomaly-only engagement columns (wishlist, klik_iklan_30h, durasi_sesi_menit)
    - Impute rata2_keranjang (missing only when frekuensi_90h == 0) using median of freq == 1
    - Smooth log1p scaling for skewed features (avoiding 9-unit log1p(0) cliff on belanja_90h)
    - CLR + PCA compression for 16 category shares (5 PCs) and 4 time shares (3 PCs)
    - Standardize + PCA(10) for clustering space
    """
    cat_porsi = [c for c in df_norm.columns if c.startswith("porsi_") and not c.startswith("porsi_jam_")]
    time_porsi = ["porsi_jam_pagi", "porsi_jam_siang", "porsi_jam_sore", "porsi_jam_malam"]

    med_keranjang_f1 = df_norm.loc[df_norm["frekuensi_90h"] == 1, "rata2_keranjang"].median()
    keranjang_imp = df_norm["rata2_keranjang"].fillna(med_keranjang_f1)

    behav_df = pd.DataFrame({
        "recency": np.log1p(np.clip(df_norm["recency_hari"].to_numpy(dtype=float), 0, None)),
        "freq": np.log1p(np.clip(df_norm["frekuensi_90h"].to_numpy(dtype=float), 0, None)),
        "belanja_smooth": np.log1p(np.clip(df_norm["belanja_90h"].to_numpy(dtype=float), 0, None) / 100000.0),
        "keranjang_smooth": np.log1p(np.clip(keranjang_imp.to_numpy(dtype=float), 0, None) / 100000.0),
        "item": np.log1p(np.clip(df_norm["item_per_pesanan"].to_numpy(dtype=float), 0, None)),
        "retur": np.log1p(np.clip(df_norm["retur_90h"].to_numpy(dtype=float), 0, None)),
        "ulasan": np.log1p(np.clip(df_norm["ulasan_90h"].to_numpy(dtype=float), 0, None)),
        "sesi": np.log1p(np.clip(df_norm["sesi_per_minggu"].to_numpy(dtype=float), 0, None)),
        "konversi": np.log1p(np.clip(df_norm["rasio_konversi"].to_numpy(dtype=float), 0, None)),
        "lama_member": np.log1p(np.clip(df_norm["lama_member_bulan"].to_numpy(dtype=float), 0, None)),
        "voucher": df_norm["persen_voucher"].to_numpy(dtype=float),
        "flash_sale": df_norm["persen_flash_sale"].to_numpy(dtype=float),
        "paylater": df_norm["persen_paylater"].to_numpy(dtype=float),
        "cod": df_norm["persen_cod"].to_numpy(dtype=float),
        "umur": df_norm["umur"].to_numpy(dtype=float),
        "is_ios": (df_norm["perangkat"] == "ios").astype(float).to_numpy(),
    })

    sc_behav = StandardScaler().fit_transform(behav_df)

    clr_cat = clr_transform(df_norm[cat_porsi].to_numpy())
    clr_time = clr_transform(df_norm[time_porsi].to_numpy())

    cat_pcs = StandardScaler().fit_transform(
        PCA(n_components=5, random_state=RANDOM_SEED).fit_transform(StandardScaler().fit_transform(clr_cat))
    )
    time_pcs = StandardScaler().fit_transform(
        PCA(n_components=3, random_state=RANDOM_SEED).fit_transform(StandardScaler().fit_transform(clr_time))
    )

    X_24d = np.hstack([sc_behav, cat_pcs, time_pcs])
    pca_final = PCA(n_components=10, random_state=RANDOM_SEED)
    X_10d = pca_final.fit_transform(X_24d)

    feature_names = list(behav_df.columns) + [f"cat_pc{i}" for i in range(1, 6)] + [f"time_pc{i}" for i in range(1, 4)]
    return X_24d, X_10d, feature_names


def evaluate_methods_and_k(X_10d: np.ndarray, k_range=range(2, 10)) -> pd.DataFrame:
    """Compare K-Means, GMM (Full), BIRCH, and Agglomerative Clustering across K."""
    rng = np.random.RandomState(RANDOM_SEED)
    idx_25k = rng.choice(len(X_10d), size=25000, replace=False)
    idx_10k = idx_25k[:10000]
    Xs = X_10d[idx_25k]
    Xs_small = X_10d[idx_10k]

    rows = []
    for k in k_range:
        # 1. K-Means
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=15).fit(Xs)
        rows.append({
            "method": "K-Means",
            "k": k,
            "silhouette": silhouette_score(Xs, km.labels_, sample_size=10000, random_state=RANDOM_SEED),
            "davies_bouldin": davies_bouldin_score(Xs, km.labels_),
            "calinski_harabasz": calinski_harabasz_score(Xs, km.labels_),
            "bic": np.nan,
        })

        # 2. GMM (Full covariance)
        gmm = GaussianMixture(
            n_components=k, covariance_type="full", random_state=RANDOM_SEED, n_init=3, reg_covar=1e-4
        ).fit(Xs)
        gmm_lbl = gmm.predict(Xs)
        rows.append({
            "method": "GMM (Full)",
            "k": k,
            "silhouette": silhouette_score(Xs, gmm_lbl, sample_size=10000, random_state=RANDOM_SEED),
            "davies_bouldin": davies_bouldin_score(Xs, gmm_lbl),
            "calinski_harabasz": calinski_harabasz_score(Xs, gmm_lbl),
            "bic": gmm.bic(Xs),
        })

        # 3. BIRCH
        birch = Birch(n_clusters=k, threshold=0.8, branching_factor=50).fit(Xs)
        rows.append({
            "method": "BIRCH",
            "k": k,
            "silhouette": silhouette_score(Xs, birch.labels_, sample_size=10000, random_state=RANDOM_SEED),
            "davies_bouldin": davies_bouldin_score(Xs, birch.labels_),
            "calinski_harabasz": calinski_harabasz_score(Xs, birch.labels_),
            "bic": np.nan,
        })

        # 4. Agglomerative (Ward) on 10k sample
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Xs_small)
        rows.append({
            "method": "Agglomerative (Ward)",
            "k": k,
            "silhouette": silhouette_score(Xs_small, agg.labels_),
            "davies_bouldin": davies_bouldin_score(Xs_small, agg.labels_),
            "calinski_harabasz": calinski_harabasz_score(Xs_small, agg.labels_),
            "bic": np.nan,
        })

    return pd.DataFrame(rows)


def evaluate_stability(X_10d: np.ndarray, k_list=(4, 5, 6, 7), seeds=(10, 21, 42, 55, 77)) -> pd.DataFrame:
    """5-run 80% subsampling stability test reporting pairwise ARI."""
    rng = np.random.RandomState(RANDOM_SEED)
    idx_30k = rng.choice(len(X_10d), size=30000, replace=False)
    Xm = X_10d[idx_30k]

    results = []
    for k in k_list:
        gmm_preds, km_preds = [], []
        for s in seeds:
            r = np.random.RandomState(s)
            sub = r.choice(len(Xm), size=24000, replace=False)
            gmm = GaussianMixture(
                n_components=k, covariance_type="full", random_state=s, n_init=3, reg_covar=1e-4
            ).fit(Xm[sub])
            km = KMeans(n_clusters=k, random_state=s, n_init=15).fit(Xm[sub])
            gmm_preds.append(gmm.predict(Xm))
            km_preds.append(km.predict(Xm))

        gmm_aris = [adjusted_rand_score(a, b) for a, b in combinations(gmm_preds, 2)]
        km_aris = [adjusted_rand_score(a, b) for a, b in combinations(km_preds, 2)]
        cross_ari = adjusted_rand_score(gmm_preds[2], km_preds[2])
        results.append({
            "k": k,
            "gmm_ari_mean": float(np.mean(gmm_aris)),
            "gmm_ari_std": float(np.std(gmm_aris)),
            "gmm_ari_min": float(np.min(gmm_aris)),
            "km_ari_mean": float(np.mean(km_aris)),
            "km_ari_min": float(np.min(km_aris)),
            "cross_ari_gmm_vs_km": float(cross_ari),
        })
    return pd.DataFrame(results)


def assign_canonical_order(df_norm: pd.DataFrame, raw_labels: np.ndarray) -> np.ndarray:
    """
    Relabel clusters 0..5 deterministically by median belanja_90h descending
    so cluster IDs are stable and interpretable across runs.
    """
    tmp = pd.DataFrame({"c": raw_labels, "belanja": df_norm["belanja_90h"].to_numpy()})
    order = tmp.groupby("c")["belanja"].median().sort_values(ascending=False).index.tolist()
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[c] for c in raw_labels], dtype=int)


def generate_plots(df_eval: pd.DataFrame, df_stab: pd.DataFrame, prof: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # 1. Method comparison (Silhouette & GMM BIC)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for method, grp in df_eval.groupby("method"):
        axes[0].plot(grp["k"], grp["silhouette"], marker="o", label=method)
    axes[0].axvline(6, color="red", linestyle="--", alpha=0.7, label="Selected K=6")
    axes[0].set_title("Perbandingan Silhouette Score Antar Metode")
    axes[0].set_xlabel("Jumlah Cluster (K)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].legend()

    gmm_rows = df_eval[df_eval["method"] == "GMM (Full)"].sort_values("k")
    axes[1].plot(gmm_rows["k"], gmm_rows["bic"], marker="o", color="tab:blue")
    axes[1].axvline(6, color="red", linestyle="--", alpha=0.7, label="Selected K=6")
    axes[1].set_title("GMM (Full Covariance) BIC vs K")
    axes[1].set_xlabel("Jumlah Cluster (K)")
    axes[1].set_ylabel("BIC (lebih rendah lebih baik)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_method_comparison_bic.png", dpi=150)
    plt.close()

    # 2. Stability ARI plot
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(df_stab["k"], df_stab["gmm_ari_mean"], marker="o", label="GMM (Full) Subsample ARI")
    ax.plot(df_stab["k"], df_stab["km_ari_mean"], marker="s", label="K-Means Subsample ARI")
    ax.plot(df_stab["k"], df_stab["cross_ari_gmm_vs_km"], marker="^", linestyle="--", label="Cross-ARI (GMM vs K-Means)")
    ax.axvline(6, color="red", linestyle=":", alpha=0.8)
    ax.set_title("Uji Stabilitas Cluster (5-Run 80% Subsampling ARI)")
    ax.set_xlabel("Jumlah Cluster (K)")
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_ylim(0.5, 1.02)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_stability_ari.png", dpi=150)
    plt.close()

    # 3. Normalized heatmap of segment profiles (clusters 0..5)
    norm_prof = prof.loc[prof.index >= 0, [
        "recency_hari", "frekuensi_90h", "belanja_90h", "rata2_keranjang",
        "item_per_pesanan", "persen_voucher", "persen_flash_sale",
        "persen_paylater", "persen_cod", "retur_90h", "sesi_per_minggu",
        "lama_member_bulan", "umur", "porsi_jam_pagi", "porsi_jam_malam", "ios_pct"
    ]]
    z_prof = (norm_prof - norm_prof.mean()) / norm_prof.std()
    fig, ax = plt.subplots(figsize=(13, 5.2))
    sns.heatmap(z_prof, annot=norm_prof.round(1), fmt="", cmap="RdYlBu_r", center=0, ax=ax, annot_kws={"size": 8})
    ax.set_title("Profil Median Segmen Pelanggan (Warna = Z-Score Antar Segmen, Angka = Nilai Asli)")
    ax.set_ylabel("Cluster ID")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_segment_profiles_heatmap.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Task 3 Clustering Pipeline (UTS ML)")
    parser.add_argument("--data", type=str, default="data/clustering.csv.gz", help="Path to clustering.csv.gz")
    parser.add_argument("--output", type=str, default="cluster.csv", help="Output path for cluster.csv")
    parser.add_argument("--artifacts-dir", type=str, default="kode/outputs_task3", help="Directory for tables & plots")
    args = parser.parse_args()

    print("=" * 80)
    print("TASK 3 - CUSTOMER SEGMENTATION (REPRODUCIBLE PIPELINE)")
    print("=" * 80)
    print(f"Python: {sys.version.split()[0]} | numpy: {np.__version__} | pandas: {pd.__version__} "
          f"| scikit-learn: {sklearn.__version__} | scipy: {scipy.__version__} | seed: {RANDOM_SEED}")

    df = pd.read_csv(args.data)
    print(f"\n[1] Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"    Missing values: {df.isna().sum()[df.isna().sum() > 0].to_dict()}")

    # Feature relevance audit
    signal_df = audit_feature_signal(df)
    noise_detected = signal_df[signal_df["max_spearman"] < 0.025].index.tolist()
    print(f"\n[2] Feature Selection Audit (Spearman max |rho| < 0.025 -> pure noise):")
    print(f"    Detected {len(noise_detected)} noise features: {noise_detected}")

    # Anomaly detection
    is_anomaly = detect_anomalies(df)
    exact_bot_check = (df["klik_iklan_30h"] > 60).to_numpy()
    overlap = int((is_anomaly & exact_bot_check).sum())
    print(f"\n[3] Anomaly Detection (IsolationForest, contamination=0.015):")
    print(f"    Flagged anomalies (-1): {int(is_anomaly.sum()):,} ({is_anomaly.mean()*100:.2f}%)")
    print(f"    Overlap with extreme bot signature (klik_iklan_30h > 60, n={int(exact_bot_check.sum())}): {overlap:,} ({overlap/exact_bot_check.sum()*100:.2f}%)")

    # Preprocessing normal customers
    df_norm = df.loc[~is_anomaly].copy().reset_index(drop=True)
    X_24d, X_10d, feat_names = build_feature_matrix(df_norm)
    print(f"\n[4] Feature Matrix Ready (Normal Customers: {len(df_norm):,}):")
    print(f"    Clean compact features: {X_24d.shape[1]} -> PCA-10 clustering space: {X_10d.shape}")

    # Baseline comparison: noisy vs clean (quantifying feature selection impact)
    rng = np.random.RandomState(RANDOM_SEED)
    samp_idx = rng.choice(len(df_norm), size=15000, replace=False)
    num_all = df_norm.select_dtypes(include=[np.number]).fillna(0).to_numpy()
    X_noisy_sc = StandardScaler().fit_transform(num_all)
    km_noisy = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10).fit(X_noisy_sc[samp_idx])
    sil_noisy = silhouette_score(X_noisy_sc[samp_idx], km_noisy.labels_, sample_size=10000, random_state=RANDOM_SEED)
    km_clean = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10).fit(X_10d[samp_idx])
    sil_clean = silhouette_score(X_10d[samp_idx], km_clean.labels_, sample_size=10000, random_state=RANDOM_SEED)
    print(f"    Feature Selection Impact (K=6 K-Means Silhouette): Noisy 58D = {sil_noisy:.4f} -> Clean PCA-10 = {sil_clean:.4f} (+{(sil_clean-sil_noisy)/sil_noisy*100:.1f}%)")

    # Method & K comparison
    print("\n[5] Comparing 4 Clustering Methods across K=2..9...")
    df_eval = evaluate_methods_and_k(X_10d, k_range=range(2, 10))
    print(df_eval[df_eval["k"].isin([4, 5, 6, 7])].to_string(index=False))

    # Stability test
    print("\n[6] 5-Run 80% Subsampling Stability Test (K=4..7)...")
    df_stab = evaluate_stability(X_10d, k_list=(4, 5, 6, 7))
    print(df_stab.to_string(index=False))

    # Final model: GMM (Full covariance, K=6) on full normal population
    print("\n[7] Training Final Model: GMM (Full Covariance, K=6) on all 118,200 normal customers...")
    final_gmm = GaussianMixture(
        n_components=6,
        covariance_type="full",
        random_state=RANDOM_SEED,
        n_init=5,
        reg_covar=1e-4,
    ).fit(X_10d)
    raw_labels = final_gmm.predict(X_10d)
    norm_labels = assign_canonical_order(df_norm, raw_labels)

    final_sil = silhouette_score(X_10d[samp_idx], norm_labels[samp_idx], sample_size=10000, random_state=RANDOM_SEED)
    final_db = davies_bouldin_score(X_10d[samp_idx], norm_labels[samp_idx])
    final_ch = calinski_harabasz_score(X_10d[samp_idx], norm_labels[samp_idx])
    print(f"    Final GMM (K=6) | Silhouette: {final_sil:.4f} | Davies-Bouldin: {final_db:.4f} | Calinski-Harabasz: {final_ch:.1f}")

    # Assemble final labels for all 120,000 customers
    full_labels = np.full(len(df), -1, dtype=int)
    full_labels[~is_anomaly] = norm_labels

    # Profiles
    df_prof = df.copy()
    df_prof["cluster"] = full_labels
    cat_porsi = [c for c in df.columns if c.startswith("porsi_") and not c.startswith("porsi_jam_")]
    cols_show = [
        "recency_hari", "frekuensi_90h", "belanja_90h", "rata2_keranjang",
        "item_per_pesanan", "persen_voucher", "persen_flash_sale", "persen_paylater",
        "persen_cod", "retur_90h", "sesi_per_minggu", "durasi_sesi_menit",
        "wishlist", "klik_iklan_30h", "lama_member_bulan", "umur",
        "porsi_jam_pagi", "porsi_jam_malam",
    ]
    prof = df_prof.groupby("cluster")[cols_show].median()
    prof.insert(0, "n", df_prof.groupby("cluster").size())
    prof.insert(1, "pct", (prof["n"] / len(df_prof) * 100).round(2))
    prof["ios_pct"] = (df_prof.groupby("cluster")["perangkat"].apply(lambda s: (s == "ios").mean() * 100)).round(1)
    top_cats = []
    for cid in prof.index:
        med_c = df_prof.loc[df_prof["cluster"] == cid, cat_porsi].median().sort_values(ascending=False)
        top_cats.append(", ".join([f"{x.replace('porsi_', '')}:{v*100:.0f}%" for x, v in med_c.head(2).items()]))
    prof["top2_cat"] = top_cats

    print("\n[8] FINAL SEGMENT PROFILES (MEDIAN):")
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 220)
    print(prof.round(2))

    # Save submission file cluster.csv
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({
        "id_pelanggan": df["id_pelanggan"],
        "cluster": full_labels,
    })
    assert len(sub) == 120000, f"Expected 120000 rows, got {len(sub)}"
    assert sub["id_pelanggan"].nunique() == 120000, "Duplicate id_pelanggan found!"
    assert sub["cluster"].isna().sum() == 0, "NaN cluster labels found!"
    assert set(sub["cluster"].unique()) == {-1, 0, 1, 2, 3, 4, 5}, f"Unexpected labels: {sub['cluster'].unique()}"
    sub.to_csv(out_path, index=False)
    print(f"\n[9] Saved valid submission file: {out_path.resolve()} ({len(sub):,} rows)")

    # Save artifacts for report
    art_dir = Path(args.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    df_eval.to_csv(art_dir / "method_comparison.csv", index=False)
    df_stab.to_csv(art_dir / "stability_ari.csv", index=False)
    prof.to_csv(art_dir / "segment_profiles.csv")
    signal_df.to_csv(art_dir / "feature_signal_audit.csv")
    generate_plots(df_eval, df_stab, prof, art_dir / "figures")

    summary = {
        "python_version": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "random_seed": RANDOM_SEED,
        "n_rows": int(len(df)),
        "n_anomalies": int(is_anomaly.sum()),
        "bot_overlap": overlap,
        "silhouette_noisy_k6": round(float(sil_noisy), 4),
        "silhouette_clean_k6": round(float(sil_clean), 4),
        "final_gmm_silhouette": round(float(final_sil), 4),
        "final_gmm_davies_bouldin": round(float(final_db), 4),
        "final_gmm_calinski_harabasz": round(float(final_ch), 2),
        "cluster_counts": {str(k): int(v) for k, v in sub["cluster"].value_counts().sort_index().items()},
    }
    (art_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"    Saved report tables & figures to: {art_dir.resolve()}")
    print("DONE!")


if __name__ == "__main__":
    main()
