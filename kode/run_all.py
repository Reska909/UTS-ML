"""
Single-command reproducible entrypoint for UTS Machine Learning (Tasks 1, 2, and 3).
Reproduces every metric, table, figure, and final CSV submission file.

Usage:
    python kode/run_all.py --data-dir data
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str], title: str) -> None:
    print(f"\n{'#' * 80}\n# {title}\n{'#' * 80}", flush=True)
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"Step failed ({title}) with exit code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run all UTS ML pipelines (Tasks 1-3)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing the .csv.gz datasets")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory for prediksi_*.csv and cluster.csv")
    args = parser.parse_args()

    py = sys.executable
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    run_step(
        [
            py, "kode/task_1_regresi.py",
            "--train", str(data_dir / "regresi_train.csv.gz"),
            "--test", str(data_dir / "regresi_test.csv.gz"),
            "--output", str(out_dir / "prediksi_regresi.csv"),
        ],
        "TASK 1 - REGRESSION (DELIVERY TIME ESTIMATION)",
    )
    run_step(
        [
            py, "kode/task_2_klasifikasi.py",
            "--train", str(data_dir / "klasifikasi_train.csv.gz"),
            "--test", str(data_dir / "klasifikasi_test.csv.gz"),
            "--output", str(out_dir / "prediksi_klasifikasi.csv"),
        ],
        "TASK 2 - CLASSIFICATION (FRAUD DETECTION)",
    )
    run_step(
        [
            py, "kode/task_3_clustering.py",
            "--data", str(data_dir / "clustering.csv.gz"),
            "--output", str(out_dir / "cluster.csv"),
        ],
        "TASK 3 - CLUSTERING (CUSTOMER SEGMENTATION)",
    )
    print("\nALL 3 TASKS COMPLETED AND REPRODUCED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
