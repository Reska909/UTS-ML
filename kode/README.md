# Kode Reproducible — UTS Machine Learning

Folder `kode/` ini berisi seluruh skrip Python untuk mereproduksi setiap angka, tabel, grafik pada `laporan.pdf`, serta ketiga file prediksi (`prediksi_regresi.csv`, `prediksi_klasifikasi.csv`, `cluster.csv`) dalam **satu perintah**.

## Lingkungan & Versi Library
- **Python**: `3.15.0` (kompatibel juga dengan Python `3.10+`)
- **Random Seed**: `42` (ditetapkan di seluruh skrip)
- **Library utama**:
  - `numpy==2.5.4`
  - `pandas==3.1.1`
  - `scikit-learn==1.9.1`
  - `scipy==1.18.1`
  - `lightgbm==4.7.0`
  - `matplotlib==3.11.2`
  - `seaborn==0.13.2`
  - `reportlab==5.1.0` (hanya untuk generasi ulang `laporan.pdf`)

Instalasi dependensi:
```bash
pip install numpy pandas scikit-learn scipy lightgbm matplotlib seaborn reportlab
```

## Persiapan Dataset
Letakkan file dataset UTS di folder `data/` pada root repository:
- `data/regresi_train.csv.gz`
- `data/regresi_test.csv.gz`
- `data/klasifikasi_train.csv.gz`
- `data/klasifikasi_test.csv.gz`
- `data/clustering.csv.gz`

## Satu Perintah untuk Reproduksi Penuh
Jalankan dari root repository:
```bash
python kode/run_all.py --data-dir data
```

Perintah di atas akan mengeksekusi secara berurutan:
1. `kode/task_1_regresi.py` → menghasilkan `prediksi_regresi.csv` serta tabel & grafik di `kode/outputs_task1/`.
2. `kode/task_2_klasifikasi.py` → menghasilkan `prediksi_klasifikasi.csv` serta tabel & grafik di `kode/outputs_task2/`.
3. `kode/task_3_clustering.py` → menghasilkan `cluster.csv` serta tabel & grafik di `kode/outputs_task3/`.

Untuk mereproduksi masing-masing tugas secara terpisah (mis. saat demo 5 menit ujian lisan):
```bash
python kode/task_1_regresi.py --train data/regresi_train.csv.gz --test data/regresi_test.csv.gz --output prediksi_regresi.csv
python kode/task_2_klasifikasi.py --train data/klasifikasi_train.csv.gz --test data/klasifikasi_test.csv.gz --output prediksi_klasifikasi.csv
python kode/task_3_clustering.py --data data/clustering.csv.gz --output cluster.csv
```
