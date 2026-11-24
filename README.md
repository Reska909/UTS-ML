# UTS Machine Learning — Regresi, Klasifikasi, dan Clustering

Repository resmi kelompok untuk Ujian Tengah Semester (UTS) Machine Learning. Seluruh angka pada `laporan.pdf` dan ketiga file prediksi dapat direproduksi dengan satu perintah melalui folder `kode/`.

## Identitas Kelompok

| No. | Nama | NIM | Pembagian Tugas |
|---|---|---|---|
| 1 | Irfan Reska Nugraha | 101032300042 | Task 1 (Regresi) & Task 2 (Klasifikasi) |
| 2 | Ruben Alfa Richie | 101032300008 | Task 3 (Clustering), pipeline reproducible, & `laporan.pdf` |

## Struktur Repository (`main`)

```text
UTS-ML/
├── README.md
├── kontribusi.md
├── laporan.pdf
├── prediksi_regresi.csv
├── prediksi_klasifikasi.csv
├── cluster.csv
└── kode/
    ├── README.md
    ├── run_all.py
    ├── task_1_regresi.py
    ├── task_2_klasifikasi.py
    └── task_3_clustering.py
```

## Cara Reproduksi (Satu Perintah)

Letakkan file dataset UTS (`regresi_train.csv.gz`, `regresi_test.csv.gz`, `klasifikasi_train.csv.gz`, `klasifikasi_test.csv.gz`, `clustering.csv.gz`) di folder `data/`, lalu jalankan:

```bash
python kode/run_all.py --data-dir data
```

Detail versi library, seed, dan cara menjalankan masing-masing task tersedia di [`kode/README.md`](kode/README.md).
