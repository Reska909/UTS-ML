# UTS Machine Learning

Repository ini berisi pengerjaan Ujian Tengah Semester (UTS) mata kuliah Machine Learning. Pengerjaan terdiri dari tiga task yang menerapkan metode machine learning untuk menyelesaikan permasalahan regresi, klasifikasi, dan clustering.

## Identitas Kelompok

| No. | Nama | NIM | Pembagian Tugas |
|---|---|---|---|
| 1 | Irfan Reska Nugraha | 101032300042 | Task 1 - Regresi dan Task 2 - Klasifikasi |
| 2 | Ruben Alfa Richie | 101032300008 | Task 3 - Clustering dan penyusunan laporan |

## Daftar Pengerjaan

### Task 1 - Regresi

Melakukan prediksi durasi pengiriman paket berdasarkan data historis pengiriman.

- Target prediksi: `durasi_jam`
- Metode: regresi dengan pendekatan quantile prediction.
- Model yang dievaluasi: Baseline, CatBoost, dan LightGBM.
- Output: prediksi durasi pengiriman pada kuantil P50 dan P90.

File notebook: [`task_1.ipynb`](task_1.ipynb)

### Task 2 - Klasifikasi

Melakukan klasifikasi transaksi untuk mengidentifikasi kemungkinan terjadinya fraud.

- Target prediksi: `is_fraud`
- Metode: supervised learning dengan klasifikasi.
- Model utama: CatBoost Classifier.
- Evaluasi mempertimbangkan PR-AUC dan biaya kesalahan klasifikasi.
- Output: probabilitas fraud dan keputusan penandaan transaksi.

File notebook: [`task 2.ipynb`](task%202.ipynb)

### Task 3 - Clustering

Melakukan segmentasi pelanggan berdasarkan karakteristik aktivitas dan perilaku transaksi.

- Metode: unsupervised learning.
- Model yang dievaluasi: K-Means, Gaussian Mixture Model (GMM), dan BIRCH.
- Jumlah cluster akhir: 3 cluster untuk pelanggan aktif.
- Analisis tambahan: profil cluster, segmentasi pelanggan tidak aktif, dan deteksi anomali.

File notebook: [`task 3.ipynb`](task%203.ipynb)

## Teknologi dan Library

Pengerjaan menggunakan Python dengan beberapa library berikut:

- NumPy: pengolahan data numerik.
- Pandas: manipulasi dan analisis data.
- Scikit-learn: preprocessing, evaluasi, dan pemodelan machine learning.
- CatBoost: pemodelan regresi dan klasifikasi.
- LightGBM: pemodelan regresi.
- Matplotlib dan Seaborn: visualisasi data.
- SHAP: interpretasi model.

## Dataset

Dataset yang digunakan merupakan dataset yang disediakan untuk keperluan tugas. Tidak menggunakan dataset eksternal.

File dataset yang diperlukan:

| Nama File | Keterangan |
|---|---|
| `regresi_train.csv.gz` | Data training Task 1 |
| `regresi_test.csv.gz` | Data testing Task 1 |
| `klasifikasi_train.csv.gz` | Data training Task 2 |
| `klasifikasi_test.csv.gz` | Data testing Task 2 |
| `clustering.csv.gz` | Data Task 3 |
| `kamus_data.md` | Keterangan variabel dataset |

Dataset diunggah melalui fitur upload pada Google Colab sesuai kebutuhan masing-masing notebook.

## Cara Menjalankan Notebook

1. Buka repository ini di GitHub.
2. Pilih notebook Task 1, Task 2, atau Task 3.
3. Buka notebook menggunakan Google Colab.
4. Siapkan dan unggah dataset yang sesuai.
5. Pastikan library yang dibutuhkan tersedia di runtime.
6. Jalankan seluruh cell secara berurutan dari awal sampai akhir.
7. Periksa hasil analisis, evaluasi model, dan file prediksi yang dihasilkan.

### Instalasi Library

Jika library belum tersedia pada runtime, jalankan perintah berikut di Google Colab:

```python
!pip install numpy pandas scikit-learn catboost lightgbm matplotlib seaborn shap
```

## Hasil Prediksi

File hasil prediksi akhir disiapkan dalam format CSV dengan ketentuan nama dan kolom berikut:

| File | Kolom |
|---|---|
| `prediksi_regresi.csv` | `id_paket`, `p50`, `p90` |
| `prediksi_klasifikasi.csv` | `id_transaksi`, `prob`, `tandai` |
| `cluster.csv` | `id_pelanggan`, `cluster` |

## Struktur Repository

```text
UTS-ML/
├── README.md
├── kontribusi.md
├── task_1.ipynb
├── task 2.ipynb
├── task 3.ipynb
├── laporan.pdf
├── prediksi_regresi.csv
├── prediksi_klasifikasi.csv
├── cluster.csv
└── kode/
    └── README.md
```

Struktur di atas merupakan struktur akhir yang dituju. File yang belum tersedia akan ditambahkan setelah proses pengerjaan dan dokumentasi selesai.

## Kontribusi Anggota

Informasi pembagian tugas, kontribusi, riwayat Pull Request, dan penggunaan AI dijelaskan dalam file [`kontribusi.md`](kontribusi.md).

## Catatan

Seluruh proses preprocessing, pelatihan model, evaluasi, analisis, dan pembuatan prediksi dijelaskan pada masing-masing notebook. Hasil akhir digunakan sebagai bahan penyusunan laporan UTS Machine Learning.
