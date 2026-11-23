# KONTRIBUSI ANGGOTA KELOMPOK

## 1. Identitas dan Pembagian Tugas

| No. | Nama | NIM | Tanggung Jawab |
|---|---|---|---|
| 1 | Irfan Reska Nugraha | 101032300042 | Pengerjaan Task 1 (Regresi), Task 2 (Klasifikasi / Fraud Detection) dan README|
| 2 | Ruben Alfa Richie | 101032300008 | Pengerjaan Task 3 (Clustering), integrasi pipeline reproducible (`kode/`), dan penyusunan laporan akhir (`laporan.pdf`) |

## 2. Rincian Kontribusi

### Irfan Reska Nugraha
- Melakukan eksplorasi awal, data audit, dan pemodelan baseline hingga gradient boosting untuk Task 1 (`task_1_regresi.ipynb`).
- Melakukan eksplorasi awal, analisis ketidakseimbangan kelas, dan pemodelan untuk Task 2 (`task_2_klasifikasi.ipynb`).
- Mengelola repository utama kelompok dan melakukan review serta merge Pull Request.

### Ruben Alfa Richie
- Melakukan audit fitur mendalam, seleksi fitur (membuang 18 fitur noise & 2 fitur redundan), deteksi anomali bot berbasis Isolation Forest, perbandingan 4 metode clustering, uji stabilitas subsampling ARI, serta pembuatan `cluster.csv` (`kode/task_3_clustering.py`).
- Menyusun pipeline reproducible 1-perintah (`kode/run_all.py`, `kode/task_1_regresi.py`, `kode/task_2_klasifikasi.py`, `kode/task_3_clustering.py`, `kode/README.md`) untuk menutup celah rubrik (audit unit error, random CV vs time-based, delayed label cutoff 21 hari, fitur behavioral kausal, kalibrasi probabilitas) dan menghasilkan ketiga file prediksi final.
- Menyusun dan memfinalisasi laporan akhir (`laporan.pdf`) beserta seluruh tabel dan visualisasi evaluasi untuk Task 1–3.

## 3. Riwayat Pull Request

Setiap anggota bekerja melalui repository fork masing-masing dan mengajukan Pull Request ke branch `main` repository utama (`Reska909/UTS-ML`).

| No. | Nama Anggota | Link Pull Request | Ringkasan | Status |
|---|---|---|---|---|
| 1 | Irfan Reska Nugraha | https://github.com/Reska909/UTS-ML/pull/1 | Inisialisasi dokumen kontribusi & notebook awal | Merged |
| 2 | Ruben Alfa Richie | https://github.com/Reska909/UTS-ML/pull/2 | Perbaikan Task 3 (`cluster.csv`), pipeline reproducible `kode/`, Task 1 & 2, dan `laporan.pdf` | Open (Siap Merge) |

## 4. Penggunaan Artificial Intelligence (AI)

Sesuai ketentuan UTS, berikut rincian penggunaan alat bantu AI oleh kelompok:

- **Alat AI yang digunakan**: Antigravity (Google Gemini) dan ChatGPT.
- **Bagian yang dibantu**:
  - Membantu audit kualitas data (identifikasi error satuan menit vs jam pada `durasi_jam`, jendela *delayed label* 21 hari pada `is_fraud`, serta identifikasi 18 fitur noise dan anomali bot pada `clustering.csv.gz`).
  - Membantu merancang skrip Python reproducible 1-perintah di folder `kode/` (termasuk transformasi CLR, fitur behavioral kausal tanpa *future leakage*, *conformal calibration*, dan kalibrasi probabilitas).
  - Membantu penyusunan tata letak dokumen `laporan.pdf` dan perbaikan dokumentasi `README.md`.
- **Cara kelompok memverifikasi output**:
  - Seluruh skrip dijalankan ulang secara lokal dari awal hingga akhir menggunakan dataset resmi UTS (`data/*.csv.gz`) dengan random seed tetap (`42`).
  - Setiap metrik evaluasi, distribusi label, *no quantile crossing* (`p90 >= p50`), rentang probabilitas (`[0, 1]`), jumlah baris, dan kecocokan ID pada `prediksi_regresi.csv`, `prediksi_klasifikasi.csv`, serta `cluster.csv` diperiksa menggunakan *assertion* otomatis dan pengecekan manual.
  - Seluruh keputusan pemodelan dan interpretasi bisnis diverifikasi oleh kedua anggota agar dapat dijelaskan dan direproduksi langsung pada ujian lisan.

## 5. Persetujuan Anggota Kelompok

Kami menyatakan bahwa pembagian tugas, kontribusi, dan rincian penggunaan AI dalam dokumen ini telah diketahui dan disetujui oleh seluruh anggota kelompok.

| Nama | NIM | Persetujuan |
|---|---|---|
| Irfan Reska Nugraha | 101032300042 | Disetujui |
| Ruben Alfa Richie | 101032300008 | Disetujui |
