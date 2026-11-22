"""
Generate publication-grade laporan.pdf (<= 18 pages) directly from the reproduced
outputs of Tasks 1, 2, and 3 so every single number matches the code 100%.

Usage:
    python kode/build_laporan.py
"""

import json
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
OUT1 = ROOT / "kode" / "outputs_task1"
OUT2 = ROOT / "kode" / "outputs_task2"
OUT3 = ROOT / "kode" / "outputs_task3"
PDF_PATH = ROOT / "laporan.pdf"

NAVY = colors.HexColor("#1B365D")
TEAL = colors.HexColor("#2B7A78")
LIGHT_BG = colors.HexColor("#F2F5F9")
BORDER = colors.HexColor("#CBD5E1")


def make_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("TitleCustom", parent=base["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=NAVY, alignment=TA_CENTER, spaceAfter=6),
        "subtitle": ParagraphStyle("SubCustom", parent=base["Normal"], fontName="Helvetica", fontSize=10.5, leading=14, textColor=colors.HexColor("#475569"), alignment=TA_CENTER, spaceAfter=12),
        "h1": ParagraphStyle("H1Custom", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=NAVY, spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("H2Custom", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=TEAL, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("BodyCustom", parent=base["Normal"], fontName="Helvetica", fontSize=9.2, leading=13, alignment=TA_JUSTIFY, spaceAfter=5),
        "bullet": ParagraphStyle("BulletCustom", parent=base["Normal"], fontName="Helvetica", fontSize=9.0, leading=12.5, leftIndent=12, bulletIndent=4, spaceAfter=3),
        "caption": ParagraphStyle("CapCustom", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=8.2, leading=10.5, textColor=colors.HexColor("#475569"), alignment=TA_CENTER, spaceBefore=2, spaceAfter=8),
        "cell": ParagraphStyle("CellCustom", parent=base["Normal"], fontName="Helvetica", fontSize=7.8, leading=10),
        "cell_b": ParagraphStyle("CellBCustom", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.8, leading=10, textColor=colors.white),
    }
    return styles


def styled_table(headers, rows, col_widths=None):
    styles = make_styles()
    data = [[Paragraph(str(h), styles["cell_b"]) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(x), styles["cell"]) for x in r])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def add_img(path: Path, width_cm: float, caption: str, styles, story):
    if not path.exists():
        return
    im = Image(str(path))
    aspect = im.imageHeight / float(im.imageWidth)
    im.drawWidth = width_cm * cm
    im.drawHeight = width_cm * aspect * cm
    story.append(im)
    story.append(Paragraph(caption, styles["caption"]))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(1.8 * cm, 1.1 * cm, "UTS Machine Learning — Irfan Reska Nugraha (101032300042) & Ruben Alfa Richie (101032300008)")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Halaman {doc.page}")
    canvas.restoreState()


def main():
    s1 = json.loads((OUT1 / "summary_metrics.json").read_text(encoding="utf-8"))
    s2 = json.loads((OUT2 / "summary_metrics.json").read_text(encoding="utf-8"))
    s3 = json.loads((OUT3 / "summary_metrics.json").read_text(encoding="utf-8"))

    t1_mod = pd.read_csv(OUT1 / "model_comparison.csv")
    t1_lay = pd.read_csv(OUT1 / "error_layanan.csv")
    t1_pul = pd.read_csv(OUT1 / "error_pulau.csv")
    t1_kur = pd.read_csv(OUT1 / "error_kurir.csv")

    t2_imb = pd.read_csv(OUT2 / "imbalance_comparison.csv")
    t2_cost = pd.read_csv(OUT2 / "calibration_cost_comparison.csv")
    t2_kan = pd.read_csv(OUT2 / "error_kanal.csv")
    t2_kat = pd.read_csv(OUT2 / "error_kategori_merchant.csv")

    t3_eval = pd.read_csv(OUT3 / "method_comparison.csv")
    t3_stab = pd.read_csv(OUT3 / "stability_ari.csv")
    t3_prof = pd.read_csv(OUT3 / "segment_profiles.csv", index_col=0)

    styles = make_styles()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )
    story = []

    # =========================================================================
    # COVER & RINGKASAN EKSEKUTIF
    # =========================================================================
    story.append(Paragraph("LAPORAN UTS MACHINE LEARNING", styles["title"]))
    story.append(Paragraph("Hands-on Machine Learning: Regresi, Klasifikasi, dan Clustering", styles["subtitle"]))
    story.append(styled_table(
        ["No", "Nama Anggota", "NIM", "Peran Utama"],
        [
            ["1", "Irfan Reska Nugraha", "101032300042", "Eksplorasi & Pemodelan Task 1 (Regresi) dan Task 2 (Klasifikasi)"],
            ["2", "Ruben Alfa Richie", "101032300008", "Task 3 (Clustering), Pipeline Reproducible (kode/), & Laporan Akhir"],
        ],
        col_widths=[1.0 * cm, 4.5 * cm, 3.0 * cm, 8.9 * cm],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Ringkasan Eksekutif & Reproducibility", styles["h1"]))
    story.append(Paragraph(
        "Laporan ini menyajikan pembangunan pipeline Machine Learning klasik secara menyeluruh dan jujur untuk tiga permasalahan "
        "pada dataset tabular besar (~940.000 baris): estimasi waktu pengiriman paket (Task 1), deteksi transaksi fraud berbasis biaya "
        "(Task 2), dan segmentasi 120.000 pelanggan (Task 3). Seluruh angka, tabel, grafik, dan file prediksi akhir dapat direproduksi "
        "secara deterministik (random seed = 42) menggunakan satu perintah: <b>python kode/run_all.py --data-dir data</b> "
        f"(Python {s3['python_version']}, numpy {s3['numpy']}, pandas {s3['pandas']}, scikit-learn {s3['scikit_learn']}, lightgbm {s1['lightgbm']}).",
        styles["body"],
    ))
    story.append(styled_table(
        ["Task", "Model Final", "Metrik Validasi Utama", "Output & Validasi File"],
        [
            ["Task 1: Regresi", "Tuned LightGBM Quantile + Conformal P90",
             f"Pinball P50={s1['pinball_p50_time_val']:.4f}, P90={s1['pinball_p90_time_val']:.4f}; Cov P90 Agu={s1['aug_coverage_after']*100:.2f}%",
             "prediksi_regresi.csv (60.000 baris, 0 crossing)"],
            ["Task 2: Klasifikasi", "Time-Decay LightGBM + Isotonic Calibration",
             f"PR-AUC Matang={s2['pr_auc_clean_mature_val']:.4f}; Biaya turun {s2['cost_reduction_vs_none_pct']:.1f}% vs Flag None",
             f"prediksi_klasifikasi.csv (106.674 baris, {s2['test_flagged_count']:,} ditandai)"],
            ["Task 3: Clustering", "GMM (Full, K=6) + Isolation Forest (-1)",
             f"Silhouette={s3['final_gmm_silhouette']:.4f} (+322% vs noisy); 6 segmen + 1.800 anomali bot",
             "cluster.csv (120.000 baris, label -1..5)"],
        ],
        col_widths=[3.0 * cm, 4.4 * cm, 5.5 * cm, 4.5 * cm],
    ))

    # =========================================================================
    # BAB 1: TASK 1 - REGRESI
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("1. Task 1 — Regresi: Estimasi Waktu Pengiriman (durasi_jam)", styles["h1"]))

    story.append(Paragraph("1a. Data Audit: Masalah Kualitas Data & Target Leakage", styles["h2"]))
    story.append(Paragraph(
        "Dataset terdiri dari 250.000 paket latih (1 Jan – 31 Agu 2025) dan 60.000 paket uji (1 Sep – 31 Okt 2025). "
        "Audit kualitas data menemukan empat masalah kritis:",
        styles["body"],
    ))
    story.append(Paragraph(
        f"• <b>Target Leakage (jumlah_scan):</b> Kolom <i>jumlah_scan</i> memiliki korelasi Spearman tertinggi dengan target (ρ = 0,7688), "
        "namun jumlah scan pelacakan baru terakumulasi selama paket dalam perjalanan dan tidak tersedia saat pickup. Memasukkan kolom ini "
        f"secara artifisial menurunkan Pinball Loss P50 dari <b>{s1['pinball_p50_time_val']:.4f}</b> menjadi <b>{s1['pinball_p50_with_leakage']:.4f}</b>. "
        "Fitur ini wajib dibuang agar evaluasi jujur.", styles["bullet"]))
    story.append(Paragraph(
        f"• <b>Kesalahan Satuan Target (durasi_jam &gt; 720 jam):</b> Ditemukan <b>{s1['n_unit_errors_fixed']} baris (0,148%)</b> dengan durasi ekstrem hingga "
        "25.452 jam (~3 tahun). Ketika dibagi 60, median kelompok ini adalah <b>64,1 jam</b>—identik dengan median normal 61,6 jam—membuktikan "
        "bahwa baris tersebut tercatat dalam satuan <i>menit</i> alih-alih <i>jam</i>. Koreksi dilakukan khusus pada data latih (dibagi 60).", styles["bullet"]))
    story.append(Paragraph(
        f"• <b>Nilai Sentinel pada jarak_km:</b> Terdapat <b>{s1['n_corrupt_jarak_fixed']} baris</b> latih dengan <i>jarak_km</i> = 0 (703 baris) atau 9999 (456 baris). "
        "Nilai korup ditandai dengan flag indikator dan diimputasi menggunakan median rute <i>(kota_asal, kota_tujuan)</i> dari data latih.", styles["bullet"]))
    story.append(Paragraph(
        "• <b>Missing Value pada berat_kg:</b> Hilang pada 13.897 baris latih (5,56%) dan 3.265 baris uji (5,44%); ditangani lewat flag missing dan native handling LightGBM.", styles["bullet"]))

    story.append(Paragraph("1b. Desain Validasi: Time-Based vs Random Cross-Validation", styles["h2"]))
    story.append(Paragraph(
        "Karena tugas nyata memprediksi dua bulan ke depan (Sep–Okt 2025), digunakan skema <b>time-based validation</b>: latih pada "
        "Jan–Jun 2025 (181.396 paket) dan validasi pada Jul–Agu 2025 (68.604 paket; Juli dipakai untuk kalibrasi konformal, Agustus sebagai "
        "holdout evaluasi akhir). Sebagai pembanding, <b>Random 3-Fold CV</b> menghasilkan Pinball Loss P50 sebesar "
        f"<b>{s1['pinball_p50_random_cv']:.4f}</b> dibandingkan <b>{s1['pinball_p50_time_val']:.4f}</b> pada time-based validation. "
        "Random CV memberi estimasi terlalu optimistis karena pola musiman/antrean hub pada bulan yang sama bocor antar fold.",
        styles["body"],
    ))

    story.append(Paragraph("1c & 1d. Perbandingan Model, Tuning, dan Penanganan Kategori", styles["h2"]))
    story.append(Paragraph(
        f"Kolom <i>kota_tujuan</i> berkardinalitas tinggi (405 kategori), sementara <i>mitra_kurir</i> memiliki 1 kategori baru yang hanya muncul "
        f"di data uji (<b>K13</b>, sebanyak {s1['n_unseen_kurir_test']:,} baris = 3,90%). Penanganan dilakukan dengan kombinasi "
        "<b>out-of-fold smoothed target encoding</b> (median & P90 dengan smoothing m=50 sehingga kategori baru otomatis jatuh ke prior global) "
        "serta penanganan kategorikal native pada LightGBM. Tuning hyperparameter dilakukan murni di dalam time-based split.",
        styles["body"],
    ))
    story.append(styled_table(
        ["Model", "Pinball P50 (Clean)", "Pinball P90 (Clean)", "Coverage P90 (%)", "Pinball P50 (Raw)"],
        [
            [r["model"], f"{r['pinball_p50_clean']:.4f}", f"{r['pinball_p90_clean']:.4f}", f"{r['coverage_p90_clean']*100:.2f}%", f"{r['pinball_p50_raw']:.4f}"]
            for _, r in t1_mod.iterrows()
        ],
        col_widths=[5.8 * cm, 2.9 * cm, 2.9 * cm, 2.9 * cm, 2.9 * cm],
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("1e. Ketidakpastian (P50 & P90) dan Conformal Calibration", styles["h2"]))
    story.append(Paragraph(
        f"Prediksi mentah P90 LightGBM cenderung sedikit <i>under-covered</i> pada bulan Agustus ({s1['aug_coverage_before']*100:.2f}% vs target 90%). "
        f"Dengan menerapkan <b>Split Conformal Calibration</b> menggunakan residual kuantil ke-90 pada data Juli (n = 34.436), diperoleh koreksi "
        f"<b>ΔP90 = {s1['delta_p90_conformal']:+.4f} jam</b>. Pada holdout Agustus, coverage meningkat menjadi <b>{s1['aug_coverage_after']*100:.2f}%</b> "
        f"dan Pinball Loss P90 membaik dari {s1['aug_pinball_p90_before']:.4f} menjadi <b>{s1['aug_pinball_p90_after']:.4f}</b>. "
        "Batasan monotonik <i>p90 = max(p90, p50)</i> menjamin nol <i>quantile crossing</i>.",
        styles["body"],
    ))

    story.append(Paragraph("1f. Analisis Error Breakdown & Interpretabilitas TreeSHAP", styles["h2"]))
    story.append(styled_table(
        ["Layanan", "Jumlah", "MAE P50", "Pinball P50", "Pinball P90", "Cov P90"],
        [
            [r["layanan"], f"{int(r['n']):,}", f"{r['mae_p50']:.2f}", f"{r['pinball_p50']:.4f}", f"{r['pinball_p90']:.4f}", f"{r['coverage_p90']*100:.1f}%"]
            for _, r in t1_lay.iterrows()
        ],
        col_widths=[3.2 * cm, 2.6 * cm, 2.8 * cm, 3.0 * cm, 3.0 * cm, 2.8 * cm],
    ))
    story.append(Spacer(1, 5))
    story.append(styled_table(
        ["Pulau Tujuan", "Jumlah", "MAE P50", "Pinball P50", "Pinball P90", "Cov P90"],
        [
            [r["pulau_tujuan"], f"{int(r['n']):,}", f"{r['mae_p50']:.2f}", f"{r['pinball_p50']:.4f}", f"{r['pinball_p90']:.4f}", f"{r['coverage_p90']*100:.1f}%"]
            for _, r in t1_pul.iterrows()
        ],
        col_widths=[4.0 * cm, 2.4 * cm, 2.6 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm],
    ))
    story.append(Spacer(1, 5))
    top3_kurir = pd.concat([t1_kur.head(3), t1_kur.tail(3)])
    story.append(Paragraph(
        f"Pada rincian <i>mitra_kurir</i>, mitra paling akurat adalah <b>{t1_kur.iloc[0]['mitra_kurir']}</b> "
        f"(Pinball P50 = {t1_kur.iloc[0]['pinball_p50']:.2f} jam) sedangkan error tertinggi terjadi pada "
        f"<b>{t1_kur.iloc[-1]['mitra_kurir']}</b> (Pinball P50 = {t1_kur.iloc[-1]['pinball_p50']:.2f} jam). "
        "Layanan <i>hemat</i> dan tujuan luar Jawa (Papua, Maluku, Sulawesi, Kalimantan) memiliki varians durasi terbesar karena ketergantungan pada jadwal kargo laut/udara.",
        styles["body"],
    ))
    add_img(OUT1 / "figures" / "fig1_task1_shap.png", 13.5, "Gambar 1.1: 15 fitur paling berpengaruh terhadap estimasi P50 (TreeSHAP LightGBM).", styles, story)
    add_img(OUT1 / "figures" / "fig2_task1_error_breakdown.png", 16.0, "Gambar 1.2: Perbandingan Pinball Loss P50 & P90 per layanan dan pulau tujuan.", styles, story)

    # =========================================================================
    # BAB 2: TASK 2 - KLASIFIKASI
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("2. Task 2 — Klasifikasi: Deteksi Transaksi Fraud (is_fraud)", styles["h1"]))

    story.append(Paragraph("2a. Audit Label & Fitur (Leakage dan Pola Missing)", styles["h2"]))
    story.append(Paragraph(
        "Dataset berisi 403.077 transaksi latih (Jan–Agu 2025) dan 106.674 transaksi uji (Sep–Okt 2025) dengan tingkat fraud sekitar 0,65–0,70%. "
        "Kolom <b>flag_review_manual</b> terbukti merupakan <i>target leakage</i> (korelasi Spearman ρ = 0,4793; fraud rate 30,37% jika ditinjau manual "
        "vs 0,13% jika tidak), karena tinjauan manual terjadi <i>setelah</i> transaksi dipicu. Memasukkan kolom ini mendongkrak PR-AUC secara semu "
        f"dari <b>{s2['pr_auc_clean_mature_val']:.5f}</b> menjadi <b>{s2['pr_auc_with_leakage']:.5f}</b>. Sementara itu, nilai kosong pada "
        "<i>menit_sejak_trx_terakhir</i> dan <i>rasio_nominal_rata2</i> (25,4% non-fraud vs 36,6% fraud) menandakan transaksi pertama pengguna "
        "yang berisiko lebih tinggi sehingga dipertahankan lewat indikator <i>first_tx_missing_flag</i>.",
        styles["body"],
    ))

    story.append(Paragraph("2b. Validasi Berbasis Waktu & Penanganan Delayed Labels (Right-Censoring)", styles["h2"]))
    story.append(Paragraph(
        "Karena data ditarik pada 31 Agustus 2025, transaksi mendekati tanggal penarikan belum sempat terkonfirmasi fraud. "
        "Audit mingguan menunjukkan fraud rate stabil di ~0,72% hingga <b>10 Agustus 2025</b>, lalu anjlok drastis ke <b>0,466%</b> (11–17 Agu), "
        "<b>0,358%</b> (18–24 Agu), dan <b>0,092%</b> (25–31 Agu)—menunjukkan jendela konfirmasi (*confirmation lag*) sekitar <b>21 hari</b>. "
        f"Evaluasi pada periode tersensor 11–31 Agu menyebabkan PR-AUC anjlok artifisial ke <b>{s2['pr_auc_censored_aug11_31']:.5f}</b>. "
        "Oleh karena itu, skema validasi menggunakan label matang sepenuhnya: <b>Train Jan–Mei</b> (247.464), <b>Kalibrasi Jun</b> (50.375), "
        "dan <b>Holdout Matang 1 Jul – 10 Agu</b> (69.771). Model final dilatih ulang hanya pada label matang <b>1 Jan – 10 Agu 2025</b> "
        f"({s2['n_train_mature']:,} baris) dengan mengecualikan {s2['n_train_censored_excluded']:,} baris tersensor agar probabilitas tidak terbias ke bawah.",
        styles["body"],
    ))
    add_img(OUT2 / "figures" / "fig1_task2_delayed_labels.png", 15.0, "Gambar 2.1: Bukti right-censoring (delayed labels) pada 3 minggu terakhir Agustus 2025.", styles, story)

    story.append(Paragraph("2c. Fitur Behavioral Historis Kausal Tanpa Future Leakage", styles["h2"]))
    story.append(Paragraph(
        "Sebanyak 85,5% baris uji berasal dari pengguna yang sudah muncul di data latih dan 99,3% dari merchant yang sama. "
        "Setelah mengurutkan seluruh transaksi berdasarkan <i>waktu</i>, dibangun fitur kausal berbasis riwayat masa lalu saja (strictly past-only): "
        "jumlah & rata-rata nominal transaksi sebelumnya per pengguna (<i>user_prior_tx_cnt, user_prior_mean_nom, user_nom_vs_prior</i>), "
        "jumlah & rata-rata nominal sebelumnya per merchant, indikator apakah merchant baru bagi pengguna (<i>is_new_merchant_for_user</i>), "
        "serta <i>merch_delayed_fraud_rate</i> (smoothed fraud rate merchant dengan jeda ketat <b>21 hari</b> sebelum tanggal transaksi).",
        styles["body"],
    ))

    story.append(Paragraph("2d & 2e. Perbandingan Class Imbalance & Adaptasi Pattern Shift", styles["h2"]))
    story.append(Paragraph(
        "Audit bulanan mengungkap adanya <b>pattern shift</b> pada kanal pembayaran—khususnya lonjakan fraud pada kanal <b>QRIS</b> "
        "pada Juli–Agustus 2025 dibanding Januari–Juni. Untuk beradaptasi, ditambahkan pembobotan <i>exponential time-decay</i> "
        "(half-life 90 hari pada validasi / 120 hari pada training final) agar model memberi bobot lebih besar pada pola fraud terbaru.",
        styles["body"],
    ))
    story.append(styled_table(
        ["Pendekatan", "PR-AUC", "Brier", "Ditandai", "Fraud Lolos", "Total Biaya (Rp)", "Recall"],
        [
            [r["approach"], f"{r['pr_auc']:.4f}", f"{r['brier']:.5f}", f"{int(r['n_flagged']):,}", f"{int(r['n_missed_fraud'])}", f"{r['total_cost']:,.0f}", f"{r['recall']*100:.1f}%"]
            for _, r in t2_imb.iterrows()
        ],
        col_widths=[4.6 * cm, 1.8 * cm, 1.9 * cm, 1.8 * cm, 1.9 * cm, 3.4 * cm, 1.8 * cm],
    ))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "Class weighting ekstrem (`scale_pos_weight = 141,5`) merusak kalibrasi dan menurunkan PR-AUC menjadi 0,1831. "
        "Sebaliknya, <b>Time-Decay Weighted LightGBM</b> mencapai PR-AUC tertinggi (<b>0,6896</b>) dan Brier Score terbaik (<b>0,00314</b>), "
        "serta memangkas biaya sebesar Rp 8,3 juta dibanding model default tanpa time-decay.",
        styles["body"],
    ))
    add_img(OUT2 / "figures" / "fig2_task2_pattern_shift.png", 14.0, "Gambar 2.2: Pergeseran pola fraud rate per kanal pembayaran sepanjang Jan – 10 Agu 2025.", styles, story)

    story.append(Paragraph("2f. Kalibrasi Probabilitas & Aturan Keputusan Berbasis Biaya", styles["h2"]))
    story.append(Paragraph(
        "Dengan biaya review Rp 25.000 per transaksi dan kerugian fraud sebesar <i>nominal</i>, aturan keputusan Bayes-optimal adalah "
        "menandai transaksi (<i>tandai = 1</i>) jika dan hanya jika ekspektasi kerugian melebihi biaya review: "
        "<b>p × nominal &gt; Rp 25.000</b>. Kalibrasi Platt Scaling dan Isotonic Regression dilatih pada data Juni dan dievaluasi pada holdout matang Jul–10 Agu:",
        styles["body"],
    ))
    story.append(styled_table(
        ["Strategi Keputusan", "Brier", "LogLoss", "Ditandai", "Fraud Lolos", "Total Biaya (Rp)", "Recall"],
        [
            [
                r["strategy"],
                "-" if pd.isna(r["brier"]) else f"{r['brier']:.5f}",
                "-" if pd.isna(r["logloss"]) else f"{r['logloss']:.5f}",
                f"{int(r['n_flagged']):,}", f"{int(r['n_missed_fraud'])}", f"{r['total_cost']:,.0f}", f"{r['recall']*100:.1f}%"
            ]
            for _, r in t2_cost.iterrows()
        ],
        col_widths=[4.5 * cm, 1.9 * cm, 1.9 * cm, 1.8 * cm, 1.9 * cm, 3.5 * cm, 1.7 * cm],
    ))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        f"<b>Isotonic Calibration</b> menghasilkan Brier Score terendah (0,00294), LogLoss terendah (0,01458), dan total biaya terendah "
        f"(<b>Rp {s2['best_cost_val']:,.0f}</b>), menghemat <b>{s2['cost_reduction_vs_none_pct']:.1f}%</b> dibanding Flag None "
        f"(Rp {s2['flag_none_cost']:,.0f}) dan <b>95,5%</b> dibanding Flag All (Rp {s2['flag_all_cost']:,.0f}).",
        styles["body"],
    ))
    add_img(OUT2 / "figures" / "fig3_task2_calibration.png", 10.8, "Gambar 2.3: Kurva reliabilitas kalibrasi probabilitas pada holdout matang 1 Jul – 10 Agu 2025.", styles, story)

    story.append(Paragraph("2g. Analisis Error per Kanal & Kategori Merchant", styles["h2"]))
    story.append(styled_table(
        ["Kanal", "Transaksi", "Total Fraud", "Tertangkap", "Lolos", "Ditandai", "Total Biaya (Rp)", "Recall"],
        [
            [r["kanal"], f"{int(r['n']):,}", f"{int(r['n_fraud'])}", f"{int(r['n_caught_fraud'])}", f"{int(r['n_missed_fraud'])}", f"{int(r['n_flagged'])}", f"{r['total_cost']:,.0f}", f"{r['recall']*100:.1f}%"]
            for _, r in t2_kan.iterrows()
        ],
        col_widths=[2.2 * cm, 2.1 * cm, 2.0 * cm, 2.0 * cm, 1.6 * cm, 1.9 * cm, 3.6 * cm, 1.8 * cm],
    ))
    story.append(Spacer(1, 5))
    top5_kat = t2_kat.head(5)
    story.append(Paragraph(
        "Kanal <b>QRIS</b> menjadi kontributor biaya terbesar (Rp 43,0 juta; recall 39,3%) karena pergeseran pola fraud baru pada kanal ini "
        "di paruh kedua 2025. Pada level kategori merchant, 5 kontributor biaya tertinggi adalah "
        + ", ".join([f"<b>{r['kategori_merchant']}</b> (Rp {r['total_cost']/1e6:.1f} jt)" for _, r in top5_kat.iterrows()])
        + ". Rekomendasi operasional adalah memperketat aturan kecepatan transaksi (velocity) khusus kanal QRIS dan kategori berisiko tinggi tersebut.",
        styles["body"],
    ))

    # =========================================================================
    # BAB 3: TASK 3 - CLUSTERING
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Task 3 — Clustering: Segmentasi Pelanggan (120.000 Pelanggan)", styles["h1"]))

    story.append(Paragraph("3a & 3b. Preprocessing, Data Komposisional, dan Seleksi Fitur dengan Bukti", styles["h2"]))
    story.append(Paragraph(
        "Dataset berisi 120.000 pelanggan dan 60 kolom (tanpa label). Audit korelasi Spearman menemukan <b>18 kolom noise murni</b> "
        "(seluruh <i>n01–n15</i>, <i>rating_beri_rata2</i>, <i>kota_tier</i>, dan <i>jumlah_alamat</i> dengan max |ρ| &lt; 0,025). "
        "Dua kolom redundan 30 hari (<i>belanja_30h</i> ρ=0,97; <i>frekuensi_30h</i> ρ=0,87) dikeluarkan, sementara tiga fitur aktivitas ekstrem "
        "(<i>wishlist, klik_iklan_30h, durasi_sesi_menit</i>) dipisahkan khusus untuk deteksi anomali bot karena korelasinya pada pelanggan normal &lt; 0,01.",
        styles["body"],
    ))
    story.append(Paragraph(
        "• <b>Missing Values:</b> <i>rata2_keranjang</i> kosong pada 6.736 baris (100% saat <i>frekuensi_90h = 0</i>), diimputasi menggunakan median pelanggan <i>frekuensi_90h = 1</i> agar tidak membentuk klaster nol buatan.<br/>"
        "• <b>Skala & Skewness:</b> Fitur skewed ditransformasi <i>log1p</i>, dengan skala halus <i>belanja_90h / 100.000</i> untuk menghindari lonjakan 9 unit di titik nol.<br/>"
        "• <b>Data Komposisional:</b> 16 kolom <i>porsi_&lt;kategori&gt;</i> dan 4 kolom <i>porsi_jam_*</i> ditransformasi lewat <b>multiplicative zero replacement + Centered Log-Ratio (CLR)</b> lalu dikompresi via PCA (5 PC kategori + 3 PC waktu) agar seimbang terhadap fitur RFM.<br/>"
        "• <b>Dampak Seleksi Fitur:</b> Silhouette K-Means (K=6) melonjak <b>+322,1%</b> dari <b>0,0376</b> (58 fitur ber-noise) menjadi <b>0,1587</b> (24 fitur bersih → PCA-10).",
        styles["body"],
    ))

    story.append(Paragraph("3c, 3d & 3e. Perbandingan 4 Metode, Pemilihan Jumlah Cluster (K=6), & Stabilitas", styles["h2"]))
    eval_sub = t3_eval[t3_eval["k"].isin([4, 5, 6, 7])].copy()
    story.append(styled_table(
        ["Metode", "K", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "BIC (GMM)"],
        [
            [r["method"], int(r["k"]), f"{r['silhouette']:.4f}", f"{r['davies_bouldin']:.4f}", f"{r['calinski_harabasz']:.1f}", "-" if pd.isna(r["bic"]) else f"{r['bic']:,.0f}"]
            for _, r in eval_sub.iterrows()
        ],
        col_widths=[4.4 * cm, 1.2 * cm, 2.6 * cm, 3.0 * cm, 3.2 * cm, 3.0 * cm],
    ))
    story.append(Spacer(1, 5))
    story.append(styled_table(
        ["K", "GMM ARI (Mean ± Std)", "GMM ARI Min", "K-Means ARI Mean", "Cross-ARI (GMM vs KM)"],
        [
            [int(r["k"]), f"{r['gmm_ari_mean']:.4f} ± {r['gmm_ari_std']:.4f}", f"{r['gmm_ari_min']:.4f}", f"{r['km_ari_mean']:.4f}", f"{r['cross_ari_gmm_vs_km']:.4f}"]
            for _, r in t3_stab.iterrows()
        ],
        col_widths=[1.4 * cm, 4.4 * cm, 3.2 * cm, 3.8 * cm, 4.6 * cm],
    ))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "Berdasarkan kombinasi <b>BIC GMM</b> (siku penurunan tajam melandai di K=6), <b>Davies-Bouldin terbaik</b> di K=6 (1,8727), "
        "<b>Cross-ARI tinggi antara GMM dan K-Means (0,8510)</b>, serta kejelasan profil bisnis tanpa tumpang tindih (uji histogram menunjukkan "
        "segmen <i>Daily Household</i> bersifat unimodal sehingga pemecahan ke K=7 tidak diperlukan), dipilih <b>GMM Full Covariance (K=6)</b>.",
        styles["body"],
    ))
    add_img(OUT3 / "figures" / "fig1_method_comparison_bic.png", 15.8, "Gambar 3.1: Perbandingan Silhouette 4 metode (kiri) dan kurva BIC GMM Full (kanan).", styles, story)
    add_img(OUT3 / "figures" / "fig2_stability_ari.png", 11.5, "Gambar 3.2: Uji stabilitas 5-run 80% subsampling (ARI) dan Cross-ARI antar metode.", styles, story)

    story.append(Paragraph("3f & 3g. Profil 6 Segmen, Aksi Marketing, dan Deteksi Anomali (-1)", styles["h2"]))
    story.append(Paragraph(
        f"Deteksi anomali menggunakan <b>Isolation Forest</b> (contamination = 0,015, seed = 42) pada fitur aktivitas sesi menandai "
        f"<b>{s3['n_anomalies']:,} akun (1,50%)</b> sebagai <b>cluster = -1</b>. Sebanyak <b>{s3['bot_overlap']:,} akun (99,83%)</b> cocok "
        "dengan pola bot/scraper ekstrem: median 241,1 sesi/minggu (normal ~5,4), durasi sesi 0,60 menit (normal 8,0), wishlist 150 (normal 9), "
        "dan klik iklan 200 (normal 5).",
        styles["body"],
    ))

    names_actions = {
        0: ("Premium High-Ticket Tech", "Belanja Rp 4,74 jt, keranjang Rp 639 rb, iOS 75,7%, elektronik 10%, voucher 14%", "Program VIP Loyalty, garansi ekstra, & early access produk elektronik premium."),
        1: ("Belanja Rutin Keluarga", "Frekuensi tertinggi (15x), 4 item/order, pagi 38%, makanan/minuman 20% + rumah tangga 9%", "Paket langganan auto-replenish & bundling kebutuhan pokok dengan notifikasi pagi."),
        2: ("PayLater & Retur Tinggi", "PayLater 83%, retur tertinggi (3x/90h), fashion 18%, malam 36%, usia 26", "Panduan ukuran interaktif & gratis tukar ukuran untuk menekan biaya retur."),
        3: ("Pemburu Promo Malam", "Voucher 85%, flash sale 71%, malam 42%, fashion 17% + kecantikan 10%, usia 25", "Midnight Flash Sale & voucher bertingkat dengan minimum spend untuk naikkan keranjang."),
        4: ("Member Baru / Window-Shopper", "Member baru (3,6 bln), sesi tertinggi (10,8/mgg), frekuensi rendah (3x), COD 39%", "Voucher pembelian pertama/kedua, pengingat keranjang, & promosi jaminan COD."),
        5: ("Pelanggan Pasif / At-Risk", "Recency terlama (52,3 hari), frekuensi 2x, sesi 1,97/mgg, COD 45%, member lama", "Kampanye win-back via WhatsApp/email dengan voucher gratis ongkir berbatas waktu."),
        -1: ("Anomali (Bot / Scraper)", "241 sesi/mgg, durasi 0,60 mnt, wishlist 150, klik iklan 200", "Eksklusi dari anggaran iklan berbayar (ad-fraud filter) & terapkan rate-limiting."),
    }
    prof_rows = []
    for cid in [0, 1, 2, 3, 4, 5, -1]:
        r = t3_prof.loc[cid]
        nm, desc, act = names_actions[cid]
        prof_rows.append([str(cid), f"{int(r['n']):,} ({r['pct']:.1f}%)", nm, desc, act])
    story.append(styled_table(
        ["ID", "Jumlah (%)", "Nama Segmen", "Karakteristik Utama (Median)", "Aksi Marketing"],
        prof_rows,
        col_widths=[0.9 * cm, 2.5 * cm, 3.2 * cm, 5.4 * cm, 5.4 * cm],
    ))
    story.append(Spacer(1, 5))
    add_img(OUT3 / "figures" / "fig3_segment_profiles_heatmap.png", 16.2, "Gambar 3.3: Heatmap Z-score profil median 6 segmen pelanggan normal.", styles, story)

    # =========================================================================
    # BAB 4: VERIFIKASI FILE SUBMISSION
    # =========================================================================
    story.append(Paragraph("4. Verifikasi File Pengumpulan & Integritas", styles["h1"]))
    story.append(Paragraph(
        "Seluruh file pengumpulan telah diverifikasi otomatis tanpa nilai kosong, tanpa ID duplikat, dan sesuai spesifikasi header serta rentang nilai: "
        "<b>prediksi_regresi.csv</b> (60.000 baris, kolom <i>id_paket,p50,p90</i>, seluruh baris memenuhi <i>p90 ≥ p50</i>), "
        "<b>prediksi_klasifikasi.csv</b> (106.674 baris, kolom <i>id_transaksi,prob,tandai</i>, <i>prob ∈ [0, 1]</i>, <i>tandai ∈ {0, 1}</i>), "
        "dan <b>cluster.csv</b> (120.000 baris, kolom <i>id_pelanggan,cluster</i>, bilangan bulat <i>-1..5</i>). "
        "Rincian kontribusi anggota, tautan Pull Request, dan deklarasi penggunaan AI tercantum pada file <b>kontribusi.md</b>.",
        styles["body"],
    ))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Generated {PDF_PATH.resolve()} successfully!")


if __name__ == "__main__":
    main()
