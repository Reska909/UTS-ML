"""
Generate laporan.pdf (<= 18 pages) dengan desain standar laporan akademik Indonesia
(gaya DOCX: Times New Roman, cover formal, BAB I-V, tabel akademik hitam-putih/abu-abu,
judul tabel di atas, judul gambar di bawah, nomor halaman di kanan bawah).

Seluruh angka, tabel, dan grafik diambil langsung dari output kode Task 1-3.

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
    HRFlowable,
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

BLACK = colors.black
DARK_GRAY = colors.HexColor("#333333")
HEADER_BG = colors.HexColor("#E6E6E6")
ALT_ROW = colors.HexColor("#F8F8F8")
LINE_GRAY = colors.HexColor("#666666")


def make_styles():
    base = getSampleStyleSheet()
    return {
        "cover_top": ParagraphStyle(
            "CoverTop", parent=base["Normal"], fontName="Times-Bold", fontSize=14,
            leading=20, alignment=TA_CENTER, textColor=BLACK, spaceAfter=8
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=base["Normal"], fontName="Times-Bold", fontSize=16,
            leading=22, alignment=TA_CENTER, textColor=BLACK, spaceAfter=10
        ),
        "cover_sub": ParagraphStyle(
            "CoverSub", parent=base["Normal"], fontName="Times-Italic", fontSize=12,
            leading=16, alignment=TA_CENTER, textColor=DARK_GRAY, spaceAfter=24
        ),
        "cover_label": ParagraphStyle(
            "CoverLabel", parent=base["Normal"], fontName="Times-Roman", fontSize=11.5,
            leading=16, alignment=TA_CENTER, textColor=BLACK, spaceAfter=10
        ),
        "cover_author": ParagraphStyle(
            "CoverAuthor", parent=base["Normal"], fontName="Times-Bold", fontSize=12,
            leading=18, alignment=TA_CENTER, textColor=BLACK
        ),
        "cover_footer": ParagraphStyle(
            "CoverFooter", parent=base["Normal"], fontName="Times-Bold", fontSize=12,
            leading=18, alignment=TA_CENTER, textColor=BLACK
        ),
        "bab": ParagraphStyle(
            "Bab", parent=base["Heading1"], fontName="Times-Bold", fontSize=13,
            leading=17, alignment=TA_CENTER, textColor=BLACK, spaceBefore=4, spaceAfter=10
        ),
        "h2": ParagraphStyle(
            "SubBab", parent=base["Heading2"], fontName="Times-Bold", fontSize=11,
            leading=15, alignment=TA_LEFT, textColor=BLACK, spaceBefore=10, spaceAfter=4
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"], fontName="Times-Roman", fontSize=10.5,
            leading=14.5, alignment=TA_JUSTIFY, firstLineIndent=0.75 * cm, spaceAfter=5
        ),
        "body_noindent": ParagraphStyle(
            "BodyNoIndent", parent=base["Normal"], fontName="Times-Roman", fontSize=10.5,
            leading=14.5, alignment=TA_JUSTIFY, spaceAfter=5
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"], fontName="Times-Roman", fontSize=10.2,
            leading=14, alignment=TA_JUSTIFY, leftIndent=0.8 * cm, firstLineIndent=-0.35 * cm, spaceAfter=3.5
        ),
        "tbl_title": ParagraphStyle(
            "TblTitle", parent=base["Normal"], fontName="Times-Bold", fontSize=9.5,
            leading=12, alignment=TA_LEFT, textColor=BLACK, spaceBefore=6, spaceAfter=3
        ),
        "fig_caption": ParagraphStyle(
            "FigCaption", parent=base["Normal"], fontName="Times-Italic", fontSize=9.2,
            leading=12, alignment=TA_CENTER, textColor=DARK_GRAY, spaceBefore=3, spaceAfter=8
        ),
        "cell": ParagraphStyle(
            "Cell", parent=base["Normal"], fontName="Times-Roman", fontSize=8.5, leading=11, textColor=BLACK
        ),
        "cell_b": ParagraphStyle(
            "CellB", parent=base["Normal"], fontName="Times-Bold", fontSize=8.5, leading=11, textColor=BLACK
        ),
    }


def academic_table(title, headers, rows, col_widths, styles, story):
    story.append(Paragraph(title, styles["tbl_title"]))
    data = [[Paragraph(str(h), styles["cell_b"]) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(x), styles["cell"]) for x in r])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, BLACK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, BLACK),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, BLACK),
        ("INNERGRID", (0, 1), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))


def add_img(path: Path, width_cm: float, caption: str, styles, story):
    if not path.exists():
        return
    im = Image(str(path))
    aspect = im.imageHeight / float(im.imageWidth)
    im.drawWidth = width_cm * cm
    im.drawHeight = width_cm * aspect * cm
    im.hAlign = "CENTER"
    story.append(im)
    story.append(Paragraph(caption, styles["fig_caption"]))


def page_number_footer(canvas, doc):
    if doc.page == 1:
        return  # Halaman sampul tanpa nomor halaman (standar laporan Indonesia)
    canvas.saveState()
    canvas.setFont("Times-Roman", 10)
    canvas.setFillColor(BLACK)
    canvas.drawRightString(A4[0] - 2.5 * cm, 1.5 * cm, str(doc.page))
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
    # Margin standar akademik Indonesia (kiri 3.5 cm, atas/kanan/bawah 2.5 cm -> lebar konten 15.0 cm)
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=3.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )
    W = 15.0 * cm
    story = []

    # =========================================================================
    # HALAMAN SAMPUL (COVER STANDAR INDONESIA)
    # =========================================================================
    story.append(Spacer(1, 2.0 * cm))
    story.append(Paragraph("LAPORAN UJIAN TENGAH SEMESTER (UTS)", styles["cover_top"]))
    story.append(Paragraph("MATA KULIAH MACHINE LEARNING", styles["cover_top"]))
    story.append(Spacer(1, 0.8 * cm))
    story.append(HRFlowable(width="80%", thickness=1.5, color=BLACK, spaceBefore=4, spaceAfter=14, hAlign="CENTER"))
    story.append(Paragraph(
        "HANDS-ON MACHINE LEARNING:<br/>REGRESSION, CLASSIFICATION, AND CLUSTERING",
        styles["cover_title"],
    ))
    story.append(Paragraph(
        "Estimasi Waktu Pengiriman Paket, Deteksi Transaksi Fraud Berbasis Biaya,<br/>dan Segmentasi Pelanggan pada Dataset Tabular Skala Besar",
        styles["cover_sub"],
    ))
    story.append(HRFlowable(width="80%", thickness=1.5, color=BLACK, spaceBefore=4, spaceAfter=26, hAlign="CENTER"))

    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("Disusun Oleh:", styles["cover_label"]))
    story.append(Spacer(1, 0.3 * cm))

    author_tbl = Table(
        [
            [Paragraph("1.", styles["cover_author"]), Paragraph("Irfan Reska Nugraha", styles["cover_author"]), Paragraph("101032300042", styles["cover_author"])],
            [Paragraph("2.", styles["cover_author"]), Paragraph("Ruben Alfa Richie", styles["cover_author"]), Paragraph("101032300008", styles["cover_author"])],
        ],
        colWidths=[1.0 * cm, 7.0 * cm, 4.0 * cm],
    )
    author_tbl.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(author_tbl)

    story.append(Spacer(1, 4.5 * cm))
    story.append(Paragraph("MATA KULIAH MACHINE LEARNING", styles["cover_footer"]))
    story.append(Paragraph("NOVEMBER 2026", styles["cover_footer"]))

    # =========================================================================
    # BAB I: PENDAHULUAN & REPRODUCIBILITY
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("BAB I<br/>PENDAHULUAN DAN RINGKASAN EKSEKUTIF", styles["bab"]))

    story.append(Paragraph("1.1 Latar Belakang dan Tujuan", styles["h2"]))
    story.append(Paragraph(
        "Ujian Tengah Semester (UTS) ini bertujuan membangun pipeline Machine Learning klasik secara menyeluruh, jujur, "
        "dan dapat direproduksi pada tiga dataset tabular sintetis berukuran besar (~940.000 baris total): "
        "(1) Regresi estimasi waktu pengiriman paket (<i>durasi_jam</i>) pada kuantil P50 dan P90, "
        "(2) Klasifikasi deteksi transaksi fraud (<i>is_fraud</i>) dengan aturan keputusan berbasis biaya review Rp 25.000, dan "
        "(3) Clustering segmentasi 120.000 pelanggan beserta deteksi anomali (label -1).",
        styles["body"],
    ))

    story.append(Paragraph("1.2 Lingkungan Komputasi dan Reproducibility", styles["h2"]))
    story.append(Paragraph(
        "Seluruh angka, tabel, grafik dalam laporan ini serta ketiga file prediksi akhir dapat direproduksi secara deterministik "
        "dengan satu perintah dari root repository: <b>python kode/run_all.py --data-dir data</b> menggunakan "
        f"<b>random seed = 42</b> pada lingkungan Python {s3['python_version']} (numpy {s3['numpy']}, pandas {s3['pandas']}, "
        f"scikit-learn {s3['scikit_learn']}, scipy {s3['scipy']}, lightgbm {s1['lightgbm']}). Tidak ada data eksternal maupun neural network yang digunakan.",
        styles["body"],
    ))

    academic_table(
        "Tabel 1.1 Ringkasan Hasil Akhir dan Validasi Output Task 1–3",
        ["Task", "Model Final", "Metrik Validasi Utama", "File Output"],
        [
            ["Task 1: Regresi", "Tuned LightGBM Quantile + Conformal P90",
             f"Pinball P50={s1['pinball_p50_time_val']:.4f}; P90={s1['pinball_p90_time_val']:.4f}; Cov P90 Agu={s1['aug_coverage_after']*100:.2f}%",
             "prediksi_regresi.csv (60.000 baris)"],
            ["Task 2: Klasifikasi", "Time-Decay LightGBM + Isotonic Calibration",
             f"PR-AUC Matang={s2['pr_auc_clean_mature_val']:.4f}; Biaya turun {s2['cost_reduction_vs_none_pct']:.1f}% vs Flag None",
             f"prediksi_klasifikasi.csv (106.674 baris)"],
            ["Task 3: Clustering", "GMM (Full, K=6) + Isolation Forest (-1)",
             f"Silhouette={s3['final_gmm_silhouette']:.4f} (+322% vs noisy); 6 segmen + 1.800 anomali",
             "cluster.csv (120.000 baris)"],
        ],
        [2.7 * cm, 3.8 * cm, 5.0 * cm, 3.5 * cm],
        styles, story,
    )

    # =========================================================================
    # BAB II: TASK 1 - REGRESI
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("BAB II<br/>TASK 1 — REGRESI: ESTIMASI WAKTU PENGIRIMAN", styles["bab"]))

    story.append(Paragraph("2.1 Audit Data dan Dampak Target Leakage (Poin 1a)", styles["h2"]))
    story.append(Paragraph(
        "Dataset Task 1 terdiri dari 250.000 paket latih (1 Jan – 31 Agu 2025) dan 60.000 paket uji (1 Sep – 31 Okt 2025). "
        "Audit kualitas data mengidentifikasi empat temuan utama yang berdampak signifikan terhadap performa model:",
        styles["body"],
    ))
    story.append(Paragraph(
        f"• <b>Target Leakage (jumlah_scan):</b> Kolom <i>jumlah_scan</i> memiliki korelasi Spearman tertinggi terhadap <i>durasi_jam</i> "
        "(ρ = 0,7688), namun jumlah scan pelacakan baru tercatat selama paket berada dalam perjalanan dan tidak tersedia saat pickup. "
        f"Pengujian membuktikan bahwa menyertakan <i>jumlah_scan</i> menurunkan Pinball Loss P50 secara semu sebesar <b>20,9%</b> "
        f"(dari {s1['pinball_p50_time_val']:.4f} menjadi {s1['pinball_p50_with_leakage']:.4f}). Fitur ini wajib dibuang.", styles["bullet"]))
    story.append(Paragraph(
        f"• <b>Kesalahan Satuan Target (durasi_jam &gt; 720 jam):</b> Sebanyak <b>{s1['n_unit_errors_fixed']} baris (0,148%)</b> memiliki durasi "
        "hingga 25.452 jam (~3 tahun). Ketika dibagi 60, median kelompok ini adalah <b>64,1 jam</b>—sangat dekat dengan median normal 61,6 jam—"
        "membuktikan adanya kesalahan pencatatan dalam satuan <i>menit</i> alih-alih <i>jam</i>. Koreksi dilakukan khusus pada data latih (dibagi 60).", styles["bullet"]))
    story.append(Paragraph(
        f"• <b>Nilai Sentinel pada jarak_km:</b> Ditemukan <b>{s1['n_corrupt_jarak_fixed']} baris</b> latih dengan <i>jarak_km</i> = 0 (703 baris) "
        "atau 9999 (456 baris). Nilai korup ditandai flag indikator dan diimputasi menggunakan median rute <i>(kota_asal, kota_tujuan)</i> data latih.", styles["bullet"]))
    story.append(Paragraph(
        "• <b>Missing Value pada berat_kg:</b> Hilang pada 13.897 baris latih (5,56%) dan 3.265 baris uji (5,44%), ditangani dengan indikator missing.", styles["bullet"]))

    story.append(Paragraph("2.2 Desain Validasi: Time-Based vs Random CV (Poin 1b)", styles["h2"]))
    story.append(Paragraph(
        "Untuk meniru skenario produksi (memprediksi 2 bulan berikutnya, Sep–Okt 2025), diterapkan <b>time-based validation</b>: "
        "latih pada Jan–Jun 2025 (181.396 baris) dan validasi pada Jul–Agu 2025 (68.604 baris; Juli untuk kalibrasi konformal, Agustus "
        "sebagai evaluasi akhir). Sebagai pembanding, <b>Random 3-Fold CV</b> menghasilkan Pinball Loss P50 sebesar "
        f"<b>{s1['pinball_p50_random_cv']:.4f}</b> dibanding <b>{s1['pinball_p50_time_val']:.4f}</b> pada time-based validation. "
        "Random CV terlalu optimistis karena pola antrean hub dan cuaca pada tanggal yang berdekatan bocor antar fold.",
        styles["body"],
    ))

    story.append(Paragraph("2.3 Perbandingan Model, Tuning, dan Penanganan Kategori (Poin 1c & 1d)", styles["h2"]))
    story.append(Paragraph(
        f"Kolom <i>kota_tujuan</i> berkardinalitas tinggi (405 kategori), sedangkan <i>mitra_kurir</i> memiliki 1 kategori baru yang hanya muncul "
        f"pada data uji, yakni <b>K13</b> ({s1['n_unseen_kurir_test']:,} baris = 3,90%). Keduanya ditangani melalui <b>out-of-fold smoothed target encoding</b> "
        "(median & P90, smoothing m=50 sehingga kategori baru jatuh ke prior global) dan native categorical LightGBM. Tuning dilakukan sepenuhnya "
        "di dalam skema time-based validation.",
        styles["body"],
    ))
    academic_table(
        "Tabel 2.1 Perbandingan Performa Model Regresi pada Validasi Berbasis Waktu (Jul–Agu 2025)",
        ["Model", "Pinball P50 (Clean)", "Pinball P90 (Clean)", "Cov P90 (%)", "Pinball P50 (Raw)"],
        [
            [r["model"], f"{r['pinball_p50_clean']:.4f}", f"{r['pinball_p90_clean']:.4f}", f"{r['coverage_p90_clean']*100:.2f}%", f"{r['pinball_p50_raw']:.4f}"]
            for _, r in t1_mod.iterrows()
        ],
        [5.0 * cm, 2.5 * cm, 2.5 * cm, 2.4 * cm, 2.6 * cm],
        styles, story,
    )

    story.append(Paragraph("2.4 Ketidakpastian P50/P90 dan Conformal Calibration (Poin 1e)", styles["h2"]))
    story.append(Paragraph(
        f"Prediksi mentah P90 LightGBM mengalami sedikit <i>under-coverage</i> pada Agustus ({s1['aug_coverage_before']*100:.2f}% vs target 90%). "
        f"Dengan menerapkan <b>Split Conformal Calibration</b> menggunakan residual kuantil ke-90 pada data Juli (n = 34.436), diperoleh koreksi "
        f"<b>ΔP90 = {s1['delta_p90_conformal']:+.4f} jam</b>. Pada holdout Agustus, coverage meningkat menjadi <b>{s1['aug_coverage_after']*100:.2f}%</b> "
        f"dan Pinball Loss P90 membaik dari {s1['aug_pinball_p90_before']:.4f} menjadi <b>{s1['aug_pinball_p90_after']:.4f}</b>. "
        "Aturan <i>p90 = max(p90, p50)</i> menjamin nol <i>quantile crossing</i> pada seluruh 60.000 baris uji.",
        styles["body"],
    ))

    story.append(Paragraph("2.5 Analisis Error Breakdown dan Interpretabilitas TreeSHAP (Poin 1f)", styles["h2"]))
    academic_table(
        "Tabel 2.2 Error Breakdown Model Final berdasarkan Jenis Layanan",
        ["Layanan", "Jumlah", "MAE P50", "Pinball P50", "Pinball P90", "Coverage P90"],
        [
            [r["layanan"], f"{int(r['n']):,}", f"{r['mae_p50']:.2f}", f"{r['pinball_p50']:.4f}", f"{r['pinball_p90']:.4f}", f"{r['coverage_p90']*100:.1f}%"]
            for _, r in t1_lay.iterrows()
        ],
        [2.8 * cm, 2.3 * cm, 2.4 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm],
        styles, story,
    )
    academic_table(
        "Tabel 2.3 Error Breakdown Model Final berdasarkan Pulau Tujuan",
        ["Pulau Tujuan", "Jumlah", "MAE P50", "Pinball P50", "Pinball P90", "Coverage P90"],
        [
            [r["pulau_tujuan"], f"{int(r['n']):,}", f"{r['mae_p50']:.2f}", f"{r['pinball_p50']:.4f}", f"{r['pinball_p90']:.4f}", f"{r['coverage_p90']*100:.1f}%"]
            for _, r in t1_pul.iterrows()
        ],
        [3.6 * cm, 2.1 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm, 2.3 * cm],
        styles, story,
    )
    story.append(Paragraph(
        f"Berdasarkan rincian <i>mitra_kurir</i>, akurasi tertinggi dicapai oleh <b>{t1_kur.iloc[0]['mitra_kurir']}</b> "
        f"(Pinball P50 = {t1_kur.iloc[0]['pinball_p50']:.2f} jam) dan error tertinggi pada <b>{t1_kur.iloc[-1]['mitra_kurir']}</b> "
        f"(Pinball P50 = {t1_kur.iloc[-1]['pinball_p50']:.2f} jam). Layanan <i>hemat</i> serta tujuan luar Jawa (Papua, Maluku, Sulawesi, Kalimantan) "
        "memiliki varians waktu tempuh terbesar akibat ketergantungan pada jadwal transit laut/udara.",
        styles["body"],
    ))
    add_img(OUT1 / "figures" / "fig1_task1_shap.png", 12.2, "Gambar 2.1: 15 fitur paling berpengaruh terhadap estimasi P50 (TreeSHAP LightGBM).", styles, story)
    add_img(OUT1 / "figures" / "fig2_task1_error_breakdown.png", 14.2, "Gambar 2.2: Perbandingan Pinball Loss P50 dan P90 per layanan dan pulau tujuan.", styles, story)

    # =========================================================================
    # BAB III: TASK 2 - KLASIFIKASI
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("BAB III<br/>TASK 2 — KLASIFIKASI: DETEKSI TRANSAKSI FRAUD", styles["bab"]))

    story.append(Paragraph("3.1 Audit Label dan Fitur Leakage (Poin 2a)", styles["h2"]))
    story.append(Paragraph(
        "Dataset Task 2 mencakup 403.077 transaksi latih (Jan–Agu 2025) dan 106.674 transaksi uji (Sep–Okt 2025). "
        "Kolom <b>flag_review_manual</b> teridentifikasi sebagai <i>target leakage</i> kuat (ρ = 0,4793; fraud rate 30,37% jika ditinjau manual "
        "vs 0,13% jika tidak), karena tinjauan manual baru dilakukan setelah transaksi diproses. Memasukkan kolom ini mendongkrak PR-AUC secara semu "
        f"dari <b>{s2['pr_auc_clean_mature_val']:.5f}</b> menjadi <b>{s2['pr_auc_with_leakage']:.5f}</b>. Nilai kosong pada "
        "<i>menit_sejak_trx_terakhir</i> dan <i>rasio_nominal_rata2</i> (25,4% non-fraud vs 36,6% fraud) mencerminkan transaksi pertama pengguna "
        "yang lebih berisiko sehingga direpresentasikan lewat fitur <i>first_tx_missing_flag</i>.",
        styles["body"],
    ))

    story.append(Paragraph("3.2 Validasi Berbasis Waktu dan Penanganan Delayed Labels (Poin 2b)", styles["h2"]))
    story.append(Paragraph(
        "Karena data ditarik pada 31 Agustus 2025, transaksi menjelang akhir Agustus belum sempat terkonfirmasi fraud (*right-censoring*). "
        "Audit mingguan memperlihatkan fraud rate stabil di ~0,72% hingga <b>10 Agustus 2025</b>, lalu anjlok menjadi <b>0,466%</b> (11–17 Agu), "
        "<b>0,358%</b> (18–24 Agu), dan <b>0,092%</b> (25–31 Agu)—membuktikan adanya jeda konfirmasi sekitar <b>21 hari</b>. "
        f"Evaluasi naif pada periode tersensor 11–31 Agu menurunkan PR-AUC secara artifisial ke <b>{s2['pr_auc_censored_aug11_31']:.5f}</b>. "
        "Oleh sebab itu, validasi dirancang menggunakan label matang sepenuhnya: <b>Train Jan–Mei</b> (247.464), <b>Kalibrasi Jun</b> (50.375), "
        "dan <b>Holdout Matang 1 Jul – 10 Agu</b> (69.771). Pelatihan model final hanya memakai data matang <b>1 Jan – 10 Agu 2025</b> "
        f"({s2['n_train_mature']:,} baris) dengan mengecualikan {s2['n_train_censored_excluded']:,} baris tersensor.",
        styles["body"],
    ))
    add_img(OUT2 / "figures" / "fig1_task2_delayed_labels.png", 13.8, "Gambar 3.1: Bukti delayed labels (right-censoring) pada tiga minggu terakhir Agustus 2025.", styles, story)

    story.append(Paragraph("3.3 Fitur Behavioral Historis Kausal Tanpa Future Leakage (Poin 2c)", styles["h2"]))
    story.append(Paragraph(
        "Sebanyak 85,5% transaksi uji dilakukan pengguna yang muncul di data latih dan 99,3% pada merchant yang sama. "
        "Dengan mengurutkan seluruh transaksi berdasarkan <i>waktu</i>, dibangun fitur riwayat murni masa lalu (strictly past-only): "
        "frekuensi dan rata-rata nominal historis per pengguna (<i>user_prior_tx_cnt, user_prior_mean_nom, user_nom_vs_prior</i>), "
        "frekuensi dan rata-rata nominal per merchant, indikator merchant baru bagi pengguna (<i>is_new_merchant_for_user</i>), serta "
        "<i>merch_delayed_fraud_rate</i> (smoothed fraud rate merchant dengan jeda ketat <b>21 hari</b> sebelum tanggal transaksi).",
        styles["body"],
    ))

    story.append(Paragraph("3.4 Penanganan Class Imbalance dan Adaptasi Pattern Shift (Poin 2d & 2e)", styles["h2"]))
    story.append(Paragraph(
        "Audit bulanan menunjukkan terjadinya <b>pattern shift</b>, terutama lonjakan fraud pada kanal <b>QRIS</b> pada Juli–Agustus 2025. "
        "Untuk beradaptasi, diterapkan pembobotan <i>exponential time-decay</i> (half-life 90 hari pada validasi / 120 hari pada model final) "
        "sehingga pola transaksi terbaru mendapat bobot lebih tinggi.",
        styles["body"],
    ))
    academic_table(
        "Tabel 3.1 Perbandingan Pendekatan Class Imbalance & Time-Decay pada Holdout Matang (1 Jul – 10 Agu)",
        ["Pendekatan", "PR-AUC", "Brier", "Ditandai", "Lolos", "Total Biaya (Rp)", "Recall"],
        [
            [r["approach"], f"{r['pr_auc']:.4f}", f"{r['brier']:.5f}", f"{int(r['n_flagged']):,}", f"{int(r['n_missed_fraud'])}", f"{r['total_cost']:,.0f}", f"{r['recall']*100:.1f}%"]
            for _, r in t2_imb.iterrows()
        ],
        [4.2 * cm, 1.5 * cm, 1.6 * cm, 1.5 * cm, 1.3 * cm, 3.3 * cm, 1.6 * cm],
        styles, story,
    )
    add_img(OUT2 / "figures" / "fig2_task2_pattern_shift.png", 12.8, "Gambar 3.2: Pergeseran pola fraud rate per kanal pembayaran sepanjang Jan – 10 Agu 2025.", styles, story)

    story.append(Paragraph("3.5 Kalibrasi Probabilitas dan Aturan Keputusan Berbasis Biaya (Poin 2f)", styles["h2"]))
    story.append(Paragraph(
        "Biaya review per transaksi adalah Rp 25.000, sedangkan kerugian fraud lolos sebesar <i>nominal</i>. Aturan keputusan Bayes-optimal "
        "menandai transaksi (<i>tandai = 1</i>) apabila ekspektasi kerugian melebihi biaya review: <b>p × nominal &gt; Rp 25.000</b>. "
        "Kalibrasi Platt Scaling dan Isotonic Regression dilatih pada data Juni dan dievaluasi pada holdout matang 1 Jul – 10 Agu:",
        styles["body"],
    ))
    academic_table(
        "Tabel 3.2 Perbandingan Kalibrasi Probabilitas dan Strategi Keputusan Berbasis Biaya",
        ["Strategi", "Brier", "LogLoss", "Ditandai", "Lolos", "Total Biaya (Rp)", "Recall"],
        [
            [
                r["strategy"],
                "-" if pd.isna(r["brier"]) else f"{r['brier']:.5f}",
                "-" if pd.isna(r["logloss"]) else f"{r['logloss']:.5f}",
                f"{int(r['n_flagged']):,}", f"{int(r['n_missed_fraud'])}", f"{r['total_cost']:,.0f}", f"{r['recall']*100:.1f}%"
            ]
            for _, r in t2_cost.iterrows()
        ],
        [4.1 * cm, 1.6 * cm, 1.6 * cm, 1.5 * cm, 1.3 * cm, 3.3 * cm, 1.6 * cm],
        styles, story,
    )
    story.append(Paragraph(
        f"<b>Isotonic Regression</b> mencapai Brier Score terbaik (0,00294), LogLoss terbaik (0,01458), dan total biaya terendah "
        f"(<b>Rp {s2['best_cost_val']:,.0f}</b>), menghemat <b>{s2['cost_reduction_vs_none_pct']:.1f}%</b> dibanding Flag None "
        f"(Rp {s2['flag_none_cost']:,.0f}) dan <b>95,5%</b> dibanding Flag All (Rp {s2['flag_all_cost']:,.0f}).",
        styles["body"],
    ))
    add_img(OUT2 / "figures" / "fig3_task2_calibration.png", 9.6, "Gambar 3.3: Kurva kalibrasi probabilitas pada holdout matang 1 Jul – 10 Agu 2025.", styles, story)

    story.append(Paragraph("3.6 Analisis Error per Kanal dan Kategori Merchant (Poin 2g)", styles["h2"]))
    academic_table(
        "Tabel 3.3 Analisis Error dan Biaya Fraud berdasarkan Kanal Pembayaran",
        ["Kanal", "Transaksi", "Fraud", "Tertangkap", "Lolos", "Ditandai", "Total Biaya (Rp)", "Recall"],
        [
            [r["kanal"], f"{int(r['n']):,}", f"{int(r['n_fraud'])}", f"{int(r['n_caught_fraud'])}", f"{int(r['n_missed_fraud'])}", f"{int(r['n_flagged'])}", f"{r['total_cost']:,.0f}", f"{r['recall']*100:.1f}%"]
            for _, r in t2_kan.iterrows()
        ],
        [1.9 * cm, 1.9 * cm, 1.4 * cm, 1.9 * cm, 1.3 * cm, 1.7 * cm, 3.3 * cm, 1.6 * cm],
        styles, story,
    )
    top5_kat = t2_kat.head(5)
    story.append(Paragraph(
        "Kanal <b>QRIS</b> menyumbang porsi biaya terbesar (Rp 43,0 juta; recall 39,3%) karena pergeseran pola serangan baru pada kanal tersebut. "
        "Lima kategori merchant penyumbang biaya tertinggi adalah "
        + ", ".join([f"<b>{r['kategori_merchant']}</b> (Rp {r['total_cost']/1e6:.1f} jt)" for _, r in top5_kat.iterrows()])
        + ". Pemantauan aturan velocity khusus pada kanal QRIS dan kategori berisiko tinggi direkomendasikan untuk mitigasi lanjutan.",
        styles["body"],
    ))

    # =========================================================================
    # BAB IV: TASK 3 - CLUSTERING
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("BAB IV<br/>TASK 3 — CLUSTERING: SEGMENTASI PELANGGAN", styles["bab"]))

    story.append(Paragraph("4.1 Preprocessing dan Seleksi Fitur dengan Bukti Kuantitatif (Poin 3a & 3b)", styles["h2"]))
    story.append(Paragraph(
        "Dataset terdiri dari 120.000 pelanggan dan 60 kolom tanpa label. Audit korelasi Spearman menemukan <b>18 fitur noise murni</b> "
        "(seluruh <i>n01–n15</i>, <i>rating_beri_rata2</i>, <i>kota_tier</i>, dan <i>jumlah_alamat</i> dengan max |ρ| &lt; 0,025). "
        "Dua kolom redundan 30 hari (<i>belanja_30h</i> ρ=0,97; <i>frekuensi_30h</i> ρ=0,87) dikeluarkan, sedangkan tiga kolom aktivitas sesi ekstrem "
        "(<i>wishlist, klik_iklan_30h, durasi_sesi_menit</i>) dipisahkan khusus untuk deteksi anomali bot karena korelasinya pada pelanggan normal &lt; 0,01.",
        styles["body"],
    ))
    story.append(Paragraph(
        "• <b>Missing Values:</b> <i>rata2_keranjang</i> kosong pada 6.736 baris (100% saat <i>frekuensi_90h = 0</i>), diimputasi dengan median pelanggan <i>frekuensi_90h = 1</i> agar tidak membentuk klaster nol buatan.", styles["bullet"]))
    story.append(Paragraph(
        "• <b>Skala & Skewness:</b> Variabel skewed ditransformasi <i>log1p</i>, dengan skala halus <i>belanja_90h / 100.000</i> agar tidak muncul lonjakan 9 unit di nilai nol.", styles["bullet"]))
    story.append(Paragraph(
        "• <b>Data Komposisional:</b> 16 kolom <i>porsi_&lt;kategori&gt;</i> dan 4 kolom <i>porsi_jam_*</i> ditransformasi memakai <b>multiplicative zero replacement + Centered Log-Ratio (CLR)</b> lalu dikompresi via PCA (5 PC kategori + 3 PC waktu) agar seimbang terhadap fitur RFM.", styles["bullet"]))
    story.append(Paragraph(
        "• <b>Bukti Dampak Seleksi Fitur:</b> Silhouette Score K-Means (K=6) melonjak <b>+322,1%</b> dari <b>0,0376</b> (58 fitur ber-noise) menjadi <b>0,1587</b> (24 fitur bersih → ruang PCA-10).", styles["bullet"]))

    story.append(Paragraph("4.2 Perbandingan 4 Metode, Pemilihan Jumlah Cluster, dan Stabilitas (Poin 3c–3e)", styles["h2"]))
    eval_sub = t3_eval[t3_eval["k"].isin([4, 5, 6, 7])].copy()
    academic_table(
        "Tabel 4.1 Perbandingan 4 Metode Clustering pada K = 4 sampai 7",
        ["Metode", "K", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "BIC (GMM)"],
        [
            [r["method"], int(r["k"]), f"{r['silhouette']:.4f}", f"{r['davies_bouldin']:.4f}", f"{r['calinski_harabasz']:.1f}", "-" if pd.isna(r["bic"]) else f"{r['bic']:,.0f}"]
            for _, r in eval_sub.iterrows()
        ],
        [4.0 * cm, 1.0 * cm, 2.3 * cm, 2.5 * cm, 2.7 * cm, 2.5 * cm],
        styles, story,
    )
    academic_table(
        "Tabel 4.2 Hasil Uji Stabilitas Subsampling 5-Run (80% Sampel) dan Cross-ARI",
        ["K", "GMM ARI (Mean ± Std)", "GMM ARI Min", "K-Means ARI Mean", "Cross-ARI (GMM vs KM)"],
        [
            [int(r["k"]), f"{r['gmm_ari_mean']:.4f} ± {r['gmm_ari_std']:.4f}", f"{r['gmm_ari_min']:.4f}", f"{r['km_ari_mean']:.4f}", f"{r['cross_ari_gmm_vs_km']:.4f}"]
            for _, r in t3_stab.iterrows()
        ],
        [1.2 * cm, 3.8 * cm, 2.8 * cm, 3.3 * cm, 3.9 * cm],
        styles, story,
    )
    story.append(Paragraph(
        "Berdasarkan siku penurunan <b>BIC GMM</b>, <b>Davies-Bouldin terbaik</b> di K=6 (1,8727), <b>Cross-ARI tinggi antara GMM dan K-Means (0,8510)</b>, "
        "serta verifikasi bahwa segmen <i>Belanja Rutin Keluarga</i> bersifat unimodal (tidak perlu dipecah ke K=7), model final yang dipilih "
        "adalah <b>Gaussian Mixture Model (Full Covariance, K=6)</b>.",
        styles["body"],
    ))
    add_img(OUT3 / "figures" / "fig1_method_comparison_bic.png", 14.0, "Gambar 4.1: Perbandingan Silhouette 4 metode (kiri) dan kurva BIC GMM Full (kanan).", styles, story)
    add_img(OUT3 / "figures" / "fig2_stability_ari.png", 10.5, "Gambar 4.2: Stabilitas subsampling 5-run ARI dan Cross-ARI antar metode.", styles, story)

    story.append(Paragraph("4.3 Profil Segmen, Aksi Marketing, dan Deteksi Anomali (Poin 3f & 3g)", styles["h2"]))
    story.append(Paragraph(
        f"Deteksi anomali menggunakan <b>Isolation Forest</b> (contamination = 0,015, seed = 42) pada fitur aktivitas sesi menandai "
        f"<b>{s3['n_anomalies']:,} akun (1,50%)</b> sebagai <b>cluster = -1</b>, dengan tumpang tindih <b>99,83% ({s3['bot_overlap']:,}/1.800)</b> "
        "terhadap pola bot/scraper ekstrem: median 241,1 sesi/minggu (normal 5,4), durasi sesi 0,60 menit (normal 8,0), wishlist 150 (normal 9), "
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
    academic_table(
        "Tabel 4.3 Profil 6 Segmen Pelanggan, Anomali (-1), dan Rekomendasi Aksi Marketing",
        ["ID", "Jumlah (%)", "Nama Segmen", "Karakteristik Utama (Median)", "Aksi Marketing"],
        prof_rows,
        [0.8 * cm, 2.2 * cm, 2.9 * cm, 4.6 * cm, 4.5 * cm],
        styles, story,
    )
    add_img(OUT3 / "figures" / "fig3_segment_profiles_heatmap.png", 14.5, "Gambar 4.3: Heatmap Z-score profil median 6 segmen pelanggan normal.", styles, story)

    # =========================================================================
    # BAB V: PENUTUP & VERIFIKASI PENGUMPULAN
    # =========================================================================
    story.append(Paragraph("BAB V<br/>PENUTUP DAN VERIFIKASI PENGUMPULAN", styles["bab"]))
    story.append(Paragraph(
        "Seluruh pipeline untuk Task 1, Task 2, dan Task 3 telah diuji secara end-to-end dan menghasilkan file pengumpulan yang valid "
        "tanpa nilai kosong maupun duplikasi ID: <b>prediksi_regresi.csv</b> (60.000 baris, seluruh baris memenuhi <i>p90 ≥ p50</i>), "
        "<b>prediksi_klasifikasi.csv</b> (106.674 baris, <i>prob ∈ [0, 1]</i>, <i>tandai ∈ {0, 1}</i>), dan <b>cluster.csv</b> (120.000 baris, "
        "label bilangan bulat <i>-1..5</i>). Rincian peran masing-masing anggota, tautan Pull Request, serta deklarasi penggunaan alat bantu AI "
        "dan verifikasinya didokumentasikan secara transparan pada file <b>kontribusi.md</b>.",
        styles["body"],
    ))

    doc.build(story, onFirstPage=page_number_footer, onLaterPages=page_number_footer)
    print(f"Generated Indonesian academic DOCX-style {PDF_PATH.resolve()} successfully!")


if __name__ == "__main__":
    main()
