import pandas as pd

# =========================
# DAFTAR KOLOM PDRB
# =========================

REQUIRED_PDRB_COLUMNS = [
    "Kabupaten",
    "Tahun",
    "Lapangan_Usaha",
    "Nilai"
]

TEXT_COLUMNS = [
    "Kabupaten",
    "Lapangan_Usaha"
]

# =========================
# 1. STANDARISASI NAMA KOLOM
# =========================

def standardize_pdrb_column_names(df):
    df = df.copy()

    rename_map = {
        "Lapangan Usaha": "Lapangan_Usaha",
        "Kategori Lapangan Usaha": "Lapangan_Usaha",
        "Sektor": "Lapangan_Usaha",
        "PDRB": "Nilai",
        "Nilai PDRB": "Nilai",
        "PDRB ADHK": "Nilai",
        "PDRB ADHK 2010": "Nilai"
    }

    df = df.rename(columns=rename_map)

    return df

# =========================
# 2. VALIDASI KOLOM WAJIB
# =========================

def validate_pdrb_columns(df):
    missing_columns = []

    for col in REQUIRED_PDRB_COLUMNS:
        if col not in df.columns:
            missing_columns.append(col)

    if len(missing_columns) > 0:
        raise ValueError(
            f"Kolom wajib PDRB belum ditemukan: {missing_columns}"
        )

# =========================
# 3. MEMBERSIHKAN KOLOM TEKS
# =========================

def clean_pdrb_text_columns(df):
    df = df.copy()

    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace("\u200b", "", regex=False)
        df[col] = df[col].str.replace("\u200c", "", regex=False)
        df[col] = df[col].str.replace("\u200d", "", regex=False)
        df[col] = df[col].str.replace("\ufeff", "", regex=False)
        df[col] = df[col].str.replace("\xa0", " ", regex=False)
        df[col] = df[col].str.strip()

    return df

# =========================
# 4. KONVERSI TIPE DATA
# =========================

def convert_pdrb_data_types(df):
    df = df.copy()

    df["Tahun"] = pd.to_numeric(
        df["Tahun"],
        errors="raise"
    )

    df["Tahun"] = df["Tahun"].astype(int)

    df["Nilai"] = pd.to_numeric(
        df["Nilai"],
        errors="coerce"
    )

    return df

# =========================
# 5. HITUNG TOTAL DAN PERSENTASE PDRB
# =========================

def add_pdrb_percentage(df):
    df = df.copy()

    df["Total_PDRB_Kabupaten_Tahun"] = df.groupby(
        ["Kabupaten", "Tahun"]
    )["Nilai"].transform("sum")

    df["Persentase_PDRB"] = (
        df["Nilai"] / df["Total_PDRB_Kabupaten_Tahun"]
    ) * 100

    return df

# =========================
# 6. MEMBUAT RINGKASAN DATA PDRB
# =========================

def make_pdrb_report(df):
    jumlah_baris = df.shape[0]
    jumlah_kolom = df.shape[1]

    jumlah_kabupaten = df["Kabupaten"].nunique()
    jumlah_tahun = df["Tahun"].nunique()
    jumlah_sektor = df["Lapangan_Usaha"].nunique()

    tahun_awal = int(df["Tahun"].min())
    tahun_akhir = int(df["Tahun"].max())

    jumlah_nilai_tersedia = df["Nilai"].notna().sum()
    jumlah_missing = df["Nilai"].isna().sum()

    pdrb_report = {
        "Jumlah Baris": jumlah_baris,
        "Jumlah Kolom": jumlah_kolom,
        "Jumlah Kabupaten": jumlah_kabupaten,
        "Jumlah Tahun": jumlah_tahun,
        "Jumlah Sektor": jumlah_sektor,
        "Tahun Awal": tahun_awal,
        "Tahun Akhir": tahun_akhir,
        "Nilai Tersedia": jumlah_nilai_tersedia,
        "Missing Value": jumlah_missing
    }

    return pdrb_report

# =========================
# 7. DATA PDRB TAHUN TERBARU
# =========================

def make_latest_pdrb_data(df):
    df = df.copy()

    tahun_terbaru = int(df["Tahun"].max())

    df_latest = df[df["Tahun"] == tahun_terbaru].copy()

    df_latest = df_latest.sort_values(
        ["Kabupaten", "Persentase_PDRB"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return df_latest

# =========================
# 8. SEKTOR DOMINAN PER KABUPATEN
# =========================

def make_dominant_sector_data(df_latest):
    df_latest = df_latest.copy()

    df_dominant = (
        df_latest
        .sort_values(
            ["Kabupaten", "Persentase_PDRB"],
            ascending=[True, False]
        )
        .groupby("Kabupaten")
        .head(3)
        .reset_index(drop=True)
    )

    df_dominant = df_dominant[
        [
            "Kabupaten",
            "Tahun",
            "Lapangan_Usaha",
            "Nilai",
            "Total_PDRB_Kabupaten_Tahun",
            "Persentase_PDRB"
        ]
    ]

    return df_dominant

# =========================
# 9. CEK TOTAL PERSENTASE PDRB
# =========================

def make_pdrb_percentage_check(df):
    df_check = (
        df.groupby(["Kabupaten", "Tahun"], as_index=False)
        .agg(
            Total_Persentase_PDRB=("Persentase_PDRB", "sum"),
            Jumlah_Sektor=("Lapangan_Usaha", "nunique")
        )
    )

    df_check["Selisih_dari_100"] = (
        df_check["Total_Persentase_PDRB"] - 100
    )

    return df_check

# =========================
# 10. PIPELINE UTAMA PREPROCESSING PDRB
# =========================

def preprocess_pdrb_data(df_pdrb_raw):
    df_pdrb = df_pdrb_raw.copy()

    df_pdrb = standardize_pdrb_column_names(df_pdrb)

    validate_pdrb_columns(df_pdrb)

    df_pdrb = clean_pdrb_text_columns(df_pdrb)

    df_pdrb = convert_pdrb_data_types(df_pdrb)

    df_pdrb = df_pdrb.sort_values(
        ["Kabupaten", "Tahun", "Lapangan_Usaha"]
    ).reset_index(drop=True)

    df_pdrb = add_pdrb_percentage(df_pdrb)

    pdrb_report = make_pdrb_report(df_pdrb)

    df_pdrb_latest = make_latest_pdrb_data(df_pdrb)

    df_pdrb_dominant = make_dominant_sector_data(df_pdrb_latest)

    df_pdrb_percentage_check = make_pdrb_percentage_check(df_pdrb)

    return (
        df_pdrb,
        pdrb_report,
        df_pdrb_latest,
        df_pdrb_dominant,
        df_pdrb_percentage_check
    )