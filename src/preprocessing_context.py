import pandas as pd


# =========================
# DAFTAR KOLOM KONTEKSTUAL
# =========================

REQUIRED_CONTEXT_COLUMNS = [
    "Kabupaten",
    "Tahun",
    "Indikator",
    "Nilai",
    "Satuan",
    "Pilar"
]


TEXT_COLUMNS = [
    "Kabupaten",
    "Indikator",
    "Satuan",
    "Pilar"
]


# =========================
# 1. VALIDASI KOLOM WAJIB
# =========================

def validate_context_columns(df):
    missing_columns = []

    for col in REQUIRED_CONTEXT_COLUMNS:
        if col not in df.columns:
            missing_columns.append(col)

    if len(missing_columns) > 0:
        raise ValueError(
            f"Kolom wajib indikator kontekstual belum ditemukan: {missing_columns}"
        )


# =========================
# 2. MEMBERSIHKAN KOLOM TEKS
# =========================

def clean_context_text_columns(df):
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
# 3. KONVERSI TIPE DATA
# =========================

def convert_context_data_types(df):
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
# 4. MEMBUAT RINGKASAN DATA KONTEKSTUAL
# =========================

def make_context_report(df):
    jumlah_baris = df.shape[0]
    jumlah_kolom = df.shape[1]

    jumlah_kabupaten = df["Kabupaten"].nunique()
    jumlah_indikator = df["Indikator"].nunique()
    jumlah_pilar = df["Pilar"].nunique()

    tahun_awal = int(df["Tahun"].min())
    tahun_akhir = int(df["Tahun"].max())

    jumlah_nilai_tersedia = df["Nilai"].notna().sum()
    jumlah_missing_resmi = df["Nilai"].isna().sum()

    context_report = {
        "Jumlah Baris": jumlah_baris,
        "Jumlah Kolom": jumlah_kolom,
        "Jumlah Kabupaten": jumlah_kabupaten,
        "Jumlah Indikator": jumlah_indikator,
        "Jumlah Pilar": jumlah_pilar,
        "Tahun Awal": tahun_awal,
        "Tahun Akhir": tahun_akhir,
        "Nilai Tersedia": jumlah_nilai_tersedia,
        "Missing Resmi BPS": jumlah_missing_resmi
    }

    return context_report

# =========================
# 5. RINGKASAN KETERSEDIAAN INDIKATOR
# =========================

def make_indicator_availability(df):
    daftar_ringkasan = []

    daftar_indikator = sorted(df["Indikator"].dropna().unique())

    for indikator in daftar_indikator:
        df_indikator = df[df["Indikator"] == indikator]

        nilai_tersedia = df_indikator["Nilai"].notna().sum()
        nilai_missing = df_indikator["Nilai"].isna().sum()

        jumlah_kabupaten = df_indikator["Kabupaten"].nunique()
        jumlah_tahun = df_indikator["Tahun"].nunique()

        data_tersedia = df_indikator[df_indikator["Nilai"].notna()]

        if data_tersedia.shape[0] > 0:
            tahun_awal_tersedia = int(data_tersedia["Tahun"].min())
            tahun_akhir_tersedia = int(data_tersedia["Tahun"].max())
        else:
            tahun_awal_tersedia = None
            tahun_akhir_tersedia = None

        daftar_satuan = df_indikator["Satuan"].dropna().unique()
        daftar_pilar = df_indikator["Pilar"].dropna().unique()

        if len(daftar_satuan) > 0:
            satuan = daftar_satuan[0]
        else:
            satuan = None

        if len(daftar_pilar) > 0:
            pilar = daftar_pilar[0]
        else:
            pilar = None

        ringkasan = {
            "Pilar": pilar,
            "Indikator": indikator,
            "Satuan": satuan,
            "Jumlah Kabupaten": jumlah_kabupaten,
            "Jumlah Tahun dalam Sheet": jumlah_tahun,
            "Nilai Tersedia": nilai_tersedia,
            "Missing Resmi BPS": nilai_missing,
            "Tahun Awal Tersedia": tahun_awal_tersedia,
            "Tahun Akhir Tersedia": tahun_akhir_tersedia
        }

        daftar_ringkasan.append(ringkasan)

    df_availability = pd.DataFrame(daftar_ringkasan)

    return df_availability

# =========================
# 6. DATA TERBARU PER KABUPATEN-INDIKATOR
# =========================

def make_latest_context_data(df):
    df = df.copy()

    df_available = df[df["Nilai"].notna()].copy()

    df_available = df_available.sort_values(
        ["Kabupaten", "Indikator", "Tahun"]
    ).reset_index(drop=True)

    df_latest = (
        df_available
        .groupby(["Kabupaten", "Indikator"])
        .tail(1)
        .reset_index(drop=True)
    )

    df_latest = df_latest.sort_values(
        ["Kabupaten", "Pilar", "Indikator"]
    ).reset_index(drop=True)

    return df_latest

# =========================
# 7. DATA TERBARU FORMAT WIDE
# =========================

def make_latest_context_wide(df_latest):
    df_wide = df_latest.pivot_table(
        index="Kabupaten",
        columns="Indikator",
        values="Nilai",
        aggfunc="first"
    ).reset_index()

    df_wide.columns.name = None

    return df_wide

# =========================
# 8. PIPELINE UTAMA PREPROCESSING KONTEKSTUAL
# =========================

def preprocess_context_data(df_context_raw):
    df_context = df_context_raw.copy()

    validate_context_columns(df_context)

    df_context = clean_context_text_columns(df_context)

    df_context = convert_context_data_types(df_context)

    df_context = df_context.sort_values(
        ["Kabupaten", "Tahun", "Pilar", "Indikator"]
    ).reset_index(drop=True)

    context_report = make_context_report(df_context)

    df_indicator_availability = make_indicator_availability(df_context)

    df_context_latest = make_latest_context_data(df_context)

    df_context_wide_latest = make_latest_context_wide(df_context_latest)

    return (
        df_context,
        context_report,
        df_indicator_availability,
        df_context_latest,
        df_context_wide_latest
    )