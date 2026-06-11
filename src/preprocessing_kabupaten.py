import pandas as pd


# =========================
# DAFTAR KOLOM REFERENSI KABUPATEN
# =========================

REQUIRED_KABUPATEN_COLUMNS = [
    "Kabupaten",
    "Provinsi",
    "Pulau_Kawasan",
    "Latitude",
    "Longitude"
]


TEXT_COLUMNS = [
    "Kabupaten",
    "Provinsi",
    "Pulau_Kawasan"
]

# =========================
# 1. STANDARISASI NAMA KOLOM
# =========================

def standardize_kabupaten_column_names(df):
    df = df.copy()

    rename_map = {
        "Pulau/Kawasan": "Pulau_Kawasan",
        "Pulau Kawasan": "Pulau_Kawasan",
        "Kawasan": "Pulau_Kawasan",
        "Lat": "Latitude",
        "Long": "Longitude",
        "Lon": "Longitude"
    }

    df = df.rename(columns=rename_map)

    return df

# =========================
# 2. VALIDASI KOLOM WAJIB
# =========================

def validate_kabupaten_columns(df):
    missing_columns = []

    for col in REQUIRED_KABUPATEN_COLUMNS:
        if col not in df.columns:
            missing_columns.append(col)

    if len(missing_columns) > 0:
        raise ValueError(
            f"Kolom wajib referensi kabupaten belum ditemukan: {missing_columns}"
        )

# =========================
# 3. MEMBERSIHKAN KOLOM TEKS
# =========================

def clean_kabupaten_text_columns(df):
    df = df.copy()

    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace("\u200b", "", regex=False)
        df[col] = df[col].str.replace("\u200c", "", regex=False)
        df[col] = df[col].str.replace("\u200d", "", regex=False)
        df[col] = df[col].str.replace("\ufeff", "", regex=False)
        df[col] = df[col].str.replace("\xa0", " ", regex=False)
        df[col] = df[col].str.strip()

    if "Catatan" in df.columns:
        df["Catatan"] = df["Catatan"].astype(str)
        df["Catatan"] = df["Catatan"].str.strip()

    return df

# =========================
# 4. KONVERSI TIPE DATA
# =========================

def convert_kabupaten_data_types(df):
    df = df.copy()

    df["Latitude"] = pd.to_numeric(
        df["Latitude"],
        errors="raise"
    )

    df["Longitude"] = pd.to_numeric(
        df["Longitude"],
        errors="raise"
    )

    return df

# =========================
# 5. MEMBUAT RINGKASAN REFERENSI KABUPATEN
# =========================

def make_kabupaten_report(df):
    jumlah_baris = df.shape[0]
    jumlah_kolom = df.shape[1]

    jumlah_kabupaten = df["Kabupaten"].nunique()
    jumlah_provinsi = df["Provinsi"].nunique()
    jumlah_pulau_kawasan = df["Pulau_Kawasan"].nunique()

    missing_latitude = df["Latitude"].isna().sum()
    missing_longitude = df["Longitude"].isna().sum()

    koordinat_lengkap = df[
        df["Latitude"].notna() &
        df["Longitude"].notna()
    ].shape[0]

    kabupaten_report = {
        "Jumlah Baris": jumlah_baris,
        "Jumlah Kolom": jumlah_kolom,
        "Jumlah Kabupaten": jumlah_kabupaten,
        "Jumlah Provinsi": jumlah_provinsi,
        "Jumlah Pulau/Kawasan": jumlah_pulau_kawasan,
        "Koordinat Lengkap": koordinat_lengkap,
        "Missing Latitude": missing_latitude,
        "Missing Longitude": missing_longitude
    }

    return kabupaten_report

# =========================
# 6. CEK KONSISTENSI KABUPATEN ANTAR DATA
# =========================

def make_kabupaten_consistency_report(
    df_panel,
    df_context,
    df_pdrb,
    df_kabupaten
):
    panel_set = set(df_panel["Kabupaten"].dropna().unique())
    context_set = set(df_context["Kabupaten"].dropna().unique())
    pdrb_set = set(df_pdrb["Kabupaten"].dropna().unique())
    kabupaten_set = set(df_kabupaten["Kabupaten"].dropna().unique())

    semua_kabupaten = sorted(
        panel_set |
        context_set |
        pdrb_set |
        kabupaten_set
    )

    daftar_status = []

    for kabupaten in semua_kabupaten:
        ada_di_panel = kabupaten in panel_set
        ada_di_context = kabupaten in context_set
        ada_di_pdrb = kabupaten in pdrb_set
        ada_di_referensi = kabupaten in kabupaten_set

        if (
            ada_di_panel and
            ada_di_context and
            ada_di_pdrb and
            ada_di_referensi
        ):
            status = "Lengkap"
        else:
            status = "Perlu Dicek"

        info = {
            "Kabupaten": kabupaten,
            "Ada di Panel": ada_di_panel,
            "Ada di Kontekstual": ada_di_context,
            "Ada di PDRB": ada_di_pdrb,
            "Ada di Referensi": ada_di_referensi,
            "Status": status
        }

        daftar_status.append(info)

    df_consistency = pd.DataFrame(daftar_status)

    return df_consistency

# =========================
# 7. DATA REFERENSI SIAP PETA
# =========================

def make_map_reference_data(df):
    df = df.copy()

    selected_columns = [
        "Kabupaten",
        "Provinsi",
        "Pulau_Kawasan",
        "Latitude",
        "Longitude"
    ]

    if "Catatan" in df.columns:
        selected_columns.append("Catatan")

    df_map_reference = df[selected_columns].copy()

    df_map_reference = df_map_reference.dropna(
        subset=["Latitude", "Longitude"]
    ).reset_index(drop=True)

    return df_map_reference

# =========================
# 8. DATA DASAR PETA DENGAN INDIKATOR PANEL TERBARU
# =========================

def make_map_base_data(df_kabupaten, df_panel):
    df_kabupaten = df_kabupaten.copy()
    df_panel = df_panel.copy()

    tahun_terbaru = int(df_panel["Tahun"].max())

    df_panel_latest = df_panel[
        df_panel["Tahun"] == tahun_terbaru
    ].copy()

    selected_panel_columns = [
        "Kabupaten",
        "Tahun",
        "Kemiskinan",
        "TPT",
        "DBH",
        "DAU",
        "DAK_Fisik",
        "DAK_Non_Fisik",
        "Dana_Desa",
        "DID",
        "KUR",
        "Total_TKD"
    ]

    df_panel_latest = df_panel_latest[selected_panel_columns]

    df_map_base = df_kabupaten.merge(
        df_panel_latest,
        on="Kabupaten",
        how="left"
    )

    return df_map_base

# =========================
# 9. PIPELINE UTAMA PREPROCESSING REFERENSI KABUPATEN
# =========================

def preprocess_kabupaten_data(
    df_kabupaten_raw,
    df_panel,
    df_context,
    df_pdrb
):
    df_kabupaten = df_kabupaten_raw.copy()

    df_kabupaten = standardize_kabupaten_column_names(df_kabupaten)

    validate_kabupaten_columns(df_kabupaten)

    df_kabupaten = clean_kabupaten_text_columns(df_kabupaten)

    df_kabupaten = convert_kabupaten_data_types(df_kabupaten)

    df_kabupaten = df_kabupaten.sort_values(
        "Kabupaten"
    ).reset_index(drop=True)

    kabupaten_report = make_kabupaten_report(df_kabupaten)

    df_kabupaten_consistency = make_kabupaten_consistency_report(
        df_panel,
        df_context,
        df_pdrb,
        df_kabupaten
    )

    df_map_reference = make_map_reference_data(df_kabupaten)

    df_map_base = make_map_base_data(
        df_kabupaten,
        df_panel
    )

    return (
        df_kabupaten,
        kabupaten_report,
        df_kabupaten_consistency,
        df_map_reference,
        df_map_base
    )