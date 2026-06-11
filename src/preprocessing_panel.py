import pandas as pd


# =========================
# DAFTAR KOLOM PANEL
# =========================

REQUIRED_COLUMNS = [
    "Kabupaten",
    "Tahun",
    "DBH",
    "DAU",
    "DAK_Fisik",
    "DAK_Non_Fisik",
    "Dana_Desa",
    "DID",
    "KUR",
    "TPT",
    "Kemiskinan"
]


NUMERIC_COLUMNS = [
    "Tahun",
    "DBH",
    "DAU",
    "DAK_Fisik",
    "DAK_Non_Fisik",
    "Dana_Desa",
    "DID",
    "KUR",
    "TPT",
    "Kemiskinan"
]


TKD_COLUMNS = [
    "DBH",
    "DAU",
    "DAK_Fisik",
    "DAK_Non_Fisik",
    "Dana_Desa",
    "DID"
]


MODEL_X_COLUMNS = [
    "DBH",
    "DAU",
    "DAK_Fisik",
    "DAK_Non_Fisik",
    "Dana_Desa",
    "DID",
    "KUR",
    "TPT"
]


LAG_SOURCE_COLUMNS = [
    "DBH",
    "DAU",
    "DAK_Fisik",
    "DAK_Non_Fisik",
    "Dana_Desa",
    "DID",
    "KUR",
    "TPT",
    "Total_TKD"
]


# =========================
# 1. STANDARISASI NAMA KOLOM
# =========================

def standardize_panel_column_names(df):
    df = df.copy()

    rename_map = {
        "DAK Fisik": "DAK_Fisik",
        "DAK Non Fisik": "DAK_Non_Fisik",
        "Dana Desa": "Dana_Desa",
        "Tingkat Pengangguran Terbuka (TPT)": "TPT",
        "Tingkat Pengangguran Terbuka": "TPT",
        "Penduduk Miskin (P0)": "Kemiskinan",
        "Persentase Penduduk Miskin": "Kemiskinan",
        "Kemiskinan (P0)": "Kemiskinan"
    }

    df = df.rename(columns=rename_map)

    return df


# =========================
# 2. VALIDASI KOLOM WAJIB
# =========================

def validate_panel_columns(df):
    missing_columns = []

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing_columns.append(col)

    if len(missing_columns) > 0:
        raise ValueError(
            f"Kolom wajib panel belum ditemukan: {missing_columns}"
        )


# =========================
# 3. KONVERSI TIPE DATA
# =========================

def convert_panel_data_types(df):
    df = df.copy()

    df["Kabupaten"] = df["Kabupaten"].astype(str).str.strip()

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["Tahun"] = df["Tahun"].astype(int)

    return df


# =========================
# 4. MEMBUAT TOTAL TKD
# =========================

def add_total_tkd(df):
    df = df.copy()

    df["Total_TKD"] = df[TKD_COLUMNS].sum(axis=1)

    return df


# =========================
# 5. MEMBUAT VARIABEL LAG 1 TAHUN
# =========================

def add_lag_variables(df):
    df = df.copy()

    df = df.sort_values(
        ["Kabupaten", "Tahun"]
    ).reset_index(drop=True)

    for col in LAG_SOURCE_COLUMNS:
        nama_kolom_lag = f"L1_{col}"

        df[nama_kolom_lag] = df.groupby("Kabupaten")[col].shift(1)

    return df


# =========================
# 6. CEK STATUS PANEL
# =========================

def check_panel_status(df):
    kabupaten_list = sorted(df["Kabupaten"].unique())
    tahun_list = sorted(df["Tahun"].unique())

    jumlah_kabupaten = len(kabupaten_list)
    jumlah_tahun = len(tahun_list)

    expected_observations = jumlah_kabupaten * jumlah_tahun

    unique_observations = df[["Kabupaten", "Tahun"]].drop_duplicates().shape[0]

    duplicate_observations = df.duplicated(
        subset=["Kabupaten", "Tahun"]
    ).sum()

    expected_index = pd.MultiIndex.from_product(
        [kabupaten_list, tahun_list],
        names=["Kabupaten", "Tahun"]
    )

    actual_index = pd.MultiIndex.from_frame(
        df[["Kabupaten", "Tahun"]].drop_duplicates()
    )

    missing_index = expected_index.difference(actual_index)

    missing_panel = missing_index.to_frame(index=False)

    missing_panel_observations = missing_panel.shape[0]

    if duplicate_observations > 0:
        status_panel = "Ada Duplikasi"

    elif missing_panel_observations > 0:
        status_panel = "Unbalanced Panel"

    else:
        status_panel = "Balanced Panel"

    panel_report = {
        "Status Panel": status_panel,
        "Jumlah Kabupaten": jumlah_kabupaten,
        "Jumlah Tahun": jumlah_tahun,
        "Expected Observations": expected_observations,
        "Unique Observations": unique_observations,
        "Duplicate Observations": duplicate_observations,
        "Missing Panel Observations": missing_panel_observations,
        "Tahun Awal": min(tahun_list),
        "Tahun Akhir": max(tahun_list)
    }

    return panel_report, missing_panel


# =========================
# 7. MEMBUAT DATASET MODEL NORMAL
# =========================

def make_model_normal_data(df):
    df = df.copy()

    selected_columns = [
        "Kabupaten",
        "Tahun",
        "Kemiskinan"
    ]

    for col in MODEL_X_COLUMNS:
        selected_columns.append(col)

    df_model_normal = df[selected_columns].copy()

    df_model_normal = df_model_normal.dropna().reset_index(drop=True)

    return df_model_normal


# =========================
# 8. MEMBUAT DATASET MODEL LAG 1
# =========================

def make_model_lag_data(df):
    df = df.copy()

    selected_columns = [
        "Kabupaten",
        "Tahun",
        "Kemiskinan"
    ]

    for col in MODEL_X_COLUMNS:
        nama_kolom_lag = f"L1_{col}"

        selected_columns.append(nama_kolom_lag)

    df_model_lag = df[selected_columns].copy()

    df_model_lag = df_model_lag.dropna().reset_index(drop=True)

    return df_model_lag


# =========================
# 9. RINGKASAN DATASET MODEL
# =========================

def make_model_dataset_summary(df_model_normal, df_model_lag):
    daftar_ringkasan = []

    ringkasan_normal = {
        "Dataset Model": "Model Normal",
        "Jumlah Observasi": df_model_normal.shape[0],
        "Jumlah Variabel": df_model_normal.shape[1],
        "Jumlah Kabupaten": df_model_normal["Kabupaten"].nunique(),
        "Tahun Awal": int(df_model_normal["Tahun"].min()),
        "Tahun Akhir": int(df_model_normal["Tahun"].max()),
        "Keterangan": "Menggunakan variabel tahun berjalan"
    }

    ringkasan_lag = {
        "Dataset Model": "Model Lag 1",
        "Jumlah Observasi": df_model_lag.shape[0],
        "Jumlah Variabel": df_model_lag.shape[1],
        "Jumlah Kabupaten": df_model_lag["Kabupaten"].nunique(),
        "Tahun Awal": int(df_model_lag["Tahun"].min()),
        "Tahun Akhir": int(df_model_lag["Tahun"].max()),
        "Keterangan": "Menggunakan variabel independen tahun sebelumnya"
    }

    daftar_ringkasan.append(ringkasan_normal)
    daftar_ringkasan.append(ringkasan_lag)

    df_ringkasan_model = pd.DataFrame(daftar_ringkasan)

    return df_ringkasan_model


# =========================
# 10. PIPELINE UTAMA PREPROCESSING PANEL
# =========================

def preprocess_panel_data(df_panel_raw):
    df_panel = df_panel_raw.copy()

    df_panel = standardize_panel_column_names(df_panel)

    validate_panel_columns(df_panel)

    df_panel = convert_panel_data_types(df_panel)

    df_panel = df_panel.sort_values(
        ["Kabupaten", "Tahun"]
    ).reset_index(drop=True)

    panel_report, missing_panel = check_panel_status(df_panel)

    df_panel = add_total_tkd(df_panel)

    df_panel = add_lag_variables(df_panel)

    df_model_normal = make_model_normal_data(df_panel)

    df_model_lag = make_model_lag_data(df_panel)

    df_ringkasan_model = make_model_dataset_summary(
        df_model_normal,
        df_model_lag
    )

    return (
        df_panel,
        panel_report,
        missing_panel,
        df_model_normal,
        df_model_lag,
        df_ringkasan_model
    )