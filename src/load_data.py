from pathlib import Path
import pandas as pd


# =========================
# LOKASI FILE DATA
# =========================

DATA_PATH = Path("data/data_flores_dashboard.xlsx")


# =========================
# FUNGSI MEMBERSIHKAN NAMA KOLOM
# =========================

def clean_column_names(df):
    df = df.copy()

    cleaned_columns = []

    for col in df.columns:
        clean_col = str(col)

        clean_col = clean_col.replace("\u200b", "")
        clean_col = clean_col.replace("\u200c", "")
        clean_col = clean_col.replace("\u200d", "")
        clean_col = clean_col.replace("\ufeff", "")
        clean_col = clean_col.replace("\xa0", " ")

        clean_col = clean_col.strip()

        cleaned_columns.append(clean_col)

    df.columns = cleaned_columns

    return df


# =========================
# FUNGSI MEMBACA SHEET PANEL
# =========================

def load_panel_data():
    df_panel = pd.read_excel(
        DATA_PATH,
        sheet_name="panel_inferensial",
        na_values=["#N/A"]
    )

    df_panel = clean_column_names(df_panel)

    return df_panel


# =========================
# FUNGSI MEMBACA SHEET INDIKATOR KONTEKSTUAL
# =========================

def load_context_data():
    df_context = pd.read_excel(
        DATA_PATH,
        sheet_name="indikator_kontekstual",
        na_values=["#N/A"]
    )

    df_context = clean_column_names(df_context)

    return df_context


# =========================
# FUNGSI MEMBACA SHEET PDRB SEKTORAL
# =========================

def load_pdrb_data():
    df_pdrb = pd.read_excel(
        DATA_PATH,
        sheet_name="pdrb_sektoral",
        na_values=["#N/A"]
    )

    df_pdrb = clean_column_names(df_pdrb)

    return df_pdrb


# =========================
# FUNGSI MEMBACA SHEET REFERENSI KABUPATEN
# =========================

def load_kabupaten_reference():
    df_kabupaten = pd.read_excel(
        DATA_PATH,
        sheet_name="referensi_kabupaten",
        na_values=["#N/A"]
    )

    df_kabupaten = clean_column_names(df_kabupaten)

    return df_kabupaten


# =========================
# FUNGSI UTAMA MEMBACA SEMUA DATA
# =========================

def load_all_data():
    df_panel = load_panel_data()
    df_context = load_context_data()
    df_pdrb = load_pdrb_data()
    df_kabupaten = load_kabupaten_reference()

    return df_panel, df_context, df_pdrb, df_kabupaten