import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from src.load_data import load_all_data
from src.preprocessing_panel import preprocess_panel_data
from src.preprocessing_context import preprocess_context_data
from src.preprocessing_pdrb import preprocess_pdrb_data
from src.preprocessing_kabupaten import preprocess_kabupaten_data

from src.model_panel import fit_all_panel_models
from src.model_panel import make_model_summary_table
from src.model_panel import make_coefficient_table
from src.model_panel import make_model_equation_text

from src.model_selection import run_panel_model_selection_tests
from src.model_selection import make_model_selection_conclusion

from src.model_diagnostics import calculate_vif
from src.model_diagnostics import make_correlation_matrix
from src.model_diagnostics import make_high_correlation_pairs
from src.model_diagnostics import make_multicollinearity_conclusion
from src.model_diagnostics import get_residual_diagnostic_data
from src.model_diagnostics import run_residual_diagnostic_tests
from src.model_diagnostics import make_residual_diagnostic_conclusion

# =========================
# KONFIGURASI HALAMAN
# =========================

st.set_page_config(
    page_title="DDAC 2026 | Fiscal-Poverty Dashboard Flores",
    page_icon="📊",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

@st.cache_data
def get_data():
    (
        df_panel_raw,
        df_context_raw,
        df_pdrb_raw,
        df_kabupaten_raw
    ) = load_all_data()

    (
        df_panel,
        panel_report,
        missing_panel,
        df_model_normal,
        df_model_lag,
        df_ringkasan_model
    ) = preprocess_panel_data(df_panel_raw)

    (
        df_context,
        context_report,
        df_indicator_availability,
        df_context_latest,
        df_context_wide_latest
    ) = preprocess_context_data(df_context_raw)

    (
        df_pdrb,
        pdrb_report,
        df_pdrb_latest,
        df_pdrb_dominant,
        df_pdrb_percentage_check
    ) = preprocess_pdrb_data(df_pdrb_raw)

    (
        df_kabupaten,
        kabupaten_report,
        df_kabupaten_consistency,
        df_map_reference,
        df_map_base
    ) = preprocess_kabupaten_data(
        df_kabupaten_raw,
        df_panel,
        df_context,
        df_pdrb
    )

    return (
        df_panel_raw,
        df_panel,
        panel_report,
        missing_panel,
        df_model_normal,
        df_model_lag,
        df_ringkasan_model,
        df_context_raw,
        df_context,
        context_report,
        df_indicator_availability,
        df_context_latest,
        df_context_wide_latest,
        df_pdrb_raw,
        df_pdrb,
        pdrb_report,
        df_pdrb_latest,
        df_pdrb_dominant,
        df_pdrb_percentage_check,
        df_kabupaten_raw,
        df_kabupaten,
        kabupaten_report,
        df_kabupaten_consistency,
        df_map_reference,
        df_map_base
    )

try:
    (
        df_panel_raw,
        df_panel,
        panel_report,
        missing_panel,
        df_model_normal,
        df_model_lag,
        df_ringkasan_model,
        df_context_raw,
        df_context,
        context_report,
        df_indicator_availability,
        df_context_latest,
        df_context_wide_latest,
        df_pdrb_raw,
        df_pdrb,
        pdrb_report,
        df_pdrb_latest,
        df_pdrb_dominant,
        df_pdrb_percentage_check,
        df_kabupaten_raw,
        df_kabupaten,
        kabupaten_report,
        df_kabupaten_consistency,
        df_map_reference,
        df_map_base
    ) = get_data()

except Exception as error:
    st.error("Data gagal dibaca atau preprocessing gagal dijalankan.")
    st.exception(error)
    st.stop()

# =========================
# FUNGSI RINGKASAN SHEET
# =========================

def make_sheet_summary(nama_sheet, df):
    jumlah_baris = df.shape[0]
    jumlah_kolom = df.shape[1]
    total_missing = df.isna().sum().sum()

    summary = {
        "Nama Sheet": nama_sheet,
        "Jumlah Baris": jumlah_baris,
        "Jumlah Kolom": jumlah_kolom,
        "Total Missing Value": total_missing
    }

    return summary


# =========================
# FUNGSI RINGKASAN KOLOM
# =========================

def make_column_summary(df):
    struktur_kolom = []

    for col in df.columns:
        info = {
            "Nama Kolom": col,
            "Tipe Data": str(df[col].dtype),
            "Jumlah Data Terisi": df[col].notna().sum(),
            "Jumlah Missing Value": df[col].isna().sum(),
            "Jumlah Nilai Unik": df[col].nunique(dropna=True)
        }

        struktur_kolom.append(info)

    struktur_data = pd.DataFrame(struktur_kolom)

    return struktur_data

# =========================
# FUNGSI RINGKASAN TAHUNAN FLORES
# =========================

def make_flores_yearly_summary(df_panel):
    df = df_panel.copy()

    df_yearly = (
        df.groupby("Tahun", as_index=False)
        .agg(
            Rata_Rata_Kemiskinan=("Kemiskinan", "mean"),
            Rata_Rata_TPT=("TPT", "mean"),
            Total_TKD=("Total_TKD", "sum"),
            Total_KUR=("KUR", "sum")
        )
    )

    # Karena data TKD dan KUR sudah dalam satuan miliar,
    # kita hanya membuat nama kolom yang lebih jelas untuk visualisasi.
    df_yearly["Total_TKD_Miliar"] = df_yearly["Total_TKD"]
    df_yearly["Total_KUR_Miliar"] = df_yearly["Total_KUR"]

    return df_yearly

# =========================
# FUNGSI PERUBAHAN KEMISKINAN KABUPATEN
# =========================

def make_poverty_change_data(df_panel):
    df = df_panel.copy()

    tahun_awal = int(df["Tahun"].min())
    tahun_akhir = int(df["Tahun"].max())

    df_awal = df[df["Tahun"] == tahun_awal].copy()
    df_akhir = df[df["Tahun"] == tahun_akhir].copy()

    df_awal = df_awal[
        [
            "Kabupaten",
            "Kemiskinan"
        ]
    ].copy()

    df_awal = df_awal.rename(
        columns={
            "Kemiskinan": "Kemiskinan_Awal"
        }
    )

    df_akhir = df_akhir[
        [
            "Kabupaten",
            "Kemiskinan",
            "TPT",
            "Total_TKD",
            "KUR"
        ]
    ].copy()

    df_akhir = df_akhir.rename(
        columns={
            "Kemiskinan": "Kemiskinan_Akhir",
            "TPT": "TPT_Akhir",
            "Total_TKD": "Total_TKD_Akhir",
            "KUR": "KUR_Akhir"
        }
    )

    df_change = df_akhir.merge(
        df_awal,
        on="Kabupaten",
        how="left"
    )

    df_change["Perubahan_Kemiskinan"] = (
        df_change["Kemiskinan_Akhir"] - df_change["Kemiskinan_Awal"]
    )

    status_list = []

    for nilai in df_change["Perubahan_Kemiskinan"]:
        if nilai < 0:
            status = "Turun"
        elif nilai > 0:
            status = "Naik"
        else:
            status = "Tetap"

        status_list.append(status)

    df_change["Status_Perubahan"] = status_list

    df_change = df_change.sort_values(
        "Perubahan_Kemiskinan",
        ascending=True
    ).reset_index(drop=True)

    return df_change

# =========================
# FUNGSI INSIGHT EXECUTIVE STORY
# =========================

def make_executive_insight(df_panel, df_change):
    tahun_awal = int(df_panel["Tahun"].min())
    tahun_akhir = int(df_panel["Tahun"].max())

    df_akhir = df_panel[df_panel["Tahun"] == tahun_akhir].copy()

    kabupaten_kemiskinan_tertinggi = df_akhir.sort_values(
        "Kemiskinan",
        ascending=False
    ).iloc[0]["Kabupaten"]

    nilai_kemiskinan_tertinggi = df_akhir.sort_values(
        "Kemiskinan",
        ascending=False
    ).iloc[0]["Kemiskinan"]

    kabupaten_penurunan_terbesar = df_change.sort_values(
        "Perubahan_Kemiskinan",
        ascending=True
    ).iloc[0]["Kabupaten"]

    nilai_penurunan_terbesar = df_change.sort_values(
        "Perubahan_Kemiskinan",
        ascending=True
    ).iloc[0]["Perubahan_Kemiskinan"]

    kabupaten_penurunan_terlambat = df_change.sort_values(
        "Perubahan_Kemiskinan",
        ascending=False
    ).iloc[0]["Kabupaten"]

    nilai_penurunan_terlambat = df_change.sort_values(
        "Perubahan_Kemiskinan",
        ascending=False
    ).iloc[0]["Perubahan_Kemiskinan"]

    insight = {
        "Tahun Awal": tahun_awal,
        "Tahun Akhir": tahun_akhir,
        "Kabupaten Kemiskinan Tertinggi": kabupaten_kemiskinan_tertinggi,
        "Nilai Kemiskinan Tertinggi": nilai_kemiskinan_tertinggi,
        "Kabupaten Penurunan Terbesar": kabupaten_penurunan_terbesar,
        "Nilai Penurunan Terbesar": nilai_penurunan_terbesar,
        "Kabupaten Penurunan Terlambat": kabupaten_penurunan_terlambat,
        "Nilai Penurunan Terlambat": nilai_penurunan_terlambat
    }

    return insight

# =========================
# FUNGSI LABEL INDIKATOR PETA
# =========================

def get_map_indicator_label(indikator):
    label_map = {
        "Kemiskinan": "Kemiskinan (%)",
        "TPT": "Tingkat Pengangguran Terbuka (%)",
        "Total_TKD": "Total TKD (Miliar Rp)",
        "KUR": "KUR (Miliar Rp)"
    }

    if indikator in label_map:
        label = label_map[indikator]
    else:
        label = indikator

    return label

# =========================
# FUNGSI WARNA PETA
# =========================

def get_map_color_scale(indikator):
    if indikator in ["Kemiskinan", "TPT"]:
        color_scale = [
            [0.0, "green"],
            [0.5, "yellow"],
            [1.0, "red"]
        ]

    elif indikator in ["Total_TKD", "KUR"]:
        color_scale = [
            [0.0, "red"],
            [0.5, "yellow"],
            [1.0, "green"]
        ]

    else:
        color_scale = [
            [0.0, "green"],
            [0.5, "yellow"],
            [1.0, "red"]
        ]

    return color_scale

## =========================
# HELPER METRIC CARD
# =========================

def show_metric_card(label, value, delta=None, help_text=None, height=105):
    with st.container(border=True):
        if help_text is None:
            st.metric(
                label=label,
                value=value,
                delta=delta,
                height=height
            )

        else:
            st.metric(
                label=label,
                value=value,
                delta=delta,
                help=help_text,
                height=height
            )

# =========================
# HELPER BARIS METRIC
# =========================

def show_metric_row(metric_list):
    jumlah_metric = len(metric_list)

    columns = st.columns(jumlah_metric)

    for index, metric in enumerate(metric_list):
        with columns[index]:
            label = metric["label"]
            value = metric["value"]

            if "delta" in metric:
                delta = metric["delta"]
            else:
                delta = None

            if "help" in metric:
                help_text = metric["help"]
            else:
                help_text = None

            show_metric_card(
                label,
                value,
                delta,
                help_text
            )

# =========================
# HELPER LAYOUT CHART
# =========================

def apply_common_chart_layout(
    fig,
    xaxis_title=None,
    yaxis_title=None,
    height=420,
    show_legend=True
):
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        hovermode="x unified",
        height=height,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20
        },
        showlegend=show_legend
    )

    if show_legend:
        fig.update_layout(
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "center",
                "x": 0.5
            }
        )

    return fig

# =========================
# HELPER HORIZONTAL BAR CHART
# =========================

def make_horizontal_bar_chart(
    df,
    x_column,
    y_column,
    title,
    xaxis_title,
    yaxis_title,
    hover_data=None,
    height=420
):
    fig = px.bar(
        df,
        x=x_column,
        y=y_column,
        orientation="h",
        title=title,
        hover_data=hover_data
    )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        }
    )

    fig = apply_common_chart_layout(
        fig,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        show_legend=False
    )

    return fig

# =========================
# HELPER DUAL AXIS LINE CHART
# =========================

def make_dual_axis_line_chart(
    df,
    x_column,
    y_left_column,
    y_right_column,
    left_name,
    right_name,
    title,
    xaxis_title,
    y_left_title,
    y_right_title,
    left_color="red",
    right_color="royalblue",
    height=460
):
    fig = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True
                }
            ]
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=df[x_column],
            y=df[y_left_column],
            mode="lines+markers",
            name=left_name,
            line={
                "color": left_color,
                "width": 3
            },
            marker={
                "size": 7
            }
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=df[x_column],
            y=df[y_right_column],
            mode="lines+markers",
            name=right_name,
            line={
                "color": right_color,
                "width": 3,
                "dash": "dash"
            },
            marker={
                "size": 7
            }
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        template="plotly_white",
        hovermode="x unified",
        height=height,
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 20
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5
        }
    )

    fig.update_yaxes(
        title_text=y_left_title,
        secondary_y=False
    )

    fig.update_yaxes(
        title_text=y_right_title,
        secondary_y=True
    )

    return fig

# =========================
# HELPER TAMPILKAN CHART
# =========================

def show_chart(fig):
    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================
# HELPER LINE CHART
# =========================

def make_line_chart(
    df,
    x_column,
    y_column,
    title,
    xaxis_title,
    yaxis_title,
    height=420
):
    fig = px.line(
        df,
        x=x_column,
        y=y_column,
        markers=True,
        title=title
    )

    fig.update_traces(
        line={
            "width": 3
        },
        marker={
            "size": 7
        }
    )

    fig = apply_common_chart_layout(
        fig,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        show_legend=False
    )

    return fig

# =========================
# FUNGSI DATA PROFIL KABUPATEN
# =========================

def make_kabupaten_profile_data(kabupaten_terpilih):
    df_panel_kabupaten = df_panel[
        df_panel["Kabupaten"] == kabupaten_terpilih
    ].copy()

    df_panel_kabupaten = df_panel_kabupaten.sort_values(
        "Tahun"
    ).reset_index(drop=True)

    df_context_kabupaten = df_context_latest[
        df_context_latest["Kabupaten"] == kabupaten_terpilih
    ].copy()

    df_context_kabupaten = df_context_kabupaten.sort_values(
        ["Pilar", "Indikator"]
    ).reset_index(drop=True)

    df_pdrb_kabupaten = df_pdrb_latest[
        df_pdrb_latest["Kabupaten"] == kabupaten_terpilih
    ].copy()

    df_pdrb_kabupaten = df_pdrb_kabupaten.sort_values(
        "Persentase_PDRB",
        ascending=False
    ).reset_index(drop=True)

    df_pdrb_dominan_kabupaten = df_pdrb_dominant[
        df_pdrb_dominant["Kabupaten"] == kabupaten_terpilih
    ].copy()

    df_pdrb_dominan_kabupaten = df_pdrb_dominan_kabupaten.sort_values(
        "Persentase_PDRB",
        ascending=False
    ).reset_index(drop=True)

    return (
        df_panel_kabupaten,
        df_context_kabupaten,
        df_pdrb_kabupaten,
        df_pdrb_dominan_kabupaten
    )

# =========================
# FUNGSI RINGKASAN INDIKATOR KONTEKSTUAL
# =========================

def make_context_indicator_summary(df_context, indikator):
    df = df_context.copy()

    df_indicator = df[
        df["Indikator"] == indikator
    ].copy()

    df_indicator_available = df_indicator[
        df_indicator["Nilai"].notna()
    ].copy()

    tahun_awal = int(df_indicator_available["Tahun"].min())
    tahun_akhir = int(df_indicator_available["Tahun"].max())

    df_latest = df_indicator_available[
        df_indicator_available["Tahun"] == tahun_akhir
    ].copy()

    rata_latest = df_latest["Nilai"].mean()
    nilai_tertinggi = df_latest["Nilai"].max()
    nilai_terendah = df_latest["Nilai"].min()

    kabupaten_tertinggi = (
        df_latest
        .sort_values("Nilai", ascending=False)
        .iloc[0]["Kabupaten"]
    )

    kabupaten_terendah = (
        df_latest
        .sort_values("Nilai", ascending=True)
        .iloc[0]["Kabupaten"]
    )

    satuan_list = df_indicator_available["Satuan"].dropna().unique()

    if len(satuan_list) > 0:
        satuan = satuan_list[0]
    else:
        satuan = "-"

    summary = {
        "Tahun Awal": tahun_awal,
        "Tahun Akhir": tahun_akhir,
        "Rata-rata Terbaru": rata_latest,
        "Nilai Tertinggi": nilai_tertinggi,
        "Nilai Terendah": nilai_terendah,
        "Kabupaten Tertinggi": kabupaten_tertinggi,
        "Kabupaten Terendah": kabupaten_terendah,
        "Satuan": satuan
    }

    return summary, df_indicator_available, df_latest

# =========================
# FUNGSI TREN TAHUNAN INDIKATOR KONTEKSTUAL
# =========================

def make_context_yearly_trend(df_indicator_available):
    df = df_indicator_available.copy()

    df_yearly = (
        df.groupby("Tahun", as_index=False)
        .agg(
            Rata_Rata_Nilai=("Nilai", "mean"),
            Nilai_Minimum=("Nilai", "min"),
            Nilai_Maksimum=("Nilai", "max"),
            Jumlah_Kabupaten=("Kabupaten", "nunique")
        )
    )

    return df_yearly

# =========================
# FUNGSI RINGKASAN PDRB KABUPATEN
# =========================

def make_pdrb_kabupaten_summary(df_pdrb, kabupaten_terpilih):
    df = df_pdrb.copy()

    df_kabupaten = df[
        df["Kabupaten"] == kabupaten_terpilih
    ].copy()

    tahun_terbaru = int(df_kabupaten["Tahun"].max())

    df_latest = df_kabupaten[
        df_kabupaten["Tahun"] == tahun_terbaru
    ].copy()

    df_latest = df_latest.sort_values(
        "Persentase_PDRB",
        ascending=False
    ).reset_index(drop=True)

    total_pdrb = df_latest["Total_PDRB_Kabupaten_Tahun"].iloc[0]

    sektor_dominan = df_latest.iloc[0]["Lapangan_Usaha"]
    kontribusi_dominan = df_latest.iloc[0]["Persentase_PDRB"]

    jumlah_sektor = df_latest["Lapangan_Usaha"].nunique()

    summary = {
        "Tahun Terbaru": tahun_terbaru,
        "Total PDRB": total_pdrb,
        "Sektor Dominan": sektor_dominan,
        "Kontribusi Sektor Dominan": kontribusi_dominan,
        "Jumlah Sektor": jumlah_sektor
    }

    return summary, df_kabupaten, df_latest

# =========================
# FUNGSI CEK STRUKTUR MODEL PANEL
# =========================

def make_model_structure_report(df_model):
    jumlah_entity = df_model["Kabupaten"].nunique()

    jumlah_observasi = df_model.shape[0]

    jumlah_variabel_independen = df_model.drop(
        columns=[
            "Kabupaten",
            "Tahun",
            "Kemiskinan"
        ]
    ).shape[1]

    jumlah_parameter_dengan_konstanta = jumlah_variabel_independen + 1

    sisa_entity_rem = jumlah_entity - jumlah_parameter_dengan_konstanta

    report = {
        "Jumlah Entity": jumlah_entity,
        "Jumlah Observasi": jumlah_observasi,
        "Jumlah Variabel Independen": jumlah_variabel_independen,
        "Jumlah Parameter dengan Konstanta": jumlah_parameter_dengan_konstanta,
        "Entity dikurangi Parameter": sisa_entity_rem
    }

    return report


# =========================
# FUNGSI PENJELASAN MODEL GAGAL
# =========================

def make_model_failure_explanation(nama_model, error_text, df_model):
    struktur_model = make_model_structure_report(df_model)

    jumlah_entity = struktur_model["Jumlah Entity"]
    jumlah_variabel = struktur_model["Jumlah Variabel Independen"]
    jumlah_parameter = struktur_model["Jumlah Parameter dengan Konstanta"]
    sisa_entity_rem = struktur_model["Entity dikurangi Parameter"]

    if nama_model == "REM / Random Effect" and sisa_entity_rem <= 0:
        explanation = f"""
        Model **REM / Random Effect** gagal dijalankan karena struktur model tidak
        memenuhi kebutuhan derajat bebas untuk estimasi komponen random effect.

        Pada dataset ini:

        - Jumlah entity/kabupaten = **{jumlah_entity}**
        - Jumlah variabel independen = **{jumlah_variabel}**
        - Jumlah parameter termasuk konstanta = **{jumlah_parameter}**
        - Entity dikurangi parameter = **{sisa_entity_rem}**

        REM membutuhkan jumlah entity yang lebih besar daripada jumlah parameter
        agar komponen variasi antar-kabupaten dapat dihitung. Karena nilainya
        tidak mencukupi, model REM tidak digunakan untuk spesifikasi ini.

        Solusi analitis yang dapat dipakai adalah menggunakan **CEM/FEM** untuk
        model lengkap, atau membuat model ringkas seperti **Total_TKD + KUR + TPT**
        jika tetap ingin membandingkan CEM, FEM, dan REM.
        """

    else:
        explanation = f"""
        Model **{nama_model}** gagal dijalankan pada spesifikasi ini.

        Kemungkinan penyebabnya antara lain:

        - jumlah observasi terlalu kecil,
        - variabel terlalu banyak,
        - terdapat multikolinearitas kuat,
        - atau terdapat kendala teknis dalam estimasi model.

        Pesan error dari Python:

        `{error_text}`
        """

    return explanation

# =========================
# FUNGSI SATUAN VARIABEL MODEL
# =========================

def get_variable_unit(variable_name):
    variable_clean = variable_name.replace("_Lag1", "")

    variabel_miliar = [
        "DBH",
        "DAU",
        "DAK_Fisik",
        "DAK_Non_Fisik",
        "Dana_Desa",
        "DID",
        "Total_TKD",
        "KUR"
    ]

    if variable_clean in variabel_miliar:
        satuan = "miliar rupiah"

    elif variable_clean == "TPT":
        satuan = "poin persentase"

    else:
        satuan = "satuan"

    return satuan


# =========================
# FUNGSI INTERPRETASI KOEFISIEN
# =========================

def make_coefficient_interpretation_table(df_coef, alpha=0.05):
    daftar_interpretasi = []

    for index, row in df_coef.iterrows():
        variabel = row["Variabel"]

        if variabel == "const":
            continue

        koefisien = row["Koefisien"]
        p_value = row["P-value"]

        if koefisien < 0:
            arah = "Negatif"
            makna_arah = "berasosiasi dengan penurunan kemiskinan"

        elif koefisien > 0:
            arah = "Positif"
            makna_arah = "berasosiasi dengan peningkatan kemiskinan"

        else:
            arah = "Netral"
            makna_arah = "tidak menunjukkan perubahan arah hubungan"

        if p_value < 0.01:
            status_signifikan = "Signifikan pada 1%"

        elif p_value < 0.05:
            status_signifikan = "Signifikan pada 5%"

        elif p_value < 0.10:
            status_signifikan = "Signifikan pada 10%"

        else:
            status_signifikan = "Tidak signifikan"

        if p_value < alpha:
            layak_dibaca = "Ya"
        else:
            layak_dibaca = "Hati-hati"

        satuan = get_variable_unit(variabel)

        interpretasi = (
            f"Kenaikan 1 {satuan} pada {variabel} {makna_arah} "
            f"sebesar {abs(koefisien):,.4f} poin kemiskinan, "
            f"dengan asumsi variabel lain tetap."
        )

        hasil = {
            "Variabel": variabel,
            "Koefisien": koefisien,
            "Arah": arah,
            "P-value": p_value,
            "Status Signifikansi": status_signifikan,
            "Layak Dibaca Substantif": layak_dibaca,
            "Interpretasi": interpretasi
        }

        daftar_interpretasi.append(hasil)

    df_interpretasi = pd.DataFrame(daftar_interpretasi)

    return df_interpretasi


# =========================
# FUNGSI RINGKASAN VARIABEL SIGNIFIKAN
# =========================

def make_significant_variable_summary(df_interpretasi, alpha=0.05):
    df_signifikan = df_interpretasi[
        df_interpretasi["P-value"] < alpha
    ].copy()

    if df_signifikan.shape[0] == 0:
        ringkasan = (
            "Tidak terdapat variabel independen yang signifikan pada alpha yang dipilih. "
            "Interpretasi koefisien tetap dapat dibaca secara arah hubungan, tetapi perlu "
            "kehati-hatian untuk menyimpulkan pengaruh statistik."
        )

    else:
        daftar_variabel = df_signifikan["Variabel"].tolist()
        teks_variabel = ", ".join(daftar_variabel)

        ringkasan = (
            f"Variabel yang signifikan pada alpha yang dipilih adalah: {teks_variabel}. "
            "Variabel tersebut dapat menjadi fokus utama dalam pembahasan hasil model "
            "dan penyusunan rekomendasi kebijakan."
        )

    return ringkasan

# =========================
# HELPER SWOT: ARAH INDIKATOR KONTEKSTUAL
# =========================

def get_context_indicator_direction(indikator):
    indikator_lower = str(indikator).lower()

    indikator_buruk_jika_naik = [
        "kemiskinan",
        "kedalaman",
        "keparahan",
        "pengangguran",
        "tpt",
        "gini",
        "stunting",
        "rawan",
        "keluhan",
        "tidak",
        "tanpa",
        "belum",
        "rusak",
        "jarak",
        "waktu tempuh"
    ]

    indikator_baik_jika_naik = [
        "ipm",
        "rls",
        "hls",
        "uhh",
        "harapan",
        "sekolah",
        "sanitasi",
        "air minum",
        "listrik",
        "internet",
        "jalan",
        "akses",
        "wisata",
        "akomodasi",
        "puskesmas",
        "dokter",
        "tenaga kesehatan",
        "produksi",
        "kunjungan"
    ]

    if any(keyword in indikator_lower for keyword in indikator_buruk_jika_naik):
        return "Buruk Jika Naik"

    elif any(keyword in indikator_lower for keyword in indikator_baik_jika_naik):
        return "Baik Jika Naik"

    else:
        return "Netral"


# =========================
# HELPER SWOT: TAMBAH BARIS SWOT
# =========================

def add_swot_row(
    daftar_swot,
    kategori,
    sumber,
    aspek,
    poin,
    dasar_data,
    implikasi
):
    daftar_swot.append(
        {
            "Kategori": kategori,
            "Sumber Analisis": sumber,
            "Aspek": aspek,
            "Poin SWOT": poin,
            "Dasar Data": dasar_data,
            "Implikasi": implikasi
        }
    )


# =========================
# HELPER SWOT: DATA PANEL UTAMA
# =========================

def make_panel_swot_signals(kabupaten_terpilih, df_panel):
    daftar_swot = []

    df_kab = df_panel[
        df_panel["Kabupaten"] == kabupaten_terpilih
    ].copy()

    df_kab = df_kab.sort_values("Tahun").reset_index(drop=True)

    tahun_awal = int(df_kab["Tahun"].min())
    tahun_akhir = int(df_kab["Tahun"].max())

    df_awal = df_kab[
        df_kab["Tahun"] == tahun_awal
    ].copy()

    df_akhir = df_kab[
        df_kab["Tahun"] == tahun_akhir
    ].copy()

    df_flores_akhir = df_panel[
        df_panel["Tahun"] == tahun_akhir
    ].copy()

    kemiskinan_awal = df_awal.iloc[0]["Kemiskinan"]
    kemiskinan_akhir = df_akhir.iloc[0]["Kemiskinan"]
    perubahan_kemiskinan = kemiskinan_akhir - kemiskinan_awal

    tpt_akhir = df_akhir.iloc[0]["TPT"]

    tkd_akhir = df_akhir.iloc[0]["Total_TKD"]

    kur_awal = df_awal.iloc[0]["KUR"]
    kur_akhir = df_akhir.iloc[0]["KUR"]
    perubahan_kur = kur_akhir - kur_awal

    rata_kemiskinan_flores = df_flores_akhir["Kemiskinan"].mean()
    rata_tpt_flores = df_flores_akhir["TPT"].mean()
    rata_tkd_flores = df_flores_akhir["Total_TKD"].mean()
    rata_kur_flores = df_flores_akhir["KUR"].mean()

    # =========================
    # KEMISKINAN
    # =========================

    if kemiskinan_akhir <= rata_kemiskinan_flores:
        add_swot_row(
            daftar_swot,
            "Strength",
            "Panel Utama",
            "Kemiskinan",
            "Tingkat kemiskinan relatif lebih rendah dibanding rata-rata Flores.",
            f"Kemiskinan {kabupaten_terpilih}: {kemiskinan_akhir:,.2f}; rata-rata Flores: {rata_kemiskinan_flores:,.2f}.",
            "Kondisi ini menjadi modal awal untuk memperkuat program pengurangan kemiskinan."
        )

    else:
        add_swot_row(
            daftar_swot,
            "Weakness",
            "Panel Utama",
            "Kemiskinan",
            "Tingkat kemiskinan relatif lebih tinggi dibanding rata-rata Flores.",
            f"Kemiskinan {kabupaten_terpilih}: {kemiskinan_akhir:,.2f}; rata-rata Flores: {rata_kemiskinan_flores:,.2f}.",
            "Wilayah ini memerlukan prioritas intervensi pengurangan kemiskinan yang lebih tajam."
        )

    if perubahan_kemiskinan < 0:
        add_swot_row(
            daftar_swot,
            "Strength",
            "Panel Utama",
            "Tren Kemiskinan",
            "Kemiskinan menunjukkan tren penurunan selama periode pengamatan.",
            f"Perubahan kemiskinan {tahun_awal}–{tahun_akhir}: {perubahan_kemiskinan:,.2f} poin.",
            "Penurunan ini dapat menjadi momentum untuk memperkuat program yang sudah berjalan."
        )

    else:
        add_swot_row(
            daftar_swot,
            "Threat",
            "Panel Utama",
            "Tren Kemiskinan",
            "Kemiskinan belum menunjukkan penurunan selama periode pengamatan.",
            f"Perubahan kemiskinan {tahun_awal}–{tahun_akhir}: {perubahan_kemiskinan:,.2f} poin.",
            "Terdapat risiko stagnasi atau peningkatan kemiskinan jika intervensi tidak diperbaiki."
        )

    # =========================
    # TPT
    # =========================

    if tpt_akhir <= rata_tpt_flores:
        add_swot_row(
            daftar_swot,
            "Strength",
            "Panel Utama",
            "Ketenagakerjaan",
            "TPT relatif lebih rendah dibanding rata-rata Flores.",
            f"TPT {kabupaten_terpilih}: {tpt_akhir:,.2f}; rata-rata Flores: {rata_tpt_flores:,.2f}.",
            "Tekanan pengangguran relatif lebih rendah sehingga program produktivitas dapat lebih diarahkan."
        )

    else:
        add_swot_row(
            daftar_swot,
            "Weakness",
            "Panel Utama",
            "Ketenagakerjaan",
            "TPT relatif lebih tinggi dibanding rata-rata Flores.",
            f"TPT {kabupaten_terpilih}: {tpt_akhir:,.2f}; rata-rata Flores: {rata_tpt_flores:,.2f}.",
            "Perlu penguatan program penciptaan kerja, pelatihan, dan perluasan usaha produktif."
        )

        add_swot_row(
            daftar_swot,
            "Threat",
            "Panel Utama",
            "Pasar Kerja",
            "TPT yang relatif tinggi dapat menghambat penurunan kemiskinan.",
            f"TPT {kabupaten_terpilih}: {tpt_akhir:,.2f}.",
            "Risiko pengangguran perlu diantisipasi melalui perluasan kesempatan kerja dan dukungan UMKM."
        )

    # =========================
    # TKD DAN KUR
    # =========================

    if tkd_akhir >= rata_tkd_flores:
        add_swot_row(
            daftar_swot,
            "Opportunity",
            "Panel Utama",
            "Kapasitas Fiskal",
            "Dukungan TKD relatif besar dibanding rata-rata kabupaten Flores.",
            f"TKD {kabupaten_terpilih}: {tkd_akhir:,.2f} miliar; rata-rata Flores: {rata_tkd_flores:,.2f} miliar.",
            "Ruang fiskal ini dapat diarahkan pada program berdampak langsung terhadap kemiskinan."
        )

    if tkd_akhir >= rata_tkd_flores and kemiskinan_akhir > rata_kemiskinan_flores:
        add_swot_row(
            daftar_swot,
            "Weakness",
            "Panel Utama",
            "Efektivitas Fiskal",
            "Dukungan TKD relatif besar, tetapi kemiskinan masih di atas rata-rata Flores.",
            f"TKD: {tkd_akhir:,.2f} miliar; kemiskinan: {kemiskinan_akhir:,.2f}.",
            "Perlu evaluasi kualitas belanja dan penajaman program agar dukungan fiskal lebih efektif."
        )

    if kur_akhir >= rata_kur_flores:
        add_swot_row(
            daftar_swot,
            "Strength",
            "Panel Utama",
            "Pembiayaan Usaha",
            "Penyaluran KUR relatif lebih tinggi dibanding rata-rata Flores.",
            f"KUR {kabupaten_terpilih}: {kur_akhir:,.2f} miliar; rata-rata Flores: {rata_kur_flores:,.2f} miliar.",
            "Akses pembiayaan usaha dapat menjadi modal penguatan ekonomi rumah tangga dan UMKM."
        )

    else:
        add_swot_row(
            daftar_swot,
            "Weakness",
            "Panel Utama",
            "Pembiayaan Usaha",
            "Penyaluran KUR relatif lebih rendah dibanding rata-rata Flores.",
            f"KUR {kabupaten_terpilih}: {kur_akhir:,.2f} miliar; rata-rata Flores: {rata_kur_flores:,.2f} miliar.",
            "Akses pembiayaan usaha rakyat perlu diperluas."
        )

    if perubahan_kur > 0:
        add_swot_row(
            daftar_swot,
            "Opportunity",
            "Panel Utama",
            "Tren KUR",
            "KUR menunjukkan peningkatan selama periode pengamatan.",
            f"Perubahan KUR {tahun_awal}–{tahun_akhir}: {perubahan_kur:,.2f} miliar.",
            "Peningkatan KUR dapat dimanfaatkan untuk memperkuat pembiayaan usaha produktif."
        )

    else:
        add_swot_row(
            daftar_swot,
            "Threat",
            "Panel Utama",
            "Tren KUR",
            "KUR tidak menunjukkan peningkatan selama periode pengamatan.",
            f"Perubahan KUR {tahun_awal}–{tahun_akhir}: {perubahan_kur:,.2f} miliar.",
            "Stagnasi pembiayaan usaha dapat membatasi pengembangan UMKM dan ekonomi rumah tangga."
        )

    summary = {
        "Tahun Awal": tahun_awal,
        "Tahun Akhir": tahun_akhir,
        "Kemiskinan Akhir": kemiskinan_akhir,
        "Perubahan Kemiskinan": perubahan_kemiskinan,
        "TPT Akhir": tpt_akhir,
        "TKD Akhir": tkd_akhir,
        "KUR Akhir": kur_akhir,
        "Perubahan KUR": perubahan_kur
    }

    return pd.DataFrame(daftar_swot), summary


# =========================
# HELPER SWOT: PDRB
# =========================

def make_pdrb_swot_signals(df_pdrb_dominan_kabupaten):
    daftar_swot = []

    if df_pdrb_dominan_kabupaten.shape[0] == 0:
        return pd.DataFrame(daftar_swot)

    sektor_dominan = df_pdrb_dominan_kabupaten.iloc[0]["Lapangan_Usaha"]
    kontribusi_dominan = df_pdrb_dominan_kabupaten.iloc[0]["Persentase_PDRB"]
    kontribusi_tiga_sektor = df_pdrb_dominan_kabupaten["Persentase_PDRB"].sum()

    add_swot_row(
        daftar_swot,
        "Strength",
        "PDRB Sektoral",
        "Struktur Ekonomi",
        f"Memiliki basis ekonomi dominan pada sektor {sektor_dominan}.",
        f"Kontribusi sektor dominan: {kontribusi_dominan:,.2f}%.",
        "Sektor dominan dapat menjadi basis penajaman program ekonomi lokal dan pembiayaan usaha."
    )

    add_swot_row(
        daftar_swot,
        "Opportunity",
        "PDRB Sektoral",
        "Sektor Unggulan",
        f"Sektor {sektor_dominan} dapat menjadi pintu masuk pengembangan ekonomi lokal.",
        f"Kontribusi sektor dominan: {kontribusi_dominan:,.2f}%.",
        "Program TKD, KUR, dan pemberdayaan ekonomi dapat diselaraskan dengan sektor unggulan daerah."
    )

    if kontribusi_tiga_sektor >= 60:
        add_swot_row(
            daftar_swot,
            "Threat",
            "PDRB Sektoral",
            "Konsentrasi Ekonomi",
            "Struktur ekonomi relatif terkonsentrasi pada tiga sektor terbesar.",
            f"Kontribusi tiga sektor terbesar: {kontribusi_tiga_sektor:,.2f}%.",
            "Ketergantungan pada sedikit sektor dapat meningkatkan kerentanan ekonomi jika sektor dominan mengalami tekanan."
        )

    return pd.DataFrame(daftar_swot)


# =========================
# HELPER SWOT: INDIKATOR KONTEKSTUAL
# =========================

def make_contextual_swot_signals(kabupaten_terpilih, df_context, jumlah_maksimal=6):
    daftar_swot = []

    kolom_wajib = [
        "Kabupaten",
        "Pilar",
        "Indikator",
        "Tahun",
        "Nilai",
        "Satuan"
    ]

    for col in kolom_wajib:
        if col not in df_context.columns:
            return pd.DataFrame(daftar_swot)

    df = df_context.copy()

    df = df[
        df["Nilai"].notna()
    ].copy()

    df_kab = df[
        df["Kabupaten"] == kabupaten_terpilih
    ].copy()

    if df_kab.shape[0] == 0:
        return pd.DataFrame(daftar_swot)

    df_kab_latest = (
        df_kab
        .sort_values("Tahun")
        .drop_duplicates(
            subset=[
                "Pilar",
                "Indikator"
            ],
            keep="last"
        )
        .reset_index(drop=True)
    )

    daftar_kandidat = []

    for index, row in df_kab_latest.iterrows():
        pilar = row["Pilar"]
        indikator = row["Indikator"]
        tahun = row["Tahun"]
        nilai_kab = row["Nilai"]
        satuan = row["Satuan"]

        arah = get_context_indicator_direction(indikator)

        if arah == "Netral":
            continue

        df_benchmark = df[
            (df["Indikator"] == indikator) &
            (df["Tahun"] == tahun)
        ].copy()

        if df_benchmark.shape[0] == 0:
            continue

        rata_flores = df_benchmark["Nilai"].mean()

        if pd.isna(rata_flores):
            continue

        selisih = nilai_kab - rata_flores
        abs_selisih = abs(selisih)

        if abs_selisih == 0:
            continue

        if arah == "Buruk Jika Naik":
            if nilai_kab > rata_flores:
                kategori = "Weakness"
                poin = f"{indikator} relatif lebih tinggi dibanding rata-rata Flores."
                implikasi = "Kondisi ini menunjukkan tekanan kontekstual yang dapat memperberat penurunan kemiskinan."
            else:
                kategori = "Strength"
                poin = f"{indikator} relatif lebih rendah dibanding rata-rata Flores."
                implikasi = "Kondisi ini dapat menjadi modal pendukung dalam pengurangan kemiskinan."

        else:
            if nilai_kab > rata_flores:
                kategori = "Strength"
                poin = f"{indikator} relatif lebih baik dibanding rata-rata Flores."
                implikasi = "Kondisi ini dapat dimanfaatkan sebagai kekuatan pendukung pembangunan sosial-ekonomi."
            else:
                kategori = "Weakness"
                poin = f"{indikator} relatif lebih rendah dibanding rata-rata Flores."
                implikasi = "Kondisi ini menunjukkan perlunya penguatan layanan dasar atau faktor pendukung pembangunan."

        dasar_data = (
            f"{indikator} {kabupaten_terpilih} tahun {tahun}: "
            f"{nilai_kab:,.2f} {satuan}; rata-rata Flores: "
            f"{rata_flores:,.2f} {satuan}."
        )

        daftar_kandidat.append(
            {
                "Kategori": kategori,
                "Sumber Analisis": "Indikator Kontekstual",
                "Aspek": pilar,
                "Poin SWOT": poin,
                "Dasar Data": dasar_data,
                "Implikasi": implikasi,
                "Abs Selisih": abs_selisih
            }
        )

    df_kandidat = pd.DataFrame(daftar_kandidat)

    if df_kandidat.shape[0] == 0:
        return pd.DataFrame(daftar_swot)

    df_kandidat = (
        df_kandidat
        .sort_values("Abs Selisih", ascending=False)
        .head(jumlah_maksimal)
        .reset_index(drop=True)
    )

    df_kandidat = df_kandidat.drop(
        columns=[
            "Abs Selisih"
        ]
    )

    return df_kandidat


# =========================
# HELPER SWOT: SINYAL MODEL PANEL
# =========================

def make_model_swot_signals(df_coef_model, alpha=0.05):
    daftar_swot = []

    if df_coef_model is None:
        return pd.DataFrame(daftar_swot)

    if df_coef_model.shape[0] == 0:
        return pd.DataFrame(daftar_swot)

    df = df_coef_model.copy()

    df = df[
        df["Variabel"] != "const"
    ].copy()

    df = df[
        df["P-value"] < alpha
    ].copy()

    if df.shape[0] == 0:
        add_swot_row(
            daftar_swot,
            "Threat",
            "Model Panel",
            "Sinyal Statistik",
            "Tidak terdapat variabel model yang signifikan pada alpha yang dipilih.",
            f"Alpha: {alpha}.",
            "Rekomendasi kebijakan perlu lebih bertumpu pada analisis deskriptif, konteks wilayah, dan teori pendukung."
        )

        return pd.DataFrame(daftar_swot)

    for index, row in df.iterrows():
        variabel = row["Variabel"]
        koefisien = row["Koefisien"]
        p_value = row["P-value"]

        if koefisien < 0:
            add_swot_row(
                daftar_swot,
                "Opportunity",
                "Model Panel",
                "Sinyal Statistik",
                f"{variabel} berasosiasi negatif dengan kemiskinan.",
                f"Koefisien: {koefisien:,.4f}; p-value: {p_value:,.4f}.",
                "Variabel ini dapat dibaca sebagai kanal kebijakan potensial untuk mendukung penurunan kemiskinan."
            )

        elif koefisien > 0:
            add_swot_row(
                daftar_swot,
                "Threat",
                "Model Panel",
                "Sinyal Statistik",
                f"{variabel} berasosiasi positif dengan kemiskinan.",
                f"Koefisien: {koefisien:,.4f}; p-value: {p_value:,.4f}.",
                "Hasil ini perlu dievaluasi lebih lanjut karena dapat mencerminkan targeting ke wilayah miskin atau efektivitas program yang belum optimal."
            )

    return pd.DataFrame(daftar_swot)


# =========================
# HELPER SWOT: GABUNG SEMUA ANALISIS
# =========================

def make_consolidated_swot_analysis(
    kabupaten_terpilih,
    df_panel,
    df_context,
    df_pdrb_dominan_kabupaten,
    df_coef_model=None,
    alpha=0.05
):
    df_panel_swot, summary = make_panel_swot_signals(
        kabupaten_terpilih,
        df_panel
    )

    df_pdrb_swot = make_pdrb_swot_signals(
        df_pdrb_dominan_kabupaten
    )

    df_context_swot = make_contextual_swot_signals(
        kabupaten_terpilih,
        df_context,
        jumlah_maksimal=6
    )

    df_model_swot = make_model_swot_signals(
        df_coef_model,
        alpha=alpha
    )

    daftar_df = [
        df_panel_swot,
        df_pdrb_swot,
        df_context_swot,
        df_model_swot
    ]

    daftar_df_valid = []

    for df_item in daftar_df:
        if df_item is not None and df_item.shape[0] > 0:
            daftar_df_valid.append(df_item)

    if len(daftar_df_valid) == 0:
        df_swot = pd.DataFrame(
            columns=[
                "Kategori",
                "Sumber Analisis",
                "Aspek",
                "Poin SWOT",
                "Dasar Data",
                "Implikasi"
            ]
        )

    else:
        df_swot = pd.concat(
            daftar_df_valid,
            ignore_index=True
        )

    urutan_kategori = {
        "Strength": 1,
        "Weakness": 2,
        "Opportunity": 3,
        "Threat": 4
    }

    df_swot["Urutan"] = df_swot["Kategori"].map(
        urutan_kategori
    )

    df_swot = (
        df_swot
        .sort_values(
            [
                "Urutan",
                "Sumber Analisis",
                "Aspek"
            ]
        )
        .drop(columns=["Urutan"])
        .reset_index(drop=True)
    )

    summary["Jumlah Strength"] = df_swot[
        df_swot["Kategori"] == "Strength"
    ].shape[0]

    summary["Jumlah Weakness"] = df_swot[
        df_swot["Kategori"] == "Weakness"
    ].shape[0]

    summary["Jumlah Opportunity"] = df_swot[
        df_swot["Kategori"] == "Opportunity"
    ].shape[0]

    summary["Jumlah Threat"] = df_swot[
        df_swot["Kategori"] == "Threat"
    ].shape[0]

    return df_swot, summary


# =========================
# HELPER SWOT: STRATEGI
# =========================

def make_swot_strategy_table(df_swot):
    def ambil_poin(kategori):
        df_kategori = df_swot[
            df_swot["Kategori"] == kategori
        ].copy()

        if df_kategori.shape[0] == 0:
            return "belum ada poin utama"

        return df_kategori.iloc[0]["Poin SWOT"].lower()

    strength = ambil_poin("Strength")
    weakness = ambil_poin("Weakness")
    opportunity = ambil_poin("Opportunity")
    threat = ambil_poin("Threat")

    df_strategy = pd.DataFrame(
        [
            {
                "Strategi": "SO",
                "Logika": "Memanfaatkan kekuatan untuk mengambil peluang.",
                "Rumusan Strategi": f"Gunakan kekuatan berupa {strength} untuk menangkap peluang berupa {opportunity}."
            },
            {
                "Strategi": "WO",
                "Logika": "Menggunakan peluang untuk memperbaiki kelemahan.",
                "Rumusan Strategi": f"Gunakan peluang berupa {opportunity} untuk mengatasi kelemahan berupa {weakness}."
            },
            {
                "Strategi": "ST",
                "Logika": "Menggunakan kekuatan untuk mengurangi ancaman.",
                "Rumusan Strategi": f"Gunakan kekuatan berupa {strength} untuk mengurangi ancaman berupa {threat}."
            },
            {
                "Strategi": "WT",
                "Logika": "Meminimalkan kelemahan dan menghindari ancaman.",
                "Rumusan Strategi": f"Kurangi kelemahan berupa {weakness} agar risiko dari ancaman berupa {threat} tidak semakin besar."
            }
        ]
    )

    return df_strategy

# =========================
# HELPER SWOT: REKOMENDASI
# =========================

def make_policy_recommendation_from_swot(df_swot, summary):
    daftar_rekomendasi = []

    jumlah_weakness = summary["Jumlah Weakness"]
    jumlah_threat = summary["Jumlah Threat"]

    if jumlah_weakness + jumlah_threat >= 6:
        prioritas_umum = "Tinggi"
    elif jumlah_weakness + jumlah_threat >= 3:
        prioritas_umum = "Menengah-Tinggi"
    else:
        prioritas_umum = "Menengah"

    daftar_rekomendasi.append(
        {
            "Prioritas": prioritas_umum,
            "Bidang": "Pengurangan Kemiskinan",
            "Rekomendasi": "Menajamkan program penurunan kemiskinan berdasarkan karakteristik kabupaten.",
            "Dasar": f"Kemiskinan terakhir {summary['Kemiskinan Akhir']:,.2f}; perubahan kemiskinan {summary['Perubahan Kemiskinan']:,.2f} poin.",
            "Tindak Lanjut": "Sinkronkan TKD, KUR, program pemberdayaan ekonomi, dan indikator kontekstual yang tertinggal."
        }
    )

    df_weakness_context = df_swot[
        (df_swot["Kategori"] == "Weakness") &
        (df_swot["Sumber Analisis"] == "Indikator Kontekstual")
    ].copy()

    if df_weakness_context.shape[0] > 0:
        isu_context = df_weakness_context.iloc[0]

        daftar_rekomendasi.append(
            {
                "Prioritas": "Menengah-Tinggi",
                "Bidang": "Indikator Kontekstual",
                "Rekomendasi": "Memperkuat intervensi pada indikator kontekstual yang tertinggal.",
                "Dasar": isu_context["Dasar Data"],
                "Tindak Lanjut": "Petakan program layanan dasar, belanja daerah, dan dukungan sektoral yang paling terkait dengan indikator tersebut."
            }
        )

    df_model_opportunity = df_swot[
        (df_swot["Kategori"] == "Opportunity") &
        (df_swot["Sumber Analisis"] == "Model Panel")
    ].copy()

    if df_model_opportunity.shape[0] > 0:
        signal_model = df_model_opportunity.iloc[0]

        daftar_rekomendasi.append(
            {
                "Prioritas": "Menengah",
                "Bidang": "Sinyal Model Panel",
                "Rekomendasi": "Memanfaatkan variabel yang terindikasi berasosiasi dengan penurunan kemiskinan sebagai kanal kebijakan.",
                "Dasar": signal_model["Dasar Data"],
                "Tindak Lanjut": "Gunakan hasil model sebagai bahan pendukung, kemudian validasi dengan konteks wilayah dan kualitas implementasi program."
            }
        )

    daftar_rekomendasi.append(
        {
            "Prioritas": "Menengah",
            "Bidang": "Pembiayaan Usaha",
            "Rekomendasi": "Mengoptimalkan KUR agar lebih terarah pada sektor produktif dan rumah tangga rentan.",
            "Dasar": f"KUR terakhir {summary['KUR Akhir']:,.2f} miliar; perubahan KUR {summary['Perubahan KUR']:,.2f} miliar.",
            "Tindak Lanjut": "Koordinasikan perbankan, pemerintah daerah, dan pendamping UMKM untuk memperbaiki akses dan kualitas pembiayaan."
        }
    )

    daftar_rekomendasi.append(
        {
            "Prioritas": "Pendukung",
            "Bidang": "Monitoring dan Evaluasi",
            "Rekomendasi": "Membangun monitoring berbasis outcome untuk membaca efektivitas TKD, KUR, dan program kemiskinan.",
            "Dasar": "SWOT menggabungkan sinyal panel, PDRB, indikator kontekstual, dan hasil model.",
            "Tindak Lanjut": "Gunakan dashboard ini sebagai alat pemantauan rutin lintas kabupaten dan lintas indikator."
        }
    )

    return pd.DataFrame(daftar_rekomendasi)

# =========================
# HALAMAN OVERVIEW
# =========================

def halaman_overview():

    st.title("Fiscal-Poverty Intelligence Dashboard Pulau Flores")

    st.markdown(
        """
        Dashboard ini menggabungkan analisis inferensial data panel dan analisis
        deskriptif kontekstual untuk membaca hubungan TKD, KUR, pengangguran,
        dan kemiskinan di wilayah Pulau Flores dan sekitarnya.
        """
    )

    st.divider()

    # =========================
    # OVERVIEW DATA UTAMA
    # =========================

    st.header("Overview Data Utama")

    jumlah_kabupaten_panel = df_panel["Kabupaten"].nunique()
    tahun_awal = int(df_panel["Tahun"].min())
    tahun_akhir = int(df_panel["Tahun"].max())

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric("Jumlah Kabupaten", jumlah_kabupaten_panel)

    with col2:
        with st.container(border=True):
            st.metric("Periode Data Panel", f"{tahun_awal}–{tahun_akhir}")

    with col3:
        with st.container(border=True):
            st.metric("Observasi Panel", f"{df_panel.shape[0]:,}")

    with col4:
        with st.container(border=True):
            st.metric("Sheet Utama", 4)

    st.caption(
        "Data #N/A dari sumber BPS diperlakukan sebagai missing value resmi, bukan sebagai nilai nol."
    )

    st.divider()

    # =========================
    # STATUS DATA PANEL
    # =========================

    st.header("Status Data Panel Inferensial")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric(
                "Status Panel",
                panel_report["Status Panel"]
            )

    with col2:
        with st.container(border=True):
            st.metric(
                "Expected Observations",
                panel_report["Expected Observations"]
            )

    with col3:
        with st.container(border=True):
            st.metric(
                "Unique Observations",
                panel_report["Unique Observations"]
            )

    with col4:
        with st.container(border=True):
            st.metric(
                "Duplicate Observations",
                panel_report["Duplicate Observations"]
            )

    if panel_report["Status Panel"] == "Balanced Panel":
        st.success(
            "Data panel inferensial sudah balanced. Setiap kabupaten memiliki observasi untuk seluruh tahun."
        )

    else:
        st.warning(
            "Data panel inferensial belum balanced atau memiliki duplikasi. Perlu dicek sebelum masuk model panel."
        )

        if missing_panel.shape[0] > 0:
            with st.expander("Lihat Kombinasi Kabupaten-Tahun yang Hilang"):
                st.dataframe(
                    missing_panel,
                    use_container_width=True
                )

    st.divider()

    # =========================
    # RINGKASAN DATASET MODEL
    # =========================

    st.header("Ringkasan Dataset Model Inferensial")

    st.dataframe(
        df_ringkasan_model,
        use_container_width=True
    )

    st.caption(
        "Model Normal menggunakan variabel tahun berjalan. Model Lag 1 menggunakan variabel independen tahun sebelumnya."
    )

    st.divider()

    # =========================
    # RINGKASAN DATA KONTEKSTUAL
    # =========================

    st.header("Ringkasan Data Kontekstual")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric(
                "Jumlah Indikator",
                context_report["Jumlah Indikator"]
            )

    with col2:
        with st.container(border=True):
            st.metric(
                "Jumlah Pilar",
                context_report["Jumlah Pilar"]
            )

    with col3:
        with st.container(border=True):
            st.metric(
                "Nilai Tersedia",
                f'{context_report["Nilai Tersedia"]:,}'
            )

    with col4:
        with st.container(border=True):
            st.metric(
                "Missing Resmi BPS",
                f'{context_report["Missing Resmi BPS"]:,}'
            )

    st.caption(
        "Missing Resmi BPS menunjukkan indikator yang memang tidak tersedia pada tahun tertentu, bukan nilai nol."
    )

    with st.expander("Ketersediaan Indikator Kontekstual"):
        st.dataframe(
            df_indicator_availability,
            use_container_width=True
        )

    with st.expander("Data Kontekstual Terbaru per Kabupaten-Indikator"):
        st.dataframe(
            df_context_latest,
            use_container_width=True
        )

    with st.expander("Data Kontekstual Terbaru Format Wide"):
        st.dataframe(
            df_context_wide_latest,
            use_container_width=True
        )

    st.divider()

    # =========================
    # RINGKASAN DATA PDRB
    # =========================

    st.header("Ringkasan Data PDRB Sektoral")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric(
                "Jumlah Kabupaten",
                pdrb_report["Jumlah Kabupaten"]
            )

    with col2:
        with st.container(border=True):
            st.metric(
                "Jumlah Tahun",
                pdrb_report["Jumlah Tahun"]
            )

    with col3:
        with st.container(border=True):
            st.metric(
                "Jumlah Sektor",
                pdrb_report["Jumlah Sektor"]
            )

    with col4:
        with st.container(border=True):
            st.metric(
                "Missing Value",
                pdrb_report["Missing Value"]
            )

    st.caption(
        "Persentase PDRB sektoral dihitung dari nilai setiap sektor terhadap total PDRB kabupaten pada tahun yang sama."
    )

    with st.expander("PDRB Tahun Terbaru"):
        st.dataframe(
            df_pdrb_latest,
            use_container_width=True
        )

    with st.expander("Tiga Sektor Dominan per Kabupaten"):
        st.dataframe(
            df_pdrb_dominant,
            use_container_width=True
        )

    with st.expander("Cek Total Persentase PDRB"):
        st.dataframe(
            df_pdrb_percentage_check,
            use_container_width=True
        )

    st.divider()

    # =========================
    # RINGKASAN REFERENSI KABUPATEN
    # =========================

    st.header("Ringkasan Referensi Kabupaten")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric(
                "Jumlah Kabupaten",
                kabupaten_report["Jumlah Kabupaten"]
            )

    with col2:
        with st.container(border=True):
            st.metric(
                "Pulau/Kawasan",
                kabupaten_report["Jumlah Pulau/Kawasan"]
            )

    with col3:
        with st.container(border=True):
            st.metric(
                "Koordinat Lengkap",
                kabupaten_report["Koordinat Lengkap"]
            )

    with col4:
        total_missing_koordinat = (
            kabupaten_report["Missing Latitude"] +
            kabupaten_report["Missing Longitude"]
        )
        
        with st.container(border=True):
            st.metric(
                "Missing Koordinat",
                total_missing_koordinat
            )

    with st.expander("Cek Konsistensi Kabupaten Antar Data"):
        st.dataframe(
            df_kabupaten_consistency,
            use_container_width=True
        )

    with st.expander("Referensi Kabupaten Siap Peta"):
        st.dataframe(
            df_map_reference,
            use_container_width=True
        )

    with st.expander("Data Dasar Peta dengan Indikator Panel Terbaru"):
        st.dataframe(
            df_map_base,
            use_container_width=True
        )

    st.divider()

    # =========================
    # RINGKASAN PER SHEET
    # =========================

    st.header("Ringkasan Sheet Data")

    daftar_ringkasan_sheet = []

    ringkasan_panel = make_sheet_summary("panel_inferensial_raw", df_panel_raw)
    ringkasan_panel_processed = make_sheet_summary("panel_inferensial_processed", df_panel)
    ringkasan_context_raw = make_sheet_summary("indikator_kontekstual_raw", df_context_raw)
    ringkasan_context_processed = make_sheet_summary("indikator_kontekstual_processed", df_context)
    ringkasan_pdrb_raw = make_sheet_summary("pdrb_sektoral_raw", df_pdrb_raw)
    ringkasan_pdrb_processed = make_sheet_summary("pdrb_sektoral_processed", df_pdrb)
    ringkasan_kabupaten_raw = make_sheet_summary("referensi_kabupaten_raw", df_kabupaten_raw)
    ringkasan_kabupaten_processed = make_sheet_summary("referensi_kabupaten_processed", df_kabupaten)

    daftar_ringkasan_sheet.append(ringkasan_panel)
    daftar_ringkasan_sheet.append(ringkasan_panel_processed)
    daftar_ringkasan_sheet.append(ringkasan_context_raw)
    daftar_ringkasan_sheet.append(ringkasan_context_processed)
    daftar_ringkasan_sheet.append(ringkasan_pdrb_raw)
    daftar_ringkasan_sheet.append(ringkasan_pdrb_processed)
    daftar_ringkasan_sheet.append(ringkasan_kabupaten_raw)
    daftar_ringkasan_sheet.append(ringkasan_kabupaten_processed)

    df_ringkasan_sheet = pd.DataFrame(daftar_ringkasan_sheet)

    st.dataframe(df_ringkasan_sheet, use_container_width=True)

    st.divider()

    # =========================
    # PREVIEW DATA
    # =========================

    st.header("Preview Data")

    with st.expander("Preview panel_inferensial_raw"):
        st.dataframe(
            df_panel_raw,
            use_container_width=True
        )

    with st.expander("Preview panel_inferensial_processed"):
        st.dataframe(
            df_panel,
            use_container_width=True
        )

    with st.expander("Preview dataset Model Normal"):
        st.dataframe(
            df_model_normal,
            use_container_width=True
        )

    with st.expander("Preview dataset Model Lag 1"):
        st.dataframe(
            df_model_lag,
            use_container_width=True
        )

    with st.expander("Preview indikator_kontekstual_raw"):
        st.dataframe(
            df_context_raw,
            use_container_width=True
        )

    with st.expander("Preview indikator_kontekstual_processed"):
        st.dataframe(
            df_context,
            use_container_width=True
        )

    with st.expander("Preview indikator_kontekstual_latest"):
        st.dataframe(
            df_context_latest,
            use_container_width=True
        )

    with st.expander("Preview indikator_kontekstual_wide_latest"):
        st.dataframe(
            df_context_wide_latest,
            use_container_width=True
        )

    with st.expander("Preview pdrb_sektoral_raw"):
        st.dataframe(
            df_pdrb_raw,
            use_container_width=True
        )

    with st.expander("Preview pdrb_sektoral_processed"):
        st.dataframe(
            df_pdrb,
            use_container_width=True
        )

    with st.expander("Preview pdrb_sektoral_latest"):
        st.dataframe(
            df_pdrb_latest,
            use_container_width=True
        )

    with st.expander("Preview pdrb_sektor_dominan"):
        st.dataframe(
            df_pdrb_dominant,
            use_container_width=True
        )
    
    with st.expander("Preview referensi_kabupaten_raw"):
        st.dataframe(
            df_kabupaten_raw,
            use_container_width=True
        )

    with st.expander("Preview referensi_kabupaten_processed"):
        st.dataframe(
            df_kabupaten,
            use_container_width=True
        )

    with st.expander("Preview map_reference"):
        st.dataframe(
            df_map_reference,
            use_container_width=True
        )

    with st.expander("Preview map_base"):
        st.dataframe(
            df_map_base,
            use_container_width=True
        )

    # =========================
    # STRUKTUR KOLOM
    # =========================

    st.header("Struktur Kolom Setiap Data")

    with st.expander("Struktur Kolom panel_inferensial_raw"):
        struktur_panel_raw = make_column_summary(df_panel_raw)

        st.dataframe(
            struktur_panel_raw,
            use_container_width=True
        )

    with st.expander("Struktur Kolom panel_inferensial_processed"):
        struktur_panel = make_column_summary(df_panel)

        st.dataframe(
            struktur_panel,
            use_container_width=True
        )

    with st.expander("Struktur Kolom dataset Model Normal"):
        struktur_model_normal = make_column_summary(df_model_normal)

        st.dataframe(
            struktur_model_normal,
            use_container_width=True
        )

    with st.expander("Struktur Kolom dataset Model Lag 1"):
        struktur_model_lag = make_column_summary(df_model_lag)

        st.dataframe(
            struktur_model_lag,
            use_container_width=True
        )

    with st.expander("Struktur Kolom indikator_kontekstual_raw"):
        struktur_context_raw = make_column_summary(df_context_raw)

        st.dataframe(
            struktur_context_raw,
            use_container_width=True
        )

    with st.expander("Struktur Kolom indikator_kontekstual_processed"):
        struktur_context = make_column_summary(df_context)

        st.dataframe(
            struktur_context,
            use_container_width=True
        )

    with st.expander("Struktur Kolom indikator_kontekstual_latest"):
        struktur_context_latest = make_column_summary(df_context_latest)

        st.dataframe(
            struktur_context_latest,
            use_container_width=True
        )

    with st.expander("Struktur Kolom indikator_kontekstual_wide_latest"):
        struktur_context_wide_latest = make_column_summary(df_context_wide_latest)

        st.dataframe(
            struktur_context_wide_latest,
            use_container_width=True
        )

    with st.expander("Struktur Kolom pdrb_sektoral_raw"):
        struktur_pdrb_raw = make_column_summary(df_pdrb_raw)

        st.dataframe(
            struktur_pdrb_raw,
            use_container_width=True
        )

    with st.expander("Struktur Kolom pdrb_sektoral_processed"):
        struktur_pdrb = make_column_summary(df_pdrb)

        st.dataframe(
            struktur_pdrb,
            use_container_width=True
        )

    with st.expander("Struktur Kolom pdrb_sektoral_latest"):
        struktur_pdrb_latest = make_column_summary(df_pdrb_latest)

        st.dataframe(
            struktur_pdrb_latest,
            use_container_width=True
        )

    with st.expander("Struktur Kolom pdrb_sektor_dominan"):
        struktur_pdrb_dominant = make_column_summary(df_pdrb_dominant)

        st.dataframe(
            struktur_pdrb_dominant,
            use_container_width=True
        )

    with st.expander("Struktur Kolom referensi_kabupaten"):
        struktur_kabupaten = make_column_summary(df_kabupaten)

        st.dataframe(
            struktur_kabupaten,
            use_container_width=True
        )

    with st.expander("Struktur Kolom referensi_kabupaten_raw"):
        struktur_kabupaten_raw = make_column_summary(df_kabupaten_raw)

        st.dataframe(
            struktur_kabupaten_raw,
            use_container_width=True
        )

    with st.expander("Struktur Kolom referensi_kabupaten_processed"):
        struktur_kabupaten = make_column_summary(df_kabupaten)

        st.dataframe(
            struktur_kabupaten,
            use_container_width=True
        )

    with st.expander("Struktur Kolom map_reference"):
        struktur_map_reference = make_column_summary(df_map_reference)

        st.dataframe(
            struktur_map_reference,
            use_container_width=True
        )

    with st.expander("Struktur Kolom map_base"):
        struktur_map_base = make_column_summary(df_map_base)

        st.dataframe(
            struktur_map_base,
            use_container_width=True
        )

# =========================
# HALAMAN FLORES INTELLIGENCE
# =========================

def halaman_flores_intelligence():

    # =========================
    # HEADER
    # =========================

    st.title("Flores Intelligence")

    st.markdown(
        """
        Halaman ini menyajikan gambaran besar Pulau Flores melalui kombinasi
        tren kemiskinan, alokasi TKD, penyaluran KUR, sebaran spasial indikator,
        dan indikator kontekstual. Halaman ini menjadi dasar awal sebelum masuk
        ke analisis inferensial dan SWOT kebijakan.
        """
    )

    st.divider()

    # =========================
    # DATA DASAR
    # =========================

    df_yearly = make_flores_yearly_summary(df_panel)
    df_change = make_poverty_change_data(df_panel)
    insight = make_executive_insight(df_panel, df_change)

    tahun_awal = insight["Tahun Awal"]
    tahun_akhir = insight["Tahun Akhir"]

    df_tahun_awal = df_panel[
        df_panel["Tahun"] == tahun_awal
    ].copy()

    df_tahun_akhir = df_panel[
        df_panel["Tahun"] == tahun_akhir
    ].copy()

    rata_kemiskinan_awal = df_tahun_awal["Kemiskinan"].mean()
    rata_kemiskinan_akhir = df_tahun_akhir["Kemiskinan"].mean()

    perubahan_rata_kemiskinan = (
        rata_kemiskinan_akhir - rata_kemiskinan_awal
    )

    rata_tpt_akhir = df_tahun_akhir["TPT"].mean()
    total_tkd_akhir = df_tahun_akhir["Total_TKD"].sum()
    total_kur_akhir = df_tahun_akhir["KUR"].sum()

    # =========================
    # METRIC UTAMA
    # =========================

    st.header("Ringkasan Makro Flores")

    show_metric_row(
        [
            {
                "label": f"Rata-rata Kemiskinan {tahun_akhir}",
                "value": f"{rata_kemiskinan_akhir:,.2f}",
                "delta": f"{perubahan_rata_kemiskinan:,.2f}"
            },
            {
                "label": f"Rata-rata TPT {tahun_akhir}",
                "value": f"{rata_tpt_akhir:,.2f}"
            },
            {
                "label": f"Total TKD {tahun_akhir}",
                "value": f"{total_tkd_akhir:,.0f} Miliar"
            },
            {
                "label": f"Total KUR {tahun_akhir}",
                "value": f"{total_kur_akhir:,.0f} Miliar"
            }
        ]
    )

    st.caption(
        "Delta pada rata-rata kemiskinan menunjukkan perubahan dibandingkan tahun awal periode data."
    )

    st.divider()

    # =========================
    # NARASI UTAMA
    # =========================

    st.header("Executive Insight")

    with st.container(border=True):
        st.markdown(
            f"""
            Pada periode **{tahun_awal}–{tahun_akhir}**, rata-rata kemiskinan kabupaten
            di wilayah Pulau Flores berubah sebesar **{perubahan_rata_kemiskinan:,.2f} poin**.

            Pada tahun **{tahun_akhir}**, kabupaten dengan tingkat kemiskinan tertinggi adalah
            **{insight["Kabupaten Kemiskinan Tertinggi"]}** dengan nilai
            **{insight["Nilai Kemiskinan Tertinggi"]:,.2f}**.

            Kabupaten dengan penurunan kemiskinan terbesar selama periode pengamatan adalah
            **{insight["Kabupaten Penurunan Terbesar"]}** dengan perubahan sebesar
            **{insight["Nilai Penurunan Terbesar"]:,.2f} poin**.

            Kabupaten dengan perubahan kemiskinan paling lambat atau paling tinggi adalah
            **{insight["Kabupaten Penurunan Terlambat"]}** dengan perubahan sebesar
            **{insight["Nilai Penurunan Terlambat"]:,.2f} poin**.
            """
        )

    st.divider()

    # =========================
    # TREN UTAMA
    # =========================

    st.header("Tren Fiskal dan Kemiskinan")

    fig_kemiskinan_tkd = make_dual_axis_line_chart(
        df=df_yearly,
        x_column="Tahun",
        y_left_column="Rata_Rata_Kemiskinan",
        y_right_column="Total_TKD_Miliar",
        left_name="Rata-rata Kemiskinan (%)",
        right_name="Total TKD (Miliar Rp)",
        title="Tren Historis: Total TKD vs Angka Kemiskinan",
        xaxis_title="Tahun",
        y_left_title="Rata-rata Kemiskinan (%)",
        y_right_title="Total TKD (Miliar Rp)"
    )

    show_chart(fig_kemiskinan_tkd)

    st.markdown(
        """
        Grafik ini menjadi visual utama untuk membaca apakah peningkatan alokasi TKD
        bergerak beriringan dengan penurunan kemiskinan secara deskriptif. Grafik ini
        belum membuktikan hubungan sebab-akibat, tetapi menjadi pintu masuk sebelum
        analisis panel.
        """
    )

    st.divider()

    # =========================
    # KREDIT USAHA RAKYAT
    # =========================

    st.header("Kredit Usaha Rakyat")

    fig_kur = px.line(
        df_yearly,
        x="Tahun",
        y="Total_KUR_Miliar",
        markers=True,
        title="Tren Total KUR Pulau Flores"
    )

    fig_kur.update_traces(
        line={
            "width": 3
        },
        marker={
            "size": 7
        }
    )

    fig_kur.update_layout(
        xaxis_title="Tahun",
        yaxis_title="Total KUR (Miliar Rp)",
        template="plotly_white",
        hovermode="x unified",
        height=420,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20
        }
    )

    st.plotly_chart(
        fig_kur,
        use_container_width=True
    )

    st.markdown(
        """
        KUR dibaca sebagai indikator pembiayaan usaha rakyat. Tren KUR membantu melihat
        apakah akses pembiayaan produktif meningkat dari waktu ke waktu di wilayah Flores.
        """
    )

    st.divider()

    # =========================
    # PRIORITAS WILAYAH
    # =========================

    st.header("Prioritas Wilayah")

    df_prioritas = df_change.sort_values(
        "Kemiskinan_Akhir",
        ascending=False
    ).reset_index(drop=True)

    df_prioritas = df_prioritas[
        [
            "Kabupaten",
            "Kemiskinan_Akhir",
            "Perubahan_Kemiskinan",
            "TPT_Akhir",
            "Total_TKD_Akhir",
            "KUR_Akhir"
        ]
    ].copy()

    df_prioritas_top = df_prioritas.head(5).copy()

    kabupaten_prioritas_utama = df_prioritas_top.iloc[0]["Kabupaten"]
    kemiskinan_prioritas_utama = df_prioritas_top.iloc[0]["Kemiskinan_Akhir"]

    rata_kemiskinan_prioritas = df_prioritas_top["Kemiskinan_Akhir"].mean()
    rata_tpt_prioritas = df_prioritas_top["TPT_Akhir"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric(
                "Prioritas Utama",
                kabupaten_prioritas_utama,
                height=105
            )

    with col2:
        with st.container(border=True):
            st.metric(
                "Kemiskinan Prioritas Utama",
                f"{kemiskinan_prioritas_utama:,.2f}",
                height=105
            )

    with col3:
        with st.container(border=True):
            st.metric(
                "Rata-rata Kemiskinan Top 5",
                f"{rata_kemiskinan_prioritas:,.2f}",
                height=105
            )

    with col4:
        with st.container(border=True):
            st.metric(
                "Rata-rata TPT Top 5",
                f"{rata_tpt_prioritas:,.2f}",
                height=105
            )

    st.caption(
        "Prioritas wilayah ditentukan berdasarkan tingkat kemiskinan tahun terakhir tertinggi."
    )

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.subheader("Top 5 Kabupaten Prioritas")

        st.dataframe(
            df_prioritas_top.style.format(
                {
                    "Kemiskinan_Akhir": "{:,.2f}",
                    "Perubahan_Kemiskinan": "{:,.2f}",
                    "TPT_Akhir": "{:,.2f}",
                    "Total_TKD_Akhir": "{:,.2f}",
                    "KUR_Akhir": "{:,.2f}"
                }
            ),
            use_container_width=True
        )

    with col2:
        df_prioritas_chart = df_prioritas_top.sort_values(
            "Kemiskinan_Akhir",
            ascending=True
        ).reset_index(drop=True)

        fig_prioritas = px.bar(
            df_prioritas_chart,
            x="Kemiskinan_Akhir",
            y="Kabupaten",
            orientation="h",
            title=f"Top 5 Kemiskinan Tertinggi {tahun_akhir}",
            hover_data=[
                "Perubahan_Kemiskinan",
                "TPT_Akhir",
                "Total_TKD_Akhir",
                "KUR_Akhir"
            ]
        )

        fig_prioritas.update_layout(
            xaxis_title="Kemiskinan (%)",
            yaxis_title="Kabupaten",
            template="plotly_white",
            height=420,
            margin={
                "l": 20,
                "r": 20,
                "t": 60,
                "b": 20
            },
            yaxis={
                "categoryorder": "total ascending"
            }
        )

        st.plotly_chart(
            fig_prioritas,
            use_container_width=True
        )

    st.markdown(
        """
        Bagian prioritas wilayah membantu mengidentifikasi kabupaten yang perlu mendapat
        perhatian lebih lanjut dalam analisis SWOT. Kabupaten dengan kemiskinan tinggi
        perlu dibaca bersama perubahan kemiskinan, TPT, dukungan TKD, KUR, serta
        indikator kontekstual seperti kedalaman kemiskinan, layanan dasar, konektivitas,
        dan struktur ekonomi.
        """
    )

    st.divider()

    # =========================
    # PETA WILAYAH
    # =========================

    st.header("Peta Sebaran Indikator Wilayah")

    df_map = df_map_base.copy()
    tahun_peta = int(df_map["Tahun"].max())

    col1, col2 = st.columns(2)

    with col1:
        indikator_warna = st.selectbox(
            "Pilih indikator warna titik",
            [
                "Kemiskinan",
                "TPT",
                "Total_TKD",
                "KUR"
            ],
            key="flores_map_color"
        )

    with col2:
        indikator_ukuran = st.selectbox(
            "Pilih indikator ukuran titik",
            [
                "Total_TKD",
                "KUR",
                "Kemiskinan",
                "TPT"
            ],
            key="flores_map_size"
        )

    label_warna = get_map_indicator_label(indikator_warna)
    label_ukuran = get_map_indicator_label(indikator_ukuran)
    warna_peta = get_map_color_scale(indikator_warna)

    rata_latitude = df_map["Latitude"].mean()
    rata_longitude = df_map["Longitude"].mean()

    fig_map = px.scatter_mapbox(
        df_map,
        lat="Latitude",
        lon="Longitude",
        color=indikator_warna,
        size=indikator_ukuran,
        hover_name="Kabupaten",
        hover_data={
            "Latitude": False,
            "Longitude": False,
            "Kemiskinan": ":,.2f",
            "TPT": ":,.2f",
            "Total_TKD": ":,.2f",
            "KUR": ":,.2f",
            "Pulau_Kawasan": True
        },
        color_continuous_scale=warna_peta,
        size_max=35,
        zoom=7,
        center={
            "lat": rata_latitude,
            "lon": rata_longitude
        },
        mapbox_style="carto-positron",
        title=f"Peta {label_warna} Tahun {tahun_peta}"
    )

    fig_map.update_layout(
        margin={
            "r": 0,
            "t": 50,
            "l": 0,
            "b": 0
        },
        height=520
    )

    show_chart(fig_map)

    if indikator_warna in ["Kemiskinan", "TPT"]:
        st.caption(
            "Untuk indikator ini, warna hijau menunjukkan nilai lebih rendah, sedangkan merah menunjukkan nilai lebih tinggi."
        )

    else:
        st.caption(
            "Untuk indikator fiskal/pembiayaan, warna hijau menunjukkan nilai lebih besar, sedangkan merah menunjukkan nilai lebih rendah."
        )

    st.divider()

    # =========================
    # ANALISIS KONTEKSTUAL
    # =========================

    st.header("Indikator Kontekstual Flores")

    st.markdown(
        """
        Bagian ini digunakan untuk membaca indikator tambahan seperti kedalaman
        kemiskinan, keparahan kemiskinan, layanan dasar, konektivitas, pariwisata,
        dan indikator kontekstual lainnya.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        daftar_pilar = sorted(df_context["Pilar"].dropna().unique())

        pilar_terpilih = st.selectbox(
            "Pilih Pilar",
            daftar_pilar,
            key="flores_context_pilar"
        )

    df_pilar = df_context[
        df_context["Pilar"] == pilar_terpilih
    ].copy()

    with col2:
        daftar_indikator = sorted(df_pilar["Indikator"].dropna().unique())

        indikator_terpilih = st.selectbox(
            "Pilih Indikator",
            daftar_indikator,
            key="flores_context_indikator"
        )

    (
        summary,
        df_indicator_available,
        df_indicator_latest
    ) = make_context_indicator_summary(
        df_context,
        indikator_terpilih
    )

    df_yearly_context = make_context_yearly_trend(
        df_indicator_available
    )

    satuan = summary["Satuan"]
    tahun_indikator = summary["Tahun Akhir"]

    show_metric_row(
        [
            {
                "label": f"Rata-rata {tahun_indikator}",
                "value": f"{summary['Rata-rata Terbaru']:,.2f}"
            },
            {
                "label": "Satuan",
                "value": satuan
            },
            {
                "label": "Kabupaten Tertinggi",
                "value": summary["Kabupaten Tertinggi"]
            },
            {
                "label": "Kabupaten Terendah",
                "value": summary["Kabupaten Terendah"]
            }
        ]
    )

    st.divider()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        fig_context_trend = make_line_chart(
            df=df_yearly_context,
            x_column="Tahun",
            y_column="Rata_Rata_Nilai",
            title=f"Tren Rata-rata {indikator_terpilih}",
            xaxis_title="Tahun",
            yaxis_title=f"{indikator_terpilih} ({satuan})"
        )

        show_chart(fig_context_trend)

    with col2:
        df_ranking_context = df_indicator_latest.sort_values(
            "Nilai",
            ascending=False
        ).reset_index(drop=True)

        fig_context_ranking = make_horizontal_bar_chart(
            df=df_ranking_context,
            x_column="Nilai",
            y_column="Kabupaten",
            title=f"Ranking {indikator_terpilih} Tahun {tahun_indikator}",
            xaxis_title=f"{indikator_terpilih} ({satuan})",
            yaxis_title="Kabupaten",
            hover_data=[
                "Tahun",
                "Satuan",
                "Pilar"
            ],
            height=420
        )

        show_chart(fig_context_ranking)

    st.markdown(
        """
        Indikator kontekstual digunakan untuk memperkaya interpretasi wilayah.
        Bagian ini belum menyimpulkan kausalitas, tetapi membantu menjelaskan
        kondisi pendukung yang dapat memengaruhi efektivitas kebijakan fiskal,
        pembiayaan usaha, dan pengentasan kemiskinan.
        """
    )

    with st.expander("Tabel Data Indikator Kontekstual Terpilih"):
        st.dataframe(
            df_indicator_available,
            use_container_width=True
        )

# =========================
# HALAMAN KABUPATEN INTELLIGENCE
# =========================

def halaman_kabupaten_intelligence():

    # =========================
    # HEADER
    # =========================

    st.title("Kabupaten Intelligence")

    st.markdown(
        """
        Halaman ini menyajikan gambaran besar masing-masing kabupaten melalui
        indikator kemiskinan, TPT, TKD, KUR, indikator kontekstual, dan struktur
        ekonomi PDRB sektoral. Halaman ini digunakan untuk membaca karakteristik
        wilayah sebelum masuk ke analisis SWOT dan rekomendasi kebijakan.
        """
    )

    st.divider()

    # =========================
    # PILIH KABUPATEN
    # =========================

    daftar_kabupaten = sorted(df_panel["Kabupaten"].dropna().unique())

    kabupaten_terpilih = st.selectbox(
        "Pilih Kabupaten",
        daftar_kabupaten,
        key="kabupaten_intelligence_selectbox"
    )

    (
        df_panel_kabupaten,
        df_context_kabupaten,
        df_pdrb_kabupaten,
        df_pdrb_dominan_kabupaten
    ) = make_kabupaten_profile_data(kabupaten_terpilih)

    (
        summary_pdrb,
        df_pdrb_all_kabupaten,
        df_pdrb_latest_kabupaten
    ) = make_pdrb_kabupaten_summary(
        df_pdrb,
        kabupaten_terpilih
    )

    tahun_awal = int(df_panel_kabupaten["Tahun"].min())
    tahun_akhir = int(df_panel_kabupaten["Tahun"].max())

    df_awal = df_panel_kabupaten[
        df_panel_kabupaten["Tahun"] == tahun_awal
    ].copy()

    df_akhir = df_panel_kabupaten[
        df_panel_kabupaten["Tahun"] == tahun_akhir
    ].copy()

    kemiskinan_awal = df_awal.iloc[0]["Kemiskinan"]
    kemiskinan_akhir = df_akhir.iloc[0]["Kemiskinan"]
    perubahan_kemiskinan = kemiskinan_akhir - kemiskinan_awal

    tpt_akhir = df_akhir.iloc[0]["TPT"]
    tkd_akhir = df_akhir.iloc[0]["Total_TKD"]
    kur_akhir = df_akhir.iloc[0]["KUR"]

    tahun_pdrb = summary_pdrb["Tahun Terbaru"]
    total_pdrb = summary_pdrb["Total PDRB"]
    sektor_dominan = summary_pdrb["Sektor Dominan"]
    kontribusi_dominan = summary_pdrb["Kontribusi Sektor Dominan"]
    jumlah_sektor = summary_pdrb["Jumlah Sektor"]

    # =========================
    # METRIC UTAMA KABUPATEN
    # =========================

    st.header(f"Ringkasan Utama {kabupaten_terpilih}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric(
                f"Kemiskinan {tahun_akhir}",
                f"{kemiskinan_akhir:,.2f}",
                delta=f"{perubahan_kemiskinan:,.2f}",
                height=105
            )

    with col2:
        with st.container(border=True):
            st.metric(
                f"TPT {tahun_akhir}",
                f"{tpt_akhir:,.2f}",
                height=105
            )

    with col3:
        with st.container(border=True):
            st.metric(
                f"Total TKD {tahun_akhir}",
                f"{tkd_akhir:,.2f} Miliar",
                height=105
            )

    with col4:
        with st.container(border=True):
            st.metric(
                f"KUR {tahun_akhir}",
                f"{kur_akhir:,.2f} Miliar",
                height=105
            )

    st.caption(
        "Delta pada metric kemiskinan menunjukkan perubahan dibandingkan tahun awal periode data."
    )

    st.divider()

    # =========================
    # NARASI PROFIL KABUPATEN
    # =========================

    st.header("Big Capture Kabupaten")

    if perubahan_kemiskinan < 0:
        arah_kemiskinan = "mengalami penurunan"

    elif perubahan_kemiskinan > 0:
        arah_kemiskinan = "mengalami peningkatan"

    else:
        arah_kemiskinan = "relatif tidak berubah"

    with st.container(border=True):
        st.markdown(
            f"""
            Selama periode **{tahun_awal}–{tahun_akhir}**, tingkat kemiskinan di
            **{kabupaten_terpilih}** {arah_kemiskinan} sebesar
            **{perubahan_kemiskinan:,.2f} poin**.

            Pada tahun **{tahun_akhir}**, kemiskinan tercatat sebesar
            **{kemiskinan_akhir:,.2f}**, dengan TPT sebesar **{tpt_akhir:,.2f}**.
            Dari sisi dukungan fiskal, total TKD mencapai **{tkd_akhir:,.2f} miliar rupiah**,
            sedangkan penyaluran KUR mencapai **{kur_akhir:,.2f} miliar rupiah**.

            Pada struktur ekonomi tahun **{tahun_pdrb}**, sektor dominan adalah
            **{sektor_dominan}** dengan kontribusi sebesar
            **{kontribusi_dominan:,.2f}%** terhadap total PDRB kabupaten.
            """
        )

    st.divider()

    # =========================
    # TREN KEMISKINAN DAN TKD
    # =========================

    st.header("Tren Kemiskinan dan Dukungan Fiskal")

    fig_kabupaten_tkd = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True
                }
            ]
        ]
    )

    fig_kabupaten_tkd.add_trace(
        go.Scatter(
            x=df_panel_kabupaten["Tahun"],
            y=df_panel_kabupaten["Kemiskinan"],
            mode="lines+markers",
            name="Kemiskinan (%)",
            line={
                "color": "red",
                "width": 3
            },
            marker={
                "size": 7
            }
        ),
        secondary_y=False
    )

    fig_kabupaten_tkd.add_trace(
        go.Scatter(
            x=df_panel_kabupaten["Tahun"],
            y=df_panel_kabupaten["Total_TKD"],
            mode="lines+markers",
            name="Total TKD (Miliar Rp)",
            line={
                "color": "royalblue",
                "width": 3,
                "dash": "dash"
            },
            marker={
                "size": 7
            }
        ),
        secondary_y=True
    )

    fig_kabupaten_tkd.update_layout(
        title=f"Tren Kemiskinan vs TKD - {kabupaten_terpilih}",
        xaxis_title="Tahun",
        template="plotly_white",
        hovermode="x unified",
        height=460,
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 20
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5
        }
    )

    fig_kabupaten_tkd.update_yaxes(
        title_text="Kemiskinan (%)",
        secondary_y=False
    )

    fig_kabupaten_tkd.update_yaxes(
        title_text="Total TKD (Miliar Rp)",
        secondary_y=True
    )

    st.plotly_chart(
        fig_kabupaten_tkd,
        use_container_width=True
    )

    st.markdown(
        """
        Grafik ini membantu membaca apakah perubahan TKD bergerak searah atau
        berlawanan dengan perubahan kemiskinan pada kabupaten terpilih. Grafik ini
        masih bersifat deskriptif dan belum menunjukkan hubungan sebab-akibat.
        """
    )

    st.divider()

    # =========================
    # KREDIT USAHA RAKYAT
    # =========================

    st.header("Kredit Usaha Rakyat")

    kur_awal = df_awal.iloc[0]["KUR"]
    perubahan_kur = kur_akhir - kur_awal

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.metric(
                f"KUR {tahun_akhir}",
                f"{kur_akhir:,.2f} Miliar",
                delta=f"{perubahan_kur:,.2f} Miliar",
                height=105
            )

    with col2:
        with st.container(border=True):
            st.metric(
                f"KUR {tahun_awal}",
                f"{kur_awal:,.2f} Miliar",
                height=105
            )

    st.caption(
        "Delta KUR menunjukkan perubahan penyaluran KUR dibandingkan tahun awal periode data."
    )

    fig_kur_kabupaten = px.line(
        df_panel_kabupaten,
        x="Tahun",
        y="KUR",
        markers=True,
        title=f"Tren KUR - {kabupaten_terpilih}"
    )

    fig_kur_kabupaten.update_traces(
        line={
            "width": 3
        },
        marker={
            "size": 7
        }
    )

    fig_kur_kabupaten.update_layout(
        xaxis_title="Tahun",
        yaxis_title="KUR (Miliar Rp)",
        template="plotly_white",
        hovermode="x unified",
        height=420,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20
        }
    )

    st.plotly_chart(
        fig_kur_kabupaten,
        use_container_width=True
    )

    st.markdown(
        """
        KUR digunakan untuk membaca dinamika akses pembiayaan usaha rakyat pada
        kabupaten terpilih. Bagian ini membantu melihat apakah pembiayaan produktif
        meningkat, menurun, atau cenderung stagnan selama periode pengamatan.
        """
    )

    st.divider()

    with st.expander("Tabel Data Panel Kabupaten"):
        st.dataframe(
            df_panel_kabupaten,
            use_container_width=True
        )
    
    st.divider()
    # =========================
    # RINGKASAN STRUKTUR PDRB
    # =========================

    st.header("Ringkasan Struktur PDRB")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.metric(
                f"Total PDRB {tahun_pdrb}",
                f"{total_pdrb:,.2f}",
                height=105
            )

    with col2:
        with st.container(border=True):
            st.metric(
                "Jumlah Sektor",
                jumlah_sektor,
                height=105
            )

    with col3:
        with st.container(border=True):
            st.metric(
                f"Kontribusi Sektor Dominan: {sektor_dominan}",
                f"{kontribusi_dominan:,.2f}%",
                height=105
            )

    st.caption(
        "Struktur PDRB menunjukkan sektor ekonomi yang paling dominan dalam pembentukan ekonomi kabupaten."
    )

    st.divider()

    col1, col2 = st.columns([0.9, 1.4])

    with col1:
        st.subheader("Tiga Sektor Terbesar")

        if df_pdrb_dominan_kabupaten.shape[0] == 0:
            st.warning(
                "Belum ada data sektor dominan untuk kabupaten ini."
            )

        else:
            for index, row in df_pdrb_dominan_kabupaten.iterrows():
                nomor = index + 1
                sektor = row["Lapangan_Usaha"]
                kontribusi = row["Persentase_PDRB"]
                nilai = row["Nilai"]

                with st.container(border=True):
                    st.markdown(f"**{nomor}. {sektor}**")
                    st.markdown(f"Kontribusi: **{kontribusi:,.2f}%**")
                    st.markdown(f"Nilai PDRB: **{nilai:,.2f}**")

        st.markdown(
            """
            Tiga sektor terbesar menunjukkan basis utama ekonomi lokal.
            Informasi ini penting untuk membaca apakah kebijakan fiskal,
            pembiayaan usaha, dan program pengurangan kemiskinan sudah selaras
            dengan struktur ekonomi kabupaten.
            """
        )

    with col2:
        st.subheader(f"Grafik Struktur PDRB Sektoral {kabupaten_terpilih} {tahun_akhir}")

        df_pdrb_chart = df_pdrb_latest_kabupaten.sort_values(
            "Persentase_PDRB",
            ascending=True
        ).reset_index(drop=True)

        fig_pdrb_lengkap = px.bar(
            df_pdrb_chart,
            x="Persentase_PDRB",
            y="Lapangan_Usaha",
            orientation="h",
            hover_data=[
                "Nilai",
                "Total_PDRB_Kabupaten_Tahun",
                "Persentase_PDRB"
            ]
        )

        fig_pdrb_lengkap.update_layout(
            xaxis_title="Kontribusi terhadap PDRB (%)",
            yaxis_title="Lapangan Usaha",
            template="plotly_white",
            height=650,
            margin={
                "l": 20,
                "r": 20,
                "t": 60,
                "b": 20
            }
        )

        st.plotly_chart(
            fig_pdrb_lengkap,
            use_container_width=True
        )

    st.markdown(
        f"""
        Pada tahun **{tahun_pdrb}**, struktur ekonomi **{kabupaten_terpilih}**
        didominasi oleh sektor **{sektor_dominan}** dengan kontribusi sebesar
        **{kontribusi_dominan:,.2f}%** terhadap total PDRB kabupaten.
        """
    )

    with st.expander("Tabel PDRB Sektoral Lengkap"):
        st.dataframe(
            df_pdrb_latest_kabupaten,
            use_container_width=True
        )

    st.divider()

    # =========================
    # INDIKATOR KONTEKSTUAL
    # =========================

    st.header("Indikator Kontekstual")

    st.markdown(
        """
        Bagian ini menampilkan tren indikator kontekstual pada kabupaten terpilih.
        Indikator ini digunakan untuk membaca faktor pendukung yang dapat memperkuat
        interpretasi kemiskinan, TKD, KUR, dan struktur ekonomi lokal.
        """
    )

    df_context_kabupaten_all = df_context[
        df_context["Kabupaten"] == kabupaten_terpilih
    ].copy()

    df_context_kabupaten_all = df_context_kabupaten_all[
        df_context_kabupaten_all["Nilai"].notna()
    ].copy()

    if df_context_kabupaten_all.shape[0] == 0:
        st.warning(
            "Belum ada data indikator kontekstual yang tersedia untuk kabupaten ini."
        )

    else:
        col1, col2 = st.columns(2)

        with col1:
            daftar_pilar_kabupaten = sorted(
                df_context_kabupaten_all["Pilar"].dropna().unique()
            )

            pilar_terpilih = st.selectbox(
                "Pilih Pilar Kontekstual",
                daftar_pilar_kabupaten,
                key="kabupaten_intelligence_pilar"
            )

        df_context_pilar = df_context_kabupaten_all[
            df_context_kabupaten_all["Pilar"] == pilar_terpilih
        ].copy()

        with col2:
            daftar_indikator_kabupaten = sorted(
                df_context_pilar["Indikator"].dropna().unique()
            )

            indikator_terpilih = st.selectbox(
                "Pilih Indikator",
                daftar_indikator_kabupaten,
                key="kabupaten_intelligence_indikator"
            )

        df_context_chart = df_context_pilar[
            df_context_pilar["Indikator"] == indikator_terpilih
        ].copy()

        df_context_chart = df_context_chart.sort_values(
            "Tahun"
        ).reset_index(drop=True)

        satuan_list = df_context_chart["Satuan"].dropna().unique()

        if len(satuan_list) > 0:
            satuan = satuan_list[0]
        else:
            satuan = "-"

        fig_context_kabupaten = px.line(
            df_context_chart,
            x="Tahun",
            y="Nilai",
            markers=True,
            title=f"Tren {indikator_terpilih} - {kabupaten_terpilih}"
        )

        fig_context_kabupaten.update_traces(
            line={
                "width": 3
            },
            marker={
                "size": 8
            }
        )

        fig_context_kabupaten.update_layout(
            xaxis_title="Tahun",
            yaxis_title=f"{indikator_terpilih} ({satuan})",
            template="plotly_white",
            hovermode="x unified",
            height=430,
            margin={
                "l": 20,
                "r": 20,
                "t": 60,
                "b": 20
            }
        )

        st.plotly_chart(
            fig_context_kabupaten,
            use_container_width=True
        )

        tahun_awal_context = int(df_context_chart["Tahun"].min())
        tahun_akhir_context = int(df_context_chart["Tahun"].max())

        nilai_awal_context = df_context_chart[
            df_context_chart["Tahun"] == tahun_awal_context
        ].iloc[0]["Nilai"]

        nilai_akhir_context = df_context_chart[
            df_context_chart["Tahun"] == tahun_akhir_context
        ].iloc[0]["Nilai"]

        perubahan_context = nilai_akhir_context - nilai_awal_context

        if perubahan_context > 0:
            arah_context = "meningkat"

        elif perubahan_context < 0:
            arah_context = "menurun"

        else:
            arah_context = "relatif tidak berubah"

        st.caption(
            f"Selama periode {tahun_awal_context}–{tahun_akhir_context}, "
            f"indikator {indikator_terpilih} di {kabupaten_terpilih} "
            f"{arah_context} sebesar {perubahan_context:,.2f} {satuan}."
        )

# =========================
# HALAMAN ESTIMASI MODEL PANEL
# =========================

def halaman_estimasi_panel():

    # =========================
    # HEADER
    # =========================

    st.title("Estimasi Model Panel")

    st.markdown(
        """
        Halaman ini digunakan untuk menjalankan estimasi awal model panel antara
        TKD, KUR, pengangguran terbuka, dan kemiskinan di Pulau Flores.
        Estimasi dilakukan pada seluruh kabupaten dan seluruh tahun sehingga
        halaman ini tidak difilter berdasarkan kabupaten tertentu.
        """
    )

    st.caption(
        "Tahap ini masih merupakan estimasi awal. Pemilihan model final melalui uji Chow, LM, dan Hausman dilakukan pada halaman/batch berikutnya."
    )

    st.divider()

    # =========================
    # PENGATURAN ESTIMASI
    # =========================

    st.header("Pengaturan Estimasi")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            pilihan_dataset = st.selectbox(
                "Pilih dataset model",
                [
                    "Model Normal",
                    "Model Lag 1"
                ]
            )

        with col2:
            pilihan_covariance = st.selectbox(
                "Pilih jenis standard error",
                [
                    "Robust",
                    "Unadjusted"
                ]
            )

        with col3:
            pilihan_model = st.selectbox(
                "Pilih model panel",
                [
                    "CEM / Pooled OLS",
                    "FEM / Fixed Effect",
                    "REM / Random Effect"
                ]
            )

    if pilihan_dataset == "Model Normal":
        df_model = df_model_normal.copy()
        keterangan_dataset = "Variabel independen menggunakan nilai tahun berjalan."

    else:
        df_model = df_model_lag.copy()
        keterangan_dataset = "Variabel independen menggunakan nilai tahun sebelumnya atau lag 1."

    if pilihan_covariance == "Robust":
        cov_type = "robust"
        keterangan_covariance = "Robust standard error digunakan agar inferensi lebih tahan terhadap indikasi heteroskedastisitas."

    else:
        cov_type = "unadjusted"
        keterangan_covariance = "Unadjusted standard error adalah standard error biasa tanpa koreksi robust."

    st.markdown(
        f"""
        **Dataset:** {pilihan_dataset}  
        **Keterangan dataset:** {keterangan_dataset}  
        **Standard error:** {pilihan_covariance}  
        **Keterangan standard error:** {keterangan_covariance}
        """
    )

    st.divider()

    # =========================
    # RINGKASAN DATASET MODEL
    # =========================

    st.header("Ringkasan Dataset Model")

    jumlah_observasi = df_model.shape[0]
    jumlah_kabupaten = df_model["Kabupaten"].nunique()
    jumlah_tahun = df_model["Tahun"].nunique()
    tahun_awal = int(df_model["Tahun"].min())
    tahun_akhir = int(df_model["Tahun"].max())

    jumlah_variabel_independen = df_model.drop(
        columns=[
            "Kabupaten",
            "Tahun",
            "Kemiskinan"
        ]
    ).shape[1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_metric_card(
            "Observasi",
            f"{jumlah_observasi:,}"
        )

    with col2:
        show_metric_card(
            "Kabupaten",
            jumlah_kabupaten
        )

    with col3:
        show_metric_card(
            "Periode",
            f"{tahun_awal}–{tahun_akhir}"
        )

    with col4:
        show_metric_card(
            "Variabel Independen",
            jumlah_variabel_independen
        )

    equation = make_model_equation_text(df_model)

    with st.container(border=True):
        st.markdown("**Persamaan umum model:**")
        st.code(equation)

    st.divider()

    # =========================
    # MENJALANKAN ESTIMASI MODEL
    # =========================

    try:
        hasil_model = fit_all_panel_models(
            df_model,
            cov_type=cov_type
        )

        df_ringkasan_model_panel = make_model_summary_table(
            hasil_model
        )

    except Exception as error:
        st.error("Estimasi model panel gagal dijalankan.")
        st.exception(error)
        st.stop()

    st.divider()

    # =========================
    # HASIL MODEL TERPILIH
    # =========================

    st.header(f"Hasil Model Terpilih: {pilihan_model}")

    result_terpilih = hasil_model[pilihan_model]

    if isinstance(result_terpilih, dict):

        st.error(
            f"{pilihan_model} gagal dijalankan pada spesifikasi model ini."
        )

        explanation = make_model_failure_explanation(
            pilihan_model,
            result_terpilih["error"],
            df_model
        )

        st.markdown(explanation)

        struktur_model = make_model_structure_report(df_model)

        df_struktur_model = pd.DataFrame(
            [
                struktur_model
            ]
        )

        with st.expander("Lihat Struktur Dataset Model"):
            st.dataframe(
                df_struktur_model,
                use_container_width=True
            )

        with st.expander("Detail Error Teknis"):
            st.code(
                result_terpilih["error"],
                language=None
            )

    else:

        # =========================
        # RINGKASAN STATISTIK MODEL
        # =========================

        st.subheader("Ringkasan Statistik Model")

        if result_terpilih.f_statistic is not None:
            nilai_f_statistic = result_terpilih.f_statistic.stat
            nilai_prob_f = result_terpilih.f_statistic.pval

            teks_f_statistic = f"{nilai_f_statistic:,.4f}"
            teks_prob_f = f"{nilai_prob_f:,.4f}"

        else:
            teks_f_statistic = "-"
            teks_prob_f = "-"

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            show_metric_card(
                "Observasi",
                f"{result_terpilih.nobs:,}"
            )

        with col2:
            show_metric_card(
                "R-squared",
                f"{result_terpilih.rsquared:,.4f}"
            )

        with col3:
            show_metric_card(
                "F-statistic",
                teks_f_statistic
            )

        with col4:
            show_metric_card(
                "Prob(F-statistic)",
                teks_prob_f
            )

        st.divider()

        # =========================
        # TABEL KOEFISIEN
        # =========================

        st.subheader("Tabel Koefisien")

        df_coef_terpilih = make_coefficient_table(
            result_terpilih
        )

        st.dataframe(
            df_coef_terpilih.style.format(
                {
                    "Koefisien": "{:,.4f}",
                    "Std. Error": "{:,.4f}",
                    "t-statistic": "{:,.4f}",
                    "P-value": "{:,.4f}"
                }
            ),
            use_container_width=True
        )

        st.markdown(
            """
            **Cara membaca tabel koefisien:**

            - Koefisien negatif menunjukkan bahwa kenaikan variabel tersebut berasosiasi dengan penurunan kemiskinan.
            - Koefisien positif menunjukkan bahwa kenaikan variabel tersebut berasosiasi dengan kenaikan kemiskinan.
            - P-value digunakan untuk membaca signifikansi statistik masing-masing variabel.
            - Tanda `*` (<0.10), `**` (<0.05), dan `***` (<0.01) menunjukkan tingkat signifikansi yang semakin kuat.
            """
        )

        with st.expander(f"Output Teknis Lengkap {pilihan_model}"):
            st.code(
                str(result_terpilih.summary),
                language=None
            )

    st.divider()
    
    # =========================
    # RINGKASAN PERBANDINGAN MODEL
    # =========================

    st.header("Ringkasan Perbandingan Model")

    df_ringkasan_tampil = df_ringkasan_model_panel.copy()

    kolom_format = [
        "R-squared",
        "F-statistic",
        "Prob(F-statistic)"
    ]

    for col in kolom_format:
        if col in df_ringkasan_tampil.columns:
            df_ringkasan_tampil[col] = df_ringkasan_tampil[col].apply(
                lambda x: "-" if pd.isna(x) else f"{x:,.4f}"
            )

    st.dataframe(
        df_ringkasan_tampil,
        use_container_width=True
    )

    jumlah_model_berhasil = (
        df_ringkasan_model_panel["Status"] == "Berhasil"
    ).sum()

    jumlah_model_gagal = (
        df_ringkasan_model_panel["Status"] == "Gagal"
    ).sum()

    if jumlah_model_gagal > 0:
        st.warning(
            f"Terdapat {jumlah_model_gagal} model yang gagal dijalankan. "
            "Model yang gagal tetap ditampilkan agar struktur estimasi dapat dibaca secara transparan."
        )

    else:
        st.success(
            f"Seluruh {jumlah_model_berhasil} model berhasil dijalankan."
        )

    st.caption(
        "Tabel ini membandingkan estimasi awal CEM, FEM, dan REM. Keputusan model terbaik belum diambil pada tahap ini."
    )

    st.divider()

    # =========================
    # CATATAN ANALISIS
    # =========================

    st.header("Catatan Analisis")

    with st.container(border=True):
        st.markdown(
            """
            Hasil pada halaman ini belum digunakan sebagai keputusan model final.
            Tahap berikutnya perlu melakukan:

            1. Uji Chow untuk memilih antara CEM dan FEM.
            2. Uji LM untuk memilih antara CEM dan REM.
            3. Uji Hausman untuk memilih antara FEM dan REM.
            4. Uji multikolinearitas untuk melihat hubungan kuat antarvariabel independen.
            5. Uji diagnostik model untuk membaca heteroskedastisitas, autokorelasi, dan dependensi antar-unit.
            """
        )

    with st.expander("Tabel Dataset yang Digunakan"):
        st.dataframe(
            df_model,
            use_container_width=True
        )

# =========================
# HALAMAN PEMILIHAN MODEL PANEL
# =========================

def halaman_pemilihan_model_panel():

    # =========================
    # HEADER
    # =========================

    st.title("Pemilihan Model Panel")

    st.markdown(
        """
        Halaman ini menjalankan uji pemilihan model panel untuk menentukan model
        yang lebih tepat antara CEM, FEM, dan REM. Uji yang digunakan meliputi
        uji Chow, uji LM Breusch-Pagan, dan uji Hausman.
        """
    )

    st.caption(
        "Pemilihan model dilakukan pada seluruh data panel, bukan per kabupaten."
    )

    st.divider()

    # =========================
    # PENGATURAN UJI
    # =========================

    st.header("Pengaturan Uji Pemilihan Model")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            pilihan_dataset = st.selectbox(
                "Pilih dataset model",
                [
                    "Model Normal",
                    "Model Lag 1"
                ],
                key="selection_dataset"
            )

        with col2:
            pilihan_covariance = st.selectbox(
                "Pilih jenis standard error",
                [
                    "Unadjusted",
                    "Robust"
                ],
                key="selection_covariance"
            )

        with col3:
            alpha = st.selectbox(
                "Pilih alpha",
                [
                    0.01,
                    0.05,
                    0.10
                ],
                index=1,
                key="selection_alpha"
            )

    if pilihan_dataset == "Model Normal":
        df_model = df_model_normal.copy()
        keterangan_dataset = "Model menggunakan variabel independen tahun berjalan."

    else:
        df_model = df_model_lag.copy()
        keterangan_dataset = "Model menggunakan variabel independen tahun sebelumnya atau lag 1."

    if pilihan_covariance == "Robust":
        cov_type = "robust"
        keterangan_covariance = "Robust standard error digunakan dalam estimasi model sebelum uji dilakukan."

    else:
        cov_type = "unadjusted"
        keterangan_covariance = "Unadjusted standard error digunakan sebagai basis awal uji pemilihan model."

    st.markdown(
        f"""
        **Dataset:** {pilihan_dataset}  
        **Keterangan dataset:** {keterangan_dataset}  
        **Standard error:** {pilihan_covariance}  
        **Keterangan standard error:** {keterangan_covariance}  
        **Alpha:** {alpha}
        """
    )

    st.divider()

    # =========================
    # RINGKASAN DATASET
    # =========================

    st.header("Ringkasan Dataset Uji")

    jumlah_observasi = df_model.shape[0]
    jumlah_kabupaten = df_model["Kabupaten"].nunique()
    jumlah_tahun = df_model["Tahun"].nunique()
    tahun_awal = int(df_model["Tahun"].min())
    tahun_akhir = int(df_model["Tahun"].max())

    jumlah_variabel_independen = df_model.drop(
        columns=[
            "Kabupaten",
            "Tahun",
            "Kemiskinan"
        ]
    ).shape[1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_metric_card(
            "Observasi",
            f"{jumlah_observasi:,}"
        )

    with col2:
        show_metric_card(
            "Kabupaten",
            jumlah_kabupaten
        )

    with col3:
        show_metric_card(
            "Periode",
            f"{tahun_awal}–{tahun_akhir}"
        )

    with col4:
        show_metric_card(
            "Variabel Independen",
            jumlah_variabel_independen
        )

    st.divider()

    # =========================
    # ESTIMASI MODEL UNTUK UJI
    # =========================

    try:
        hasil_model = fit_all_panel_models(
            df_model,
            cov_type=cov_type
        )

        df_ringkasan_model_panel = make_model_summary_table(
            hasil_model
        )

        df_hasil_uji = run_panel_model_selection_tests(
            hasil_model,
            alpha=alpha
        )

        conclusion = make_model_selection_conclusion(
            df_hasil_uji,
            hasil_model
        )

    except Exception as error:
        st.error("Uji pemilihan model gagal dijalankan.")
        st.exception(error)
        st.stop()

    # =========================
    # KESIMPULAN UTAMA
    # =========================

    st.header("Kesimpulan Pemilihan Model")

    with st.container(border=True):
        st.metric(
            "Model Akhir Sementara",
            conclusion["Model Akhir Sementara"],
            height=105
        )

        st.markdown(
            f"""
            **Alasan:** {conclusion["Alasan"]}
            """
        )

    st.caption(
        "Kesimpulan ini masih bersifat sementara dan perlu dibaca bersama uji diagnostik model."
    )

    st.divider()

    # =========================
    # TABEL HASIL UJI
    # =========================

    st.header("Hasil Uji Pemilihan Model")

    df_hasil_uji_tampil = df_hasil_uji.copy()

    kolom_format = [
        "Statistik",
        "P-value",
        "Alpha"
    ]

    for col in kolom_format:
        df_hasil_uji_tampil[col] = df_hasil_uji_tampil[col].apply(
            lambda x: "-" if pd.isna(x) else f"{x:,.4f}"
        )

    st.dataframe(
        df_hasil_uji_tampil,
        use_container_width=True
    )

    st.markdown(
        """
        **Cara membaca hasil uji:**

        - **Uji Chow** digunakan untuk memilih antara CEM dan FEM.
        - **Uji LM Breusch-Pagan** digunakan untuk memilih antara CEM dan REM.
        - **Uji Hausman** digunakan untuk memilih antara FEM dan REM.
        - Jika p-value lebih kecil dari alpha, maka H0 ditolak.
        """
    )

    st.divider()

    # =========================
    # RINGKASAN ESTIMASI MODEL
    # =========================

    st.header("Ringkasan Estimasi Model")

    df_ringkasan_tampil = df_ringkasan_model_panel.copy()

    kolom_format_model = [
        "R-squared",
        "F-statistic",
        "Prob(F-statistic)"
    ]

    for col in kolom_format_model:
        if col in df_ringkasan_tampil.columns:
            df_ringkasan_tampil[col] = df_ringkasan_tampil[col].apply(
                lambda x: "-" if pd.isna(x) else f"{x:,.4f}"
            )

    st.dataframe(
        df_ringkasan_tampil,
        use_container_width=True
    )

    with st.expander("Tabel Dataset yang Digunakan"):
        st.dataframe(
            df_model,
            use_container_width=True
        )

# =========================
# HALAMAN DIAGNOSTIK MODEL
# =========================

def halaman_diagnostik_model():

    # =========================
    # HEADER
    # =========================

    st.title("Diagnostik Model")

    st.markdown(
        """
        Halaman ini digunakan untuk mengevaluasi kelayakan model panel sebelum
        hasil estimasi digunakan untuk interpretasi kebijakan. Diagnostik mencakup
        multikolinearitas antarvariabel independen dan diagnostik residual model.
        """
    )

    st.caption(
        "Multikolinearitas dibaca dari variabel independen, sedangkan diagnostik residual bergantung pada model panel yang dipilih."
    )

    st.divider()

    # =========================
    # PENGATURAN DIAGNOSTIK
    # =========================

    st.header("Pengaturan Diagnostik")

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            pilihan_dataset = st.selectbox(
                "Pilih dataset model",
                [
                    "Model Normal",
                    "Model Lag 1"
                ],
                key="diagnostic_dataset"
            )

        with col2:
            pilihan_model = st.selectbox(
                "Pilih model panel",
                [
                    "CEM / Pooled OLS",
                    "FEM / Fixed Effect",
                    "REM / Random Effect"
                ],
                key="diagnostic_model"
            )

        with col3:
            pilihan_covariance = st.selectbox(
                "Pilih standard error",
                [
                    "Robust",
                    "Unadjusted"
                ],
                key="diagnostic_covariance"
            )

        with col4:
            alpha = st.selectbox(
                "Pilih alpha",
                [
                    0.01,
                    0.05,
                    0.10
                ],
                index=1,
                key="diagnostic_alpha"
            )

    if pilihan_dataset == "Model Normal":
        df_model = df_model_normal.copy()
        keterangan_dataset = "Variabel independen menggunakan nilai tahun berjalan."

    else:
        df_model = df_model_lag.copy()
        keterangan_dataset = "Variabel independen menggunakan nilai tahun sebelumnya atau lag 1."

    if pilihan_covariance == "Robust":
        cov_type = "robust"
        keterangan_covariance = "Robust standard error digunakan dalam estimasi model diagnostik."
    else:
        cov_type = "unadjusted"
        keterangan_covariance = "Unadjusted standard error digunakan dalam estimasi model diagnostik."

    st.markdown(
        f"""
        **Dataset:** {pilihan_dataset}  
        **Keterangan dataset:** {keterangan_dataset}  
        **Model panel:** {pilihan_model}  
        **Standard error:** {pilihan_covariance}  
        **Keterangan standard error:** {keterangan_covariance}  
        **Alpha:** {alpha}
        """
    )

    st.divider()

    # =========================
    # RINGKASAN DATASET
    # =========================

    st.header("Ringkasan Dataset Diagnostik")

    jumlah_observasi = df_model.shape[0]
    jumlah_kabupaten = df_model["Kabupaten"].nunique()
    jumlah_tahun = df_model["Tahun"].nunique()

    tahun_awal = int(df_model["Tahun"].min())
    tahun_akhir = int(df_model["Tahun"].max())

    jumlah_variabel_independen = df_model.drop(
        columns=[
            "Kabupaten",
            "Tahun",
            "Kemiskinan"
        ]
    ).shape[1]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_metric_card(
            "Observasi",
            f"{jumlah_observasi:,}"
        )

    with col2:
        show_metric_card(
            "Kabupaten",
            jumlah_kabupaten
        )

    with col3:
        show_metric_card(
            "Periode",
            f"{tahun_awal}–{tahun_akhir}"
        )

    with col4:
        show_metric_card(
            "Variabel Independen",
            jumlah_variabel_independen
        )

    st.divider()

    # =========================
    # HITUNG DIAGNOSTIK AWAL
    # =========================

    try:
        df_vif = calculate_vif(df_model)

        df_corr = make_correlation_matrix(df_model)

        df_high_corr = make_high_correlation_pairs(
            df_model,
            threshold=0.80
        )

        conclusion_multicollinearity = make_multicollinearity_conclusion(
            df_vif,
            df_high_corr
        )

    except Exception as error:
        st.error("Diagnostik multikolinearitas gagal dijalankan.")
        st.exception(error)
        st.stop()

    try:
        hasil_model_diagnostik = fit_all_panel_models(
            df_model,
            cov_type=cov_type
        )

        result_diagnostik = hasil_model_diagnostik[pilihan_model]

    except Exception as error:
        st.error("Estimasi model untuk diagnostik residual gagal dijalankan.")
        st.exception(error)
        st.stop()

    # =========================
    # TAB DIAGNOSTIK
    # =========================

    tab1, tab2, tab3 = st.tabs(
        [
            "Multikolinearitas",
            "Diagnostik Residual",
            "Kesimpulan Diagnostik"
        ]
    )

    # =========================
    # TAB 1: MULTIKOLINEARITAS
    # =========================

    with tab1:

        st.header("Uji Multikolinearitas")

        st.markdown(
            """
            Multikolinearitas terjadi ketika variabel independen saling berkorelasi
            sangat kuat. Kondisi ini dapat membuat koefisien regresi menjadi tidak stabil
            dan sulit diinterpretasikan secara individual.
            """
        )

        st.subheader("Kesimpulan Multikolinearitas")

        with st.container(border=True):
            st.metric(
                "Status",
                conclusion_multicollinearity["Status"],
                height=105
            )

            st.markdown(
                f"""
                **Kesimpulan:** {conclusion_multicollinearity["Kesimpulan"]}

                **Rekomendasi:** {conclusion_multicollinearity["Rekomendasi"]}
                """
            )

        st.divider()

        max_vif = conclusion_multicollinearity.get(
            "Max VIF",
            None
        )

        jumlah_vif_tinggi = conclusion_multicollinearity.get(
            "Jumlah VIF Tinggi",
            0
        )

        jumlah_korelasi_tinggi = conclusion_multicollinearity.get(
            "Jumlah Korelasi Tinggi",
            0
        )

        if max_vif is None:
            teks_max_vif = "-"
        else:
            teks_max_vif = f"{max_vif:,.2f}"

        col1, col2, col3 = st.columns(3)

        with col1:
            show_metric_card(
                "Max VIF",
                teks_max_vif
            )

        with col2:
            show_metric_card(
                "Variabel VIF Tinggi",
                jumlah_vif_tinggi
            )

        with col3:
            show_metric_card(
                "Pasangan Korelasi Tinggi",
                jumlah_korelasi_tinggi
            )

        st.divider()

        st.subheader("Grafik VIF")

        df_vif_chart = df_vif.copy()

        df_vif_chart["VIF_Chart"] = pd.to_numeric(
            df_vif_chart["VIF"],
            errors="coerce"
        )

        df_vif_chart = df_vif_chart[
            df_vif_chart["VIF_Chart"].notna()
        ].copy()

        df_vif_chart = df_vif_chart[
            df_vif_chart["VIF_Chart"] != float("inf")
        ].copy()

        if df_vif_chart.shape[0] == 0:
            st.warning(
                "Grafik VIF tidak dapat ditampilkan karena nilai VIF tidak valid atau tidak terbatas."
            )

        else:
            df_vif_chart = df_vif_chart.sort_values(
                "VIF_Chart",
                ascending=True
            ).reset_index(drop=True)

            fig_vif = px.bar(
                df_vif_chart,
                x="VIF_Chart",
                y="Variabel",
                orientation="h",
                color="Status",
                title="Nilai VIF Variabel Independen",
                hover_data=[
                    "VIF",
                    "Status"
                ]
            )

            fig_vif.update_layout(
                xaxis_title="VIF",
                yaxis_title="Variabel",
                template="plotly_white",
                height=430,
                margin={
                    "l": 20,
                    "r": 20,
                    "t": 60,
                    "b": 20
                }
            )

            st.plotly_chart(
                fig_vif,
                use_container_width=True
            )

        st.divider()

        st.subheader("Tabel VIF")

        st.dataframe(
            df_vif.style.format(
                {
                    "VIF": "{:,.4f}"
                }
            ),
            use_container_width=True
        )

        st.markdown(
            """
            **Panduan membaca VIF:**

            - VIF di bawah 5: relatif aman.
            - VIF antara 5 sampai di bawah 10: perlu perhatian.
            - VIF 10 atau lebih: indikasi multikolinearitas tinggi.
            """
        )

        st.divider()

        st.subheader("Korelasi Antarvariabel Independen")

        with st.expander("Lihat Matriks Korelasi"):
            st.dataframe(
                df_corr.style.format("{:,.4f}"),
                use_container_width=True
            )

        if df_high_corr.shape[0] == 0:
            st.success(
                "Tidak terdapat pasangan variabel dengan korelasi absolut minimal 0,80."
            )

        else:
            st.warning(
                "Terdapat pasangan variabel independen dengan korelasi absolut minimal 0,80."
            )

            st.dataframe(
                df_high_corr.style.format(
                    {
                        "Korelasi": "{:,.4f}",
                        "Abs Korelasi": "{:,.4f}"
                    }
                ),
                use_container_width=True
            )

    # =========================
    # TAB 2: DIAGNOSTIK RESIDUAL
    # =========================

    with tab2:

        st.header("Diagnostik Residual")

        st.markdown(
            """
            Diagnostik residual digunakan untuk membaca apakah error model memiliki
            masalah seperti heteroskedastisitas, autokorelasi, dependensi antar-kabupaten,
            atau penyimpangan dari normalitas.
            """
        )

        if isinstance(result_diagnostik, dict):

            st.error(
                f"{pilihan_model} gagal dijalankan sehingga diagnostik residual tidak dapat dilakukan."
            )

            explanation = make_model_failure_explanation(
                pilihan_model,
                result_diagnostik["error"],
                df_model
            )

            st.markdown(explanation)

            with st.expander("Detail Error Teknis"):
                st.code(
                    result_diagnostik["error"],
                    language=None
                )

        else:

            try:
                df_residual_tests = run_residual_diagnostic_tests(
                    result_diagnostik,
                    df_model,
                    alpha=alpha
                )

                conclusion_residual = make_residual_diagnostic_conclusion(
                    df_residual_tests
                )

                df_residual_plot = get_residual_diagnostic_data(
                    result_diagnostik,
                    df_model
                )

            except Exception as error:
                st.error("Diagnostik residual gagal dijalankan.")
                st.exception(error)
                st.stop()

            st.subheader("Kesimpulan Diagnostik Residual")

            with st.container(border=True):
                st.metric(
                    "Status",
                    conclusion_residual["Status"],
                    height=105
                )

                st.markdown(
                    f"""
                    **Kesimpulan:** {conclusion_residual["Kesimpulan"]}

                    **Rekomendasi:** {conclusion_residual["Rekomendasi"]}
                    """
                )

            st.divider()

            jumlah_uji_berhasil = conclusion_residual.get(
                "Jumlah Uji Berhasil",
                0
            )

            jumlah_masalah = conclusion_residual.get(
                "Jumlah Masalah",
                0
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                show_metric_card(
                    "Model Diagnostik",
                    pilihan_model
                )

            with col2:
                show_metric_card(
                    "Uji Berhasil",
                    jumlah_uji_berhasil
                )

            with col3:
                show_metric_card(
                    "Masalah Terdeteksi",
                    jumlah_masalah
                )

            st.divider()

            st.subheader("Tabel Hasil Diagnostik Residual")

            df_residual_tests_tampil = df_residual_tests.copy()

            kolom_format = [
                "Statistik",
                "P-value",
                "Alpha"
            ]

            for col in kolom_format:
                if col in df_residual_tests_tampil.columns:
                    df_residual_tests_tampil[col] = df_residual_tests_tampil[col].apply(
                        lambda x: "-" if pd.isna(x) else f"{x:,.4f}"
                    )

            st.dataframe(
                df_residual_tests_tampil,
                use_container_width=True
            )

            st.markdown(
                """
                **Cara membaca hasil uji:**

                - Jika p-value lebih kecil dari alpha, maka H0 ditolak.
                - Breusch-Pagan dan White membaca indikasi heteroskedastisitas.
                - Residual AR(1) membaca indikasi autokorelasi residual.
                - Pesaran CD membaca indikasi dependensi residual antar-kabupaten.
                - Jarque-Bera membaca normalitas residual.
                """
            )

            st.divider()

            st.subheader("Visual Residual")

            col1, col2 = st.columns(2)

            with col1:
                fig_residual_time = px.line(
                    df_residual_plot,
                    x="Tahun",
                    y="Residual",
                    color="Kabupaten",
                    markers=True,
                    title=f"Residual per Tahun - {pilihan_model}"
                )

                fig_residual_time.update_layout(
                    xaxis_title="Tahun",
                    yaxis_title="Residual",
                    template="plotly_white",
                    hovermode="x unified",
                    height=420,
                    margin={
                        "l": 20,
                        "r": 20,
                        "t": 60,
                        "b": 20
                    }
                )

                st.plotly_chart(
                    fig_residual_time,
                    use_container_width=True
                )

            with col2:
                fig_residual_hist = px.histogram(
                    df_residual_plot,
                    x="Residual",
                    nbins=12,
                    title=f"Distribusi Residual - {pilihan_model}"
                )

                fig_residual_hist.update_layout(
                    xaxis_title="Residual",
                    yaxis_title="Frekuensi",
                    template="plotly_white",
                    height=420,
                    margin={
                        "l": 20,
                        "r": 20,
                        "t": 60,
                        "b": 20
                    }
                )

                st.plotly_chart(
                    fig_residual_hist,
                    use_container_width=True
                )

    # =========================
    # TAB 3: KESIMPULAN DIAGNOSTIK
    # =========================

    with tab3:

        st.header("Kesimpulan Diagnostik Model")

        if isinstance(result_diagnostik, dict):

            status_residual = "Tidak Dapat Diuji"
            kesimpulan_residual = (
                f"{pilihan_model} gagal dijalankan sehingga diagnostik residual tidak tersedia."
            )

            rekomendasi_residual = (
                "Gunakan model lain yang berhasil diestimasi atau sederhanakan spesifikasi model."
            )

        else:
            df_residual_tests = run_residual_diagnostic_tests(
                result_diagnostik,
                df_model,
                alpha=alpha
            )

            conclusion_residual = make_residual_diagnostic_conclusion(
                df_residual_tests
            )

            status_residual = conclusion_residual["Status"]
            kesimpulan_residual = conclusion_residual["Kesimpulan"]
            rekomendasi_residual = conclusion_residual["Rekomendasi"]

        df_kesimpulan = pd.DataFrame(
            [
                {
                    "Aspek": "Multikolinearitas",
                    "Status": conclusion_multicollinearity["Status"],
                    "Kesimpulan": conclusion_multicollinearity["Kesimpulan"],
                    "Rekomendasi": conclusion_multicollinearity["Rekomendasi"]
                },
                {
                    "Aspek": "Diagnostik Residual",
                    "Status": status_residual,
                    "Kesimpulan": kesimpulan_residual,
                    "Rekomendasi": rekomendasi_residual
                }
            ]
        )

        st.dataframe(
            df_kesimpulan,
            use_container_width=True
        )

        st.divider()

        st.subheader("Narasi Akhir Diagnostik")

        with st.container(border=True):
            st.markdown(
                f"""
                Berdasarkan diagnostik yang dijalankan pada **{pilihan_dataset}**
                dengan model **{pilihan_model}**, hasil diagnostik menunjukkan bahwa
                aspek multikolinearitas berada pada status
                **{conclusion_multicollinearity["Status"]}**.

                Untuk aspek residual, status diagnostik adalah **{status_residual}**.

                Dengan demikian, interpretasi hasil estimasi perlu memperhatikan
                rekomendasi diagnostik, terutama terkait penggunaan standard error
                robust dan kehati-hatian dalam membaca koefisien secara individual.
                """
            )

    with st.expander("Tabel Dataset yang Digunakan"):
        st.dataframe(
            df_model,
            use_container_width=True
        )

# =========================
# HALAMAN MODEL FINAL DAN INTERPRETASI
# =========================

def halaman_model_final():

    # =========================
    # HEADER
    # =========================

    st.title("Model Final & Interpretasi")

    st.markdown(
        """
        Halaman ini merangkum hasil akhir analisis panel. Model final sementara
        ditentukan berdasarkan uji pemilihan model, kemudian koefisiennya
        diinterpretasikan untuk mendukung pembahasan kebijakan.
        """
    )

    st.caption(
        "Model final pada halaman ini tetap perlu dibaca bersama hasil diagnostik model."
    )

    st.divider()

    # =========================
    # PENGATURAN MODEL FINAL
    # =========================

    st.header("Pengaturan Model Final")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            pilihan_dataset = st.selectbox(
                "Pilih dataset model",
                [
                    "Model Normal",
                    "Model Lag 1"
                ],
                key="final_dataset"
            )

        with col2:
            pilihan_covariance = st.selectbox(
                "Pilih standard error",
                [
                    "Robust",
                    "Unadjusted"
                ],
                key="final_covariance"
            )

        with col3:
            alpha = st.selectbox(
                "Pilih alpha",
                [
                    0.01,
                    0.05,
                    0.10
                ],
                index=1,
                key="final_alpha"
            )

    if pilihan_dataset == "Model Normal":
        df_model = df_model_normal.copy()
        keterangan_dataset = "Variabel independen menggunakan nilai tahun berjalan."

    else:
        df_model = df_model_lag.copy()
        keterangan_dataset = "Variabel independen menggunakan nilai tahun sebelumnya atau lag 1."

    if pilihan_covariance == "Robust":
        cov_type = "robust"
        keterangan_covariance = "Robust standard error digunakan untuk interpretasi utama."

    else:
        cov_type = "unadjusted"
        keterangan_covariance = "Unadjusted standard error digunakan untuk interpretasi utama."

    st.markdown(
        f"""
        **Dataset:** {pilihan_dataset}  
        **Keterangan dataset:** {keterangan_dataset}  
        **Standard error:** {pilihan_covariance}  
        **Keterangan standard error:** {keterangan_covariance}  
        **Alpha:** {alpha}
        """
    )

    st.divider()

    # =========================
    # ESTIMASI DAN PEMILIHAN MODEL
    # =========================

    try:
        hasil_model = fit_all_panel_models(
            df_model,
            cov_type=cov_type
        )

        df_hasil_uji = run_panel_model_selection_tests(
            hasil_model,
            alpha=alpha
        )

        conclusion_selection = make_model_selection_conclusion(
            df_hasil_uji,
            hasil_model
        )

    except Exception as error:
        st.error("Model final gagal dihitung.")
        st.exception(error)
        st.stop()

    model_akhir = conclusion_selection["Model Akhir Sementara"]

    daftar_model_berhasil = []

    for nama_model, result in hasil_model.items():
        if not isinstance(result, dict):
            daftar_model_berhasil.append(nama_model)

    if model_akhir == "Belum dapat ditentukan":
        st.warning(
            "Model akhir belum dapat ditentukan secara otomatis. Pilih model yang berhasil diestimasi untuk ditampilkan."
        )

        if len(daftar_model_berhasil) == 0:
            st.error(
                "Tidak ada model yang berhasil diestimasi."
            )
            st.stop()

        model_ditampilkan = st.selectbox(
            "Pilih model untuk ditampilkan",
            daftar_model_berhasil,
            key="manual_final_model"
        )

    else:
        model_ditampilkan = model_akhir

    result_final = hasil_model[model_ditampilkan]

    if isinstance(result_final, dict):
        st.error(
            f"{model_ditampilkan} gagal dijalankan sehingga tidak dapat ditampilkan sebagai model final."
        )

        explanation = make_model_failure_explanation(
            model_ditampilkan,
            result_final["error"],
            df_model
        )

        st.markdown(explanation)
        st.stop()

    # =========================
    # RINGKASAN MODEL FINAL
    # =========================

    st.header("Ringkasan Model Final")

    if result_final.f_statistic is not None:
        teks_f_statistic = f"{result_final.f_statistic.stat:,.4f}"
        teks_prob_f = f"{result_final.f_statistic.pval:,.4f}"

    else:
        teks_f_statistic = "-"
        teks_prob_f = "-"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_metric_card(
            "Model Final Sementara",
            model_ditampilkan
        )

    with col2:
        show_metric_card(
            "R-squared",
            f"{result_final.rsquared:,.4f}"
        )

    with col3:
        show_metric_card(
            "F-statistic",
            teks_f_statistic
        )

    with col4:
        show_metric_card(
            "Prob(F-statistic)",
            teks_prob_f
        )

    with st.container(border=True):
        st.markdown(
            f"""
            **Alasan pemilihan model:** {conclusion_selection["Alasan"]}
            """
        )

    st.divider()

    # =========================
    # TABEL HASIL UJI PEMILIHAN MODEL
    # =========================

    with st.expander("Lihat Hasil Uji Pemilihan Model"):
        df_hasil_uji_tampil = df_hasil_uji.copy()

        for col in [
            "Statistik",
            "P-value",
            "Alpha"
        ]:
            if col in df_hasil_uji_tampil.columns:
                df_hasil_uji_tampil[col] = df_hasil_uji_tampil[col].apply(
                    lambda x: "-" if pd.isna(x) else f"{x:,.4f}"
                )

        st.dataframe(
            df_hasil_uji_tampil,
            use_container_width=True
        )

    st.divider()

    # =========================
    # KOEFISIEN MODEL FINAL
    # =========================

    st.header("Koefisien Model Final")

    df_coef_final = make_coefficient_table(
        result_final
    )

    st.dataframe(
        df_coef_final.style.format(
            {
                "Koefisien": "{:,.4f}",
                "Std. Error": "{:,.4f}",
                "t-statistic": "{:,.4f}",
                "P-value": "{:,.4f}"
            }
        ),
        use_container_width=True
    )

    st.caption(
        "Tanda signifikansi: *** signifikan 1%, ** signifikan 5%, * signifikan 10%."
    )

    st.divider()

    # =========================
    # GRAFIK KOEFISIEN
    # =========================

    st.header("Visualisasi Arah Koefisien")

    df_coef_chart = df_coef_final[
        df_coef_final["Variabel"] != "const"
    ].copy()

    df_coef_chart = df_coef_chart.sort_values(
        "Koefisien",
        ascending=True
    ).reset_index(drop=True)

    fig_coef = px.bar(
        df_coef_chart,
        x="Koefisien",
        y="Variabel",
        orientation="h",
        color="Signifikansi",
        title=f"Arah Koefisien Model Final: {model_ditampilkan}",
        hover_data=[
            "Std. Error",
            "t-statistic",
            "P-value",
            "Signifikansi"
        ]
    )

    fig_coef.update_layout(
        xaxis_title="Koefisien",
        yaxis_title="Variabel",
        template="plotly_white",
        height=460,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20
        }
    )

    st.plotly_chart(
        fig_coef,
        use_container_width=True
    )

    st.markdown(
        """
        Batang di sisi negatif menunjukkan variabel yang berasosiasi dengan penurunan
        kemiskinan. Batang di sisi positif menunjukkan variabel yang berasosiasi dengan
        peningkatan kemiskinan. Interpretasi substantif tetap perlu memperhatikan
        signifikansi statistik dan hasil diagnostik.
        """
    )

    st.divider()

    # =========================
    # INTERPRETASI KOEFISIEN
    # =========================

    st.header("Interpretasi Koefisien")

    df_interpretasi = make_coefficient_interpretation_table(
        df_coef_final,
        alpha=alpha
    )

    st.dataframe(
        df_interpretasi.style.format(
            {
                "Koefisien": "{:,.4f}",
                "P-value": "{:,.4f}"
            }
        ),
        use_container_width=True
    )

    ringkasan_signifikan = make_significant_variable_summary(
        df_interpretasi,
        alpha=alpha
    )

    with st.container(border=True):
        st.markdown(
            f"""
            **Ringkasan variabel signifikan:**  
            {ringkasan_signifikan}
            """
        )

    st.divider()

    # =========================
    # NARASI AKADEMIK AWAL
    # =========================

    st.header("Narasi Akademik Awal")

    df_signifikan = df_interpretasi[
        df_interpretasi["P-value"] < alpha
    ].copy()

    if df_signifikan.shape[0] == 0:
        st.markdown(
            """
            Berdasarkan hasil model final sementara, belum terdapat variabel independen
            yang signifikan pada tingkat alpha yang dipilih. Oleh karena itu, pembahasan
            hasil perlu difokuskan pada arah koefisien, kelayakan model, serta hasil
            diagnostik, tanpa menyimpulkan pengaruh statistik yang terlalu kuat.
            """
        )

    else:
        st.markdown(
            f"""
            Berdasarkan hasil model final sementara, terdapat **{df_signifikan.shape[0]}**
            variabel yang signifikan pada tingkat alpha **{alpha}**. Variabel-variabel
            tersebut menjadi sinyal utama dalam membaca hubungan antara kebijakan fiskal,
            pembiayaan usaha, pengangguran, dan kemiskinan di wilayah Pulau Flores.
            """
        )

        for index, row in df_signifikan.iterrows():
            st.markdown(
                f"""
                - **{row["Variabel"]}** memiliki koefisien **{row["Koefisien"]:,.4f}**
                  dengan p-value **{row["P-value"]:,.4f}**. {row["Interpretasi"]}
                """
            )

    st.divider()

    # =========================
    # OUTPUT TEKNIS DAN DATASET
    # =========================

    with st.expander(f"Output Teknis Lengkap {model_ditampilkan}"):
        st.code(
            str(result_final.summary),
            language=None
        )

    with st.expander("Tabel Dataset yang Digunakan"):
        st.dataframe(
            df_model,
            use_container_width=True
        )

# =========================
# HALAMAN SWOT DAN REKOMENDASI
# =========================

def halaman_swot_rekomendasi():

    # =========================
    # HEADER
    # =========================

    st.title("SWOT & Rekomendasi Kebijakan")

    st.markdown(
        """
        Halaman ini menyusun SWOT dan rekomendasi kebijakan dengan menggabungkan
        seluruh hasil dashboard: data panel utama, struktur PDRB, indikator kontekstual,
        dan sinyal model panel.
        """
    )

    st.caption(
        "SWOT ini bersifat analitis dan digunakan sebagai jembatan dari hasil data menuju rekomendasi kebijakan."
    )

    st.divider()

    # =========================
    # PENGATURAN
    # =========================

    st.header("Pengaturan Analisis")

    daftar_kabupaten = sorted(
        df_panel["Kabupaten"].dropna().unique()
    )

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            kabupaten_terpilih = st.selectbox(
                "Pilih Kabupaten",
                daftar_kabupaten,
                key="swot_kabupaten"
            )

        with col2:
            pilihan_dataset = st.selectbox(
                "Pilih dataset model",
                [
                    "Model Normal",
                    "Model Lag 1"
                ],
                key="swot_dataset"
            )

        with col3:
            pilihan_covariance = st.selectbox(
                "Pilih standard error",
                [
                    "Robust",
                    "Unadjusted"
                ],
                key="swot_covariance"
            )

        with col4:
            alpha = st.selectbox(
                "Pilih alpha",
                [
                    0.01,
                    0.05,
                    0.10
                ],
                index=1,
                key="swot_alpha"
            )

    if pilihan_dataset == "Model Normal":
        df_model = df_model_normal.copy()
    else:
        df_model = df_model_lag.copy()

    if pilihan_covariance == "Robust":
        cov_type = "robust"
    else:
        cov_type = "unadjusted"

    # =========================
    # DATA KABUPATEN
    # =========================

    (
        df_panel_kabupaten,
        df_context_kabupaten,
        df_pdrb_kabupaten,
        df_pdrb_dominan_kabupaten
    ) = make_kabupaten_profile_data(
        kabupaten_terpilih
    )

    # =========================
    # SINYAL MODEL PANEL
    # =========================

    df_coef_model = None
    model_dibaca = "-"
    alasan_model = "Model panel belum dapat dibaca."

    try:
        hasil_model = fit_all_panel_models(
            df_model,
            cov_type=cov_type
        )

        df_hasil_uji = run_panel_model_selection_tests(
            hasil_model,
            alpha=alpha
        )

        conclusion_selection = make_model_selection_conclusion(
            df_hasil_uji,
            hasil_model
        )

        model_akhir = conclusion_selection["Model Akhir Sementara"]
        alasan_model = conclusion_selection["Alasan"]

        daftar_model_berhasil = []

        for nama_model, result in hasil_model.items():
            if not isinstance(result, dict):
                daftar_model_berhasil.append(nama_model)

        if model_akhir == "Belum dapat ditentukan":
            if len(daftar_model_berhasil) > 0:
                model_dibaca = daftar_model_berhasil[0]
            else:
                model_dibaca = "-"

        else:
            model_dibaca = model_akhir

        if model_dibaca != "-":
            result_model = hasil_model[model_dibaca]

            if not isinstance(result_model, dict):
                df_coef_model = make_coefficient_table(
                    result_model
                )

    except Exception:
        df_coef_model = None
        model_dibaca = "-"
        alasan_model = "Sinyal model panel tidak tersedia karena estimasi atau pemilihan model gagal."

    # =========================
    # SWOT KONSOLIDASI
    # =========================

    df_swot, summary_swot = make_consolidated_swot_analysis(
        kabupaten_terpilih=kabupaten_terpilih,
        df_panel=df_panel,
        df_context=df_context,
        df_pdrb_dominan_kabupaten=df_pdrb_dominan_kabupaten,
        df_coef_model=df_coef_model,
        alpha=alpha
    )

    df_strategy = make_swot_strategy_table(
        df_swot
    )

    df_rekomendasi = make_policy_recommendation_from_swot(
        df_swot,
        summary_swot
    )

    st.divider()

    # =========================
    # RINGKASAN UTAMA
    # =========================

    st.header(f"Ringkasan SWOT {kabupaten_terpilih}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_metric_card(
            "Strength",
            summary_swot["Jumlah Strength"]
        )

    with col2:
        show_metric_card(
            "Weakness",
            summary_swot["Jumlah Weakness"]
        )

    with col3:
        show_metric_card(
            "Opportunity",
            summary_swot["Jumlah Opportunity"]
        )

    with col4:
        show_metric_card(
            "Threat",
            summary_swot["Jumlah Threat"]
        )

    st.caption(
        f"SWOT menggabungkan data panel, PDRB sektoral, indikator kontekstual, dan sinyal model panel. Periode utama: {summary_swot['Tahun Awal']}–{summary_swot['Tahun Akhir']}."
    )

    st.divider()

    # =========================
    # STATUS MODEL PANEL
    # =========================

    st.header("Sinyal Model Panel yang Digunakan")

    with st.container(border=True):
        st.markdown(
            f"""
            **Model dibaca:** {model_dibaca}  
            **Dataset:** {pilihan_dataset}  
            **Standard error:** {pilihan_covariance}  
            **Alpha:** {alpha}  
            **Alasan:** {alasan_model}
            """
        )

    if df_coef_model is None:
        st.warning(
            "Sinyal model panel belum masuk ke SWOT. SWOT tetap dibentuk dari data panel utama, PDRB, dan indikator kontekstual."
        )

    else:
        with st.expander("Lihat Koefisien Model yang Masuk Pertimbangan"):
            st.dataframe(
                df_coef_model,
                use_container_width=True
            )

    st.divider()

    # =========================
    # MATRIKS SWOT
    # =========================

    st.header("Matriks SWOT Konsolidasi")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Strength")

        df_strength = df_swot[
            df_swot["Kategori"] == "Strength"
        ].copy()

        if df_strength.shape[0] == 0:
            st.info("Belum ada strength yang teridentifikasi.")

        else:
            for index, row in df_strength.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['Aspek']}**")
                    st.markdown(row["Poin SWOT"])
                    st.caption(f"{row['Sumber Analisis']} | {row['Dasar Data']}")

    with col2:
        st.subheader("Weakness")

        df_weakness = df_swot[
            df_swot["Kategori"] == "Weakness"
        ].copy()

        if df_weakness.shape[0] == 0:
            st.info("Belum ada weakness yang teridentifikasi.")

        else:
            for index, row in df_weakness.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['Aspek']}**")
                    st.markdown(row["Poin SWOT"])
                    st.caption(f"{row['Sumber Analisis']} | {row['Dasar Data']}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Opportunity")

        df_opportunity = df_swot[
            df_swot["Kategori"] == "Opportunity"
        ].copy()

        if df_opportunity.shape[0] == 0:
            st.info("Belum ada opportunity yang teridentifikasi.")

        else:
            for index, row in df_opportunity.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['Aspek']}**")
                    st.markdown(row["Poin SWOT"])
                    st.caption(f"{row['Sumber Analisis']} | {row['Dasar Data']}")

    with col2:
        st.subheader("Threat")

        df_threat = df_swot[
            df_swot["Kategori"] == "Threat"
        ].copy()

        if df_threat.shape[0] == 0:
            st.info("Belum ada threat yang teridentifikasi.")

        else:
            for index, row in df_threat.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['Aspek']}**")
                    st.markdown(row["Poin SWOT"])
                    st.caption(f"{row['Sumber Analisis']} | {row['Dasar Data']}")

    st.divider()

    # =========================
    # TABEL SWOT PER SUMBER
    # =========================

    st.header("Konsolidasi SWOT per Sumber Analisis")

    sumber_terpilih = st.multiselect(
        "Pilih sumber analisis yang ditampilkan",
        sorted(df_swot["Sumber Analisis"].dropna().unique()),
        default=sorted(df_swot["Sumber Analisis"].dropna().unique()),
        key="swot_sumber_filter"
    )

    df_swot_filtered = df_swot[
        df_swot["Sumber Analisis"].isin(sumber_terpilih)
    ].copy()

    st.dataframe(
        df_swot_filtered,
        use_container_width=True
    )

    st.divider()

    # =========================
    # STRATEGI SWOT
    # =========================

    st.header("Matriks Strategi SWOT")

    st.dataframe(
        df_strategy,
        use_container_width=True
    )

    st.markdown(
        """
        Strategi SO memanfaatkan kekuatan untuk menangkap peluang. Strategi WO
        menggunakan peluang untuk memperbaiki kelemahan. Strategi ST menggunakan
        kekuatan untuk mengurangi ancaman. Strategi WT meminimalkan kelemahan dan
        menghindari ancaman.
        """
    )

    st.divider()

    # =========================
    # REKOMENDASI KEBIJAKAN
    # =========================

    st.header("Rekomendasi Kebijakan")

    st.dataframe(
        df_rekomendasi,
        use_container_width=True
    )

    st.markdown(
        """
        Rekomendasi disusun dari gabungan hasil panel, konteks wilayah, struktur ekonomi,
        dan sinyal model. Oleh karena itu, rekomendasi tidak hanya bertumpu pada koefisien
        model, tetapi juga pada kondisi faktual kabupaten.
        """
    )

    st.divider()

    # =========================
    # DATA PENDUKUNG
    # =========================

    with st.expander("Tabel Lengkap SWOT Konsolidasi"):
        st.dataframe(
            df_swot,
            use_container_width=True
        )

    with st.expander("Data Panel Kabupaten"):
        st.dataframe(
            df_panel_kabupaten,
            use_container_width=True
        )

    with st.expander("Indikator Kontekstual Kabupaten"):
        st.dataframe(
            df_context_kabupaten,
            use_container_width=True
        )

    with st.expander("PDRB Dominan Kabupaten"):
        st.dataframe(
            df_pdrb_dominan_kabupaten,
            use_container_width=True
        )

# =========================
# NAVIGASI SIDEBAR
# =========================

pages = {
    "Main Menu": [
        st.Page(
            halaman_overview,
            title="Overview"
        ),
        st.Page(
            halaman_flores_intelligence,
            title="Flores Intelligence"
        ),
        st.Page(
            halaman_kabupaten_intelligence,
            title="Kabupaten Intelligence"
        ),
        st.Page(
            halaman_estimasi_panel,
            title="Estimasi Panel"
        ),
        st.Page(
            halaman_pemilihan_model_panel,
            title="Pemilihan Model"
        ),
        st.Page(
            halaman_diagnostik_model,
            title="Diagnostik Model"
        ),
        st.Page(
            halaman_model_final,
            title="Model Final"
        ),
        st.Page(
            halaman_swot_rekomendasi,
            title="SWOT & Rekomendasi"
        )
    ]
}

current_page = st.navigation(
    pages,
    position="sidebar"
)

current_page.run()