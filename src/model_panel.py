import pandas as pd
import statsmodels.api as sm

from linearmodels.panel import PooledOLS
from linearmodels.panel import PanelOLS
from linearmodels.panel import RandomEffects


# =========================
# 1. MENYIAPKAN DATA MODEL PANEL
# =========================

def prepare_panel_model_data(df_model):
    df = df_model.copy()

    df = df.sort_values(
        ["Kabupaten", "Tahun"]
    ).reset_index(drop=True)

    df = df.set_index(
        ["Kabupaten", "Tahun"]
    )

    y = df["Kemiskinan"]

    x_columns = []

    for col in df.columns:
        if col != "Kemiskinan":
            x_columns.append(col)

    X = df[x_columns].copy()

    X = sm.add_constant(
        X,
        has_constant="add"
    )

    return y, X, x_columns

# =========================
# 2. ESTIMASI CEM / POOLED OLS
# =========================

def fit_cem_model(df_model, cov_type="robust"):
    y, X, x_columns = prepare_panel_model_data(df_model)

    model = PooledOLS(
        y,
        X
    )

    result = model.fit(
        cov_type=cov_type
    )

    return result


# =========================
# 3. ESTIMASI FEM / FIXED EFFECT
# =========================

def fit_fem_model(df_model, cov_type="robust"):
    y, X, x_columns = prepare_panel_model_data(df_model)

    model = PanelOLS(
        y,
        X,
        entity_effects=True,
        drop_absorbed=True
    )

    result = model.fit(
        cov_type=cov_type
    )

    return result


# =========================
# 4. ESTIMASI REM / RANDOM EFFECT
# =========================

def fit_rem_model(df_model, cov_type="robust"):
    y, X, x_columns = prepare_panel_model_data(df_model)

    model = RandomEffects(
        y,
        X
    )

    result = model.fit(
        cov_type=cov_type
    )

    return result

# =========================
# 5. MENJALANKAN SEMUA MODEL PANEL
# =========================

def fit_all_panel_models(df_model, cov_type="robust"):
    hasil_model = {}

    try:
        hasil_model["CEM / Pooled OLS"] = fit_cem_model(
            df_model,
            cov_type=cov_type
        )

    except Exception as error:
        hasil_model["CEM / Pooled OLS"] = {
            "error": str(error)
        }

    try:
        hasil_model["FEM / Fixed Effect"] = fit_fem_model(
            df_model,
            cov_type=cov_type
        )

    except Exception as error:
        hasil_model["FEM / Fixed Effect"] = {
            "error": str(error)
        }

    try:
        hasil_model["REM / Random Effect"] = fit_rem_model(
            df_model,
            cov_type=cov_type
        )

    except Exception as error:
        hasil_model["REM / Random Effect"] = {
            "error": str(error)
        }

    return hasil_model

# =========================
# 6. TABEL RINGKASAN MODEL
# =========================

def make_model_summary_table(hasil_model):
    daftar_ringkasan = []

    for nama_model, result in hasil_model.items():

        if isinstance(result, dict):
            ringkasan = {
                "Model": nama_model,
                "Status": "Gagal",
                "Observasi": None,
                "R-squared": None,
                "F-statistic": None,
                "Prob(F-statistic)": None,
                "Jumlah Parameter": None,
                "Keterangan Error": result["error"]
            }

        else:
            if result.f_statistic is not None:
                f_statistic = result.f_statistic.stat
                f_pvalue = result.f_statistic.pval
            else:
                f_statistic = None
                f_pvalue = None

            ringkasan = {
                "Model": nama_model,
                "Status": "Berhasil",
                "Observasi": result.nobs,
                "R-squared": result.rsquared,
                "F-statistic": f_statistic,
                "Prob(F-statistic)": f_pvalue,
                "Jumlah Parameter": len(result.params),
                "Keterangan Error": ""
            }

        daftar_ringkasan.append(ringkasan)

    df_ringkasan = pd.DataFrame(daftar_ringkasan)

    return df_ringkasan

# =========================
# 7. TABEL KOEFISIEN MODEL
# =========================

def make_coefficient_table(result):
    df_coef = pd.DataFrame({
        "Variabel": result.params.index,
        "Koefisien": result.params.values,
        "Std. Error": result.std_errors.values,
        "t-statistic": result.tstats.values,
        "P-value": result.pvalues.values
    })

    signifikan_list = []

    for pvalue in df_coef["P-value"]:
        if pvalue < 0.01:
            signifikan = "***"
        elif pvalue < 0.05:
            signifikan = "**"
        elif pvalue < 0.10:
            signifikan = "*"
        else:
            signifikan = ""

        signifikan_list.append(signifikan)

    df_coef["Signifikansi"] = signifikan_list

    return df_coef

# =========================
# 7. TABEL KOEFISIEN MODEL
# =========================

def make_coefficient_table(result):
    df_coef = pd.DataFrame({
        "Variabel": result.params.index,
        "Koefisien": result.params.values,
        "Std. Error": result.std_errors.values,
        "t-statistic": result.tstats.values,
        "P-value": result.pvalues.values
    })

    signifikan_list = []

    for pvalue in df_coef["P-value"]:
        if pvalue < 0.01:
            signifikan = "***"
        elif pvalue < 0.05:
            signifikan = "**"
        elif pvalue < 0.10:
            signifikan = "*"
        else:
            signifikan = ""

        signifikan_list.append(signifikan)

    df_coef["Signifikansi"] = signifikan_list

    return df_coef

# =========================
# 8. MEMBUAT TEKS PERSAMAAN MODEL
# =========================

def make_model_equation_text(df_model):
    x_columns = []

    for col in df_model.columns:
        if col not in ["Kabupaten", "Tahun", "Kemiskinan"]:
            x_columns.append(col)

    bagian_x = " + ".join(x_columns)

    equation = f"Kemiskinan_it = α + {bagian_x} + ε_it"

    return equation