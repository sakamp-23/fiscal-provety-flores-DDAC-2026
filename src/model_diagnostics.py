import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor


# =========================
# 1. MENGAMBIL VARIABEL INDEPENDEN
# =========================

def get_independent_variables(df_model):
    x_columns = []

    for col in df_model.columns:
        if col not in ["Kabupaten", "Tahun", "Kemiskinan"]:
            x_columns.append(col)

    return x_columns


# =========================
# 2. MEMBUAT DATA X UNTUK DIAGNOSTIK
# =========================

def prepare_x_diagnostic_data(df_model):
    df = df_model.copy()

    x_columns = get_independent_variables(df)

    X = df[x_columns].copy()

    X = X.dropna().reset_index(drop=True)

    return X, x_columns


# =========================
# 3. INTERPRETASI NILAI VIF
# =========================

def interpret_vif_value(vif_value):
    if pd.isna(vif_value):
        status = "Tidak Dapat Dihitung"

    elif np.isinf(vif_value):
        status = "Sangat Tinggi"

    elif vif_value < 5:
        status = "Aman"

    elif vif_value < 10:
        status = "Perlu Perhatian"

    else:
        status = "Tinggi"

    return status


# =========================
# 4. MENGHITUNG VIF
# =========================

def calculate_vif(df_model):
    X, x_columns = prepare_x_diagnostic_data(df_model)

    X_const = sm.add_constant(
        X,
        has_constant="add"
    )

    daftar_vif = []

    for index, col in enumerate(X_const.columns):

        if col == "const":
            continue

        try:
            vif_value = variance_inflation_factor(
                X_const.values,
                index
            )

        except Exception:
            vif_value = np.nan

        status = interpret_vif_value(vif_value)

        row = {
            "Variabel": col,
            "VIF": vif_value,
            "Status": status
        }

        daftar_vif.append(row)

    df_vif = pd.DataFrame(daftar_vif)

    df_vif = df_vif.sort_values(
        "VIF",
        ascending=False
    ).reset_index(drop=True)

    return df_vif


# =========================
# 5. MEMBUAT MATRIKS KORELASI
# =========================

def make_correlation_matrix(df_model):
    X, x_columns = prepare_x_diagnostic_data(df_model)

    df_corr = X.corr()

    return df_corr


# =========================
# 6. MENGAMBIL PASANGAN KORELASI TINGGI
# =========================

def make_high_correlation_pairs(df_model, threshold=0.80):
    df_corr = make_correlation_matrix(df_model)

    columns = df_corr.columns.tolist()

    daftar_korelasi = []

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            var_1 = columns[i]
            var_2 = columns[j]

            nilai_korelasi = df_corr.loc[var_1, var_2]
            abs_korelasi = abs(nilai_korelasi)

            if abs_korelasi >= threshold:
                row = {
                    "Variabel 1": var_1,
                    "Variabel 2": var_2,
                    "Korelasi": nilai_korelasi,
                    "Abs Korelasi": abs_korelasi
                }

                daftar_korelasi.append(row)

    df_high_corr = pd.DataFrame(daftar_korelasi)

    if df_high_corr.shape[0] > 0:
        df_high_corr = df_high_corr.sort_values(
            "Abs Korelasi",
            ascending=False
        ).reset_index(drop=True)

    return df_high_corr


# =========================
# 7. KESIMPULAN MULTIKOLINEARITAS
# =========================

def make_multicollinearity_conclusion(df_vif, df_high_corr):
    df_vif_valid = df_vif[
        df_vif["VIF"].notna()
    ].copy()

    if df_vif_valid.shape[0] == 0:
        conclusion = {
            "Status": "Tidak Dapat Disimpulkan",
            "Kesimpulan": "Nilai VIF tidak dapat dihitung.",
            "Rekomendasi": "Periksa kembali struktur data dan variabel independen."
        }

        return conclusion

    max_vif = df_vif_valid["VIF"].max()

    jumlah_vif_tinggi = df_vif[
        df_vif["Status"].isin(["Tinggi", "Sangat Tinggi"])
    ].shape[0]

    jumlah_vif_perhatian = df_vif[
        df_vif["Status"] == "Perlu Perhatian"
    ].shape[0]

    jumlah_korelasi_tinggi = df_high_corr.shape[0]

    if jumlah_vif_tinggi > 0:
        status = "Multikolinearitas Tinggi"

        kesimpulan = (
            "Terdapat variabel independen dengan nilai VIF tinggi. "
            "Hal ini menunjukkan adanya indikasi multikolinearitas kuat antarvariabel independen."
        )

        rekomendasi = (
            "Pertimbangkan penyederhanaan spesifikasi model, penggabungan variabel yang sejenis, "
            "atau penggunaan model ringkas seperti Total_TKD + KUR + TPT."
        )

    elif jumlah_vif_perhatian > 0:
        status = "Perlu Perhatian"

        kesimpulan = (
            "Terdapat variabel independen dengan nilai VIF pada area perhatian. "
            "Model masih dapat dibaca, tetapi interpretasi koefisien perlu dilakukan secara hati-hati."
        )

        rekomendasi = (
            "Periksa korelasi antarvariabel dan pastikan variabel yang digunakan memiliki dasar teoritis yang kuat."
        )

    else:
        status = "Relatif Aman"

        kesimpulan = (
            "Nilai VIF seluruh variabel berada pada kategori aman. "
            "Tidak terdapat indikasi multikolinearitas kuat berdasarkan VIF."
        )

        rekomendasi = (
            "Model dapat dilanjutkan ke diagnostik residual seperti heteroskedastisitas, autokorelasi, dan cross-section dependence."
        )

    if jumlah_korelasi_tinggi > 0:
        rekomendasi = (
            rekomendasi +
            " Selain itu, terdapat pasangan variabel dengan korelasi tinggi yang perlu diperhatikan."
        )

    conclusion = {
        "Status": status,
        "Kesimpulan": kesimpulan,
        "Rekomendasi": rekomendasi,
        "Max VIF": max_vif,
        "Jumlah VIF Tinggi": jumlah_vif_tinggi,
        "Jumlah Korelasi Tinggi": jumlah_korelasi_tinggi
    }

    return conclusion

from scipy.stats import norm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import jarque_bera


# =========================
# 8. MEMBUAT DATA RESIDUAL UNTUK DIAGNOSTIK
# =========================

def get_residual_diagnostic_data(result, df_model):
    residual = result.resids.copy()

    df_residual = residual.reset_index()

    df_residual = df_residual.rename(
        columns={
            df_residual.columns[0]: "Kabupaten",
            df_residual.columns[1]: "Tahun",
            df_residual.columns[2]: "Residual"
        }
    )

    df_base = df_model.copy()

    df_diagnostic = df_residual.merge(
        df_base,
        on=[
            "Kabupaten",
            "Tahun"
        ],
        how="left"
    )

    df_diagnostic = df_diagnostic.dropna(
        subset=[
            "Residual"
        ]
    ).copy()

    return df_diagnostic


# =========================
# 9. MEMBUAT BARIS HASIL DIAGNOSTIK RESIDUAL
# =========================

def make_residual_test_row(
    uji,
    aspek,
    h0,
    h1,
    statistik,
    p_value,
    alpha,
    status,
    keterangan
):
    if status == "Berhasil":
        if p_value < alpha:
            keputusan = "Tolak H0"
        else:
            keputusan = "Gagal Tolak H0"
    else:
        keputusan = "-"

    row = {
        "Uji": uji,
        "Aspek": aspek,
        "H0": h0,
        "H1": h1,
        "Statistik": statistik,
        "P-value": p_value,
        "Alpha": alpha,
        "Keputusan": keputusan,
        "Status": status,
        "Keterangan": keterangan
    }

    return row


# =========================
# 10. UJI HETEROSKEDASTISITAS BREUSCH-PAGAN
# =========================

def run_breusch_pagan_test(result, df_model, alpha=0.05):
    uji = "Breusch-Pagan"
    aspek = "Heteroskedastisitas"
    h0 = "Tidak terdapat heteroskedastisitas"
    h1 = "Terdapat heteroskedastisitas"

    try:
        df_diagnostic = get_residual_diagnostic_data(
            result,
            df_model
        )

        x_columns = get_independent_variables(df_model)

        df_test = df_diagnostic[
            [
                "Residual"
            ] + x_columns
        ].copy()

        df_test = df_test.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        ).dropna()

        if df_test.shape[0] <= len(x_columns) + 1:
            row = make_residual_test_row(
                uji,
                aspek,
                h0,
                h1,
                None,
                None,
                alpha,
                "Gagal",
                "Observasi tidak mencukupi untuk menjalankan uji Breusch-Pagan."
            )

            return row

        X = sm.add_constant(
            df_test[x_columns],
            has_constant="add"
        )

        residual = df_test["Residual"]

        lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(
            residual,
            X
        )

        if lm_pvalue < alpha:
            keterangan = "H0 ditolak. Terdapat indikasi heteroskedastisitas."
        else:
            keterangan = "H0 gagal ditolak. Tidak terdapat indikasi kuat heteroskedastisitas."

        row = make_residual_test_row(
            uji,
            aspek,
            h0,
            h1,
            lm_stat,
            lm_pvalue,
            alpha,
            "Berhasil",
            keterangan
        )

        return row

    except Exception as error:
        row = make_residual_test_row(
            uji,
            aspek,
            h0,
            h1,
            None,
            None,
            alpha,
            "Gagal",
            str(error)
        )

        return row


# =========================
# 11. UJI HETEROSKEDASTISITAS WHITE
# =========================

def run_white_test(result, df_model, alpha=0.05):
    uji = "White"
    aspek = "Heteroskedastisitas"
    h0 = "Tidak terdapat heteroskedastisitas"
    h1 = "Terdapat heteroskedastisitas"

    try:
        df_diagnostic = get_residual_diagnostic_data(
            result,
            df_model
        )

        x_columns = get_independent_variables(df_model)

        df_test = df_diagnostic[
            [
                "Residual"
            ] + x_columns
        ].copy()

        df_test = df_test.replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        ).dropna()

        if df_test.shape[0] <= len(x_columns) + 1:
            row = make_residual_test_row(
                uji,
                aspek,
                h0,
                h1,
                None,
                None,
                alpha,
                "Gagal",
                "Observasi tidak mencukupi untuk menjalankan uji White."
            )

            return row

        X = sm.add_constant(
            df_test[x_columns],
            has_constant="add"
        )

        residual = df_test["Residual"]

        lm_stat, lm_pvalue, f_stat, f_pvalue = het_white(
            residual,
            X
        )

        if lm_pvalue < alpha:
            keterangan = "H0 ditolak. Terdapat indikasi heteroskedastisitas."
        else:
            keterangan = "H0 gagal ditolak. Tidak terdapat indikasi kuat heteroskedastisitas."

        row = make_residual_test_row(
            uji,
            aspek,
            h0,
            h1,
            lm_stat,
            lm_pvalue,
            alpha,
            "Berhasil",
            keterangan
        )

        return row

    except Exception as error:
        row = make_residual_test_row(
            uji,
            aspek,
            h0,
            h1,
            None,
            None,
            alpha,
            "Gagal",
            str(error)
        )

        return row


# =========================
# 12. UJI AUTOKORELASI RESIDUAL AR(1)
# =========================

def run_panel_ar1_residual_test(result, df_model, alpha=0.05):
    uji = "Residual AR(1)"
    aspek = "Autokorelasi"
    h0 = "Tidak terdapat autokorelasi residual"
    h1 = "Terdapat autokorelasi residual"

    try:
        df_diagnostic = get_residual_diagnostic_data(
            result,
            df_model
        )

        df_test = df_diagnostic[
            [
                "Kabupaten",
                "Tahun",
                "Residual"
            ]
        ].copy()

        df_test = df_test.sort_values(
            [
                "Kabupaten",
                "Tahun"
            ]
        ).reset_index(drop=True)

        df_test["Residual_Lag"] = (
            df_test
            .groupby("Kabupaten")["Residual"]
            .shift(1)
        )

        df_test = df_test.dropna(
            subset=[
                "Residual",
                "Residual_Lag"
            ]
        ).copy()

        if df_test.shape[0] < 5:
            row = make_residual_test_row(
                uji,
                aspek,
                h0,
                h1,
                None,
                None,
                alpha,
                "Gagal",
                "Observasi residual lag terlalu sedikit untuk menjalankan uji AR(1)."
            )

            return row

        X = sm.add_constant(
            df_test["Residual_Lag"],
            has_constant="add"
        )

        y = df_test["Residual"]

        model_ar1 = sm.OLS(
            y,
            X
        ).fit()

        statistik = model_ar1.tvalues["Residual_Lag"]
        p_value = model_ar1.pvalues["Residual_Lag"]
        koefisien_lag = model_ar1.params["Residual_Lag"]

        if p_value < alpha:
            keterangan = (
                f"H0 ditolak. Terdapat indikasi autokorelasi residual. "
                f"Koefisien residual lag = {koefisien_lag:,.4f}."
            )
        else:
            keterangan = (
                f"H0 gagal ditolak. Tidak terdapat indikasi kuat autokorelasi residual. "
                f"Koefisien residual lag = {koefisien_lag:,.4f}."
            )

        row = make_residual_test_row(
            uji,
            aspek,
            h0,
            h1,
            statistik,
            p_value,
            alpha,
            "Berhasil",
            keterangan
        )

        return row

    except Exception as error:
        row = make_residual_test_row(
            uji,
            aspek,
            h0,
            h1,
            None,
            None,
            alpha,
            "Gagal",
            str(error)
        )

        return row


# =========================
# 13. UJI CROSS-SECTION DEPENDENCE PESARAN CD
# =========================

def run_pesaran_cd_test(result, df_model, alpha=0.05):
    uji = "Pesaran CD"
    aspek = "Cross-section Dependence"
    h0 = "Tidak terdapat dependensi antar-kabupaten"
    h1 = "Terdapat dependensi antar-kabupaten"

    try:
        df_diagnostic = get_residual_diagnostic_data(
            result,
            df_model
        )

        df_residual = df_diagnostic[
            [
                "Kabupaten",
                "Tahun",
                "Residual"
            ]
        ].copy()

        df_pivot = df_residual.pivot(
            index="Tahun",
            columns="Kabupaten",
            values="Residual"
        )

        daftar_kabupaten = df_pivot.columns.tolist()
        jumlah_kabupaten = len(daftar_kabupaten)

        if jumlah_kabupaten < 2:
            row = make_residual_test_row(
                uji,
                aspek,
                h0,
                h1,
                None,
                None,
                alpha,
                "Gagal",
                "Jumlah kabupaten kurang dari dua sehingga uji Pesaran CD tidak dapat dijalankan."
            )

            return row

        total_component = 0
        jumlah_pasangan_valid = 0

        for i in range(jumlah_kabupaten):
            for j in range(i + 1, jumlah_kabupaten):
                kabupaten_i = daftar_kabupaten[i]
                kabupaten_j = daftar_kabupaten[j]

                df_pair = df_pivot[
                    [
                        kabupaten_i,
                        kabupaten_j
                    ]
                ].dropna()

                jumlah_tahun_pair = df_pair.shape[0]

                if jumlah_tahun_pair >= 3:
                    korelasi = df_pair[kabupaten_i].corr(
                        df_pair[kabupaten_j]
                    )

                    if pd.notna(korelasi):
                        total_component = total_component + (
                            np.sqrt(jumlah_tahun_pair) * korelasi
                        )

                        jumlah_pasangan_valid = jumlah_pasangan_valid + 1

        if jumlah_pasangan_valid == 0:
            row = make_residual_test_row(
                uji,
                aspek,
                h0,
                h1,
                None,
                None,
                alpha,
                "Gagal",
                "Tidak terdapat pasangan kabupaten dengan observasi residual yang cukup."
            )

            return row

        statistik = np.sqrt(
            2 / (jumlah_kabupaten * (jumlah_kabupaten - 1))
        ) * total_component

        p_value = 2 * (
            1 - norm.cdf(abs(statistik))
        )

        if p_value < alpha:
            keterangan = "H0 ditolak. Terdapat indikasi dependensi residual antar-kabupaten."
        else:
            keterangan = "H0 gagal ditolak. Tidak terdapat indikasi kuat dependensi residual antar-kabupaten."

        row = make_residual_test_row(
            uji,
            aspek,
            h0,
            h1,
            statistik,
            p_value,
            alpha,
            "Berhasil",
            keterangan
        )

        return row

    except Exception as error:
        row = make_residual_test_row(
            uji,
            aspek,
            h0,
            h1,
            None,
            None,
            alpha,
            "Gagal",
            str(error)
        )

        return row


# =========================
# 14. UJI NORMALITAS JARQUE-BERA
# =========================

def run_jarque_bera_test(result, df_model, alpha=0.05):
    uji = "Jarque-Bera"
    aspek = "Normalitas Residual"
    h0 = "Residual berdistribusi normal"
    h1 = "Residual tidak berdistribusi normal"

    try:
        df_diagnostic = get_residual_diagnostic_data(
            result,
            df_model
        )

        residual = df_diagnostic["Residual"].dropna()

        if residual.shape[0] < 5:
            row = make_residual_test_row(
                uji,
                aspek,
                h0,
                h1,
                None,
                None,
                alpha,
                "Gagal",
                "Observasi residual terlalu sedikit untuk menjalankan uji Jarque-Bera."
            )

            return row

        jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(
            residual
        )

        if jb_pvalue < alpha:
            keterangan = (
                f"H0 ditolak. Residual tidak berdistribusi normal. "
                f"Skewness = {skewness:,.4f}; Kurtosis = {kurtosis:,.4f}."
            )
        else:
            keterangan = (
                f"H0 gagal ditolak. Tidak terdapat bukti kuat bahwa residual menyimpang dari normalitas. "
                f"Skewness = {skewness:,.4f}; Kurtosis = {kurtosis:,.4f}."
            )

        row = make_residual_test_row(
            uji,
            aspek,
            h0,
            h1,
            jb_stat,
            jb_pvalue,
            alpha,
            "Berhasil",
            keterangan
        )

        return row

    except Exception as error:
        row = make_residual_test_row(
            uji,
            aspek,
            h0,
            h1,
            None,
            None,
            alpha,
            "Gagal",
            str(error)
        )

        return row


# =========================
# 15. MENJALANKAN SEMUA DIAGNOSTIK RESIDUAL
# =========================

def run_residual_diagnostic_tests(result, df_model, alpha=0.05):
    daftar_hasil = []

    daftar_hasil.append(
        run_breusch_pagan_test(
            result,
            df_model,
            alpha=alpha
        )
    )

    daftar_hasil.append(
        run_white_test(
            result,
            df_model,
            alpha=alpha
        )
    )

    daftar_hasil.append(
        run_panel_ar1_residual_test(
            result,
            df_model,
            alpha=alpha
        )
    )

    daftar_hasil.append(
        run_pesaran_cd_test(
            result,
            df_model,
            alpha=alpha
        )
    )

    daftar_hasil.append(
        run_jarque_bera_test(
            result,
            df_model,
            alpha=alpha
        )
    )

    df_hasil = pd.DataFrame(
        daftar_hasil
    )

    return df_hasil


# =========================
# 16. KESIMPULAN DIAGNOSTIK RESIDUAL
# =========================

def make_residual_diagnostic_conclusion(df_residual_tests):
    df_success = df_residual_tests[
        df_residual_tests["Status"] == "Berhasil"
    ].copy()

    if df_success.shape[0] == 0:
        conclusion = {
            "Status": "Tidak Dapat Disimpulkan",
            "Kesimpulan": "Seluruh uji diagnostik residual gagal dijalankan.",
            "Rekomendasi": "Periksa kembali struktur model dan data residual."
        }

        return conclusion

    df_reject = df_success[
        df_success["Keputusan"] == "Tolak H0"
    ].copy()

    aspek_bermasalah = df_reject["Aspek"].unique().tolist()

    ada_heteroskedastisitas = "Heteroskedastisitas" in aspek_bermasalah
    ada_autokorelasi = "Autokorelasi" in aspek_bermasalah
    ada_dependensi = "Cross-section Dependence" in aspek_bermasalah
    ada_tidak_normal = "Normalitas Residual" in aspek_bermasalah

    if (
        ada_heteroskedastisitas is False
        and ada_autokorelasi is False
        and ada_dependensi is False
        and ada_tidak_normal is False
    ):
        status = "Relatif Aman"

        kesimpulan = (
            "Tidak terdapat indikasi kuat pelanggaran diagnostik residual berdasarkan uji yang berhasil dijalankan."
        )

        rekomendasi = (
            "Model dapat dibaca lebih lanjut, tetapi tetap perlu mempertimbangkan teori dan ukuran sampel."
        )

    else:
        status = "Perlu Koreksi / Kehati-hatian"

        daftar_masalah = []

        if ada_heteroskedastisitas:
            daftar_masalah.append("heteroskedastisitas")

        if ada_autokorelasi:
            daftar_masalah.append("autokorelasi")

        if ada_dependensi:
            daftar_masalah.append("dependensi antar-kabupaten")

        if ada_tidak_normal:
            daftar_masalah.append("residual tidak normal")

        kesimpulan = (
            "Terdapat indikasi masalah diagnostik residual, yaitu: "
            + ", ".join(daftar_masalah)
            + "."
        )

        rekomendasi = (
            "Gunakan robust standard error sebagai batas minimum. "
            "Jika autokorelasi atau cross-section dependence muncul kuat, pertimbangkan pendekatan standard error yang lebih sesuai pada tahap lanjutan."
        )

    conclusion = {
        "Status": status,
        "Kesimpulan": kesimpulan,
        "Rekomendasi": rekomendasi,
        "Jumlah Uji Berhasil": df_success.shape[0],
        "Jumlah Masalah": df_reject.shape[0]
    }

    return conclusion