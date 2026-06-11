import numpy as np
import pandas as pd

from scipy.stats import chi2


# =========================
# 1. MEMBUAT BARIS HASIL UJI
# =========================

def make_test_row(
    nama_uji,
    perbandingan,
    h0,
    h1,
    statistik,
    p_value,
    alpha,
    model_terindikasi,
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
        "Uji": nama_uji,
        "Perbandingan": perbandingan,
        "H0": h0,
        "H1": h1,
        "Statistik": statistik,
        "P-value": p_value,
        "Alpha": alpha,
        "Keputusan": keputusan,
        "Model Terindikasi": model_terindikasi,
        "Status": status,
        "Keterangan": keterangan
    }

    return row


# =========================
# 2. UJI CHOW / F-TEST POOLABILITY
# =========================

def run_chow_test(hasil_model, alpha=0.05):
    nama_uji = "Uji Chow / F-test Poolability"
    perbandingan = "CEM vs FEM"
    h0 = "CEM lebih tepat"
    h1 = "FEM lebih tepat"

    fem_result = hasil_model["FEM / Fixed Effect"]

    if isinstance(fem_result, dict):
        row = make_test_row(
            nama_uji,
            perbandingan,
            h0,
            h1,
            None,
            None,
            alpha,
            "-",
            "Gagal",
            "FEM gagal diestimasi sehingga uji Chow tidak dapat dijalankan."
        )

        return row

    if not hasattr(fem_result, "f_pooled"):
        row = make_test_row(
            nama_uji,
            perbandingan,
            h0,
            h1,
            None,
            None,
            alpha,
            "-",
            "Gagal",
            "Objek hasil FEM tidak memiliki atribut f_pooled."
        )

        return row

    statistik = fem_result.f_pooled.stat
    p_value = fem_result.f_pooled.pval

    if p_value < alpha:
        model_terindikasi = "FEM / Fixed Effect"
        keterangan = "H0 ditolak. Terdapat efek individu sehingga FEM lebih tepat dibanding CEM."

    else:
        model_terindikasi = "CEM / Pooled OLS"
        keterangan = "H0 gagal ditolak. CEM masih dapat digunakan dibanding FEM."

    row = make_test_row(
        nama_uji,
        perbandingan,
        h0,
        h1,
        statistik,
        p_value,
        alpha,
        model_terindikasi,
        "Berhasil",
        keterangan
    )

    return row


# =========================
# 3. UJI LM BREUSCH-PAGAN
# =========================

def run_lm_test(hasil_model, alpha=0.05):
    nama_uji = "Uji LM Breusch-Pagan"
    perbandingan = "CEM vs REM"
    h0 = "CEM lebih tepat"
    h1 = "REM lebih tepat"

    cem_result = hasil_model["CEM / Pooled OLS"]

    if isinstance(cem_result, dict):
        row = make_test_row(
            nama_uji,
            perbandingan,
            h0,
            h1,
            None,
            None,
            alpha,
            "-",
            "Gagal",
            "CEM gagal diestimasi sehingga uji LM tidak dapat dijalankan."
        )

        return row

    try:
        residual = cem_result.resids.copy()

        df_residual = residual.reset_index()
        df_residual.columns = [
            "Kabupaten",
            "Tahun",
            "Residual"
        ]

        jumlah_entity = df_residual["Kabupaten"].nunique()

        observasi_per_entity = (
            df_residual
            .groupby("Kabupaten")["Tahun"]
            .nunique()
        )

        jumlah_tahun_unik = observasi_per_entity.nunique()

        if jumlah_tahun_unik == 1:
            jumlah_tahun = observasi_per_entity.iloc[0]

        else:
            jumlah_tahun = observasi_per_entity.mean()

        if jumlah_tahun <= 1:
            row = make_test_row(
                nama_uji,
                perbandingan,
                h0,
                h1,
                None,
                None,
                alpha,
                "-",
                "Gagal",
                "Jumlah periode terlalu kecil untuk menjalankan uji LM."
            )

            return row

        residual_sum_per_entity = (
            df_residual
            .groupby("Kabupaten")["Residual"]
            .sum()
        )

        total_residual_square = (
            df_residual["Residual"] ** 2
        ).sum()

        bagian_dalam = (
            (residual_sum_per_entity ** 2).sum() /
            total_residual_square
        ) - 1

        statistik = (
            jumlah_entity * jumlah_tahun /
            (2 * (jumlah_tahun - 1))
        ) * (bagian_dalam ** 2)

        p_value = chi2.sf(
            statistik,
            1
        )

        if p_value < alpha:
            model_terindikasi = "REM / Random Effect"
            keterangan = "H0 ditolak. Terdapat indikasi random effect sehingga REM lebih tepat dibanding CEM."

        else:
            model_terindikasi = "CEM / Pooled OLS"
            keterangan = "H0 gagal ditolak. CEM masih dapat digunakan dibanding REM."

        row = make_test_row(
            nama_uji,
            perbandingan,
            h0,
            h1,
            statistik,
            p_value,
            alpha,
            model_terindikasi,
            "Berhasil",
            keterangan
        )

        return row

    except Exception as error:
        row = make_test_row(
            nama_uji,
            perbandingan,
            h0,
            h1,
            None,
            None,
            alpha,
            "-",
            "Gagal",
            str(error)
        )

        return row


# =========================
# 4. UJI HAUSMAN
# =========================

def run_hausman_test(hasil_model, alpha=0.05):
    nama_uji = "Uji Hausman"
    perbandingan = "FEM vs REM"
    h0 = "REM lebih tepat"
    h1 = "FEM lebih tepat"

    fem_result = hasil_model["FEM / Fixed Effect"]
    rem_result = hasil_model["REM / Random Effect"]

    if isinstance(fem_result, dict):
        row = make_test_row(
            nama_uji,
            perbandingan,
            h0,
            h1,
            None,
            None,
            alpha,
            "-",
            "Gagal",
            "FEM gagal diestimasi sehingga uji Hausman tidak dapat dijalankan."
        )

        return row

    if isinstance(rem_result, dict):
        row = make_test_row(
            nama_uji,
            perbandingan,
            h0,
            h1,
            None,
            None,
            alpha,
            "-",
            "Gagal",
            "REM gagal diestimasi sehingga uji Hausman tidak dapat dijalankan."
        )

        return row

    try:
        common_variables = []

        for var in fem_result.params.index:
            if var in rem_result.params.index and var != "const":
                common_variables.append(var)

        if len(common_variables) == 0:
            row = make_test_row(
                nama_uji,
                perbandingan,
                h0,
                h1,
                None,
                None,
                alpha,
                "-",
                "Gagal",
                "Tidak ada variabel koefisien yang dapat dibandingkan antara FEM dan REM."
            )

            return row

        beta_fem = fem_result.params[common_variables]
        beta_rem = rem_result.params[common_variables]

        cov_fem = fem_result.cov.loc[
            common_variables,
            common_variables
        ]

        cov_rem = rem_result.cov.loc[
            common_variables,
            common_variables
        ]

        beta_diff = beta_fem - beta_rem
        cov_diff = cov_fem - cov_rem

        inv_cov_diff = np.linalg.pinv(
            cov_diff.values
        )

        statistik = float(
            beta_diff.values.T @ inv_cov_diff @ beta_diff.values
        )

        if statistik < 0:
            statistik = 0

        df_test = len(common_variables)

        p_value = chi2.sf(
            statistik,
            df_test
        )

        if p_value < alpha:
            model_terindikasi = "FEM / Fixed Effect"
            keterangan = "H0 ditolak. FEM lebih tepat karena terdapat indikasi korelasi antara efek individu dan variabel independen."

        else:
            model_terindikasi = "REM / Random Effect"
            keterangan = "H0 gagal ditolak. REM lebih tepat karena tidak terdapat bukti kuat adanya korelasi efek individu dengan variabel independen."

        row = make_test_row(
            nama_uji,
            perbandingan,
            h0,
            h1,
            statistik,
            p_value,
            alpha,
            model_terindikasi,
            "Berhasil",
            keterangan
        )

        return row

    except Exception as error:
        row = make_test_row(
            nama_uji,
            perbandingan,
            h0,
            h1,
            None,
            None,
            alpha,
            "-",
            "Gagal",
            str(error)
        )

        return row


# =========================
# 5. MENJALANKAN SEMUA UJI PEMILIHAN MODEL
# =========================

def run_panel_model_selection_tests(hasil_model, alpha=0.05):
    daftar_hasil = []

    hasil_chow = run_chow_test(
        hasil_model,
        alpha=alpha
    )

    hasil_lm = run_lm_test(
        hasil_model,
        alpha=alpha
    )

    hasil_hausman = run_hausman_test(
        hasil_model,
        alpha=alpha
    )

    daftar_hasil.append(hasil_chow)
    daftar_hasil.append(hasil_lm)
    daftar_hasil.append(hasil_hausman)

    df_hasil_uji = pd.DataFrame(daftar_hasil)

    return df_hasil_uji


# =========================
# 6. MENGAMBIL MODEL TERINDIKASI DARI HASIL UJI
# =========================

def get_indicated_model(df_hasil_uji, nama_uji):
    df_uji = df_hasil_uji[
        df_hasil_uji["Uji"] == nama_uji
    ].copy()

    if df_uji.shape[0] == 0:
        return None

    row = df_uji.iloc[0]

    if row["Status"] != "Berhasil":
        return None

    return row["Model Terindikasi"]


# =========================
# 7. MEMBUAT KESIMPULAN PEMILIHAN MODEL
# =========================

def make_model_selection_conclusion(df_hasil_uji, hasil_model):
    model_chow = get_indicated_model(
        df_hasil_uji,
        "Uji Chow / F-test Poolability"
    )

    model_lm = get_indicated_model(
        df_hasil_uji,
        "Uji LM Breusch-Pagan"
    )

    model_hausman = get_indicated_model(
        df_hasil_uji,
        "Uji Hausman"
    )

    rem_result = hasil_model["REM / Random Effect"]

    if isinstance(rem_result, dict):
        rem_berhasil = False
    else:
        rem_berhasil = True

    model_akhir = "Belum dapat ditentukan"
    alasan = "Tidak seluruh uji pemilihan model dapat digunakan."

    if rem_berhasil is False:
        if model_chow == "FEM / Fixed Effect":
            model_akhir = "FEM / Fixed Effect"
            alasan = "REM gagal diestimasi, sedangkan uji Chow mengarah pada FEM."

        elif model_chow == "CEM / Pooled OLS":
            model_akhir = "CEM / Pooled OLS"
            alasan = "REM gagal diestimasi dan uji Chow tidak menunjukkan kebutuhan fixed effect."

        else:
            model_akhir = "Belum dapat ditentukan"
            alasan = "REM gagal diestimasi dan uji Chow tidak dapat memberikan keputusan."

    else:
        if model_chow == "CEM / Pooled OLS" and model_lm == "CEM / Pooled OLS":
            model_akhir = "CEM / Pooled OLS"
            alasan = "Uji Chow dan uji LM sama-sama mengarah pada CEM."

        elif model_chow == "CEM / Pooled OLS" and model_lm == "REM / Random Effect":
            model_akhir = "REM / Random Effect"
            alasan = "Uji Chow tidak memilih FEM, sedangkan uji LM menunjukkan adanya random effect."

        elif model_chow == "FEM / Fixed Effect" and model_lm == "CEM / Pooled OLS":
            model_akhir = "FEM / Fixed Effect"
            alasan = "Uji Chow menunjukkan adanya efek individu sehingga FEM lebih tepat."

        elif model_chow == "FEM / Fixed Effect" and model_lm == "REM / Random Effect":
            if model_hausman is not None:
                model_akhir = model_hausman
                alasan = "Uji Chow dan LM menunjukkan model panel lebih tepat daripada CEM, sehingga keputusan akhir menggunakan uji Hausman."

            else:
                model_akhir = "Belum dapat ditentukan"
                alasan = "FEM dan REM sama-sama terindikasi, tetapi uji Hausman gagal dijalankan."

    conclusion = {
        "Model Akhir Sementara": model_akhir,
        "Alasan": alasan
    }

    return conclusion