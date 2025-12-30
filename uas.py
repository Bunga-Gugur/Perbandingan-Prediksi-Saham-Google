import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================
# CONFIG STREAMLIT
# ==========================
st.set_page_config(
    page_title="Dashboard Prediksi Saham",
    layout="wide"
)

st.title("📈 Dashboard Prediksi Harga Saham")
st.markdown("Aplikasi prediksi harga saham menggunakan **Regresi Linier** dan data dari **Yahoo Finance**.")

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("⚙️ Pengaturan")

ticker = st.sidebar.text_input(
    "Masukkan Kode Saham (Ticker)",
    value="GOOGL"
)

forecast_days = st.sidebar.slider(
    "Jumlah Hari Prediksi ke Depan",
    min_value=1,
    max_value=30,
    value=5
)

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Tentang Aplikasi**\n\n"
    "Dashboard ini mengambil data saham historis dari Yahoo Finance "
    "dan memprediksi harga penutupan (Close) menggunakan model Regresi Linier."
)

# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2018-01-01")
    return data

try:
    data = load_data(ticker)

    if data.empty:
        st.error("Data tidak ditemukan. Periksa kembali kode saham.")
        st.stop()

    st.subheader(f"📊 Data Historis Saham: {ticker}")
    st.dataframe(data.tail())

    # ==========================
    # DOWNLOAD DATA CSV
    # ==========================
    csv = data.to_csv().encode("utf-8")
    st.download_button(
        label="⬇️ Download Data Historis (CSV)",
        data=csv,
        file_name=f"{ticker}_historical_data.csv",
        mime="text/csv"
    )

    # ==========================
    # PREPROCESSING
    # ==========================
    df = data[['Close']].copy()
    df['Prediction'] = df['Close'].shift(-forecast_days)
    df.dropna(inplace=True)

    X = np.array(df[['Close']])
    y = np.array(df['Prediction'])

    # ==========================
    # TRAIN TEST SPLIT
    # ==========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ==========================
    # MODELING
    # ==========================
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ==========================
    # METRICS
    # ==========================
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")

    # ==========================
    # VISUALIZATION
    # ==========================
    st.subheader("📉 Visualisasi Harga Asli vs Prediksi")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=y_test,
        mode='lines',
        name='Harga Asli'
    ))

    fig.add_trace(go.Scatter(
        y=y_pred,
        mode='lines',
        name='Harga Prediksi'
    ))

    fig.update_layout(
        xaxis_title="Index Data",
        yaxis_title="Harga Saham",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # FUTURE PREDICTION
    # ==========================
    st.subheader(f"🔮 Prediksi {forecast_days} Hari ke Depan")

    last_close = np.array(df[['Close']].tail(forecast_days))
    future_prediction = model.predict(last_close)

    future_df = pd.DataFrame({
        "Hari": range(1, forecast_days + 1),
        "Harga Prediksi": future_prediction
    })

    st.dataframe(future_df)

    # ==========================
    # EXPLANATION
    # ==========================
    st.markdown("---")
    st.subheader("📘 Cara Kerja Regresi Linier")

    st.markdown(
        """
        **Regresi Linier** adalah metode statistik yang digunakan untuk memodelkan hubungan
        antara variabel independen (harga penutupan sebelumnya) dan variabel dependen
        (harga di masa depan).

        Dalam aplikasi ini:
        - Model mempelajari pola hubungan harga saham berdasarkan data historis.
        - Harga *Close* digunakan sebagai input utama.
        - Data digeser (*shift*) untuk memprediksi harga beberapa hari ke depan.
        - Model menghasilkan garis tren linier yang digunakan untuk melakukan prediksi.

        ⚠️ **Catatan:**  
        Prediksi ini bersifat edukatif dan tidak dapat dijadikan satu-satunya dasar
        pengambilan keputusan investasi.
        """
    )

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
