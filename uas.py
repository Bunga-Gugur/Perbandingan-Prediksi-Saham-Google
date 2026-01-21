import json
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px


def load_and_normalize(path: Path, model_name: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # detect columns
    pred_col = next((c for c in df.columns if "pred" in c.lower()), None)
    mae_col = next((c for c in df.columns if c.lower().startswith("mae")), None)
    rmse_col = next((c for c in df.columns if "rmse" in c.lower()), None)
    rename_map = {}
    if pred_col:
        rename_map[pred_col] = f"predicted_{model_name}"
    if mae_col:
        rename_map[mae_col] = f"mae_{model_name}"
    if rmse_col:
        rename_map[rmse_col] = f"rmse_{model_name}"
    df = df.rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date"])
    # ensure real_price exists
    if "real_price" not in df.columns:
        st.warning(f"'real_price' column not found in {path}")
    cols = [c for c in ["date", "real_price"] + list(rename_map.values()) if c in df.columns]
    return df[cols]


def main():
    st.title("Perbandingan Model SimpleRNN vs GRU vs LSTM")

    base = Path(__file__).parent
    files = {
        "SimpleRNN": base / "hasil_simplernn2.json",
        "GRU": base / "hasil_gru2.json",
        "LSTM": base / "hasil_lstm2.json",
    }

    st.sidebar.header("Sumber data")
    for name, p in files.items():
        st.sidebar.write(f"- {name}: {p.name} ({'found' if p.exists() else 'missing'})")

    dfs = {}
    for name, p in files.items():
        if p.exists():
            dfs[name] = load_and_normalize(p, name.lower())
        else:
            st.error(f"File not found: {p}")
            return

    # Merge on date
    merged = dfs["SimpleRNN"][["date", "real_price"]].copy()
    for name, df in dfs.items():
        # drop duplicate real_price if present
        to_merge = df.drop(columns=["real_price"]) if "real_price" in df.columns else df
        merged = merged.merge(to_merge, on="date", how="left")

    merged = merged.sort_values("date").reset_index(drop=True)

    st.subheader("Ringkasan metrik rata-rata per model")
    metrics = []
    for name in files.keys():
        mae_col = f"mae_{name.lower()}"
        rmse_col = f"rmse_{name.lower()}"
        pred_col = f"predicted_{name.lower()}"
        if pred_col in merged.columns:
            mape = ((merged[pred_col] - merged["real_price"]).abs() / merged["real_price"]).mean() * 100
        else:
            mape = None
        metrics.append({
            "model": name,
            "mae": merged[mae_col].mean() if mae_col in merged.columns else None,
            "rmse": merged[rmse_col].mean() if rmse_col in merged.columns else None,
            "mape_%": mape,
        })

    metrics_df = pd.DataFrame(metrics)
    st.table(metrics_df.set_index("model"))

    st.subheader("Grafik: Harga riil vs Prediksi")
    # prepare long format for plotting
    plot_df = merged[["date", "real_price"] + [c for c in merged.columns if c.startswith("predicted_")]]
    long = plot_df.melt(id_vars="date", var_name="series", value_name="price")
    # rename series to nicer labels
    long["model"] = long["series"].str.replace("predicted_", "", regex=False)
    fig = px.line(long, x="date", y="price", color="model", title="Real vs Predicted")
    # add real price as separate series if not included
    fig.update_traces()
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Perbandingan MAE / RMSE")
    metric_plot = metrics_df.melt(id_vars="model", value_vars=["mae", "rmse"], var_name="metric", value_name="value")
    fig2 = px.bar(metric_plot, x="model", y="value", color="metric", barmode="group", title="MAE & RMSE rata-rata")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Tabel data (potong 200 baris teratas)")
    st.dataframe(merged.head(200))

    st.subheader("Analisis residual singkat")
    select_model = st.selectbox("Pilih model untuk residual", list(files.keys()))
    pred_col = f"predicted_{select_model.lower()}"
    if pred_col in merged.columns:
        merged["residual"] = merged[pred_col] - merged["real_price"]
        fig3 = px.histogram(merged, x="residual", nbins=50, title=f"Distribusi residual - {select_model}")
        st.plotly_chart(fig3, use_container_width=True)
        fig4 = px.scatter(merged, x="real_price", y=pred_col, trendline="ols", title=f"Prediksi vs Riil - {select_model}")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Prediksi untuk model ini tidak tersedia di data.")


if __name__ == "__main__":
    main()
