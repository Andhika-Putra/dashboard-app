# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------
# Setup halaman
# --------------------------
st.set_page_config(page_title="Bike Sharing Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

# --------------------------
# Load & prep data
# --------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dteday"] = pd.to_datetime(df["dteday"])
    # Mapping kategori
    season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    weather_map = {
        1: "Clear",
        2: "Mist/Cloudy",
        3: "Light Rain/Snow",
        4: "Heavy Rain/Snow",
    }
    df["season"] = df["season"].map(season_map)
    df["weathersit"] = df["weathersit"].map(weather_map)
    df["month"] = df["dteday"].dt.month
    df["year"] = df["dteday"].dt.year
    df["workingday_label"] = df["workingday"].map({0: "Akhir Pekan/Hari Libur", 1: "Hari Kerja"})
    return df

try:
    df_raw = load_data("day.csv")
except FileNotFoundError:
    st.error("File 'day.csv' tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py.")
    st.stop()

df = df_raw.copy()

# --------------------------
# Sidebar filter
# --------------------------
st.sidebar.header("Filter Data")
tahun = st.sidebar.selectbox("Pilih Tahun", options=["All"] + sorted(df["year"].unique().tolist()))
musim_options = ["Spring", "Summer", "Fall", "Winter"]
musim = st.sidebar.multiselect("Pilih Musim", options=musim_options, default=musim_options)

# Terapkan filter
if tahun != "All":
    df = df[df["year"] == tahun]
df = df[df["season"].isin(musim)]

# --------------------------
# Header
# --------------------------
st.title("Dashboard Analisis Bike Sharing (2011–2012)")
st.write(
    "Dashboard ini menampilkan hasil analisis **Bike Sharing Dataset**: "
    "pengaruh faktor cuaca terhadap peminjaman serta perbedaan pola peminjaman antara hari kerja dan akhir pekan. "
    "Gunakan filter di **sidebar** untuk eksplorasi per tahun dan musim."
)

# --------------------------
# Ringkasan metrik
# --------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Hari", len(df))
with col2:
    st.metric("Rata-rata Peminjaman / Hari", f"{df['cnt'].mean():,.0f}")
with col3:
    st.metric("Maksimum Peminjaman / Hari", f"{df['cnt'].max():,}")
with col4:
    st.metric("Minimum Peminjaman / Hari", f"{df['cnt'].min():,}")

st.divider()

# --------------------------
# Pertanyaan 1: Faktor Cuaca
# --------------------------
st.subheader("Pertanyaan 1: Faktor cuaca apa yang paling berpengaruh terhadap jumlah peminjaman sepeda?")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Suhu vs Jumlah Peminjaman**")
    fig, ax = plt.subplots()
    sns.scatterplot(x="temp", y="cnt", data=df, alpha=0.6, ax=ax)
    ax.set_xlabel("Suhu (normalized)")
    ax.set_ylabel("Jumlah Peminjaman (cnt)")
    st.pyplot(fig, use_container_width=True)

with c2:
    st.markdown("**Kelembaban vs Jumlah Peminjaman**")
    fig, ax = plt.subplots()
    sns.scatterplot(x="hum", y="cnt", data=df, alpha=0.6, color="red", ax=ax)
    ax.set_xlabel("Kelembaban (normalized)")
    ax.set_ylabel("Jumlah Peminjaman (cnt)")
    st.pyplot(fig, use_container_width=True)

st.info(
    "Insight: **Suhu** menunjukkan hubungan positif yang jelas dengan `cnt` (semakin hangat, peminjaman meningkat). "
    "**Kelembaban** cenderung menurunkan peminjaman saat nilainya tinggi."
)

st.divider()

# --------------------------
# Pertanyaan 2: Hari Kerja vs Akhir Pekan
# --------------------------
st.subheader("Pertanyaan 2: Bagaimana perbedaan pola peminjaman sepeda antara hari kerja dan akhir pekan?")

fig, ax = plt.subplots()
sns.boxplot(x="workingday_label", y="cnt", data=df, ax=ax)
ax.set_xlabel("Tipe Hari")
ax.set_ylabel("Jumlah Peminjaman (cnt)")
st.pyplot(fig, use_container_width=True)

st.info(
    "Insight: **Akhir pekan/hari libur** cenderung memiliki peminjaman lebih tinggi dan variasi lebih besar "
    "dibandingkan **hari kerja**."
)

st.divider()

# --------------------------
# Tren Bulanan
# --------------------------
st.subheader("Tren Rata-rata Peminjaman per Bulan")
monthly = df.groupby("month", as_index=False)["cnt"].mean()

fig, ax = plt.subplots()
sns.lineplot(x="month", y="cnt", data=monthly, marker="o", ax=ax)
ax.set_xlabel("Bulan (1–12)")
ax.set_ylabel("Rata-rata Peminjaman (cnt)")
st.pyplot(fig, use_container_width=True)

st.caption("Tren menunjukkan nilai tinggi di musim panas dan rendah di musim dingin.")

st.divider()

# --------------------------
# Analisis Lanjutan: Clustering (Binning Quantile)
# --------------------------
st.subheader("Analisis Lanjutan: Clustering Harian Berdasarkan Jumlah Peminjaman")

# Jika data sedikit karena filter, fallback ke cut biasa agar tidak error
try:
    df["cnt_cluster"] = pd.qcut(df["cnt"], q=3, labels=["Rendah", "Sedang", "Tinggi"])
except ValueError:
    # Misalnya data terlalu sedikit / banyak nilai duplikat di batas quantile
    df["cnt_cluster"] = pd.cut(
        df["cnt"],
        bins=3,
        labels=["Rendah", "Sedang", "Tinggi"],
        include_lowest=True
    )

c3, c4 = st.columns(2)

with c3:
    st.markdown("**Distribusi Cluster (Rendah–Sedang–Tinggi)**")
    fig, ax = plt.subplots()
    sns.countplot(x="cnt_cluster", data=df, palette="Set2", ax=ax)
    ax.set_xlabel("Cluster Peminjaman")
    ax.set_ylabel("Jumlah Hari")
    st.pyplot(fig, use_container_width=True)

with c4:
    st.markdown("**Rata-rata Peminjaman per Musim berdasarkan Cluster**")
    fig, ax = plt.subplots()
    sns.barplot(x="season", y="cnt", hue="cnt_cluster", data=df, ci=None, ax=ax)
    ax.set_xlabel("Musim")
    ax.set_ylabel("Rata-rata Peminjaman (cnt)")
    st.pyplot(fig, use_container_width=True)

st.info(
    "Interpretasi: **Cluster Tinggi** lebih sering terjadi pada musim panas; **Cluster Rendah** banyak muncul di musim dingin/cuaca buruk. "
    "Informasi ini membantu perencanaan operasional (penambahan/penyebaran armada)."
)

st.divider()

# --------------------------
# Catatan & Footer
# --------------------------
with st.expander("Catatan Data"):
    st.write(
        "- `temp`, `hum`, `windspeed` berada pada skala **normalized (0–1)**.\n"
        "- `cnt` = `casual` + `registered`.\n"
        "- Data mencakup tahun **2011–2012** (Capital Bikeshare, Washington D.C.)."
    )

st.caption("© 2025 Bike Sharing Analysis • Dibuat dengan Streamlit")
