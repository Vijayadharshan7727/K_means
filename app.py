import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🌙",
    layout="wide"
)

# ================= DARK THEME CSS =================
st.markdown("""
<style>

/* App background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #f8fafc;
}

/* Title */
.title {
    font-size: 52px;
    font-weight: 900;
    background: linear-gradient(90deg,#38bdf8,#a78bfa,#f472b6);
    -webkit-background-clip: text;
    color: transparent;
    text-align: center;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #cbd5f5;
    margin-bottom: 25px;
}

/* KPI Cards */
.card {
    background: linear-gradient(145deg,#020617,#020617);
    padding: 30px;
    border-radius: 22px;
    box-shadow: 0px 15px 35px rgba(56,189,248,0.15);
    text-align: center;
    border: 1px solid #1e293b;
    transition: 0.3s;
    color: #f8fafc;
}

.card:hover {
    transform: translateY(-6px) scale(1.03);
    box-shadow: 0px 20px 45px rgba(168,139,250,0.3);
}

.card b {
    color: #e5e7eb;
    font-size: 18px;
}

.card h2 {
    color: #ffffff;
    font-size: 40px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020617,#020617);
    border-right: 2px solid #1e293b;
}

/* Sidebar text */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #f8fafc !important;
    font-weight: 600;
}

/* Slider value */
section[data-testid="stSidebar"] .stSlider p {
    color: #e5e7eb !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg,#38bdf8,#a78bfa);
    color: #020617;
    font-size: 18px;
    font-weight: 800;
    border-radius: 14px;
    padding: 12px 28px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.06);
    box-shadow: 0px 12px 25px rgba(56,189,248,0.5);
}

/* Section headers */
h3 {
    color: #f8fafc;
    font-size: 26px;
    font-weight: 800;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 15px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<div class='title'>🌙 Mall Customer Segmentation</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Dark Theme | KMeans Clustering Dashboard</div>", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

data = load_data()

X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# ================= SCALING & MODEL =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
data["Cluster"] = kmeans.fit_predict(X_scaled)

# ================= KPI CARDS =================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"<div class='card'>👥<br><b>Total Customers</b><h2>{len(data)}</h2></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'>📊<br><b>Features Used</b><h2>2</h2></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'>🎯<br><b>Clusters</b><h2>5</h2></div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='card'>🤖<br><b>Algorithm</b><h2>KMeans</h2></div>", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("🎛️ Customer Input")

income = st.sidebar.slider(
    "💰 Annual Income (k$)",
    int(X["Annual Income (k$)"].min()),
    int(X["Annual Income (k$)"].max()),
    60
)

spending = st.sidebar.slider(
    "🛒 Spending Score (1–100)",
    int(X["Spending Score (1-100)"].min()),
    int(X["Spending Score (1-100)"].max()),
    50
)

# ================= VISUALIZATION =================
st.subheader("📈 Customer Distribution")

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7,5))
sns.scatterplot(
    data=data,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="bright",
    ax=ax
)
ax.set_title("Customer Clusters", color="white")
ax.set_xlabel("Annual Income (k$)", color="white")
ax.set_ylabel("Spending Score", color="white")
st.pyplot(fig)

# ================= PREDICTION =================
st.subheader("🔮 Predict Customer Segment")

if st.button("✨ Predict Customer Type"):
    user_df = pd.DataFrame(
        [[income, spending]],
        columns=["Annual Income (k$)", "Spending Score (1-100)"]
    )

    user_scaled = scaler.transform(user_df)
    cluster = kmeans.predict(user_scaled)[0]

    st.success(f"🎯 Customer belongs to **Cluster {cluster}**")

    cluster_desc = {
        0: "🧠 Careful & balanced customers",
        1: "💰 High income, low spending customers",
        2: "🔥 Premium customers",
        3: "📉 Low income, low spending customers",
        4: "🎉 Low income, high spending customers"
    }

    st.info(cluster_desc[cluster])

# ================= FOOTER =================
st.divider()
st.markdown("<div class='footer'>Made with ❤️ by <b>Vijay</b> | Dark Theme Data Science Project</div>", unsafe_allow_html=True)
