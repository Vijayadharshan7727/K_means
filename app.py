import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
body {
    background-color: #f6f7fb;
}
.title {
    font-size: 50px;
    font-weight: 800;
    background: linear-gradient(90deg,#ff4b4b,#ff914d);
    -webkit-background-clip: text;
    color: transparent;
    text-align: center;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6c757d;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    text-align: center;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<div class='title'>🛍️ Mall Customer Segmentation</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Creative KMeans Clustering | Data Science Dashboard</div>", unsafe_allow_html=True)
st.write("")

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

st.write("")

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

fig, ax = plt.subplots(figsize=(7,5))
sns.scatterplot(
    data=data,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set2",
    ax=ax
)
ax.set_title("Customer Clusters")
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
        2: "🔥 Premium customers (high income & spending)",
        3: "📉 Low income, low spending customers",
        4: "🎉 Low income, high spending customers"
    }

    st.info(cluster_desc[cluster])

# ================= FOOTER =================
st.divider()
st.markdown("<div class='footer'>Made with ❤️ by <b>Vijay</b> | Data Science Project</div>", unsafe_allow_html=True)
