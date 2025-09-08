import requests, pandas as pd, time
import streamlit as st

st.set_page_config(page_title="BTC Top Signals", layout="wide")

@st.cache_data(ttl=300)
def cg_global():
    return requests.get("https://api.coingecko.com/api/v3/global").json()["data"]

@st.cache_data(ttl=300)
def cg_simple(ids="bitcoin,ethereum,tether", vs="usd"):
    url = "https://api.coingecko.com/api/v3/simple/price"
    return requests.get(url, params={
        "ids": ids, "vs_currencies": vs, "include_market_cap": "true"
    }).json()

@st.cache_data(ttl=300)
def get_dxy():
    # Stooq CSV
    csv = requests.get("https://stooq.com/q/d/l/?s=dxy&i=d").text.strip().splitlines()
    last = csv[-1].split(",")
    return float(last[4])  # Close

def compute_metrics():
    g = cg_global()
    prices = cg_simple()
    total = g["total_market_cap"]["usd"]
    btc_m = prices["bitcoin"]["usd_market_cap"]
    eth_m = prices["ethereum"]["usd_market_cap"]
    usdt_m = prices["tether"]["usd_market_cap"]
    return {
        "btc_price_usd": prices["bitcoin"]["usd"],
        "eth_price_usd": prices["ethereum"]["usd"],
        "total_mcap_usd": total,
        "btc_mcap_usd": btc_m,
        "eth_mcap_usd": eth_m,
        "usdt_mcap_usd": usdt_m,
        "btc_dominance": g["market_cap_percentage"]["bitcoin"]/100.0,
        "usdt_dominance": usdt_m/total,
        "alt_mcap_usd": total - btc_m - eth_m,
        "dxy": get_dxy()
    }

# === UI ===
st.title("BTC TOP Signals — Live")

# Threshold editor (persist in session)
defaults = {
    "btc_dominance <": 0.45,
    "usdt_dominance <": 0.03,
    "alt_mcap_usd >=": 1_200_000_000_000,
    "dxy <": 95
}
with st.sidebar:
    st.header("Signal thresholds")
    thr = {}
    thr["btc_dominance <"] = st.number_input("BTC dom <", value=defaults["btc_dominance <"], step=0.01, format="%.4f")
    thr["usdt_dominance <"] = st.number_input("USDT dom <", value=defaults["usdt_dominance <"], step=0.001, format="%.4f")
    thr["alt_mcap_usd >="] = st.number_input("Alt mcap >=", value=defaults["alt_mcap_usd >="], step=1e9, format="%.0f")
    thr["dxy <"] = st.number_input("DXY <", value=defaults["dxy <"], step=1.0, format="%.2f")

m = compute_metrics()
st.subheader("Current metrics")
st.json(m)

def check(val, op, t):
    if op == "<":  return val < t
    if op == "<=": return val <= t
    if op == ">":  return val > t
    if op == ">=": return val >= t
    if op == "==": return val == t
    return False

signals = [
    ("BTC dominance under 45%",        "btc_dominance", "<",  thr["btc_dominance <"]),
    ("USDT dominance below 3%",        "usdt_dominance","<",  thr["usdt_dominance <"]),
    ("Alt mcap crosses threshold",     "alt_mcap_usd",  ">=", thr["alt_mcap_usd >="]),
    ("DXY falls below threshold",      "dxy",           "<",  thr["dxy <"])
]

st.subheader("Signals")
sig_df = pd.DataFrame([{
    "signal": s[0],
    "metric_key": s[1],
    "value": m[s[1]],
    "operator": s[2],
    "threshold": s[3],
    "status": "√" if check(m[s[1]], s[2], s[3]) else "X"
} for s in signals])

st.dataframe(sig_df, use_container_width=True)

st.subheader("Visuals")
c1, c2 = st.columns(2)
with c1:
    st.line_chart(pd.DataFrame({"BTC Dominance": [m["btc_dominance"]]}))
with c2:
    st.bar_chart(pd.DataFrame({"Alt Market Cap (USD)": [m["alt_mcap_usd"]]}))

st.caption("Updates cached every 5 minutes.")

