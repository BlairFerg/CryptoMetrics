import requests
import pandas as pd
import streamlit as st

# ------------------ Page & Settings ------------------
st.set_page_config(page_title="BTC Top Signals — Live", layout="wide")
st.write("🔧 Initialising…")  # ensures something renders even if later fails
st.set_option("client.showErrorDetails", True)

UA = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitBot/1.0)"}

# ------------------ Data Fetch (robust + cached) ------------------
@st.cache_data(ttl=300)
def cg_global():
    """CoinGecko global market data."""
    url = "https://api.coingecko.com/api/v3/global"
    r = requests.get(url, timeout=20, headers=UA)
    r.raise_for_status()
    j = r.json()
    if "data" not in j:
        raise RuntimeError("CoinGecko /global returned no 'data' key")
    return j["data"]

@st.cache_data(ttl=300)
def cg_simple(ids="bitcoin,ethereum,tether", vs="usd"):
    """CoinGecko simple price with market caps."""
    url = "https://api.coingecko.com/api/v3/simple/price"
    r = requests.get(
        url,
        params={"ids": ids, "vs_currencies": vs, "include_market_cap": "true"},
        timeout=20,
        headers=UA,
    )
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=300)
def get_dxy():
    """
    Free daily DXY CSV (Stooq). May be flaky; caller must handle None.
    Format: Date,Open,High,Low,Close
    """
    url = "https://stooq.com/q/d/l/?s=dxy&i=d"
    r = requests.get(url, timeout=20, headers=UA)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError("DXY CSV empty")
    last = lines[-1].split(",")
    return float(last[4])  # Close

def compute_metrics_safe(include_dxy=True):
    """
    Fetch metrics, degrade gracefully if any source fails.
    Returns (metrics: dict, issues: list[str])
    """
    issues = []
    g = p = None
    dxy_val = None

    # CoinGecko global
    try:
        g = cg_global()
    except Exception as e:
        issues.append(f"CoinGecko global error: {e}")

    # CoinGecko simple prices
    try:
        p = cg_simple()
    except Exception as e:
        issues.append(f"CoinGecko prices error: {e}")

    # Optional: DXY
    if include_dxy:
        try:
            dxy_val = get_dxy()
        except Exception as e:
            issues.append(f"DXY warning: {e}")
            dxy_val = None

    metrics = {}
    if g and p:
        total = g["total_market_cap"].get("usd")
        btc_m = p.get("bitcoin", {}).get("usd_market_cap")
        eth_m = p.get("ethereum", {}).get("usd_market_cap")
        usdt_m = p.get("tether", {}).get("usd_market_cap")
        btc_price = p.get("bitcoin", {}).get("usd")
        eth_price = p.get("ethereum", {}).get("usd")

        try:
            metrics = {
                "btc_price_usd": btc_price,
                "eth_price_usd": eth_price,
                "total_mcap_usd": total,
                "btc_mcap_usd": btc_m,
                "eth_mcap_usd": eth_m,
                "usdt_mcap_usd": usdt_m,
                "btc_dominance": (
                    g["market_cap_percentage"]["bitcoin"] / 100.0
                    if g.get("market_cap_percentage", {}).get("bitcoin") is not None
                    else None
                ),  # 0..1
                "usdt_dominance": (usdt_m / total) if (usdt_m and total) else None,  # 0..1
                "alt_mcap_usd": (total - btc_m - eth_m) if (total and btc_m and eth_m) else None,
                "dxy": dxy_val,
            }
        except Exception as e:
            issues.append(f"Metric assembly error: {e}")

    return metrics, issues

# ------------------ Helpers ------------------
def tick(ok: bool) -> str:
    return "✅" if ok else "❌"

def check(val, op, thr):
    if val is None or thr is None:
        return False
    if op == "<":  return val < thr
    if op == "<=": return val <= thr
    if op == ">":  return val > thr
    if op == ">=": return val >= thr
    if op == "==": return val == thr
    return False

def fmt(v, kind="num"):
    if v is None:
        return "—"
    if kind == "usd0":
        return f"{v:,.0f}"
    if kind == "usdT":
        return f"{v/1e12:,.2f}T"
    if kind == "pct":
        return f"{v*100:,.2f}%"
    if kind == "num2":
        return f"{v:,.2f}"
    return f"{v}"

# ------------------ Sidebar Controls ------------------
with st.sidebar:
    st.header("Signal thresholds")
    st.caption("Edit thresholds to flip signals in real time.")

    btc_dom_thr = st.number_input(
        "BTC dominance <",
        min_value=0.0, max_value=1.0,
        value=float(0.45), step=float(0.01), format="%.4f"
    )

    usdt_dom_thr = st.number_input(
        "USDT dominance <",
        min_value=0.0, max_value=1.0,
        value=float(0.03), step=float(0.001), format="%.4f"
    )

    alt_mcap_thr = st.number_input(
        "Alt mcap ≥ (USD)",
        min_value=0.0, max_value=float(10_000_000_000_000),
        value=float(1_200_000_000_000), step=float(1_000_000_000),
        format="%.0f"
    )

    dxy_thr = st.number_input(
        "DXY <",
        min_value=0.0, max_value=200.0,
        value=float(95.0), step=float(1.0), format="%.2f"
    )

    st.markdown("---")
    use_dxy = st.checkbox("Use DXY signal", value=True, help="Untick if the free DXY feed is flaky.")
    refresh_s = st.slider("Auto-refresh (seconds)", 10, 300, 60)
    st.caption("If the screen ever goes blank: Menu ▸ Rerun, or disable DXY.")

st.title("BTC TOP Signals — Live Dashboard")

# ------------------ Data Fetch ------------------
metrics, issues = compute_metrics_safe(include_dxy=use_dxy)

if issues:
    st.warning(" · ".join(issues))

if not metrics:
    st.error("Live data unavailable right now. The app is up, but data sources failed. Try again shortly.")
    st.stop()

# ------------------ KPI Row ------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("BTC Price (USD)", fmt(metrics.get("btc_price_usd"), "usd0"))
k2.metric("BTC Dominance",   fmt(metrics.get("btc_dominance"), "pct"))
k3.metric("USDT Dominance",  fmt(metrics.get("usdt_dominance"), "pct"))
k4.metric("Alt Market Cap",  fmt(metrics.get("alt_mcap_usd"), "usdT"))
k5.metric("DXY",             fmt(metrics.get("dxy"), "num2"))

# ------------------ Signals Table ------------------
signals = [
    ("BTC dominance under 45%",  "btc_dominance", "<",  btc_dom_thr),
    ("USDT dominance below 3%",  "usdt_dominance","<",  usdt_dom_thr),
    ("Alt market cap ≥ target",  "alt_mcap_usd",  ">=", alt_mcap_thr),
    ("DXY falls below target",   "dxy",           "<",  dxy_thr if use_dxy else None),
]

sig_df = pd.DataFrame([{
    "Signal": s[0],
    "Metric key": s[1],
    "Value": metrics.get(s[1]),
    "Operator": s[2],
    "Threshold": s[3],
    "Status": tick(check(metrics.get(s[1]), s[2], s[3]) if s[3] is not None else False)
} for s in signals])

st.subheader("Signals")
st.dataframe(sig_df, use_container_width=True)

# ------------------ Session History (ephemeral) ------------------
if "hist" not in st.session_state:
    st.session_state.hist = []

row = {"ts": pd.Timestamp.utcnow()}
row.update({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})
st.session_state.hist.append(row)
st.session_state.hist = st.session_state.hist[-500:]  # keep last 500 samples

hist_df = pd.DataFrame(st.session_state.hist).set_index("ts")

st.subheader("Trends")
c1, c2, c3 = st.columns(3)
with c1:
    st.line_chart(hist_df[["btc_dominance"]], height=220)
with c2:
    st.line_chart(hist_df[["usdt_dominance"]], height=220)
with c3:
    if use_dxy and "dxy" in hist_df:
        st.line_chart(hist_df[["dxy"]], height=220)
    else:
        st.info("DXY disabled; enable from the sidebar to view.")

st.line_chart(
    hist_df[["alt_mcap_usd", "btc_mcap_usd", "eth_mcap_usd"]].dropna(how="all"),
    height=280
)

st.caption("Data cached for 5 minutes. Adjust thresholds on the left. If free sources hiccup, you'll see a warning instead of a blank page.")

# ------------------ Simple Auto-refresh ------------------
st.markdown(f'<meta http-equiv="refresh" content="{int(refresh_s)}">', unsafe_allow_html=True)
