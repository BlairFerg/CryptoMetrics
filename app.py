import math
import random
import time
import requests
import pandas as pd
import streamlit as st
import yfinance as yf  # NEW: for reliable DXY

st.set_page_config(page_title="BTC Top Signals — Live", layout="wide")
st.write("🔧 Initialising…")
st.set_option("client.showErrorDetails", True)

UA = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitBot/1.0)"}

# ------------------ Generic fetch with retries ------------------
def fetch_json(url, params=None, timeout=20, max_retries=4, backoff=0.6):
    last_err = None
    for i in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"{r.status_code} {r.reason}: {r.url}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff * (2 ** i) + random.uniform(0, 0.3))
    raise last_err

# ------------------ Primary data sources: CoinPaprika ------------------
def paprika_global():
    return fetch_json("https://api.coinpaprika.com/v1/global")

def paprika_ticker(coin_id):
    return fetch_json(f"https://api.coinpaprika.com/v1/tickers/{coin_id}")

# ------------------ Fallback for spot prices: Binance ------------------
def binance_price(symbol="BTCUSDT"):
    j = fetch_json("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol})
    return float(j["price"])

# ------------------ DXY via yfinance (with Stooq fallback) -------------
def get_dxy_yf():
    """
    Try common Yahoo Finance symbols for the US Dollar Index.
    Order: ^DXY (index), DX-Y.NYB (ICE index), DX=F (futures).
    Returns latest close as float.
    """
    symbols = ["^DXY", "DX-Y.NYB", "DX=F"]
    for sym in symbols:
        try:
            df = yf.download(sym, period="5d", interval="1d", progress=False, auto_adjust=False)
            if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df:
                val = float(df["Close"].dropna().iloc[-1])
                if val and val == val:
                    return val
        except Exception:
            continue
    raise RuntimeError("yfinance DXY unavailable for symbols ^DXY/DX-Y.NYB/DX=F")

def get_dxy_stooq():
    url = "https://stooq.com/q/d/l/?s=dxy&i=d"
    r = requests.get(url, timeout=20, headers=UA)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError("DXY CSV empty")
    last = lines[-1].split(",")
    return float(last[4])

@st.cache_data(ttl=300)
def get_dxy_cached():
    try:
        return get_dxy_yf()
    except Exception:
        # fallback (non-fatal)
        return get_dxy_stooq()

# ------------------ Cached wrappers for crypto ------------------
@st.cache_data(ttl=300)
def get_global_cached():
    return paprika_global()

@st.cache_data(ttl=300)
def get_ticker_cached(coin_id):
    return paprika_ticker(coin_id)

@st.cache_data(ttl=120)
def get_binance_price_cached(symbol):
    return binance_price(symbol)

# ------------------ Compute metrics (robust) ------------------
def compute_metrics_safe(include_dxy=True):
    issues = []
    metrics = {}

    # Global (total mcap + BTC dominance)
    try:
        g = get_global_cached()
        total_mcap_usd = g.get("market_cap_usd")
        btc_dom_pct = g.get("bitcoin_dominance_percentage")  # %
        btc_dominance = (btc_dom_pct / 100.0) if btc_dom_pct is not None else None
    except Exception as e:
        issues.append(f"Global (CoinPaprika) error: {e}")
        g, total_mcap_usd, btc_dominance = None, None, None

    # Tickers
    btc_mcap = eth_mcap = usdt_mcap = None
    btc_price = eth_price = None

    try:
        tb = get_ticker_cached("btc-bitcoin")
        btc_mcap = tb.get("quotes", {}).get("USD", {}).get("market_cap")
        btc_price = tb.get("quotes", {}).get("USD", {}).get("price") or get_binance_price_cached("BTCUSDT")
    except Exception as e:
        issues.append(f"BTC ticker/price error: {e}")
        try:
            btc_price = get_binance_price_cached("BTCUSDT")
        except Exception as e2:
            issues.append(f"BTC price fallback error: {e2}")

    try:
        te = get_ticker_cached("eth-ethereum")
        eth_mcap = te.get("quotes", {}).get("USD", {}).get("market_cap")
        eth_price = te.get("quotes", {}).get("USD", {}).get("price") or get_binance_price_cached("ETHUSDT")
    except Exception as e:
        issues.append(f"ETH ticker/price error: {e}")
        try:
            eth_price = get_binance_price_cached("ETHUSDT")
        except Exception as e2:
            issues.append(f"ETH price fallback error: {e2}")

    try:
        tu = get_ticker_cached("usdt-tether")
        usdt_mcap = tu.get("quotes", {}).get("USD", {}).get("market_cap")
    except Exception as e:
        issues.append(f"USDT ticker error: {e}")

    # DXY (optional)
    dxy_val = None
    if include_dxy:
        try:
            dxy_val = get_dxy_cached()
        except Exception as e:
            issues.append(f"DXY warning: {e}")
            dxy_val = None

    # Assemble
    try:
        alt_mcap = (total_mcap_usd - btc_mcap - eth_mcap) if (total_mcap_usd and btc_mcap and eth_mcap) else None
        usdt_dom = (usdt_mcap / total_mcap_usd) if (total_mcap_usd and usdt_mcap) else None

        metrics = {
            "btc_price_usd": btc_price,
            "eth_price_usd": eth_price,
            "total_mcap_usd": total_mcap_usd,
            "btc_mcap_usd": btc_mcap,
            "eth_mcap_usd": eth_mcap,
            "usdt_mcap_usd": usdt_mcap,
            "btc_dominance": btc_dominance,  # 0..1
            "usdt_dominance": usdt_dom,      # 0..1
            "alt_mcap_usd": alt_mcap,
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

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("Signal thresholds")
    st.caption("Edit thresholds to flip signals in real time.")

    btc_dom_thr = st.number_input("BTC dominance <", 0.0, 1.0, float(0.45), step=float(0.01), format="%.4f")
    usdt_dom_thr = st.number_input("USDT dominance <", 0.0, 1.0, float(0.03), step=float(0.001), format="%.4f")
    alt_mcap_thr = st.number_input("Alt mcap ≥ (USD)", 0.0, float(10_000_000_000_000), float(1_200_000_000_000), step=float(1_000_000_000), format="%.0f")
    dxy_thr = st.number_input("DXY <", 0.0, 200.0, float(95.0), step=float(1.0), format="%.2f")

    st.markdown("---")
    use_dxy = st.checkbox("Use DXY signal", value=True, help="Disable if sources hiccup.")
    refresh_s = st.slider("Auto-refresh (seconds)", 10, 300, 60)

st.title("BTC TOP Signals — Live Dashboard")

# ------------------ Data Fetch ------------------
metrics, issues = compute_metrics_safe(include_dxy=use_dxy)
if issues:
    st.warning(" · ".join(issues))
if not metrics:
    st.error("Live data unavailable right now. Try again shortly.")
    st.stop()

# ------------------ KPIs ------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("BTC Price (USD)", fmt(metrics.get("btc_price_usd"), "usd0"))
k2.metric("BTC Dominance",   fmt(metrics.get("btc_dominance"), "pct"))
k3.metric("USDT Dominance",  fmt(metrics.get("usdt_dominance"), "pct"))
k4.metric("Alt Market Cap",  fmt(metrics.get("alt_mcap_usd"), "usdT"))
k5.metric("DXY",             fmt(metrics.get("dxy"), "num2"))

# ------------------ Signals table ------------------
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

# ------------------ Session history (ephemeral) ------------------
if "hist" not in st.session_state:
    st.session_state.hist = []
row = {"ts": pd.Timestamp.utcnow()}
row.update({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})
st.session_state.hist.append(row)
st.session_state.hist = st.session_state.hist[-500:]
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

st.caption("DXY via Yahoo Finance with Stooq fallback. Crypto via CoinPaprika + Binance price fallback.")
st.markdown(f'<meta http-equiv="refresh" content="{int(refresh_s)}">', unsafe_allow_html=True)
