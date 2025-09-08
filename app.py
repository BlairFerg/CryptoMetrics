# app.py — BTC TOP Signals + Hugging Face Space assistant (free data only)
# Data: CoinPaprika (global+tickers), Binance (price fallback), yfinance (DXY + BTC MAs)
# Assistant: calls your public HF Space via gradio_client (prefer the .hf.space URL to avoid Hub API 429s)

import json
import random
import time
import threading
import queue
from typing import Tuple

import requests
import pandas as pd
import streamlit as st
import yfinance as yf
from gradio_client import Client

st.set_page_config(page_title="BTC Top Signals — Live", layout="wide")
st.write("🔧 Initialising…")
st.set_option("client.showErrorDetails", True)

UA = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitBot/1.0)"}

# ------------------ HTTP helper (retries) ------------------
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

# ------------------ Crypto (CoinPaprika + Binance fallback) ------------------
def paprika_global():
    return fetch_json("https://api.coinpaprika.com/v1/global")

def paprika_ticker(coin_id):
    return fetch_json(f"https://api.coinpaprika.com/v1/tickers/{coin_id}")

def binance_price(symbol="BTCUSDT"):
    j = fetch_json("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol})
    return float(j["price"])

# ------------------ DXY via yfinance (with Stooq fallback) ------------------
def get_dxy_yf():
    # Prefer ICE index, then futures, then ^DXY (often blocked)
    for sym in ["DX-Y.NYB", "DX=F", "^DXY"]:
        try:
            df = yf.download(sym, period="5d", interval="1d", progress=False, auto_adjust=False)
            close_series = df["Close"].dropna()
            if not close_series.empty:
                last = close_series.iloc[-1]
                return float(last.item() if hasattr(last, "item") else last)
        except Exception:
            continue
    raise RuntimeError("yfinance DXY unavailable for DX-Y.NYB/DX=F/^DXY")

def get_dxy_stooq():
    url = "https://stooq.com/q/d/l/?s=dxy&i=d"
    r = requests.get(url, timeout=20, headers=UA)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError("DXY CSV empty")
    return float(lines[-1].split(",")[4])

@st.cache_data(ttl=300)
def get_dxy_cached():
    try:
        return get_dxy_yf()
    except Exception:
        try:
            return get_dxy_stooq()
        except Exception:
            return None

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

# ------------------ BTC history for moving averages ------------------
@st.cache_data(ttl=600)
def get_btc_history():
    # 400d for SMA200/50; extract scalars without FutureWarnings
    df = yf.download("BTC-USD", period="400d", interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("BTC-USD history empty")

    close = df["Close"].dropna()
    sma50_series  = close.rolling(50).mean().dropna()
    sma200_series = close.rolling(200).mean().dropna()

    if sma50_series.empty or sma200_series.empty:
        raise RuntimeError("Not enough data for SMA50/SMA200 yet")

    price  = close.iloc[-1]
    sma50  = sma50_series.iloc[-1]
    sma200 = sma200_series.iloc[-1]

    def to_float(x):
        return float(x.item() if hasattr(x, "item") else x)

    price_f  = to_float(price)
    sma50_f  = to_float(sma50)
    sma200_f = to_float(sma200)

    return {
        "btc_close": price_f,
        "sma50": sma50_f,
        "sma200": sma200_f,
        "price_to_sma200": (price_f / sma200_f) if sma200_f else None
    }

# ------------------ Compute metrics (free + robust) ------------------
def compute_metrics_safe(include_dxy=True):
    issues, metrics = [], {}
    # Global (total mcap + BTC dominance)
    try:
        g = get_global_cached()
        total_mcap = g.get("market_cap_usd")
        btc_dom_pct = g.get("bitcoin_dominance_percentage")  # %
        btc_dominance = (btc_dom_pct / 100.0) if btc_dom_pct is not None else None
    except Exception as e:
        issues.append(f"Global (CoinPaprika) error: {e}")
        total_mcap, btc_dominance = None, None

    # Tickers (BTC/ETH/USDT/USDC/DAI)
    btc_mcap = eth_mcap = usdt_mcap = usdc_mcap = dai_mcap = None
    btc_price = eth_price = None

    try:
        tb = get_ticker_cached("btc-bitcoin")
        btc_mcap = tb.get("quotes", {}).get("USD", {}).get("market_cap")
        btc_price = tb.get("quotes", {}).get("USD", {}).get("price") or get_binance_price_cached("BTCUSDT")
    except Exception as e:
        issues.append(f"BTC ticker error: {e}")
        try:
            btc_price = get_binance_price_cached("BTCUSDT")
        except Exception as e2:
            issues.append(f"BTC price fallback error: {e2}")

    try:
        te = get_ticker_cached("eth-ethereum")
        eth_mcap = te.get("quotes", {}).get("USD", {}).get("market_cap")
        eth_price = te.get("quotes", {}).get("USD", {}).get("price") or get_binance_price_cached("ETHUSDT")
    except Exception as e:
        issues.append(f"ETH ticker error: {e}")
        try:
            eth_price = get_binance_price_cached("ETHUSDT")
        except Exception as e2:
            issues.append(f"ETH price fallback error: {e2}")

    try:
        tu = get_ticker_cached("usdt-tether")
        usdt_mcap = tu.get("quotes", {}).get("USD", {}).get("market_cap")
    except Exception as e:
        issues.append(f"USDT ticker error: {e}")

    try:
        tuc = get_ticker_cached("usdc-usd-coin")
        usdc_mcap = tuc.get("quotes", {}).get("USD", {}).get("market_cap")
    except Exception as e:
        issues.append(f"USDC ticker error: {e}")

    try:
        tdai = get_ticker_cached("dai-dai")
        dai_mcap = tdai.get("quotes", {}).get("USD", {}).get("market_cap")
    except Exception as e:
        issues.append(f"DAI ticker error: {e}")

    # DXY (optional)
    dxy_val = None
    if include_dxy:
        dxy_val = get_dxy_cached()
        if dxy_val is None:
            issues.append("DXY warning: source unavailable")

    # Derived
    alt_mcap = (total_mcap - (btc_mcap or 0) - (eth_mcap or 0)) if total_mcap else None
    stable_mcap = sum(v for v in [usdt_mcap, usdc_mcap, dai_mcap] if v)
    stable_dom = (stable_mcap / total_mcap) if (stable_mcap and total_mcap) else None
    eth_btc_ratio = (eth_price / btc_price) if (eth_price and btc_price) else None

    # Moving averages (BTC)
    try:
        hist = get_btc_history()
    except Exception as e:
        issues.append(f"SMA fetch error: {e}")
        hist = {"btc_close": None, "sma50": None, "sma200": None, "price_to_sma200": None}

    sma50_over_sma200 = None
    if hist["sma50"] and hist["sma200"]:
        sma50_over_sma200 = 1.0 if hist["sma50"] >= hist["sma200"] else 0.0

    metrics.update({
        "btc_price_usd": btc_price,
        "eth_price_usd": eth_price,
        "total_mcap_usd": total_mcap,
        "btc_mcap_usd": btc_mcap,
        "eth_mcap_usd": eth_mcap,
        "usdt_mcap_usd": usdt_mcap,
        "alt_mcap_usd": alt_mcap,
        "stable_mcap_usd": stable_mcap,
        "btc_dominance": btc_dominance,     # 0..1
        "usdt_dominance": (usdt_mcap / total_mcap) if (usdt_mcap and total_mcap) else None,  # 0..1
        "stable_dominance": stable_dom,     # 0..1
        "eth_btc_ratio": eth_btc_ratio,
        "dxy": dxy_val,
        "btc_sma50": hist["sma50"],
        "btc_sma200": hist["sma200"],
        "btc_price_to_sma200": hist["price_to_sma200"],  # >= 1 => above 200DMA
        "sma50_over_sma200": sma50_over_sma200          # 1 if golden-cross state
    })

    return metrics, issues

# ------------------ Logic & formatting ------------------
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
    if v is None: return "—"
    if kind == "usd0": return f"{v:,.0f}"
    if kind == "usdT": return f"{v/1e12:,.2f}T"
    if kind == "pct":  return f"{v*100:,.2f}%"
    if kind == "num2": return f"{v:,.2f}"
    return f"{v}"

def pretty_value(key, val):
    if val is None:
        return "—"
    if key in ("btc_dominance", "usdt_dominance", "stable_dominance"):
        return f"{val*100:.2f}%"
    if key in ("alt_mcap_usd","btc_mcap_usd","eth_mcap_usd","stable_mcap_usd","total_mcap_usd"):
        return f"{val/1e12:.2f}T" if val >= 1e12 else f"{val/1e9:.2f}B"
    if key in ("eth_btc_ratio","btc_price_to_sma200"):
        return f"{val:.4f}"
    if key == "sma50_over_sma200":
        return "On" if val and val >= 1 else "Off"
    if key == "dxy":
        return f"{val:.2f}"
    return f"{val}"

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("Signal thresholds")
    st.caption("Edit thresholds to flip signals in real time.")

    btc_dom_thr    = st.number_input("BTC dominance <", 0.0, 1.0, 0.45, step=0.01, format="%.4f")
    usdt_dom_thr   = st.number_input("USDT dominance <", 0.0, 1.0, 0.03,  step=0.001, format="%.4f")
    stable_dom_thr = st.number_input("Stablecoin dominance <", 0.0, 1.0, 0.08, step=0.001, format="%.4f")
    alt_mcap_thr   = st.number_input("Alt mcap ≥ (USD)", 0.0, float(10_000_000_000_000), float(1_200_000_000_000), step=float(1_000_000_000), format="%.0f")
    dxy_thr        = st.number_input("DXY <", 0.0, 200.0, 95.0, step=1.0, format="%.2f")
    ethbtc_thr     = st.number_input("ETH/BTC ratio ≥", 0.0, 1.0, 0.06, step=0.001, format="%.4f")
    sma200_thr     = st.number_input("BTC price / 200DMA ≥", 0.0, 5.0, 1.00, step=0.01, format="%.2f")
    golden_thr     = st.number_input("Golden cross (SMA50 ≥ SMA200) ≥", 0.0, 1.0, 1.0, step=1.0, format="%.0f")

    st.markdown("---")
    use_dxy   = st.checkbox("Use DXY signal", value=True)
    refresh_s = st.slider("Auto-refresh (seconds)", 10, 300, 60)

# ------------------ Data fetch & KPIs ------------------
st.title("BTC TOP Signals — Live Dashboard")

pull_ts_utc = pd.Timestamp.utcnow()
metrics, issues = compute_metrics_safe(include_dxy=use_dxy)
if issues:
    st.warning(" · ".join(issues))
if not metrics:
    st.error("Live data unavailable right now. Try again shortly.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("BTC Price (USD)", fmt(metrics.get("btc_price_usd"), "usd0"))
k2.metric("BTC Dominance",   fmt(metrics.get("btc_dominance"), "pct"))
k3.metric("USDT Dominance",  fmt(metrics.get("usdt_dominance"), "pct"))
k4.metric("Alt Market Cap",  fmt(metrics.get("alt_mcap_usd"), "usdT"))
k5.metric("DXY",             fmt(metrics.get("dxy"), "num2"))

# NEW: show last data pull time (UTC)
st.caption(f"Last data pull: {pull_ts_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")

# ------------------ Signals (free-only) ------------------
signal_defs = [
    ("BTC dominance under 45%",           "btc_dominance",        "<",  btc_dom_thr),
    ("USDT dominance below 3%",           "usdt_dominance",       "<",  usdt_dom_thr),
    ("Stablecoin dominance below 8%",     "stable_dominance",     "<",  stable_dom_thr),
    ("Alt market cap ≥ target",           "alt_mcap_usd",         ">=", alt_mcap_thr),
    ("DXY falls below target",            "dxy",                  "<",  dxy_thr if use_dxy else None),
    ("ETH/BTC ratio ≥ target",            "eth_btc_ratio",        ">=", ethbtc_thr),
    ("BTC price / 200DMA ≥ target",       "btc_price_to_sma200",  ">=", sma200_thr),
    ("Golden cross (SMA50 ≥ SMA200) on",  "sma50_over_sma200",    ">=", golden_thr),
]

rows = []
for (label, key, op, thr) in signal_defs:
    val = metrics.get(key)
    rows.append({
        "Signal": label,
        "Metric key": key,
        "Value": pretty_value(key, val),
        "Operator": op,
        "Threshold": thr,
        "Status": "✅" if (thr is not None and check(val, op, thr)) else "❌"
    })
sig_df = pd.DataFrame(rows)

st.subheader("Signals")
st.dataframe(sig_df, use_container_width=True)

# ------------------ Session history & charts ------------------
if "hist" not in st.session_state:
    st.session_state.hist = []

row = {"ts": pull_ts_utc}
row.update({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})
st.session_state.hist.append(row)
st.session_state.hist = st.session_state.hist[-500:]
hist_df = pd.DataFrame(st.session_state.hist).set_index("ts")

st.subheader("Trends")
c1, c2, c3 = st.columns(3)
with c1:
    st.line_chart(hist_df[["btc_dominance"]], height=220)
with c2:
    st.line_chart(hist_df[["usdt_dominance", "stable_dominance"]].dropna(how="all"), height=220)
with c3:
    if use_dxy and "dxy" in hist_df:
        st.line_chart(hist_df[["dxy"]], height=220)
    else:
        st.info("DXY disabled; enable from the sidebar to view.")

st.line_chart(
    hist_df[["alt_mcap_usd", "btc_mcap_usd", "eth_mcap_usd"]].dropna(how="all"),
    height=280
)

st.caption("Free, live data only: CoinPaprika + Binance fallback, Yahoo Finance for DXY & BTC moving averages.")
st.markdown(f'<meta http-equiv="refresh" content="{int(refresh_s)}">', unsafe_allow_html=True)


