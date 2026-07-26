import hashlib
import hmac
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from core import DEFAULT_TICKERS, HISTORY_PATH, load_results, run_scan, save_results

st.set_page_config(page_title="Vektra Alpha", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root { --brand:#0b5fff; }
.block-container {max-width:1180px; padding-top:1rem; padding-bottom:4rem;}
.hero {padding:1.2rem 1.4rem; border-radius:20px; background:linear-gradient(135deg,#071529,#0b5fff); color:white; margin-bottom:1rem;}
.card {border:1px solid rgba(128,128,128,.25); border-radius:16px; padding:1rem; margin:.65rem 0;}
.badge {display:inline-block; border-radius:999px; padding:.2rem .65rem; font-size:.78rem; font-weight:700; background:rgba(11,95,255,.12);}
.small {font-size:.86rem; opacity:.72;}
@media(max-width:700px){.block-container{padding-left:.75rem;padding-right:.75rem}.hero h1{font-size:1.65rem}}
</style>
""", unsafe_allow_html=True)


def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return os.getenv(name, default)


def login_gate() -> None:
    password_hash = secret("APP_PASSWORD_SHA256")
    if not password_hash:
        st.warning("App login is not configured. Add APP_PASSWORD_SHA256 to Streamlit secrets before sharing this app.")
        return
    if st.session_state.get("authenticated"):
        return
    st.markdown('<div class="hero"><h1>Vektra Alpha</h1><p>Private market intelligence dashboard</p></div>', unsafe_allow_html=True)
    entered = st.text_input("Password", type="password")
    if st.button("Log in", type="primary", use_container_width=True):
        digest = hashlib.sha256(entered.encode()).hexdigest()
        if hmac.compare_digest(digest, password_hash):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password")
    st.stop()


login_gate()

st.markdown('<div class="hero"><h1>📈 Vektra Alpha</h1><p>Evidence-led global news and market signal monitor</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Controls")
    watchlist = st.text_area("Watchlist", value=secret("WATCHLIST", ",".join(DEFAULT_TICKERS)))
    threshold = st.slider("High-conviction threshold", 0.50, 0.90, float(secret("ALERT_THRESHOLD", "0.68")), 0.01)
    if st.button("Log out"):
        st.session_state.authenticated = False
        st.rerun()

col_a, col_b = st.columns([2, 1])
with col_a:
    if st.button("Run a fresh scan", type="primary", use_container_width=True):
        tickers = [t.strip().upper() for t in watchlist.split(",") if t.strip()]
        with st.spinner("Scanning live news and prices..."):
            records = run_scan(tickers, secret("FINNHUB_API_KEY"))
            save_results(records)
        st.success(f"Scan complete: {len(records)} stocks ranked.")
with col_b:
    st.caption("Background scans run through GitHub Actions. This button performs an immediate scan.")

records = load_results()
if not records:
    st.info("No saved scan exists yet. Press **Run a fresh scan**.")
    st.stop()

df = pd.DataFrame([{k:v for k,v in r.items() if k != "news"} for r in records])
df["probability_up"] = pd.to_numeric(df["probability_up"], errors="coerce").fillna(0)
df["net_expected_return"] = pd.to_numeric(df["net_expected_return"], errors="coerce").fillna(0)
df["ai_score"] = pd.to_numeric(df.get("ai_score", 0), errors="coerce").fillna(0)
df = df.sort_values(["ai_score", "probability_up"], ascending=False)

top = df.iloc[0]
c1,c2,c3,c4 = st.columns(4)
c1.metric("Top stock", top["ticker"])
c2.metric("AI Score", f"{top['ai_score']:.0f}/100")
c3.metric("Probability up", f"{top['probability_up']:.1%}")
c4.metric("Signals above threshold", int((df["probability_up"] >= threshold).sum()))

st.subheader("Ranked opportunities")
for record in records:
    probability = float(record.get("probability_up", 0))
    ai_score = float(record.get("ai_score", 0))
    coverage = float(record.get("evidence_coverage", 0))
    news = record.get("news", [])
    st.markdown(f'''<div class="card"><span class="badge">{record.get('ai_label','UNSCORED')}</span><h3>{record.get('ticker','')} &nbsp; {ai_score:.0f}/100</h3><b>Probability up:</b> {probability:.1%} &nbsp; <b>Net expected return:</b> {float(record.get('net_expected_return',0)):.2%} &nbsp; <b>Evidence coverage:</b> {coverage:.0f}%<br><span class="small">{record.get('reasons','')}</span></div>''', unsafe_allow_html=True)
    with st.expander(f"AI Score breakdown — {record.get('ticker','')}"):
        components = record.get("score_components", {})
        if components:
            for name, component in components.items():
                available = bool(component.get("available", False))
                score = float(component.get("score", 50))
                weight = float(component.get("weight", 0))
                note = component.get("note", "")
                status = "Available" if available else "Not available — excluded from score"
                st.markdown(f"**{name}: {score:.0f}/100** · Weight {weight:.0f}% · {status}")
                if available:
                    st.progress(int(max(0, min(100, score))))
                st.caption(note)
        st.markdown("---")
        st.markdown("**Latest supporting news**")
        if not news:
            st.write("No articles returned by the configured feed.")
        for item in news:
            headline = item.get("headline", "Untitled")
            url = item.get("url", "")
            source = item.get("source", "Unknown")
            st.markdown(f"**[{headline}]({url})**  \n{source}" if url else f"**{headline}**  \n{source}")


st.subheader("AI Score table")
score_table = df[[
    "ticker", "ai_score", "ai_label", "evidence_coverage",
    "probability_up", "return_30m", "volume_ratio_30m"
]].copy()
st.dataframe(
    score_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "ai_score": st.column_config.ProgressColumn("AI Score", min_value=0, max_value=100, format="%.0f"),
        "evidence_coverage": st.column_config.ProgressColumn("Evidence coverage", min_value=0, max_value=100, format="%.0f%%"),
        "probability_up": st.column_config.ProgressColumn("Probability up", min_value=0, max_value=1, format="%.1%%"),
        "return_30m": st.column_config.NumberColumn("30-minute return", format="%.2%%"),
        "volume_ratio_30m": st.column_config.NumberColumn("Relative volume", format="%.2fx"),
    },
)

st.subheader("Portfolio")
upload = st.file_uploader("Upload a Trading 212 CSV export", type=["csv"])
if upload:
    portfolio = pd.read_csv(upload)
    st.dataframe(portfolio, use_container_width=True, hide_index=True)
    numeric = portfolio.select_dtypes(include="number")
    if not numeric.empty:
        st.caption("CSV imported. Column names vary by Trading 212 export, so totals are shown only after mapping is confirmed.")

st.subheader("Signal history")
if HISTORY_PATH.exists():
    history = pd.read_csv(HISTORY_PATH)
    st.dataframe(history.tail(200), use_container_width=True, hide_index=True)
else:
    st.caption("History will appear after the scanner has run.")

st.divider()
st.caption("Research and decision-support tool only. Probabilities are model estimates, not guarantees or personalised financial advice.")
