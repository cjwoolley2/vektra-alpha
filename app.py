import hashlib
import hmac
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from core import (
    DEFAULT_TICKERS,
    HISTORY_PATH,
    fetch_market_overview,
    fetch_price_history,
    load_results,
    run_scan,
    save_results,
)

st.set_page_config(
    page_title="Vektra Alpha",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root { --brand:#1265f4; --ink:#e9f0ff; --muted:#9badca; --panel:#0d1728; --line:rgba(255,255,255,.10); }
html, body, [data-testid="stAppViewContainer"] {background:#07101e; color:var(--ink);}
.block-container {max-width:1240px; padding-top:.65rem; padding-bottom:4rem;}
.hero {padding:1.3rem 1.45rem; border-radius:22px; background:linear-gradient(135deg,#071529 0%,#0d2e68 55%,#1265f4 100%); color:white; margin-bottom:1rem; box-shadow:0 14px 40px rgba(0,0,0,.28);}
.hero h1 {margin:0; letter-spacing:.02em;}
.hero p {margin:.35rem 0 0; opacity:.82;}
.section-title {margin-top:1.25rem; margin-bottom:.45rem; font-size:1.22rem; font-weight:800;}
.market-card,.opportunity-card,.panel {background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:1rem; box-shadow:0 8px 24px rgba(0,0,0,.16);}
.market-card {min-height:112px;}
.market-name {font-size:.82rem; color:var(--muted); font-weight:700; text-transform:uppercase; letter-spacing:.06em;}
.market-value {font-size:1.45rem; font-weight:850; margin-top:.4rem;}
.positive {color:#42d392; font-weight:800;}
.negative {color:#ff6b7a; font-weight:800;}
.neutral {color:#c5d1e8; font-weight:800;}
.opportunity-card {margin:.55rem 0;}
.rank {font-size:.82rem; color:var(--muted); font-weight:700;}
.ticker {font-size:1.45rem; font-weight:900; margin:.15rem 0;}
.score-pill {display:inline-block; background:rgba(18,101,244,.18); color:#8fb8ff; border:1px solid rgba(143,184,255,.28); padding:.23rem .65rem; border-radius:999px; font-size:.78rem; font-weight:850;}
.mood {font-size:1.6rem; font-weight:900; margin:.2rem 0;}
.small {font-size:.84rem; color:var(--muted);}
.news-item {padding:.72rem 0; border-bottom:1px solid var(--line);}
.news-item:last-child {border-bottom:0;}
[data-testid="stMetric"] {background:var(--panel); border:1px solid var(--line); padding:.75rem; border-radius:16px;}
[data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:16px; overflow:hidden;}
@media(max-width:700px){
 .block-container{padding-left:.7rem;padding-right:.7rem}
 .hero{padding:1rem}.hero h1{font-size:1.65rem}
 .market-card{min-height:96px;padding:.8rem}.market-value{font-size:1.18rem}
}
</style>
""",
    unsafe_allow_html=True,
)


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


def pct_class(value: float) -> str:
    if value > 0.0001:
        return "positive"
    if value < -0.0001:
        return "negative"
    return "neutral"


def market_mood(df: pd.DataFrame, overview: list[dict]) -> tuple[str, str, int]:
    available_changes = [float(x["change"]) for x in overview if x.get("available") and x.get("name") != "VIX"]
    index_signal = sum(1 if x > 0 else -1 if x < 0 else 0 for x in available_changes)
    breadth = float((df["return_1d"] > 0).mean()) if "return_1d" in df else 0.5
    mean_score = float(df["ai_score"].mean()) if not df.empty else 50.0
    combined = 0.40 * ((index_signal + 3) / 6) + 0.35 * breadth + 0.25 * (mean_score / 100)
    confidence = int(max(50, min(95, 50 + abs(combined - 0.5) * 90)))
    if combined >= 0.62:
        return "Bullish", "🟢", confidence
    if combined <= 0.40:
        return "Bearish", "🔴", confidence
    return "Mixed", "🟠", confidence


login_gate()

with st.sidebar:
    st.header("Controls")
    watchlist = st.text_area("Watchlist", value=secret("WATCHLIST", ",".join(DEFAULT_TICKERS)))
    threshold = st.slider("High-conviction threshold", 0.50, 0.90, float(secret("ALERT_THRESHOLD", "0.68")), 0.01)
    chart_period = st.selectbox("Chart period", ["5d", "1mo", "3mo", "6mo"], index=1)
    if st.button("Log out"):
        st.session_state.authenticated = False
        st.rerun()

hour = datetime.now().hour
greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
st.markdown(
    f'<div class="hero"><h1>📈 VEKTRA ALPHA</h1><p>{greeting}, Chris. Your evidence-led market intelligence dashboard.</p></div>',
    unsafe_allow_html=True,
)

scan_col, note_col = st.columns([1.1, 2])
with scan_col:
    if st.button("Run a fresh scan", type="primary", use_container_width=True):
        tickers = [t.strip().upper() for t in watchlist.split(",") if t.strip()]
        with st.spinner("Scanning live news, prices and supporting signals..."):
            records = run_scan(tickers, secret("FINNHUB_API_KEY"))
            save_results(records)
        st.success(f"Scan complete: {len(records)} stocks ranked.")
with note_col:
    st.caption("The dashboard uses the latest saved scan. Background scans can continue through GitHub Actions.")

records = load_results()
if not records:
    st.info("No saved scan exists yet. Press **Run a fresh scan**.")
    st.stop()

flat_records = []
for record in records:
    row = {k: v for k, v in record.items() if k not in {"news", "score_components"}}
    flat_records.append(row)
df = pd.DataFrame(flat_records)
for column in ["probability_up", "net_expected_return", "ai_score", "return_30m", "return_1d", "volume_ratio_30m", "evidence_coverage", "last_price"]:
    if column in df:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
df = df.sort_values(["ai_score", "probability_up"], ascending=False).reset_index(drop=True)

st.markdown('<div class="section-title">US Market Overview</div>', unsafe_allow_html=True)
overview = fetch_market_overview()
market_cols = st.columns(4)
for col, item in zip(market_cols, overview):
    with col:
        if item.get("available"):
            value = f'{item["last"]:,.2f}'
            delta = f'{item["change"]:+.2%}'
        else:
            value, delta = "Unavailable", "—"
        st.markdown(
            f'<div class="market-card"><div class="market-name">{item["name"]}</div><div class="market-value">{value}</div><div class="{pct_class(float(item.get("change",0)))}">{delta}</div></div>',
            unsafe_allow_html=True,
        )

mood_label, mood_icon, mood_confidence = market_mood(df, overview)
summary_left, summary_right = st.columns([2.2, 1])
with summary_left:
    st.markdown('<div class="section-title">🔥 Top Opportunities</div>', unsafe_allow_html=True)
    top_records = sorted(records, key=lambda r: (float(r.get("ai_score", 0)), float(r.get("probability_up", 0))), reverse=True)[:5]
    for index, record in enumerate(top_records, start=1):
        st.markdown(
            f'''<div class="opportunity-card">
                <div class="rank">#{index} TODAY</div>
                <div class="ticker">{record.get('ticker','')}</div>
                <span class="score-pill">AI SCORE {float(record.get('ai_score',0)):.0f}/100 · {record.get('ai_label','')}</span>
                <p><b>Probability up:</b> {float(record.get('probability_up',0)):.1%} &nbsp; <b>30m:</b> {float(record.get('return_30m',0)):+.2%} &nbsp; <b>Volume:</b> {float(record.get('volume_ratio_30m',0)):.2f}×</p>
                <div class="small">{record.get('reasons','')}</div>
            </div>''',
            unsafe_allow_html=True,
        )
with summary_right:
    st.markdown('<div class="section-title">Market Mood</div>', unsafe_allow_html=True)
    positive_count = int((df["return_1d"] > 0).sum()) if "return_1d" in df else 0
    strong_count = int((df["ai_score"] >= 75).sum())
    st.markdown(
        f'''<div class="panel">
            <div class="mood">{mood_icon} {mood_label}</div>
            <p><b>Confidence:</b> {mood_confidence}%</p>
            <p><b>Watchlist breadth:</b> {positive_count}/{len(df)} rising</p>
            <p><b>Strong AI signals:</b> {strong_count}</p>
            <div class="small">This is a composite dashboard indicator, not a market forecast.</div>
        </div>''',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Watchlist Heat Map</div>', unsafe_allow_html=True)
heat = df[["ticker", "ai_score", "return_1d", "volume_ratio_30m", "probability_up"]].copy()
heat = heat.set_index("ticker")
st.dataframe(
    heat,
    use_container_width=True,
    column_config={
        "ai_score": st.column_config.ProgressColumn("AI Score", min_value=0, max_value=100, format="%.0f"),
        "return_1d": st.column_config.NumberColumn("1-day move", format="%+.2f%%"),
        "volume_ratio_30m": st.column_config.NumberColumn("Relative volume", format="%.2fx"),
        "probability_up": st.column_config.ProgressColumn("Probability up", min_value=0, max_value=1, format="%.1f%%"),
    },
)

st.markdown('<div class="section-title">Stock Detail & Live Chart</div>', unsafe_allow_html=True)
selected_ticker = st.selectbox("Choose a stock", df["ticker"].tolist(), index=0)
selected_record = next((r for r in records if r.get("ticker") == selected_ticker), None)
if selected_record:
    detail_a, detail_b, detail_c, detail_d = st.columns(4)
    detail_a.metric("Last price", f'${float(selected_record.get("last_price",0)):,.2f}')
    detail_b.metric("AI Score", f'{float(selected_record.get("ai_score",0)):.0f}/100')
    detail_c.metric("Probability up", f'{float(selected_record.get("probability_up",0)):.1%}')
    detail_d.metric("Evidence coverage", f'{float(selected_record.get("evidence_coverage",0)):.0f}%')
    history = fetch_price_history(selected_ticker, period=chart_period, interval="1d")
    if not history.empty:
        st.line_chart(history, use_container_width=True)
    else:
        st.caption("Chart data is temporarily unavailable.")

    with st.expander("AI Score breakdown", expanded=True):
        components = selected_record.get("score_components", {})
        for name, component in components.items():
            score = float(component.get("score", 50))
            available = bool(component.get("available", False))
            weight = float(component.get("weight", 0))
            status = "Available" if available else "Excluded: data unavailable"
            st.markdown(f"**{name}: {score:.0f}/100** · Weight {weight:.0f}% · {status}")
            if available:
                st.progress(int(max(0, min(100, score))))
            st.caption(component.get("note", ""))

news_col, portfolio_col = st.columns([1.35, 1])
with news_col:
    st.markdown('<div class="section-title">📰 Latest Market News</div>', unsafe_allow_html=True)
    news_count = 0
    for record in top_records:
        for item in record.get("news", [])[:2]:
            if news_count >= 10:
                break
            headline = item.get("headline", "Untitled")
            source = item.get("source", "Unknown source")
            url = item.get("url", "")
            title = f'<a href="{url}" target="_blank">{headline}</a>' if url else headline
            st.markdown(f'<div class="news-item"><b>{record.get("ticker")}</b> · {title}<br><span class="small">{source}</span></div>', unsafe_allow_html=True)
            news_count += 1
    if news_count == 0:
        st.caption("No current articles were returned by the configured news feed.")

with portfolio_col:
    st.markdown('<div class="section-title">📊 My Portfolio</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Upload a Trading 212 CSV export", type=["csv"])
    if upload:
        portfolio = pd.read_csv(upload)
        st.dataframe(portfolio, use_container_width=True, hide_index=True)
        numeric = portfolio.select_dtypes(include="number")
        if not numeric.empty:
            st.caption("CSV imported. We will map your exact Trading 212 columns in the portfolio-analysis step.")
    else:
        st.markdown('<div class="panel"><b>No portfolio loaded</b><p class="small">Upload your Trading 212 CSV to begin portfolio analysis.</p></div>', unsafe_allow_html=True)

with st.expander("Full signal table"):
    cols = [c for c in ["ticker", "ai_score", "ai_label", "evidence_coverage", "probability_up", "last_price", "return_30m", "return_1d", "volume_ratio_30m", "action"] if c in df]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

with st.expander("Signal history"):
    if HISTORY_PATH.exists():
        history_df = pd.read_csv(HISTORY_PATH)
        st.dataframe(history_df.tail(200), use_container_width=True, hide_index=True)
    else:
        st.caption("History will appear after the scanner has run.")

st.divider()
st.caption("Research and decision-support tool only. Probabilities are model estimates, not guarantees or personalised financial advice.")
