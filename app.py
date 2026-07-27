import hashlib
import hmac
import os
import textwrap
from datetime import datetime

import pandas as pd
import streamlit as st

from core import (
    BROAD_UK_UNIVERSE,
    BROAD_US_UNIVERSE,
    GLOBAL_UNIVERSE,
    DEFAULT_TICKERS,
    HISTORY_PATH,
    evaluate_buy_today,
    fetch_market_overview,
    get_market_universe,
    fetch_price_history,
    load_market_scan,
    load_results,
    rank_buy_today_candidates,
    run_broad_market_scan,
    run_scan,
    save_market_scan,
    save_results,
    select_buy_today,
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

.news-card {
    border: 1px solid rgba(148,163,184,.20);
    background: linear-gradient(145deg, rgba(15,23,42,.96), rgba(20,30,50,.90));
    border-radius: 16px;
    padding: 15px;
    margin-bottom: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,.16);
}
.news-topline {
    display:flex;
    justify-content:space-between;
    gap:8px;
    align-items:center;
    margin-bottom:8px;
}
.news-badge {
    display:inline-block;
    border-radius:999px;
    padding:4px 9px;
    margin-right:5px;
    background:rgba(59,130,246,.16);
    border:1px solid rgba(96,165,250,.24);
    font-size:.72rem;
    font-weight:700;
}
.impact-high {color:#4ade80;font-weight:800;}
.impact-medium {color:#facc15;font-weight:800;}
.impact-low {color:#cbd5e1;font-weight:800;}
.news-headline {font-size:1rem;font-weight:750;line-height:1.35;margin:.35rem 0;}
.news-meta {font-size:.79rem;opacity:.70;}


.briefing {
    border:1px solid rgba(96,165,250,.30);
    background:linear-gradient(145deg, rgba(7,21,41,.98), rgba(13,46,104,.75));
    border-radius:20px;
    padding:18px;
    margin:12px 0 18px;
    box-shadow:0 14px 34px rgba(0,0,0,.22);
}
.briefing-title {font-size:1.3rem;font-weight:900;margin-bottom:3px;}
.briefing-sub {color:#9badca;font-size:.86rem;margin-bottom:14px;}
.brief-row {
    display:grid;
    grid-template-columns:36px 72px 1fr 90px;
    gap:8px;
    align-items:center;
    padding:10px 0;
    border-bottom:1px solid rgba(255,255,255,.08);
}
.brief-row:last-child {border-bottom:0;}
.brief-rank {font-weight:900;color:#8fb8ff;}
.brief-ticker {font-weight:900;font-size:1.06rem;}
.brief-catalyst {font-size:.84rem;color:#c5d1e8;}
.brief-score {font-weight:900;text-align:right;}
.risk-flag {
    display:inline-block;
    border-radius:999px;
    padding:3px 8px;
    margin-top:4px;
    font-size:.70rem;
    font-weight:800;
    background:rgba(250,204,21,.12);
    border:1px solid rgba(250,204,21,.22);
    color:#fde68a;
}
@media(max-width:700px){
 .brief-row{grid-template-columns:28px 58px 1fr 66px;gap:5px}
 .brief-catalyst{font-size:.76rem}
 .brief-score{font-size:.82rem}
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


st.markdown('<div class="section-title">🌍 USA + UK Market Scanner</div>', unsafe_allow_html=True)
market_scan_a, market_scan_b, market_scan_c, market_scan_d = st.columns([1.1, 1.1, 1, 1.8])

with market_scan_a:
    selected_market = st.selectbox("Market", ["USA + UK", "USA", "UK"], index=0)

with market_scan_b:
    candidate_count = st.selectbox(
        "Deep-analysis candidates",
        [10, 15, 20],
        index=1,
        help="Only the strongest pre-screened candidates receive the full news and AI analysis.",
    )

with market_scan_c:
    run_market_scan = st.button(
        "Scan broad market",
        type="secondary",
        use_container_width=True,
    )

with market_scan_d:
    selected_universe = get_market_universe(selected_market)
    st.caption(
        f"Pre-screens {len(selected_universe)} liquid {selected_market} stocks, then deeply analyses the strongest candidates."
    )

if run_market_scan:
    with st.spinner(f"Pre-screening {selected_market}, then running deep AI analysis..."):
        market_records, market_candidates = run_broad_market_scan(
            secret("FINNHUB_API_KEY"),
            candidate_count=int(candidate_count),
            market=selected_market,
        )
        save_market_scan(market_records, market_candidates)
    st.success(
        f"Broad scan complete: {len(market_candidates)} candidates pre-screened and "
        f"{len(market_records)} deeply ranked."
    )

saved_market_scan = load_market_scan()
if saved_market_scan:
    market_records = saved_market_scan.get("records", [])
    market_candidates = saved_market_scan.get("candidates", [])
    generated_at = saved_market_scan.get("generated_at", "")

    st.markdown("#### Emerging opportunities")
    if market_records:
        emerging_cols = st.columns(min(5, len(market_records[:5])))
        for col, item in zip(emerging_cols, market_records[:5]):
            with col:
                pre = item.get("market_prefilter", {})
                st.metric(
                    item.get("ticker", ""),
                    f'{float(item.get("ai_score", 0)):.0f}/100',
                    f'{float(item.get("return_30m", 0)):+.2%} in 30m',
                )
                st.caption(
                    f'Probability {float(item.get("probability_up", 0)):.0%} · '
                    f'Volume {float(pre.get("relative_volume", 0)):.2f}×'
                )
    else:
        st.caption("No deeply scored market candidates are saved yet.")

    with st.expander("View fast pre-screen results"):
        if market_candidates:
            market_candidate_df = pd.DataFrame(market_candidates)
            st.dataframe(
                market_candidate_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "prefilter_score": st.column_config.ProgressColumn(
                        "Pre-screen score", min_value=0, max_value=100, format="%.0f"
                    ),
                    "return_30m": st.column_config.NumberColumn("30m move", format="%+.2f%%"),
                    "return_1d": st.column_config.NumberColumn("1-day move", format="%+.2f%%"),
                    "relative_volume": st.column_config.NumberColumn("Relative volume", format="%.2fx"),
                },
            )
        st.caption(f"Last broad scan: {generated_at or 'unknown'}")


# ---------------------------------------------------------------------------
# AI BUY TODAY DECISION GATE
# ---------------------------------------------------------------------------
if saved_market_scan and saved_market_scan.get("records"):
    decision_records = saved_market_scan.get("records", [])
    buy_today = select_buy_today(decision_records)
    ranked_buy_decisions = rank_buy_today_candidates(decision_records, limit=5)

    st.markdown('<div class="section-title">🎯 AI Buy Today Decision</div>', unsafe_allow_html=True)

    decision_code = buy_today.get("decision_code", "NO_TRADE")
    if decision_code == "BUY_CANDIDATE":
        st.success(
            f'BUY CANDIDATE TODAY: {buy_today.get("ticker")} — '
            f'{float(buy_today.get("gate_score", 0)):.0f}% of decision conditions passed.'
        )
    elif buy_today.get("decision") == "NO QUALIFYING BUY TODAY":
        st.warning(
            f'NO QUALIFYING BUY TODAY. Closest candidate: {buy_today.get("ticker")} — '
            f'{float(buy_today.get("gate_score", 0)):.0f}% of conditions passed.'
        )
    else:
        st.info(str(buy_today.get("decision", "NO TRADE")))

    buy_a, buy_b, buy_c, buy_d = st.columns(4)
    buy_a.metric("Stock", buy_today.get("ticker", "—"))
    buy_b.metric("AI Score", f'{float(buy_today.get("ai_score", 0)):.0f}/100')
    buy_c.metric("Probability up", f'{float(buy_today.get("probability_up", 0)):.0%}')
    buy_d.metric("Decision gate", f'{float(buy_today.get("gate_score", 0)):.0f}%')

    st.markdown(f'**Catalyst:** {buy_today.get("catalyst", "No qualifying catalyst")}.')
    st.caption(
        f'Market: {buy_today.get("market", "—")} · '
        f'30m: {float(buy_today.get("return_30m", 0)):+.2%} · '
        f'1 day: {float(buy_today.get("return_1d", 0)):+.2%} · '
        f'Relative volume: {float(buy_today.get("relative_volume", 0)):.2f}× · '
        f'Evidence: {float(buy_today.get("evidence_coverage", 0)):.0f}%'
    )

    explain_left, explain_right = st.columns(2)
    with explain_left:
        st.markdown("**Conditions passed**")
        strengths = buy_today.get("strengths", [])
        if strengths:
            for item in strengths:
                st.write(f"✅ {item}")
        else:
            st.write("No qualifying strengths recorded.")

    with explain_right:
        st.markdown("**Conditions not met / risk flags**")
        failures = list(buy_today.get("failed_checks", []))
        failures.extend(
            item for item in buy_today.get("hard_failures", []) if item not in failures
        )
        if failures:
            for item in failures:
                st.write(f"⚠️ {item}")
        else:
            st.write("No model risk flag triggered.")

    with st.expander("Compare the top five decision results"):
        decision_df = pd.DataFrame(ranked_buy_decisions)
        display_columns = [
            "ticker", "market", "decision", "gate_score", "ai_score",
            "probability_up", "evidence_coverage", "return_30m",
            "relative_volume", "catalyst_impact",
        ]
        display_columns = [c for c in display_columns if c in decision_df.columns]
        st.dataframe(
            decision_df[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "gate_score": st.column_config.ProgressColumn("Decision gate", min_value=0, max_value=100, format="%.0f"),
                "ai_score": st.column_config.ProgressColumn("AI Score", min_value=0, max_value=100, format="%.0f"),
                "probability_up": st.column_config.ProgressColumn("Probability", min_value=0, max_value=1, format="%.0f%%"),
                "return_30m": st.column_config.NumberColumn("30m move", format="%+.2f%%"),
                "relative_volume": st.column_config.NumberColumn("Relative volume", format="%.2fx"),
            },
        )

    st.caption(
        "Research decision support only. BUY CANDIDATE does not mean guaranteed profit and is not personalised financial advice or an instruction to trade."
    )

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

# ---------------------------------------------------------------------------
# STEP 6 — DAILY AI MARKET BRIEFING
# ---------------------------------------------------------------------------
briefing_records = sorted(
    records,
    key=lambda r: (
        float(r.get("ai_score", 0)),
        float(r.get("probability_up", 0)),
    ),
    reverse=True,
)[:5]

briefing_lines = [
    f"{greeting}, Chris.",
    "",
    f"Market mood: {mood_label} ({mood_confidence}% confidence)",
    "",
    "Today's five highest-ranked watchlist opportunities:",
]

briefing_rows_html = []
for rank, record in enumerate(briefing_records, start=1):
    ticker = record.get("ticker", "")
    score = float(record.get("ai_score", 0))
    probability = float(record.get("probability_up", 0))
    coverage = float(record.get("evidence_coverage", 0))
    volume = float(record.get("volume_ratio_30m", 0))
    move_30m = float(record.get("return_30m", 0))

    articles = sorted(
        record.get("news", []),
        key=lambda item: float(item.get("impact_score", 0)),
        reverse=True,
    )
    if articles:
        top_article = articles[0]
        catalyst = (
            f'{top_article.get("event_type", "News")} · '
            f'{top_article.get("sentiment_label", "Neutral")} · '
            f'{float(top_article.get("impact_score", 0)):.0f}/100 impact'
        )
    else:
        catalyst = "No qualifying live catalyst returned"

    flags = []
    if coverage < 55:
        flags.append("limited evidence")
    if volume > 2.5:
        flags.append("high-volume move")
    if abs(move_30m) > 0.05:
        flags.append("possible price exhaustion")
    if probability < threshold:
        flags.append("below alert threshold")
    risk_text = " · ".join(flags) if flags else "no major model flag"

    briefing_rows_html.append(
        (
            f'<div class="brief-row">'
            f'<div class="brief-rank">#{rank}</div>'
            f'<div class="brief-ticker">{ticker}</div>'
            f'<div class="brief-catalyst">'
            f'{catalyst}<br>'
            f'<span class="risk-flag">{risk_text}</span>'
            f'</div>'
            f'<div class="brief-score">{score:.0f}/100<br>'
            f'<span class="small">{probability:.0%}</span></div>'
            f'</div>'
        )
    )
    briefing_lines.append(
        f"{rank}. {ticker} — AI Score {score:.0f}/100; "
        f"probability up {probability:.0%}; {catalyst}; risk: {risk_text}."
    )

positive_breadth = int((df["return_1d"] > 0).sum()) if "return_1d" in df else 0
briefing_lines.extend(
    [
        "",
        f"Watchlist breadth: {positive_breadth} of {len(df)} stocks are rising today.",
        "",
        "This is research and decision support, not a trade instruction.",
    ]
)
briefing_text = "\n".join(briefing_lines)

briefing_html = (
    f'<div class="briefing">'
    f'<div class="briefing-title">☀️ Daily AI Market Briefing</div>'
    f'<div class="briefing-sub">'
    f'Generated from the latest saved scan · {datetime.now().strftime("%d %b %Y, %H:%M")}'
    f'</div>'
    f'<p><b>{mood_icon} Market mood:</b> {mood_label} · {mood_confidence}% confidence</p>'
    f'{"".join(briefing_rows_html)}'
    f'</div>'
)

st.markdown(briefing_html, unsafe_allow_html=True)

brief_a, brief_b = st.columns([1, 1])
with brief_a:
    st.download_button(
        "Download today's briefing",
        data=briefing_text,
        file_name=f"vektra_alpha_briefing_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
    )
with brief_b:
    with st.expander("Read briefing as plain text"):
        st.text(briefing_text)
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
    st.markdown('<div class="section-title">📰 Intelligent News Cards</div>', unsafe_allow_html=True)

    filter_a, filter_b = st.columns(2)
    with filter_a:
        news_direction = st.selectbox(
            "Direction",
            ["All", "Bullish", "Neutral", "Bearish"],
            label_visibility="collapsed",
        )
    with filter_b:
        minimum_impact = st.selectbox(
            "Minimum impact",
            ["All impact", "Moderate+", "High+"],
            label_visibility="collapsed",
        )

    all_news = []
    for record in records:
        for item in record.get("news", []):
            enriched = dict(item)
            enriched.setdefault("ticker", record.get("ticker", ""))
            all_news.append(enriched)

    all_news.sort(
        key=lambda item: (
            float(item.get("impact_score", 0)),
            float(item.get("datetime", 0) or 0),
        ),
        reverse=True,
    )

    shown = 0
    for item in all_news:
        direction = item.get("sentiment_label", "Neutral")
        impact = float(item.get("impact_score", 0))

        if news_direction != "All" and direction != news_direction:
            continue
        if minimum_impact == "Moderate+" and impact < 50:
            continue
        if minimum_impact == "High+" and impact < 65:
            continue
        if shown >= 12:
            break

        headline = item.get("headline", "Untitled")
        source = item.get("source", "Unknown source")
        url = item.get("url", "")
        ticker = item.get("ticker", "")
        event_type = item.get("event_type", "Other")
        age_text = item.get("age_text", "Unknown age")
        quality = item.get("source_quality", "Unknown")
        impact_label = item.get("impact_label", "Low")
        reason = item.get("reason", "No explanation available")
        sentiment_icon = item.get("sentiment_icon", "🟠")

        if impact >= 65:
            impact_class = "impact-high"
        elif impact >= 50:
            impact_class = "impact-medium"
        else:
            impact_class = "impact-low"

        headline_html = (
            f'<a href="{url}" target="_blank">{headline}</a>'
            if url else headline
        )

        card_html = f"""
        <div class="news-card">
            <div class="news-topline">
                <div>
                    <span class="news-badge">{ticker}</span>
                    <span class="news-badge">{event_type}</span>
                </div>
                <div class="{impact_class}">
                    {impact_label} impact · {impact:.0f}/100
                </div>
            </div>
            <div class="news-headline">
                {sentiment_icon} {headline_html}
            </div>
            <div class="news-meta">
                {source} · {quality} source quality · {age_text}
            </div>
            <div class="small" style="margin-top:8px">
                <b>Why it matters:</b> {reason}
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        shown += 1

    if shown == 0:
        st.caption("No articles match the selected filters or the news feed returned no current articles.")

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
