import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from vektra_global_news_alpha_engine import (
    GlobalNewsAlphaEngine,
    MarketSnapshot,
    NewsItem,
    RuleBasedFinanceTextModel,
)

st.set_page_config(page_title="Vektra Alpha", page_icon="📈", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 1rem; padding-bottom: 4rem; max-width: 1200px;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
.signal-card {border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:16px;margin-bottom:12px;}
.small-muted {opacity:.70;font-size:.88rem;}
.badge {display:inline-block;padding:3px 8px;border-radius:999px;border:1px solid rgba(128,128,128,.35);margin-right:5px;font-size:.78rem;}
@media(max-width:700px){.block-container{padding-left:.75rem;padding-right:.75rem}h1{font-size:1.75rem!important}}
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD"]
FINNHUB = "https://finnhub.io/api/v1"


@st.cache_resource
def get_engine():
    return GlobalNewsAlphaEngine(RuleBasedFinanceTextModel())


engine = get_engine()


def safe_float(value, default=0.0):
    try:
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return float(value)
    except Exception:
        return default


def api_get(path: str, api_key: str, params: Dict | None = None):
    if not api_key:
        return None
    payload = dict(params or {})
    payload["token"] = api_key
    try:
        response = requests.get(f"{FINNHUB}/{path}", params=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def fetch_news(ticker: str, api_key: str, lookback_days: int = 3) -> List[Dict]:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days)
    data = api_get("company-news", api_key, {"symbol": ticker, "from": start.isoformat(), "to": end.isoformat()})
    return data[:30] if isinstance(data, list) else []


def fetch_analyst_actions(ticker: str, api_key: str) -> Dict:
    data = api_get("stock/recommendation", api_key, {"symbol": ticker})
    return data[0] if isinstance(data, list) and data else {}


def fetch_insider_transactions(ticker: str, api_key: str) -> List[Dict]:
    data = api_get("stock/insider-transactions", api_key, {"symbol": ticker})
    if isinstance(data, dict):
        return data.get("data", [])[:30]
    return []


def fetch_social_sentiment(ticker: str, api_key: str) -> Dict:
    start = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
    data = api_get("stock/social-sentiment", api_key, {"symbol": ticker, "from": start})
    return data if isinstance(data, dict) else {}


def options_activity_proxy(ticker: str) -> Dict:
    """Free-data options proxy. It is not a professional unusual-options feed."""
    try:
        obj = yf.Ticker(ticker)
        expiries = obj.options
        if not expiries:
            return {"score": 0.5, "label": "No options data", "call_put_ratio": None, "unusual": False}
        chain = obj.option_chain(expiries[0])
        call_volume = safe_float(chain.calls.get("volume", pd.Series(dtype=float)).fillna(0).sum())
        put_volume = safe_float(chain.puts.get("volume", pd.Series(dtype=float)).fillna(0).sum())
        call_oi = safe_float(chain.calls.get("openInterest", pd.Series(dtype=float)).fillna(0).sum())
        put_oi = safe_float(chain.puts.get("openInterest", pd.Series(dtype=float)).fillna(0).sum())
        ratio = (call_volume + 1) / (put_volume + 1)
        volume_oi = (call_volume + put_volume) / max(call_oi + put_oi, 1)
        score = float(np.clip(0.50 + 0.18 * np.tanh(np.log(ratio)) + 0.15 * np.tanh(volume_oi - 0.25), 0, 1))
        unusual = bool(volume_oi > 0.65)
        return {
            "score": score,
            "label": "Call-biased" if ratio > 1.3 else "Put-biased" if ratio < 0.77 else "Balanced",
            "call_put_ratio": ratio,
            "unusual": unusual,
        }
    except Exception:
        return {"score": 0.5, "label": "Unavailable", "call_put_ratio": None, "unusual": False}


def analyst_score(data: Dict) -> Tuple[float, str]:
    if not data:
        return 0.5, "Unavailable"
    positive = safe_float(data.get("strongBuy")) * 2 + safe_float(data.get("buy"))
    negative = safe_float(data.get("strongSell")) * 2 + safe_float(data.get("sell"))
    hold = safe_float(data.get("hold"))
    total = positive + negative + hold
    score = 0.5 if total == 0 else np.clip((positive + 0.5 * hold) / total, 0, 1)
    return float(score), f'{int(data.get("strongBuy",0))} strong buy / {int(data.get("buy",0))} buy'


def insider_score(items: List[Dict]) -> Tuple[float, str]:
    if not items:
        return 0.5, "No recent data"
    net = 0.0
    buys = sells = 0
    for x in items:
        change = safe_float(x.get("change"))
        price = safe_float(x.get("transactionPrice"), 1.0)
        value = change * max(price, 1.0)
        net += value
        if change > 0:
            buys += 1
        elif change < 0:
            sells += 1
    score = float(np.clip(0.5 + 0.2 * np.tanh(net / 1_000_000), 0, 1))
    return score, f"{buys} buys / {sells} sells"


def social_score(data: Dict) -> Tuple[float, str]:
    rows = []
    for key in ("reddit", "twitter"):
        values = data.get(key, []) if isinstance(data, dict) else []
        if values:
            rows.extend(values[-5:])
    if not rows:
        return 0.5, "Unavailable"
    positive = sum(safe_float(r.get("positiveMention")) for r in rows)
    negative = sum(safe_float(r.get("negativeMention")) for r in rows)
    mentions = positive + negative
    score = 0.5 if mentions == 0 else positive / mentions
    return float(np.clip(score, 0, 1)), f"{int(mentions)} classified mentions"


def fetch_market_snapshot(ticker: str) -> Tuple[MarketSnapshot | None, Dict]:
    try:
        intraday = yf.download(ticker, period="1mo", interval="30m", progress=False, auto_adjust=True, threads=False)
        daily = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True, threads=False)
        for frame in (intraday, daily):
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = [c[0] for c in frame.columns]
        if intraday.empty or len(intraday) < 5:
            return None, {}
        close = intraday["Close"].dropna()
        volume = intraday["Volume"].fillna(0)
        dclose = daily["Close"].dropna()
        dvolume = daily["Volume"].fillna(0)
        ret30 = safe_float(close.pct_change().iloc[-1])
        vol_ratio = safe_float(volume.iloc[-1] / max(volume.tail(20).mean(), 1), 1)
        realised = safe_float(dclose.pct_change().tail(20).std() * np.sqrt(252), .35)
        adv = safe_float((dclose.tail(20) * dvolume.tail(20)).mean(), 5_000_000)
        # OBV and accumulation proxy — labelled as a proxy, not verified institutional flow.
        signed_volume = np.sign(dclose.diff().fillna(0)) * dvolume.reindex(dclose.index).fillna(0)
        recent_flow = safe_float(signed_volume.tail(10).sum())
        normaliser = max(safe_float(dvolume.tail(20).sum()), 1)
        flow_score = float(np.clip(0.5 + 2.0 * recent_flow / normaliser, 0, 1))
        momentum_20d = safe_float(dclose.pct_change(20).iloc[-1]) if len(dclose) > 21 else 0
        snapshot = MarketSnapshot(
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            last_price=safe_float(close.iloc[-1]),
            return_5m=ret30 * .35,
            return_30m=ret30,
            return_1d=safe_float(close.pct_change(13).iloc[-1]) if len(close) > 13 else ret30,
            market_return_30m=0,
            sector_return_30m=0,
            volume_ratio_30m=vol_ratio,
            spread_bps=15,
            realised_volatility_20d=max(.05, min(realised, 2)),
            average_daily_value_gbp=max(100_000, adv),
        )
        return snapshot, {"flow_score": flow_score, "momentum_20d": momentum_20d}
    except Exception:
        return None, {}


def ingest_news(ticker: str, api_key: str) -> List[Dict]:
    items = fetch_news(ticker, api_key)
    for item in items:
        published = datetime.fromtimestamp(item.get("datetime", 0), tz=timezone.utc)
        article = NewsItem(
            article_id=str(item.get("id") or item.get("url") or f"{ticker}-{published.timestamp()}"),
            published_at=published,
            source=item.get("source", "unknown"),
            headline=item.get("headline", ""),
            body=item.get("summary", ""),
            url=item.get("url", ""),
            mentioned_tickers=(ticker,),
        )
        engine.analyse_article(article, ticker, "major_wire", datetime.now(timezone.utc))
    return items


def score_tickers(tickers: List[str], api_key: str):
    snapshots, details, articles = [], {}, {}
    for ticker in tickers:
        articles[ticker] = ingest_news(ticker, api_key)
        market, market_extra = fetch_market_snapshot(ticker)
        if not market:
            continue
        snapshots.append(market)
        analyst_raw = fetch_analyst_actions(ticker, api_key)
        insider_raw = fetch_insider_transactions(ticker, api_key)
        social_raw = fetch_social_sentiment(ticker, api_key)
        a_score, a_text = analyst_score(analyst_raw)
        i_score, i_text = insider_score(insider_raw)
        s_score, s_text = social_score(social_raw)
        o = options_activity_proxy(ticker)
        details[ticker] = {
            "analyst_score": a_score, "analyst_text": a_text,
            "insider_score": i_score, "insider_text": i_text,
            "social_score": s_score, "social_text": s_text,
            "options_score": o["score"], "options_text": o["label"], "options_ratio": o["call_put_ratio"],
            "unusual_options": o["unusual"],
            **market_extra,
        }
    if not snapshots:
        return pd.DataFrame(), details, articles
    ranked = engine.rank_watchlist(snapshots, market_breadth=0, volatility_percentile=.5, index_trend=0)
    rows = []
    for _, row in ranked.iterrows():
        ticker = row["ticker"]
        d = details.get(ticker, {})
        evidence = np.mean([
            d.get("analyst_score", .5), d.get("insider_score", .5),
            d.get("social_score", .5), d.get("options_score", .5), d.get("flow_score", .5),
        ])
        # Evidence adjusts, but cannot overpower the core news/market probability.
        enhanced_probability = float(np.clip(row["probability_up"] + .18 * (evidence - .5), .02, .98))
        enhanced_score = row["signal_score"] * (.75 + .5 * evidence)
        new = row.to_dict()
        new.update(d)
        new["base_probability"] = row["probability_up"]
        new["probability_up"] = enhanced_probability
        new["evidence_score"] = evidence
        new["signal_score"] = enhanced_score
        if enhanced_probability >= .68 and row["net_expected_return"] > .004:
            new["action"] = "HIGH-CONVICTION WATCH"
        elif enhanced_probability >= .58:
            new["action"] = "WATCH"
        else:
            new["action"] = "IGNORE"
        rows.append(new)
    return pd.DataFrame(rows).sort_values(["signal_score", "probability_up"], ascending=False), details, articles


st.title("📈 Vektra Alpha")
st.caption("AI-assisted news, market, options, insider, analyst and sentiment intelligence")

with st.expander("Settings", expanded=False):
    try:
        default_key = st.secrets.get("FINNHUB_API_KEY", "")
    except Exception:
        default_key = os.getenv("FINNHUB_API_KEY", "")
    api_key = st.text_input("Finnhub API key", value=default_key, type="password")
    tickers_text = st.text_input("Watchlist", value=",".join(DEFAULT_TICKERS))
    refresh_seconds = st.selectbox("Automatic refresh", [0, 60, 180, 300, 900], index=3,
                                   format_func=lambda x: "Off" if x == 0 else f"Every {x//60} minute(s)")

tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]
run = st.button("Run enhanced live scan", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
    st.session_state.articles = {}

if run or (refresh_seconds and st.session_state.results.empty):
    with st.spinner("Scanning news, prices, options, analysts, insiders and social signals..."):
        result, detail, article_map = score_tickers(tickers, api_key)
        st.session_state.results = result
        st.session_state.articles = article_map
        st.session_state.last_scan = datetime.now().strftime("%d %b %Y, %H:%M:%S")

results = st.session_state.results
if results.empty:
    st.info("Add your Finnhub key, choose tickers and press **Run enhanced live scan**.")
else:
    top = results.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top stock", top["ticker"])
    c2.metric("Enhanced probability", f'{top["probability_up"]:.1%}')
    c3.metric("Evidence score", f'{top["evidence_score"]:.1%}')
    c4.metric("Net expected return", f'{top["net_expected_return"]:.2%}')
    st.caption(f'Last scan: {st.session_state.get("last_scan", "—")}')

    st.subheader("Ranked opportunities")
    for _, row in results.iterrows():
        ticker = row["ticker"]
        badges = [
            f'Analysts {row.get("analyst_score",.5):.0%}',
            f'Insiders {row.get("insider_score",.5):.0%}',
            f'Social {row.get("social_score",.5):.0%}',
            f'Options {row.get("options_score",.5):.0%}',
            f'Flow proxy {row.get("flow_score",.5):.0%}',
        ]
        badge_html = "".join(f'<span class="badge">{b}</span>' for b in badges)
        st.markdown(f"""
        <div class="signal-card">
          <h3 style="margin:0">{ticker} — {row['action']}</h3>
          <p><b>Probability:</b> {row['probability_up']:.1%} &nbsp; <b>Base model:</b> {row['base_probability']:.1%}
          &nbsp; <b>Net expected return:</b> {row['net_expected_return']:.2%}</p>
          <div>{badge_html}</div>
          <p class="small-muted">{row['reasons']}</p>
        </div>""", unsafe_allow_html=True)
        with st.expander(f"Evidence and latest news — {ticker}"):
            st.write(f"**Analyst consensus:** {row.get('analyst_text','Unavailable')}")
            st.write(f"**Insider transactions:** {row.get('insider_text','Unavailable')}")
            st.write(f"**Social sentiment:** {row.get('social_text','Unavailable')}")
            ratio = row.get("options_ratio")
            ratio_text = "Unavailable" if pd.isna(ratio) or ratio is None else f"{ratio:.2f} call/put volume ratio"
            st.write(f"**Options proxy:** {row.get('options_text','Unavailable')} — {ratio_text}")
            st.write(f"**Accumulation/flow proxy:** {row.get('flow_score',.5):.0%} (price-volume estimate, not verified institutional orders)")
            for article in st.session_state.articles.get(ticker, [])[:8]:
                headline, source, url = article.get("headline", "Untitled"), article.get("source", "Unknown"), article.get("url", "")
                st.markdown(f"**[{headline}]({url})**  \n{source}" if url else f"**{headline}**  \n{source}")

    table_cols = ["ticker", "probability_up", "base_probability", "evidence_score", "net_expected_return", "signal_score", "action"]
    st.subheader("Comparison table")
    st.dataframe(results[table_cols], use_container_width=True, hide_index=True)

st.divider()
st.caption("Research and decision-support tool only. Options and institutional-flow readings are proxies unless connected to licensed professional feeds. No probability is a guarantee of price appreciation.")

if refresh_seconds:
    time.sleep(refresh_seconds)
    st.rerun()
