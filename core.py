from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from vektra_global_news_alpha_engine import (
    GlobalNewsAlphaEngine,
    MarketSnapshot,
    NewsItem,
    RuleBasedFinanceTextModel,
)

DATA_DIR = Path(os.getenv("VEKTRA_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_PATH = DATA_DIR / "latest_signals.json"
HISTORY_PATH = DATA_DIR / "signal_history.csv"
ALERT_STATE_PATH = DATA_DIR / "alert_state.json"
MARKET_SCAN_PATH = DATA_DIR / "market_scan.json"

DEFAULT_TICKERS = [
    "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "META", "AMZN", "GOOGL",
    "PLTR", "INTC", "ORCL", "MU", "AVGO", "CRWD", "COIN", "RIVN",
    "SOFI", "RKLB", "SMCI", "HOOD",
]



# Broad liquid US universe for the first production scanner release.
BROAD_US_UNIVERSE = [
    "AAPL","ABBV","ABNB","ADBE","ADI","ADP","ADSK","AEP","AMAT","AMD",
    "AMGN","AMT","AMZN","ANET","APP","ARM","ASML","AVGO","AXP","BA",
    "BAC","BKNG","BLK","BMY","BRK-B","C","CAT","CEG","CHTR","CMCSA",
    "COF","COIN","COP","COST","CRM","CRWD","CSCO","CVS","CVX","DASH",
    "DE","DELL","DIS","DKNG","DOCU","DUK","EA","EBAY","ELF","ENPH",
    "ETN","F","FCX","FDX","GE","GILD","GM","GOOG","GOOGL","GS",
    "HD","HON","HOOD","IBM","INTC","ISRG","JNJ","JPM","KO","LLY",
    "LOW","LRCX","LULU","MA","MARA","MCD","MCO","MDB","MDLZ","META",
    "MGM","MMM","MRK","MRNA","MRVL","MS","MSFT","MSTR","MU","NFLX",
    "NKE","NOW","NVDA","NVO","NXPI","ORCL","PANW","PEP","PFE","PG",
    "PLTR","PYPL","QCOM","RDDT","RBLX","RIVN","RKLB","ROKU","RTX",
    "SBUX","SHOP","SMCI","SNOW","SOFI","SPOT","SQ","T","TEAM","TEM",
    "TGT","TJX","TMO","TSLA","TSM","TTD","UBER","UNH","UPS","V",
    "VRT","VRTX","WFC","WMT","XOM","ZM","ZS"
]



# Broad, liquid UK universe using Yahoo Finance's London suffix (.L).
# UK-listed share prices returned by Yahoo Finance are commonly quoted in GBp
# (pence sterling), so the market metadata below identifies them accordingly.
BROAD_UK_UNIVERSE = [
    # Pharmaceuticals, healthcare and consumer staples
    "AZN.L", "GSK.L", "HLN.L", "HIK.L", "SN.L", "ULVR.L", "DGE.L",
    "BATS.L", "IMB.L", "ABF.L", "TSCO.L", "SBRY.L", "MKS.L", "NXT.L",

    # Energy, utilities and natural resources
    "SHEL.L", "BP.L", "RIO.L", "GLEN.L", "AAL.L", "ANTO.L", "FRES.L",
    "HOC.L", "S32.L", "SSE.L", "NG.L", "CNA.L", "SVT.L", "UU.L", "PNN.L",

    # Banks, insurance, asset management and financial services
    "HSBA.L", "BARC.L", "LLOY.L", "NWG.L", "STAN.L", "PRU.L", "AV.L",
    "LGEN.L", "MNG.L", "SDR.L", "III.L", "SMT.L", "LSEG.L",

    # Aerospace, defence, automotive and industrial engineering
    "RR.L", "BA.L", "QQ.L", "BAB.L", "MRO.L", "WEIR.L", "IMI.L",
    "SPX.L", "HLMA.L", "ROL.L", "DPLM.L", "BNZL.L", "RS1.L", "SXS.L",
    "VSVS.L", "GKN.L",

    # Technology, software, telecoms and media
    "SGE.L", "AUTO.L", "WPP.L", "ITV.L", "REL.L", "BT.A.L", "VOD.L",
    "MONY.L", "TRST.L", "BYIT.L", "OXIG.L", "NCC.L",

    # Travel, leisure, retail and services
    "IAG.L", "EZJ.L", "WIZZ.L", "CCL.L", "TUI.L", "IHG.L", "WTB.L",
    "CPG.L", "ENT.L", "FLTR.L", "JD.L", "FRAS.L", "BME.L", "KGF.L",

    # Property, construction and infrastructure
    "SGRO.L", "LAND.L", "BLND.L", "UTG.L", "BBOX.L", "PSN.L", "TW.L",
    "BDEV.L", "VTY.L", "CRST.L", "MGNS.L", "KIE.L", "BREE.L",
]

GLOBAL_UNIVERSE = list(dict.fromkeys(BROAD_US_UNIVERSE + BROAD_UK_UNIVERSE))

MARKET_UNIVERSES = {
    "USA": BROAD_US_UNIVERSE,
    "UK": BROAD_UK_UNIVERSE,
    "USA + UK": GLOBAL_UNIVERSE,
    "GLOBAL": GLOBAL_UNIVERSE,
}


def market_for_ticker(ticker: str) -> str:
    return "UK" if str(ticker).upper().endswith(".L") else "USA"


def currency_for_ticker(ticker: str) -> str:
    # Yahoo Finance normally returns LSE share prices in pence sterling.
    return "GBp" if market_for_ticker(ticker) == "UK" else "USD"


def get_market_universe(market: str = "USA + UK") -> list[str]:
    """Return a copy of the selected market universe."""
    key = str(market or "USA + UK").strip().upper()
    aliases = {
        "US": "USA",
        "UNITED STATES": "USA",
        "UNITED KINGDOM": "UK",
        "GB": "UK",
        "BOTH": "USA + UK",
        "USA+UK": "USA + UK",
        "US + UK": "USA + UK",
        "GLOBAL": "GLOBAL",
    }
    normalised = aliases.get(key, key)
    return list(MARKET_UNIVERSES.get(normalised, GLOBAL_UNIVERSE))


def finnhub_symbol_candidates(ticker: str) -> list[str]:
    """Return sensible Finnhub symbol fallbacks for company-level data.

    Some Finnhub plans accept the Yahoo-style .L symbol and others expose
    company data under the unsuffixed underlying symbol. The original symbol
    is always attempted first.
    """
    ticker = str(ticker).strip().upper()
    candidates = [ticker]
    if ticker.endswith(".L"):
        candidates.append(ticker[:-2])
    return list(dict.fromkeys(candidates))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if hasattr(value, "iloc"):
            value = value.iloc[-1]
        return float(value)
    except (TypeError, ValueError, IndexError):
        return default


def fetch_finnhub_news(ticker: str, api_key: str, lookback_days: int = 2) -> list[dict]:
    if not api_key:
        return []

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days)

    for symbol in finnhub_symbol_candidates(ticker):
        try:
            response = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": symbol,
                    "from": start.isoformat(),
                    "to": end.isoformat(),
                    "token": api_key,
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list) and payload:
                return payload[:30]
        except Exception:
            continue

    return []



def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


def logistic_score(value: float, midpoint: float, steepness: float = 1.0) -> float:
    """Map an unbounded input to a stable 0–100 score."""
    x = max(-60.0, min(60.0, steepness * (value - midpoint)))
    return 100.0 / (1.0 + math.exp(-x))


def calculate_rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) <= period:
        return 50.0
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(period).mean()
    losses = (-delta.clip(upper=0)).rolling(period).mean()
    loss = safe_float(losses.iloc[-1], 0.0)
    if loss <= 0:
        return 70.0
    rs = safe_float(gains.iloc[-1], 0.0) / loss
    return 100.0 - (100.0 / (1.0 + rs))


def fetch_analyst_score(ticker: str, api_key: str) -> tuple[float, bool, str]:
    """Finnhub recommendation consensus. Returns neutral if unavailable."""
    if not api_key:
        return 50.0, False, "Analyst feed not configured"
    try:
        response = requests.get(
            "https://finnhub.io/api/v1/stock/recommendation",
            params={"symbol": ticker, "token": api_key},
            timeout=12,
        )
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list) or not rows:
            return 50.0, False, "No analyst consensus returned"
        row = rows[0]
        strong_buy = safe_float(row.get("strongBuy"))
        buy = safe_float(row.get("buy"))
        hold = safe_float(row.get("hold"))
        sell = safe_float(row.get("sell"))
        strong_sell = safe_float(row.get("strongSell"))
        total = strong_buy + buy + hold + sell + strong_sell
        if total <= 0:
            return 50.0, False, "No analyst votes returned"
        weighted = (100*strong_buy + 78*buy + 50*hold + 22*sell + 0*strong_sell) / total
        return clamp(weighted), True, f"{int(strong_buy + buy)} buy vs {int(sell + strong_sell)} sell ratings"
    except Exception:
        return 50.0, False, "Analyst feed unavailable"


def fetch_insider_score(ticker: str, api_key: str) -> tuple[float, bool, str]:
    """Recent disclosed insider transactions; treated cautiously."""
    if not api_key:
        return 50.0, False, "Insider feed not configured"
    try:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=90)
        response = requests.get(
            "https://finnhub.io/api/v1/stock/insider-transactions",
            params={"symbol": ticker, "from": start.isoformat(), "to": end.isoformat(), "token": api_key},
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if not rows:
            return 50.0, False, "No recent insider data returned"
        bought = 0.0
        sold = 0.0
        for row in rows[:100]:
            change = safe_float(row.get("change"), 0.0)
            transaction_code = str(row.get("transactionCode", "")).upper()
            # P is commonly open-market purchase and S open-market sale.
            if transaction_code == "P" or change > 0:
                bought += abs(change)
            elif transaction_code == "S" or change < 0:
                sold += abs(change)
        total = bought + sold
        if total <= 0:
            return 50.0, False, "Insider transactions were not directional"
        score = 50.0 + 45.0 * ((bought - sold) / total)
        return clamp(score), True, f"Reported buys {bought:,.0f} shares; sells {sold:,.0f}"
    except Exception:
        return 50.0, False, "Insider feed unavailable"


def fetch_options_score(ticker: str) -> tuple[float, bool, str]:
    """Nearest-expiry call/put volume proxy. This is not verified institutional flow."""
    try:
        instrument = yf.Ticker(ticker)
        expiries = instrument.options
        if not expiries:
            return 50.0, False, "No listed options returned"
        chain = instrument.option_chain(expiries[0])
        call_volume = safe_float(chain.calls.get("volume", pd.Series(dtype=float)).fillna(0).sum(), 0.0)
        put_volume = safe_float(chain.puts.get("volume", pd.Series(dtype=float)).fillna(0).sum(), 0.0)
        total = call_volume + put_volume
        if total <= 0:
            return 50.0, False, "No options volume returned"
        call_share = call_volume / total
        score = clamp(15.0 + 70.0 * call_share)
        return score, True, f"Call volume share {call_share:.0%} (nearest expiry)"
    except Exception:
        return 50.0, False, "Options proxy unavailable"


def calculate_ai_score(components: dict[str, dict[str, Any]]) -> tuple[float, float]:
    """Renormalise weights across available evidence; do not invent missing data."""
    available = [item for item in components.values() if item.get("available", False)]
    used_weight = sum(float(item["weight"]) for item in available)
    total_weight = sum(float(item["weight"]) for item in components.values())
    if used_weight <= 0:
        return 50.0, 0.0
    score = sum(float(item["score"]) * float(item["weight"]) for item in available) / used_weight
    coverage = 100.0 * used_weight / total_weight
    return clamp(score), clamp(coverage)


def ai_label(score: float) -> str:
    if score >= 85:
        return "VERY STRONG"
    if score >= 75:
        return "STRONG"
    if score >= 65:
        return "POSITIVE"
    if score >= 50:
        return "NEUTRAL"
    if score >= 40:
        return "WEAK"
    return "VERY WEAK"

def fetch_market_snapshot(ticker: str) -> tuple[MarketSnapshot | None, dict[str, float]]:
    metrics: dict[str, float] = {
        "rsi": 50.0,
        "sma20_gap": 0.0,
        "sma50_gap": 0.0,
        "return_5d": 0.0,
    }
    try:
        intraday = yf.download(ticker, period="1mo", interval="30m", progress=False, auto_adjust=True, threads=False)
        if intraday is None or intraday.empty:
            return None, metrics
        if isinstance(intraday.columns, pd.MultiIndex):
            intraday.columns = [c[0] for c in intraday.columns]
        close = intraday["Close"].dropna()
        volume = intraday["Volume"].dropna()
        if len(close) < 4:
            return None, metrics

        daily = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True, threads=False)
        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = [c[0] for c in daily.columns]

        realised_vol = 0.35
        adv = 5_000_000.0
        if daily is not None and not daily.empty:
            dclose = daily["Close"].dropna()
            dvolume = daily["Volume"].dropna()
            if len(dclose) > 20:
                realised_vol = safe_float(dclose.pct_change().tail(20).std() * (252 ** 0.5), 0.35)
                metrics["rsi"] = calculate_rsi(dclose)
                sma20 = safe_float(dclose.tail(20).mean(), safe_float(dclose.iloc[-1]))
                metrics["sma20_gap"] = safe_float(dclose.iloc[-1] / sma20 - 1.0) if sma20 else 0.0
            if len(dclose) > 50:
                sma50 = safe_float(dclose.tail(50).mean(), safe_float(dclose.iloc[-1]))
                metrics["sma50_gap"] = safe_float(dclose.iloc[-1] / sma50 - 1.0) if sma50 else 0.0
            if len(dclose) > 5:
                metrics["return_5d"] = safe_float(dclose.pct_change(5).iloc[-1], 0.0)
            if len(dclose) and len(dvolume):
                adv = safe_float((dclose.tail(20) * dvolume.tail(20)).mean(), adv)

        ret_30m = safe_float(close.pct_change().iloc[-1])
        ret_1d = safe_float(close.pct_change(13).iloc[-1]) if len(close) > 13 else ret_30m
        base_volume = safe_float(volume.tail(10).mean(), 1.0)
        volume_ratio = safe_float(volume.iloc[-1] / base_volume, 1.0) if base_volume else 1.0

        snapshot = MarketSnapshot(
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            last_price=safe_float(close.iloc[-1]),
            return_5m=ret_30m * 0.35,
            return_30m=ret_30m,
            return_1d=ret_1d,
            market_return_30m=0.0,
            sector_return_30m=0.0,
            volume_ratio_30m=volume_ratio,
            spread_bps=15.0,
            realised_volatility_20d=max(0.05, min(realised_vol, 2.0)),
            average_daily_value_gbp=max(100_000.0, adv),
        )
        return snapshot, metrics
    except Exception:
        return None, metrics



NEWS_SOURCE_TIERS = {
    "reuters": ("High", 95),
    "bloomberg": ("High", 94),
    "associated press": ("High", 92),
    "wall street journal": ("High", 92),
    "financial times": ("High", 92),
    "cnbc": ("High", 86),
    "marketwatch": ("Medium", 76),
    "seekingalpha": ("Medium", 67),
    "benzinga": ("Medium", 66),
    "yahoo": ("Medium", 64),
}


def source_quality(source: str) -> tuple[str, int]:
    text = str(source or "").lower()
    for key, value in NEWS_SOURCE_TIERS.items():
        if key in text:
            return value
    if text in {"system", "unknown", ""}:
        return "Unknown", 40
    return "Medium", 62


def news_age_label(timestamp: int | float | None) -> tuple[str, float]:
    try:
        published = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        hours = max(0.0, (datetime.now(timezone.utc) - published).total_seconds() / 3600)
    except Exception:
        return "Unknown age", 0.45

    if hours < 1:
        return f"{max(1, int(hours * 60))} min ago", 1.00
    if hours < 24:
        return f"{int(hours)} hr ago", max(0.55, 1.0 - hours / 48.0)
    days = int(hours / 24)
    return f"{days} day{'s' if days != 1 else ''} ago", max(0.15, 0.55 - days * 0.10)


def intelligent_news_card(ticker: str, item: dict) -> dict:
    """Create transparent metadata for each financial-news card."""
    headline = str(item.get("headline", "") or "")
    summary = str(item.get("summary", "") or "")
    source = str(item.get("source", "Unknown source") or "Unknown source")
    text = f"{headline}. {summary}".strip()

    model = RuleBasedFinanceTextModel()
    analysis = model.analyse(text, ticker)

    direction = float(analysis.get("event_direction", 0.0))
    materiality = float(analysis.get("materiality", 0.25))
    surprise = float(analysis.get("expectation_surprise", 0.0))
    manipulation = float(analysis.get("manipulation_risk", 0.0))
    event_type = str(analysis.get("event_type", "other")).replace("_", " ").title()

    quality_label, quality_score = source_quality(source)
    age_text, freshness = news_age_label(item.get("datetime"))

    direction_score = 50.0 + 50.0 * direction
    impact_score = clamp(
        0.34 * direction_score
        + 0.27 * (100.0 * materiality)
        + 0.18 * (100.0 * surprise)
        + 0.13 * quality_score
        + 0.08 * (100.0 * freshness)
        - 35.0 * manipulation
    )

    if direction >= 0.28:
        sentiment_label, sentiment_icon = "Bullish", "🟢"
    elif direction <= -0.28:
        sentiment_label, sentiment_icon = "Bearish", "🔴"
    else:
        sentiment_label, sentiment_icon = "Neutral", "🟠"

    if impact_score >= 80:
        impact_label = "Very high"
    elif impact_score >= 65:
        impact_label = "High"
    elif impact_score >= 50:
        impact_label = "Moderate"
    else:
        impact_label = "Low"

    reasons = list(analysis.get("explanation", []))
    reason = reasons[0] if reasons else f"{event_type} event detected"

    return {
        **item,
        "ticker": ticker,
        "event_type": event_type,
        "sentiment_label": sentiment_label,
        "sentiment_icon": sentiment_icon,
        "impact_score": round(impact_score, 1),
        "impact_label": impact_label,
        "materiality_score": round(100.0 * materiality, 1),
        "source_quality": quality_label,
        "source_quality_score": quality_score,
        "freshness": round(100.0 * freshness, 1),
        "age_text": age_text,
        "reason": reason,
        "manipulation_risk": round(100.0 * manipulation, 1),
    }


def build_news_cards(ticker: str, articles: list[dict]) -> list[dict]:
    cards = [intelligent_news_card(ticker, item) for item in articles]
    cards.sort(
        key=lambda item: (
            float(item.get("impact_score", 0)),
            float(item.get("datetime", 0) or 0),
        ),
        reverse=True,
    )
    return cards


def run_scan(tickers: list[str], finnhub_api_key: str) -> list[dict]:
    engine = GlobalNewsAlphaEngine(RuleBasedFinanceTextModel())
    snapshots: list[MarketSnapshot] = []
    news_by_ticker: dict[str, list[dict]] = {}
    technical_by_ticker: dict[str, dict[str, float]] = {}
    auxiliary_by_ticker: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        try:
            articles = fetch_finnhub_news(ticker, finnhub_api_key)
        except Exception as exc:
            articles = [{"headline": f"News feed error: {exc}", "source": "system", "url": ""}]
        news_by_ticker[ticker] = build_news_cards(ticker, articles[:10])

        for item in articles:
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

        snapshot, technical = fetch_market_snapshot(ticker)
        if snapshot:
            snapshots.append(snapshot)
            technical_by_ticker[ticker] = technical

        analyst_score, analyst_available, analyst_note = fetch_analyst_score(ticker, finnhub_api_key)
        insider_score, insider_available, insider_note = fetch_insider_score(ticker, finnhub_api_key)
        options_score, options_available, options_note = fetch_options_score(ticker)
        auxiliary_by_ticker[ticker] = {
            "analyst": (analyst_score, analyst_available, analyst_note),
            "insider": (insider_score, insider_available, insider_note),
            "options": (options_score, options_available, options_note),
        }

    if not snapshots:
        return []

    ranked = engine.rank_watchlist(
        snapshots,
        market_breadth=0.0,
        volatility_percentile=0.5,
        index_trend=0.0,
    )
    now = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    snapshot_map = {s.ticker: s for s in snapshots}

    for row in ranked.to_dict(orient="records"):
        ticker = row["ticker"]
        market = snapshot_map[ticker]
        technical = technical_by_ticker.get(ticker, {})
        event_alpha, _ = engine.aggregate_event_alpha(ticker, now=datetime.now(timezone.utc))

        news_score = clamp(20.0 + 80.0 * event_alpha) if news_by_ticker.get(ticker) else 35.0
        momentum_raw = 0.50 * market.return_30m + 0.30 * market.return_1d + 0.20 * technical.get("return_5d", 0.0)
        momentum_score = logistic_score(momentum_raw, midpoint=0.0, steepness=45.0)
        volume_score = logistic_score(market.volume_ratio_30m, midpoint=1.0, steepness=1.8)

        rsi = technical.get("rsi", 50.0)
        rsi_score = 100.0 - min(100.0, abs(rsi - 60.0) * 3.0)  # rewards constructive, not extreme, RSI
        trend_score = clamp(
            50.0
            + 500.0 * technical.get("sma20_gap", 0.0)
            + 250.0 * technical.get("sma50_gap", 0.0)
        )
        technical_score = clamp(0.70 * trend_score + 0.30 * rsi_score)

        analyst_score, analyst_available, analyst_note = auxiliary_by_ticker[ticker]["analyst"]
        insider_score, insider_available, insider_note = auxiliary_by_ticker[ticker]["insider"]
        options_score, options_available, options_note = auxiliary_by_ticker[ticker]["options"]

        components = {
            "News quality": {"score": news_score, "weight": 25.0, "available": bool(news_by_ticker.get(ticker)), "note": f"Event alpha {event_alpha:.2f}"},
            "Price momentum": {"score": momentum_score, "weight": 20.0, "available": True, "note": f"30m {market.return_30m:.2%}; 1d {market.return_1d:.2%}"},
            "Relative volume": {"score": volume_score, "weight": 15.0, "available": True, "note": f"{market.volume_ratio_30m:.2f}× recent intraday average"},
            "Technical trend": {"score": technical_score, "weight": 15.0, "available": True, "note": f"RSI {rsi:.1f}; 20d gap {technical.get('sma20_gap',0):.2%}"},
            "Analyst activity": {"score": analyst_score, "weight": 10.0, "available": analyst_available, "note": analyst_note},
            "Insider activity": {"score": insider_score, "weight": 10.0, "available": insider_available, "note": insider_note},
            "Options positioning": {"score": options_score, "weight": 5.0, "available": options_available, "note": options_note},
        }
        ai_score, evidence_coverage = calculate_ai_score(components)

        records.append({
            **row,
            "scan_time": now,
            "last_price": market.last_price,
            "return_30m": market.return_30m,
            "return_1d": market.return_1d,
            "volume_ratio_30m": market.volume_ratio_30m,
            "ai_score": round(ai_score, 1),
            "ai_label": ai_label(ai_score),
            "evidence_coverage": round(evidence_coverage, 1),
            "score_components": components,
            "news": news_by_ticker.get(ticker, []),
        })

    records.sort(key=lambda item: (float(item.get("ai_score", 0)), float(item.get("probability_up", 0))), reverse=True)
    return records



def _normalise_batch_download(data: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    if data is None or data.empty:
        return result

    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(str(x) for x in data.columns.get_level_values(0))
        if {"Close", "Volume"}.intersection(level0):
            for ticker in tickers:
                try:
                    frame = data.xs(ticker, axis=1, level=1, drop_level=True).copy()
                    if not frame.empty:
                        result[ticker] = frame
                except Exception:
                    continue
        else:
            for ticker in tickers:
                try:
                    frame = data.xs(ticker, axis=1, level=0, drop_level=True).copy()
                    if not frame.empty:
                        result[ticker] = frame
                except Exception:
                    continue
    elif len(tickers) == 1:
        result[tickers[0]] = data.copy()

    return result


def prefilter_market_universe(
    tickers: list[str] | None = None,
    candidate_count: int = 15,
) -> list[dict[str, Any]]:
    """Fast broad-market pre-screen using movement, volume and trend."""
    tickers = list(dict.fromkeys(tickers or GLOBAL_UNIVERSE))
    rows: list[dict[str, Any]] = []

    for start in range(0, len(tickers), 40):
        batch = tickers[start:start + 40]
        try:
            data = yf.download(
                batch,
                period="5d",
                interval="30m",
                group_by="column",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception:
            continue

        for ticker, frame in _normalise_batch_download(data, batch).items():
            try:
                close = frame["Close"].dropna()
                volume = frame["Volume"].dropna()
                if len(close) < 8 or len(volume) < 8:
                    continue

                last_price = safe_float(close.iloc[-1])
                if last_price < 3.0:
                    continue

                return_30m = safe_float(close.pct_change().iloc[-1])
                return_1d = safe_float(close.pct_change(13).iloc[-1]) if len(close) > 13 else 0.0
                baseline_volume = safe_float(volume.tail(20).iloc[:-1].median(), 0.0)
                relative_volume = (
                    safe_float(volume.iloc[-1]) / baseline_volume
                    if baseline_volume > 0 else 1.0
                )

                trend = 0.0
                if len(close) >= 20:
                    fast = safe_float(close.tail(6).mean())
                    slow = safe_float(close.tail(20).mean())
                    trend = (fast / slow - 1.0) if slow else 0.0

                movement_score = min(abs(return_30m) / 0.035, 1.0)
                daily_score = min(abs(return_1d) / 0.08, 1.0)
                volume_score = min(max(relative_volume - 1.0, 0.0) / 3.0, 1.0)
                trend_score = min(abs(trend) / 0.04, 1.0)

                prefilter_score = 100.0 * (
                    0.35 * movement_score
                    + 0.30 * volume_score
                    + 0.20 * daily_score
                    + 0.15 * trend_score
                )

                rows.append({
                    "ticker": ticker,
                    "market": market_for_ticker(ticker),
                    "currency": currency_for_ticker(ticker),
                    "prefilter_score": round(prefilter_score, 1),
                    "last_price": round(last_price, 4),
                    "return_30m": round(return_30m, 6),
                    "return_1d": round(return_1d, 6),
                    "relative_volume": round(relative_volume, 2),
                    "trend": round(trend, 6),
                })
            except Exception:
                continue

    rows.sort(key=lambda item: float(item.get("prefilter_score", 0)), reverse=True)
    return rows[:max(1, int(candidate_count))]


def run_broad_market_scan(
    finnhub_api_key: str,
    universe: list[str] | None = None,
    candidate_count: int = 15,
    market: str = "USA + UK",
) -> tuple[list[dict], list[dict]]:
    selected_universe = universe if universe is not None else get_market_universe(market)
    candidates = prefilter_market_universe(selected_universe, candidate_count)
    candidate_tickers = [item["ticker"] for item in candidates]
    detailed = run_scan(candidate_tickers, finnhub_api_key) if candidate_tickers else []

    prefilter_lookup = {item["ticker"]: item for item in candidates}
    for record in detailed:
        ticker = str(record.get("ticker", ""))
        record["market"] = market_for_ticker(ticker)
        record["currency"] = currency_for_ticker(ticker)
        record["market_prefilter"] = prefilter_lookup.get(ticker, {})
        record["discovery_source"] = f'{record["market"]} broad market scanner' 

    detailed.sort(
        key=lambda item: (
            float(item.get("ai_score", 0)),
            float(item.get("probability_up", 0)),
            float(item.get("market_prefilter", {}).get("prefilter_score", 0)),
        ),
        reverse=True,
    )
    return detailed, candidates


def save_market_scan(records: list[dict], candidates: list[dict]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "markets": sorted({
            str(item.get("market", market_for_ticker(item.get("ticker", ""))))
            for item in candidates
        }),
        "records": records,
        "candidates": candidates,
    }
    MARKET_SCAN_PATH.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def load_market_scan() -> dict[str, Any]:
    if not MARKET_SCAN_PATH.exists():
        return {}
    try:
        return json.loads(MARKET_SCAN_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_results(records: list[dict]) -> None:
    SIGNALS_PATH.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    if records:
        flat = [{k: v for k, v in r.items() if k != "news"} for r in records]
        frame = pd.DataFrame(flat)
        frame.to_csv(HISTORY_PATH, mode="a", index=False, header=not HISTORY_PATH.exists())


def load_results() -> list[dict]:
    if not SIGNALS_PATH.exists():
        return []
    try:
        return json.loads(SIGNALS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def send_pushover(title: str, message: str, url: str = "") -> None:
    token = os.getenv("PUSHOVER_APP_TOKEN", "")
    user = os.getenv("PUSHOVER_USER_KEY", "")
    if not token or not user:
        return
    payload = {"token": token, "user": user, "title": title[:250], "message": message[:1024]}
    if url:
        payload.update({"url": url[:512], "url_title": "Open Vektra Alpha"})
    response = requests.post("https://api.pushover.net/1/messages.json", data=payload, timeout=20)
    response.raise_for_status()


def send_new_signal_alerts(records: list[dict], threshold: float, app_url: str = "") -> int:
    state = {}
    if ALERT_STATE_PATH.exists():
        try:
            state = json.loads(ALERT_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            state = {}

    sent = 0
    for record in records:
        probability = float(record.get("probability_up", 0))
        ticker = record.get("ticker", "")
        prior = float(state.get(ticker, 0))
        if probability >= threshold and prior < threshold:
            send_pushover(
                f"Vektra Alpha: {ticker}",
                f"{record.get('action', 'WATCH')} | Probability up {probability:.1%} | Net expected return {float(record.get('net_expected_return', 0)):.2%}",
                app_url,
            )
            sent += 1
        state[ticker] = probability

    ALERT_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return sent


def fetch_market_overview() -> list[dict[str, Any]]:
    """Return a compact US market overview using liquid index proxies."""
    instruments = [
        ("S&P 500", "^GSPC"),
        ("Nasdaq", "^IXIC"),
        ("Dow", "^DJI"),
        ("VIX", "^VIX"),
    ]
    rows: list[dict[str, Any]] = []
    for name, ticker in instruments:
        try:
            data = yf.download(
                ticker,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0] for c in data.columns]
            close = data["Close"].dropna() if data is not None and not data.empty else pd.Series(dtype=float)
            if len(close) < 2:
                raise ValueError("Insufficient market data")
            last = safe_float(close.iloc[-1])
            change = safe_float(close.pct_change().iloc[-1])
            rows.append({"name": name, "ticker": ticker, "last": last, "change": change, "available": True})
        except Exception:
            rows.append({"name": name, "ticker": ticker, "last": 0.0, "change": 0.0, "available": False})
    return rows


def fetch_price_history(ticker: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    """Return a simple chart-ready price history."""
    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if data is None or data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        close = data[["Close"]].dropna().copy()
        close.columns = [ticker]
        return close
    except Exception:
        return pd.DataFrame()
