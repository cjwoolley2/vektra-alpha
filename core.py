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
AI_NEWS_RADAR_PATH = DATA_DIR / "ai_news_radar.json"
AI_NEWS_RADAR_STATE_PATH = DATA_DIR / "ai_news_radar_state.json"

# ---------------------------------------------------------------------------
# VEKTRA ALPHA — AI ECOSYSTEM UNIVERSE
# ---------------------------------------------------------------------------
# The scanner is intentionally focused on companies with meaningful exposure
# to the AI value chain: compute, semiconductors, memory, networking,
# photonics, data centres, power/cooling, cloud platforms, AI software,
# cybersecurity and selected AI-adjacent industrial technology.
#
# Compatibility note:
# BROAD_US_UNIVERSE / BROAD_UK_UNIVERSE / GLOBAL_UNIVERSE are retained because
# the existing Streamlit dashboard imports those names. They now point only to
# the curated AI ecosystem rather than the general stock market.

AI_THEME_MAP = {
    # Compute / accelerators / CPU
    "NVDA": "AI Compute",
    "AMD": "AI Compute",
    "ARM": "AI Compute",
    "INTC": "AI Compute",
    "QCOM": "AI Compute",

    # Custom silicon / semiconductor design
    "AVGO": "Custom AI Silicon",
    "MRVL": "Custom AI Silicon",
    "ADI": "Semiconductor Infrastructure",
    "NXPI": "Semiconductor Infrastructure",

    # Foundry / semiconductor equipment
    "TSM": "Foundry & Manufacturing",
    "ASML": "Foundry & Manufacturing",
    "AMAT": "Foundry & Manufacturing",
    "LRCX": "Foundry & Manufacturing",
    "KLAC": "Foundry & Manufacturing",
    "TER": "Foundry & Manufacturing",

    # Memory / storage
    "MU": "AI Memory",
    "WDC": "AI Storage",
    "STX": "AI Storage",

    # Networking / connectivity
    "ANET": "AI Networking",
    "CSCO": "AI Networking",
    "CIEN": "AI Networking",
    "CRDO": "AI Networking",
    "ALAB": "AI Networking",

    # Photonics / optical interconnect
    "COHR": "Photonics & Optics",
    "LITE": "Photonics & Optics",
    "AAOI": "Photonics & Optics",

    # Servers / systems
    "SMCI": "AI Servers",
    "DELL": "AI Servers",
    "HPE": "AI Servers",

    # Data-centre infrastructure / cooling / electrical
    "VRT": "Data Centre Infrastructure",
    "ETN": "Data Centre Infrastructure",
    "PWR": "Data Centre Infrastructure",
    "NVT": "Data Centre Infrastructure",
    "TT": "Data Centre Infrastructure",
    "FIX": "Data Centre Infrastructure",

    # Power generation / grid capacity
    "CEG": "AI Power",
    "VST": "AI Power",
    "GEV": "AI Power",
    "NRG": "AI Power",
    "TLN": "AI Power",

    # Hyperscalers / AI platforms
    "MSFT": "Hyperscaler AI",
    "GOOGL": "Hyperscaler AI",
    "GOOG": "Hyperscaler AI",
    "AMZN": "Hyperscaler AI",
    "META": "Hyperscaler AI",
    "ORCL": "Hyperscaler AI",
    "IBM": "Enterprise AI",

    # AI software / data / applications
    "PLTR": "AI Software",
    "SNOW": "AI Software",
    "NOW": "AI Software",
    "CRM": "AI Software",
    "ADBE": "AI Software",
    "APP": "AI Applications",
    "PATH": "AI Automation",
    "AI": "AI Software",
    "SOUN": "AI Applications",
    "BBAI": "AI Applications",
    "UPST": "AI Applications",
    "TEM": "AI Healthcare",
    "RXRX": "AI Healthcare",

    # Cybersecurity increasingly driven by AI workloads / inference
    "CRWD": "AI Cybersecurity",
    "PANW": "AI Cybersecurity",
    "ZS": "AI Cybersecurity",
    "FTNT": "AI Cybersecurity",

    # Neocloud / AI compute infrastructure
    "CRWV": "AI Cloud",
    "NBIS": "AI Cloud",
    "APLD": "AI Cloud",

    # Robotics / autonomy / edge AI
    "TSLA": "AI Robotics & Autonomy",
    "ISRG": "AI Robotics & Autonomy",
    "SYM": "AI Robotics & Automation",
    "SERV": "AI Robotics & Automation",
    "RKLB": "AI Aerospace & Autonomy",
    "AVAV": "AI Aerospace & Autonomy",

    # AI-enabled semiconductor / test / engineering adjacency
    "AEHR": "Semiconductor Test",
    "ONTO": "Semiconductor Test",
    "ACLS": "Semiconductor Equipment",

    # UK-listed AI / data / software / technology exposure
    "REL.L": "UK AI Data & Analytics",
    "SGE.L": "UK AI Software",
    "BYIT.L": "UK AI Software",
    "OXIG.L": "UK AI Technology",
    "NCC.L": "UK AI Cybersecurity",
    "AUTO.L": "UK AI Digital Platforms",
    "TRST.L": "UK AI Software",
    "CCC.L": "UK AI IT Services",
    "SCT.L": "UK AI Software",
    "KNOS.L": "UK AI Software",
}

AI_US_UNIVERSE = [
    ticker for ticker in AI_THEME_MAP
    if not ticker.endswith(".L")
]

AI_UK_UNIVERSE = [
    ticker for ticker in AI_THEME_MAP
    if ticker.endswith(".L")
]

AI_GLOBAL_UNIVERSE = list(dict.fromkeys(AI_US_UNIVERSE + AI_UK_UNIVERSE))

# Existing dashboard compatibility names.
DEFAULT_TICKERS = [
    "NVDA", "AMD", "AVGO", "ARM", "TSM", "MU", "ANET", "VRT",
    "MSFT", "GOOGL", "AMZN", "META", "PLTR", "ORCL", "SMCI",
    "CRWV", "NBIS", "COHR", "LITE", "CEG",
]

BROAD_US_UNIVERSE = AI_US_UNIVERSE
BROAD_UK_UNIVERSE = AI_UK_UNIVERSE
GLOBAL_UNIVERSE = AI_GLOBAL_UNIVERSE

MARKET_UNIVERSES = {
    "USA": AI_US_UNIVERSE,
    "UK": AI_UK_UNIVERSE,
    "USA + UK": AI_GLOBAL_UNIVERSE,
    "GLOBAL": AI_GLOBAL_UNIVERSE,
    "AI": AI_GLOBAL_UNIVERSE,
}


def ai_theme_for_ticker(ticker: str) -> str:
    return AI_THEME_MAP.get(str(ticker).upper(), "AI Ecosystem")


def market_for_ticker(ticker: str) -> str:
    return "UK" if str(ticker).upper().endswith(".L") else "USA"


def currency_for_ticker(ticker: str) -> str:
    return "GBp" if market_for_ticker(ticker) == "UK" else "USD"


def get_market_universe(market: str = "AI") -> list[str]:
    """Return the selected AI-focused universe."""
    key = str(market or "AI").strip().upper()
    aliases = {
        "US": "USA",
        "UNITED STATES": "USA",
        "UNITED KINGDOM": "UK",
        "GB": "UK",
        "BOTH": "USA + UK",
        "USA+UK": "USA + UK",
        "US + UK": "USA + UK",
        "GLOBAL": "GLOBAL",
        "AI ONLY": "AI",
        "AI ECOSYSTEM": "AI",
    }
    normalised = aliases.get(key, key)
    return list(MARKET_UNIVERSES.get(normalised, AI_GLOBAL_UNIVERSE))


def finnhub_symbol_candidates(ticker: str) -> list[str]:
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



# ---------------------------------------------------------------------------
# AI NEWS + GEOPOLITICAL EARLY-WARNING RADAR
# ---------------------------------------------------------------------------

AI_NEWS_TOPICS = {
    "AI Compute": [
        "nvidia", "gpu", "accelerator", "cuda", "ai chip", "inference chip",
        "training chip", "blackwell", "rubin", "mi300", "mi350",
    ],
    "Semiconductor Supply": [
        "semiconductor", "foundry", "wafer", "fab", "tsmc", "asml",
        "chip equipment", "advanced packaging", "cowos", "hbm",
    ],
    "AI Memory": [
        "hbm", "high bandwidth memory", "dram", "nand", "memory chip", "micron",
    ],
    "AI Networking": [
        "ethernet", "infiniband", "networking", "optical interconnect",
        "switch", "transceiver", "silicon photonics", "coherent", "lumentum",
    ],
    "AI Data Centres": [
        "data center", "data centre", "hyperscale", "hyperscaler", "ai factory",
        "server rack", "liquid cooling", "cooling", "power density",
    ],
    "AI Power": [
        "nuclear", "power plant", "grid", "electricity", "power generation",
        "natural gas", "reactor", "smr", "data center power", "data centre power",
    ],
    "AI Cloud": [
        "cloud", "neocloud", "gpu cloud", "compute capacity", "oracle cloud",
        "azure", "aws", "google cloud",
    ],
    "AI Software": [
        "artificial intelligence", "generative ai", "agentic ai", "ai agent",
        "copilot", "large language model", "llm", "foundation model",
    ],
    "AI Robotics": [
        "robot", "robotics", "humanoid", "autonomous", "autonomy", "physical ai",
    ],
    "AI Cybersecurity": [
        "cybersecurity", "cyber attack", "cyberattack", "ransomware",
        "data breach", "ai security",
    ],
}

GEOPOLITICAL_TOPICS = {
    "US-China Technology": [
        "export control", "export restriction", "china chip", "chinese chip",
        "commerce department", "entity list", "technology restriction",
        "advanced semiconductor restriction", "chip ban",
    ],
    "Taiwan Risk": [
        "taiwan", "taiwan strait", "tsmc", "pla", "military exercise",
        "blockade", "invasion", "strait tension",
    ],
    "Tariffs & Trade": [
        "tariff", "trade war", "import duty", "customs duty", "retaliatory tariff",
    ],
    "Sanctions": [
        "sanction", "sanctions", "blacklist", "entity list", "restricted party",
    ],
    "Middle East / Energy": [
        "iran", "israel", "red sea", "strait of hormuz", "middle east",
        "oil supply", "shipping disruption", "missile", "drone attack",
    ],
    "Critical Minerals": [
        "rare earth", "gallium", "germanium", "antimony", "critical mineral",
        "graphite", "mineral export restriction",
    ],
    "AI Regulation": [
        "ai regulation", "ai act", "artificial intelligence regulation",
        "model safety", "ai safety law", "data sovereignty",
    ],
}

GEOPOLITICAL_IMPACT_MAP = {
    "US-China Technology": {
        "positive_themes": ["Foundry & Manufacturing", "AI Memory", "AI Networking"],
        "negative_themes": ["AI Compute", "Custom AI Silicon", "Foundry & Manufacturing"],
    },
    "Taiwan Risk": {
        "positive_themes": ["AI Power", "Data Centre Infrastructure"],
        "negative_themes": ["Foundry & Manufacturing", "AI Compute", "Custom AI Silicon"],
    },
    "Tariffs & Trade": {
        "positive_themes": [],
        "negative_themes": ["AI Compute", "Foundry & Manufacturing", "Semiconductor Infrastructure"],
    },
    "Sanctions": {
        "positive_themes": [],
        "negative_themes": ["AI Compute", "Custom AI Silicon", "Foundry & Manufacturing"],
    },
    "Middle East / Energy": {
        "positive_themes": ["AI Power"],
        "negative_themes": ["Data Centre Infrastructure", "AI Servers"],
    },
    "Critical Minerals": {
        "positive_themes": [],
        "negative_themes": ["Foundry & Manufacturing", "Semiconductor Infrastructure"],
    },
    "AI Regulation": {
        "positive_themes": ["AI Cybersecurity"],
        "negative_themes": ["AI Software", "Hyperscaler AI"],
    },
}

HIGH_CREDIBILITY_NEWS_SOURCES = {
    "reuters": 96,
    "associated press": 94,
    "bloomberg": 94,
    "financial times": 93,
    "wall street journal": 93,
    "bbc": 91,
    "cnbc": 86,
    "the guardian": 84,
    "marketwatch": 78,
    "yahoo": 68,
}


def fetch_finnhub_market_news(api_key: str, min_id: int = 0) -> list[dict]:
    if not api_key:
        return []
    try:
        response = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "minId": int(min_id), "token": api_key},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def fetch_gdelt_ai_geopolitical_news(max_records: int = 75) -> list[dict]:
    query = (
        '("artificial intelligence" OR "AI chips" OR semiconductor OR Nvidia OR TSMC '
        'OR "data center" OR "data centre" OR "export controls" OR Taiwan '
        'OR "rare earths" OR "critical minerals" OR "AI regulation")'
    )
    try:
        response = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={
                "query": query,
                "mode": "artlist",
                "maxrecords": min(max(10, int(max_records)), 250),
                "format": "json",
                "sort": "datedesc",
            },
            timeout=25,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    articles = payload.get("articles", []) if isinstance(payload, dict) else []
    output = []
    for item in articles:
        output.append({
            "id": str(item.get("url", "")),
            "datetime": item.get("seendate", ""),
            "headline": item.get("title", ""),
            "summary": "",
            "source": item.get("domain", "GDELT"),
            "url": item.get("url", ""),
            "related": "",
            "source_feed": "GDELT global news",
        })
    return output


def _radar_text(item: dict) -> str:
    return " ".join([
        str(item.get("headline", "") or ""),
        str(item.get("summary", "") or ""),
        str(item.get("related", "") or ""),
    ]).lower()


def _keyword_hits(text: str, taxonomy: dict[str, list[str]]) -> dict[str, list[str]]:
    results = {}
    for topic, keywords in taxonomy.items():
        hits = [keyword for keyword in keywords if keyword in text]
        if hits:
            results[topic] = hits
    return results


def _source_credibility_score(source: str) -> float:
    source_lower = str(source or "").lower()
    for name, score in HIGH_CREDIBILITY_NEWS_SOURCES.items():
        if name in source_lower:
            return float(score)
    if source_lower.endswith(".gov") or ".gov." in source_lower:
        return 96.0
    return 62.0


def _news_timestamp(item: dict) -> datetime:
    raw = item.get("datetime")
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)

    text = str(raw or "")
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _freshness_score(item: dict) -> tuple[float, str]:
    published = _news_timestamp(item)
    age_minutes = max(
        0.0,
        (datetime.now(timezone.utc) - published).total_seconds() / 60.0,
    )
    if age_minutes <= 15:
        return 100.0, f"{int(age_minutes)}m"
    if age_minutes <= 60:
        return 92.0, f"{int(age_minutes)}m"
    if age_minutes <= 180:
        return 80.0, f"{age_minutes / 60:.1f}h"
    if age_minutes <= 720:
        return 65.0, f"{age_minutes / 60:.1f}h"
    if age_minutes <= 1440:
        return 50.0, f"{age_minutes / 60:.1f}h"
    return 25.0, f"{age_minutes / 1440:.1f}d"


def _themes_to_tickers(themes: list[str], limit: int = 12) -> list[str]:
    themes_set = set(themes)
    return [
        ticker for ticker, theme in AI_THEME_MAP.items()
        if theme in themes_set
    ][:limit]


def _direct_ticker_mentions(text: str) -> list[str]:
    mentions = []
    for ticker in AI_GLOBAL_UNIVERSE:
        raw = ticker.replace(".L", "")
        if raw.lower() in text:
            mentions.append(ticker)
    return list(dict.fromkeys(mentions))


def classify_ai_market_news(item: dict) -> dict | None:
    text = _radar_text(item)
    ai_hits = _keyword_hits(text, AI_NEWS_TOPICS)
    geo_hits = _keyword_hits(text, GEOPOLITICAL_TOPICS)

    if not ai_hits and not geo_hits:
        return None

    source = str(item.get("source", "Unknown") or "Unknown")
    credibility = _source_credibility_score(source)
    freshness, age_label = _freshness_score(item)
    direct_mentions = _direct_ticker_mentions(text)

    ai_hit_count = sum(len(values) for values in ai_hits.values())
    geo_hit_count = sum(len(values) for values in geo_hits.values())

    materiality = clamp(
        40
        + min(ai_hit_count, 6) * 5
        + min(geo_hit_count, 5) * 7
        + (10 if direct_mentions else 0)
    )

    surprise_terms = (
        "unexpected", "emergency", "ban", "restriction", "halt", "suspend",
        "approval", "deal", "investment", "billion", "trillion", "attack",
        "sanction", "tariff", "shortage", "disruption", "warning",
    )
    surprise = clamp(30 + 10 * sum(term in text for term in surprise_terms))

    positive_themes = []
    negative_themes = []
    for topic in geo_hits:
        mapping = GEOPOLITICAL_IMPACT_MAP.get(topic, {})
        positive_themes.extend(mapping.get("positive_themes", []))
        negative_themes.extend(mapping.get("negative_themes", []))

    topic_theme_map = {
        "AI Compute": ["AI Compute", "Custom AI Silicon"],
        "Semiconductor Supply": ["Foundry & Manufacturing", "Semiconductor Infrastructure"],
        "AI Memory": ["AI Memory", "AI Storage"],
        "AI Networking": ["AI Networking", "Photonics & Optics"],
        "AI Data Centres": ["Data Centre Infrastructure", "AI Servers"],
        "AI Power": ["AI Power", "Data Centre Infrastructure"],
        "AI Cloud": ["AI Cloud", "Hyperscaler AI"],
        "AI Software": ["AI Software", "Hyperscaler AI", "Enterprise AI"],
        "AI Robotics": ["AI Robotics & Autonomy", "AI Robotics & Automation"],
        "AI Cybersecurity": ["AI Cybersecurity"],
    }
    relevant_ai_themes = []
    for topic in ai_hits:
        relevant_ai_themes.extend(topic_theme_map.get(topic, []))

    relevant_ai_themes = list(dict.fromkeys(relevant_ai_themes))
    positive_themes = list(dict.fromkeys(positive_themes))
    negative_themes = list(dict.fromkeys(negative_themes))

    relevance = clamp(
        45
        + min(len(relevant_ai_themes), 4) * 10
        + min(len(direct_mentions), 4) * 8
        + min(len(geo_hits), 3) * 6
    )

    impact_score = clamp(
        0.28 * materiality
        + 0.20 * surprise
        + 0.22 * relevance
        + 0.15 * credibility
        + 0.15 * freshness
    )

    if impact_score >= 90:
        impact_label, impact_icon = "CRITICAL", "🚨"
    elif impact_score >= 80:
        impact_label, impact_icon = "HIGH IMPACT", "🔴"
    elif impact_score >= 65:
        impact_label, impact_icon = "IMPORTANT", "🟠"
    elif impact_score >= 50:
        impact_label, impact_icon = "WATCH", "🟡"
    else:
        impact_label, impact_icon = "INFORMATION", "⚪"

    return {
        **item,
        "published_at": _news_timestamp(item).isoformat(),
        "age": age_label,
        "impact_score": round(impact_score, 1),
        "impact_label": impact_label,
        "impact_icon": impact_icon,
        "source_credibility": round(credibility, 1),
        "materiality": round(materiality, 1),
        "surprise": round(surprise, 1),
        "ai_relevance": round(relevance, 1),
        "ai_topics": list(ai_hits.keys()),
        "geopolitical_topics": list(geo_hits.keys()),
        "relevant_ai_themes": relevant_ai_themes,
        "positive_themes": positive_themes,
        "negative_themes": negative_themes,
        "positive_tickers": _themes_to_tickers(positive_themes),
        "negative_tickers": _themes_to_tickers(negative_themes),
        "direct_ticker_mentions": direct_mentions,
        "evidence_type": "Breaking fact / reported event",
    }


def _deduplicate_radar_events(events: list[dict]) -> list[dict]:
    groups = []
    for event in events:
        headline_tokens = set(re.findall(r"[a-z0-9]+", str(event.get("headline", "")).lower()))
        matched = False
        for group in groups:
            reference_tokens = set(
                re.findall(r"[a-z0-9]+", str(group[0].get("headline", "")).lower())
            )
            if not headline_tokens or not reference_tokens:
                continue
            overlap = len(headline_tokens & reference_tokens) / max(
                1, len(headline_tokens | reference_tokens)
            )
            if overlap >= 0.55:
                group.append(event)
                matched = True
                break
        if not matched:
            groups.append([event])

    output = []
    for group in groups:
        group.sort(
            key=lambda item: (
                float(item.get("source_credibility", 0)),
                float(item.get("impact_score", 0)),
            ),
            reverse=True,
        )
        leader = dict(group[0])
        sources = list(dict.fromkeys(str(x.get("source", "")) for x in group))
        leader["corroborating_sources"] = sources
        leader["corroboration_count"] = len(sources)
        bonus = 6 if len(sources) >= 3 else 3 if len(sources) == 2 else 0
        leader["impact_score"] = round(
            clamp(float(leader.get("impact_score", 0)) + bonus), 1
        )
        output.append(leader)

    output.sort(
        key=lambda item: (
            float(item.get("impact_score", 0)),
            item.get("published_at", ""),
        ),
        reverse=True,
    )
    return output


def detect_emerging_ai_signals(events: list[dict]) -> list[dict]:
    buckets = {}
    for event in events:
        topics = list(event.get("ai_topics", [])) + list(event.get("geopolitical_topics", []))
        for topic in topics:
            buckets.setdefault(topic, []).append(event)

    signals = []
    now = datetime.now(timezone.utc)

    for topic, items in buckets.items():
        recent = []
        for item in items:
            try:
                published = datetime.fromisoformat(item["published_at"])
            except Exception:
                continue
            if (now - published).total_seconds() <= 6 * 3600:
                recent.append(item)

        sources = list(dict.fromkeys(str(x.get("source", "")) for x in recent))
        if len(recent) < 2 or len(sources) < 2:
            continue

        avg_impact = sum(float(x.get("impact_score", 0)) for x in recent) / len(recent)
        signal_score = clamp(avg_impact + min(15, 4 * len(sources)))

        signals.append({
            "topic": topic,
            "signal_score": round(signal_score, 1),
            "article_count": len(recent),
            "independent_sources": len(sources),
            "sources": sources[:8],
            "latest_headlines": [str(x.get("headline", "")) for x in recent[:5]],
            "status": "⚡ EMERGING AI SIGNAL",
            "evidence_type": "Multi-source inference",
        })

    signals.sort(
        key=lambda item: float(item.get("signal_score", 0)),
        reverse=True,
    )
    return signals


def run_ai_news_radar(finnhub_api_key: str, include_gdelt: bool = True) -> dict:
    raw_news = []

    for item in fetch_finnhub_market_news(finnhub_api_key):
        enriched = dict(item)
        enriched["source_feed"] = "Finnhub general market news"
        raw_news.append(enriched)

    if include_gdelt:
        raw_news.extend(fetch_gdelt_ai_geopolitical_news())

    classified = []
    for item in raw_news:
        event = classify_ai_market_news(item)
        if event is not None:
            classified.append(event)

    events = _deduplicate_radar_events(classified)
    emerging = detect_emerging_ai_signals(events)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "event_count": len(events),
        "critical_count": sum(float(x.get("impact_score", 0)) >= 90 for x in events),
        "high_impact_count": sum(float(x.get("impact_score", 0)) >= 80 for x in events),
        "emerging_signal_count": len(emerging),
        "events": events[:100],
        "emerging_signals": emerging[:20],
    }
    save_ai_news_radar(payload)
    return payload


def save_ai_news_radar(payload: dict) -> None:
    AI_NEWS_RADAR_PATH.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def load_ai_news_radar() -> dict:
    if not AI_NEWS_RADAR_PATH.exists():
        return {}
    try:
        return json.loads(AI_NEWS_RADAR_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def radar_alert_candidates(payload: dict, minimum_impact: float = 85.0) -> list[dict]:
    return [
        event for event in payload.get("events", [])
        if float(event.get("impact_score", 0)) >= minimum_impact
    ]


def send_new_ai_radar_alerts(
    payload: dict,
    app_url: str = "",
    minimum_impact: float = 85.0,
) -> int:
    state = {}
    if AI_NEWS_RADAR_STATE_PATH.exists():
        try:
            state = json.loads(AI_NEWS_RADAR_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            state = {}

    seen = set(state.get("seen_event_ids", []))
    sent = 0

    for event in radar_alert_candidates(payload, minimum_impact):
        event_id = str(event.get("id") or event.get("url") or event.get("headline", ""))
        if not event_id or event_id in seen:
            continue

        message = (
            f'{event.get("impact_icon", "🚨")} '
            f'{event.get("impact_label", "AI NEWS")} '
            f'{float(event.get("impact_score", 0)):.0f}/100\n'
            f'{event.get("headline", "")}\n'
            f'AI: {", ".join(event.get("ai_topics", [])[:3]) or "—"}\n'
            f'Geo: {", ".join(event.get("geopolitical_topics", [])[:2]) or "—"}'
        )

        try:
            send_pushover(
                "Vektra Alpha AI News Radar",
                message[:1024],
                app_url,
            )
            sent += 1
            seen.add(event_id)
        except Exception:
            continue

    state["seen_event_ids"] = list(seen)[-1000:]
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    AI_NEWS_RADAR_STATE_PATH.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )
    return sent


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
                    "ai_theme": ai_theme_for_ticker(ticker),
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
    market: str = "AI",
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
        record["ai_theme"] = ai_theme_for_ticker(ticker)
        record["market_prefilter"] = prefilter_lookup.get(ticker, {})
        record["discovery_source"] = "Vektra Alpha AI ecosystem scanner" 

    detailed.sort(
        key=lambda item: (
            float(item.get("ai_score", 0)),
            float(item.get("probability_up", 0)),
            float(item.get("market_prefilter", {}).get("prefilter_score", 0)),
        ),
        reverse=True,
    )
    return detailed, candidates



def calculate_ai_theme_rotation(records: list[dict]) -> list[dict[str, Any]]:
    """Rank AI themes by current signal strength.

    Uses only the information already calculated by Vektra Alpha:
    AI Score, model probability, 30-minute momentum and relative volume.
    """
    buckets: dict[str, list[dict]] = {}

    for record in records:
        ticker = str(record.get("ticker", ""))
        theme = str(record.get("ai_theme") or ai_theme_for_ticker(ticker))
        buckets.setdefault(theme, []).append(record)

    results: list[dict[str, Any]] = []

    for theme, items in buckets.items():
        if not items:
            continue

        avg_ai = sum(float(x.get("ai_score", 0)) for x in items) / len(items)
        avg_probability = (
            sum(float(x.get("probability_up", 0.5)) for x in items) / len(items)
        )
        avg_momentum = (
            sum(float(x.get("return_30m", 0)) for x in items) / len(items)
        )

        volume_values = []
        for item in items:
            pre = item.get("market_prefilter", {}) or {}
            volume_values.append(
                float(pre.get("relative_volume", item.get("volume_ratio_30m", 1.0)))
            )
        avg_volume = sum(volume_values) / len(volume_values)

        probability_score = clamp(avg_probability * 100.0)
        momentum_score = clamp(50.0 + 700.0 * avg_momentum)
        volume_score = clamp(35.0 + 25.0 * max(0.0, avg_volume - 1.0))

        rotation_score = (
            0.45 * avg_ai
            + 0.25 * probability_score
            + 0.15 * momentum_score
            + 0.15 * volume_score
        )

        results.append({
            "theme": theme,
            "rotation_score": round(rotation_score, 1),
            "average_ai_score": round(avg_ai, 1),
            "average_probability": round(avg_probability, 4),
            "average_30m_return": round(avg_momentum, 6),
            "average_relative_volume": round(avg_volume, 2),
            "stocks": [str(x.get("ticker", "")) for x in items],
            "stock_count": len(items),
        })

    results.sort(
        key=lambda item: float(item.get("rotation_score", 0)),
        reverse=True,
    )
    return results


def ai_ecosystem_summary(records: list[dict]) -> dict[str, Any]:
    rotation = calculate_ai_theme_rotation(records)
    leaders = sorted(
        records,
        key=lambda item: (
            float(item.get("ai_score", 0)),
            float(item.get("probability_up", 0)),
        ),
        reverse=True,
    )

    return {
        "universe_size": len(AI_GLOBAL_UNIVERSE),
        "us_stocks": len(AI_US_UNIVERSE),
        "uk_stocks": len(AI_UK_UNIVERSE),
        "themes": len(set(AI_THEME_MAP.values())),
        "hottest_themes": rotation[:5],
        "top_stocks": [
            {
                "ticker": item.get("ticker", ""),
                "theme": item.get("ai_theme", ai_theme_for_ticker(item.get("ticker", ""))),
                "ai_score": round(float(item.get("ai_score", 0)), 1),
                "probability_up": round(float(item.get("probability_up", 0)), 4),
            }
            for item in leaders[:10]
        ],
    }


def save_market_scan(records: list[dict], candidates: list[dict]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scanner_mode": "AI ecosystem only",
        "universe_size": len(AI_GLOBAL_UNIVERSE),
        "markets": sorted({
            str(item.get("market", market_for_ticker(item.get("ticker", ""))))
            for item in candidates
        }),
        "theme_rotation": calculate_ai_theme_rotation(records),
        "ai_ecosystem": ai_ecosystem_summary(records),
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


def evaluate_buy_today(record: dict) -> dict[str, Any]:
    """Evaluate whether one stock qualifies as a same-day BUY CANDIDATE.

    This is a transparent research gate, not a personalised recommendation or
    execution instruction. A high AI Score alone cannot create a BUY result.
    """
    ticker = str(record.get("ticker", ""))
    ai_score = safe_float(record.get("ai_score"), 0.0)
    probability = safe_float(record.get("probability_up"), 0.0)
    evidence = safe_float(record.get("evidence_coverage"), 0.0)
    move_30m = safe_float(record.get("return_30m"), 0.0)
    move_1d = safe_float(record.get("return_1d"), 0.0)
    volume = safe_float(record.get("volume_ratio_30m"), 0.0)
    net_return = safe_float(record.get("net_expected_return"), 0.0)
    liquidity = safe_float(record.get("market_prefilter", {}).get("relative_volume"), volume)

    articles = sorted(
        record.get("news", []),
        key=lambda item: safe_float(item.get("impact_score"), 0.0),
        reverse=True,
    )
    best_article = articles[0] if articles else {}
    catalyst_direction = str(best_article.get("sentiment_label", "Neutral"))
    catalyst_impact = safe_float(best_article.get("impact_score"), 0.0)
    catalyst = (
        best_article.get("headline")
        or best_article.get("reason")
        or "No qualifying live catalyst"
    )

    checks = {
        "AI score at least 78": ai_score >= 78,
        "Probability at least 60%": probability >= 0.60,
        "Evidence coverage at least 60%": evidence >= 60,
        "Positive 30-minute momentum": move_30m > 0,
        "Move is not already exhausted": move_30m <= 0.045,
        "Positive or stable daily trend": move_1d >= -0.005,
        "Relative volume at least 1.5x": max(volume, liquidity) >= 1.5,
        "Relative volume below extreme level": max(volume, liquidity) <= 10.0,
        "Positive expected return after costs": net_return > 0,
        "Bullish material catalyst": catalyst_direction == "Bullish" and catalyst_impact >= 55,
    }

    passed = sum(checks.values())
    total = len(checks)
    gate_score = 100.0 * passed / total

    hard_failures = []
    if probability < 0.55:
        hard_failures.append("model probability below 55%")
    if evidence < 45:
        hard_failures.append("insufficient evidence coverage")
    if move_30m <= 0:
        hard_failures.append("no positive intraday confirmation")
    if move_30m > 0.06:
        hard_failures.append("price may already be overextended")
    if max(volume, liquidity) > 12:
        hard_failures.append("extreme volume may indicate unstable trading")
    if net_return <= 0:
        hard_failures.append("expected edge does not exceed estimated costs")

    if not hard_failures and passed >= 9 and probability >= 0.64 and ai_score >= 82:
        decision = "BUY CANDIDATE TODAY"
        decision_code = "BUY_CANDIDATE"
    elif not hard_failures and passed >= 7 and ai_score >= 70:
        decision = "WATCH — CONFIRMATION NEEDED"
        decision_code = "WATCH"
    else:
        decision = "NO TRADE"
        decision_code = "NO_TRADE"

    strengths = [name for name, ok in checks.items() if ok]
    failed_checks = [name for name, ok in checks.items() if not ok]

    return {
        "ticker": ticker,
        "market": record.get("market", market_for_ticker(ticker)),
        "currency": record.get("currency", currency_for_ticker(ticker)),
        "decision": decision,
        "decision_code": decision_code,
        "gate_score": round(gate_score, 1),
        "checks_passed": passed,
        "checks_total": total,
        "ai_score": round(ai_score, 1),
        "probability_up": round(probability, 4),
        "evidence_coverage": round(evidence, 1),
        "return_30m": round(move_30m, 6),
        "return_1d": round(move_1d, 6),
        "relative_volume": round(max(volume, liquidity), 2),
        "net_expected_return": round(net_return, 6),
        "catalyst": str(catalyst)[:220],
        "catalyst_direction": catalyst_direction,
        "catalyst_impact": round(catalyst_impact, 1),
        "strengths": strengths,
        "failed_checks": failed_checks,
        "hard_failures": hard_failures,
        "checks": checks,
        "disclaimer": "Research decision support only; not personalised financial advice or an instruction to trade.",
    }


def rank_buy_today_candidates(records: list[dict], limit: int = 5) -> list[dict[str, Any]]:
    """Evaluate and rank available stocks by decision quality."""
    evaluations = [evaluate_buy_today(record) for record in records]
    priority = {"BUY_CANDIDATE": 2, "WATCH": 1, "NO_TRADE": 0}
    evaluations.sort(
        key=lambda item: (
            priority.get(str(item.get("decision_code")), 0),
            safe_float(item.get("gate_score")),
            safe_float(item.get("probability_up")),
            safe_float(item.get("ai_score")),
        ),
        reverse=True,
    )
    return evaluations[:max(1, int(limit))]


def select_buy_today(records: list[dict]) -> dict[str, Any]:
    """Return the strongest qualifying candidate, or a transparent no-trade result."""
    ranked = rank_buy_today_candidates(records, limit=max(1, len(records)))
    qualifying = [item for item in ranked if item.get("decision_code") == "BUY_CANDIDATE"]
    if qualifying:
        return qualifying[0]
    if ranked:
        best = dict(ranked[0])
        best["decision"] = "NO QUALIFYING BUY TODAY"
        best["decision_code"] = "NO_TRADE"
        return best
    return {
        "ticker": "—",
        "decision": "NO DATA",
        "decision_code": "NO_TRADE",
        "gate_score": 0.0,
        "strengths": [],
        "failed_checks": ["Run a market scan first"],
        "hard_failures": ["No current scan data"],
        "disclaimer": "Research decision support only; not personalised financial advice or an instruction to trade.",
    }


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
