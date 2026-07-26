"""
VEKTRA Global News Alpha Engine
================================
A research-grade prototype for ranking stocks by the probability of a
positive abnormal return after newly detected market-moving news.

This is NOT financial advice and does not guarantee profitable trades.
Production use requires licensed real-time data, robust backtesting,
transaction-cost modelling, monitoring, and regulatory review.

Core idea
---------
The engine does not buy stocks simply because an article is positive.
It scores the complete causal chain:

    credible source
      -> material company event
      -> genuinely new information
      -> positive surprise versus expectations
      -> corroboration across independent sources
      -> price/volume confirmation
      -> favourable market regime
      -> sufficient liquidity and acceptable risk

The model returns:
    1. probability of a positive abnormal return over a chosen horizon;
    2. confidence in that probability;
    3. expected return after estimated costs;
    4. a ranked watchlist, not an automatic trade instruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from math import exp, log
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NewsItem:
    article_id: str
    published_at: datetime
    source: str
    headline: str
    body: str
    language: str = "en"
    url: str = ""
    mentioned_tickers: Tuple[str, ...] = ()
    source_country: str = ""

    @property
    def text(self) -> str:
        return f"{self.headline}. {self.body}".strip()


@dataclass(frozen=True)
class MarketSnapshot:
    ticker: str
    timestamp: datetime
    last_price: float
    return_5m: float
    return_30m: float
    return_1d: float
    market_return_30m: float
    sector_return_30m: float
    volume_ratio_30m: float
    spread_bps: float
    realised_volatility_20d: float
    average_daily_value_gbp: float
    short_interest_zscore: float = 0.0
    options_implied_move: float = 0.0


@dataclass
class ArticleFeatures:
    ticker: str
    article_id: str
    event_type: str
    event_direction: float
    sentiment: float
    materiality: float
    novelty: float
    source_credibility: float
    entity_relevance: float
    expectation_surprise: float
    corroboration: float
    contradiction_penalty: float
    age_decay: float
    manipulation_risk: float
    explanation: List[str] = field(default_factory=list)


@dataclass
class StockSignal:
    ticker: str
    timestamp: datetime
    probability_up: float
    calibrated_confidence: float
    expected_abnormal_return: float
    estimated_cost: float
    net_expected_return: float
    signal_score: float
    action: str
    reasons: List[str]


class TextModel(Protocol):
    """Replace with a finance-tuned transformer or LLM classifier."""

    def analyse(
        self, text: str, ticker: str
    ) -> Mapping[str, float | str | List[str]]:
        ...


class PriceProvider(Protocol):
    def snapshot(self, ticker: str, at: datetime) -> MarketSnapshot:
        ...


# ---------------------------------------------------------------------------
# Baseline NLP model
# ---------------------------------------------------------------------------

class RuleBasedFinanceTextModel:
    """
    Lightweight baseline so the prototype runs without external AI services.

    In production, replace this with models trained for:
      - entity linking;
      - event classification;
      - financial sentiment;
      - materiality;
      - expectation surprise;
      - misinformation/manipulation detection.
    """

    POSITIVE = {
        "beats": 1.0,
        "beat expectations": 1.0,
        "raises guidance": 1.0,
        "upgrade": 0.6,
        "contract awarded": 0.8,
        "approval": 0.8,
        "acquisition offer": 0.9,
        "record revenue": 0.7,
        "profit rises": 0.6,
        "buyback": 0.5,
        "dividend increase": 0.5,
        "successful trial": 0.7,
        "partnership": 0.4,
        "order": 0.4,
    }

    NEGATIVE = {
        "misses": -1.0,
        "cuts guidance": -1.0,
        "downgrade": -0.6,
        "investigation": -0.7,
        "fraud": -1.0,
        "recall": -0.8,
        "bankruptcy": -1.0,
        "profit warning": -1.0,
        "data breach": -0.7,
        "lawsuit": -0.5,
        "chief executive resigns": -0.6,
        "offering": -0.4,
        "dilution": -0.6,
    }

    EVENT_PATTERNS = {
        "earnings": ("earnings", "revenue", "eps", "profit", "guidance"),
        "merger_acquisition": ("acquire", "acquisition", "merger", "takeover", "bid"),
        "regulatory": ("approval", "regulator", "fda", "authorisation", "license"),
        "contract_order": ("contract", "order", "award", "customer"),
        "product_technology": ("launch", "patent", "trial", "product", "technology"),
        "capital_structure": ("buyback", "dividend", "offering", "debt", "refinancing"),
        "leadership": ("chief executive", "ceo", "cfo", "chair", "resigns"),
        "legal_risk": ("lawsuit", "investigation", "fraud", "recall"),
    }

    def analyse(
        self, text: str, ticker: str
    ) -> Mapping[str, float | str | List[str]]:
        t = text.lower()

        weighted_hits: List[Tuple[str, float]] = []
        for phrase, weight in {**self.POSITIVE, **self.NEGATIVE}.items():
            if phrase in t:
                weighted_hits.append((phrase, weight))

        raw_sentiment = sum(weight for _, weight in weighted_hits)
        sentiment = float(np.tanh(raw_sentiment))

        event_type = "other"
        best_hits = 0
        for name, patterns in self.EVENT_PATTERNS.items():
            hits = sum(pattern in t for pattern in patterns)
            if hits > best_hits:
                event_type, best_hits = name, hits

        # Materiality is intentionally not identical to sentiment.
        material_words = (
            "guidance", "earnings", "acquisition", "approval", "contract",
            "recall", "investigation", "bankruptcy", "offering", "dividend",
            "buyback", "chief executive", "strategic review"
        )
        materiality = min(1.0, 0.15 + 0.16 * sum(w in t for w in material_words))

        explicit_ticker = ticker.lower() in t
        entity_relevance = 0.95 if explicit_ticker else 0.70

        surprise_words = (
            "unexpected", "surprise", "ahead of expectations",
            "below expectations", "raises guidance", "cuts guidance",
            "beats", "misses"
        )
        surprise_hits = sum(w in t for w in surprise_words)
        expectation_surprise = min(1.0, 0.25 * surprise_hits)

        manipulation_words = (
            "rumour", "unconfirmed", "anonymous source", "could soar",
            "guaranteed", "massive upside", "penny stock", "sponsored"
        )
        manipulation_risk = min(1.0, 0.20 * sum(w in t for w in manipulation_words))

        explanation = [f"Detected phrase: {phrase}" for phrase, _ in weighted_hits[:5]]
        if event_type != "other":
            explanation.append(f"Classified event: {event_type}")

        return {
            "event_type": event_type,
            "event_direction": sentiment,
            "sentiment": sentiment,
            "materiality": materiality,
            "entity_relevance": entity_relevance,
            "expectation_surprise": expectation_surprise,
            "manipulation_risk": manipulation_risk,
            "explanation": explanation,
        }


# ---------------------------------------------------------------------------
# Feature utilities
# ---------------------------------------------------------------------------

SOURCE_PRIORS: Dict[str, float] = {
    # Illustrative priors only. Production values should be learned from
    # historical accuracy, correction rate, latency and independence.
    "regulatory_filing": 0.99,
    "company_rns": 0.96,
    "company_ir": 0.93,
    "major_wire": 0.90,
    "major_newspaper": 0.86,
    "specialist_trade_press": 0.78,
    "broker_note": 0.75,
    "social_media_verified": 0.55,
    "social_media_unverified": 0.20,
    "unknown": 0.40,
}


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def stable_text_hash(text: str) -> str:
    normalised = " ".join(text.lower().split())
    return sha256(normalised.encode("utf-8")).hexdigest()


def token_jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def exponential_age_decay(
    published_at: datetime,
    now: datetime,
    half_life_minutes: float = 90.0,
) -> float:
    age_minutes = max(0.0, (now - published_at).total_seconds() / 60.0)
    return 0.5 ** (age_minutes / half_life_minutes)


def compute_novelty(
    current: NewsItem,
    recent_articles: Sequence[NewsItem],
    lookback_hours: float = 72.0,
) -> float:
    """
    Novelty falls when substantially similar text has already appeared.
    This helps avoid repeatedly reacting to syndicated or recycled stories.
    """
    similarities: List[float] = []
    for prior in recent_articles:
        hours = abs((current.published_at - prior.published_at).total_seconds()) / 3600
        if prior.article_id != current.article_id and hours <= lookback_hours:
            similarities.append(token_jaccard(current.text, prior.text))

    max_similarity = max(similarities, default=0.0)
    return clamp(1.0 - max_similarity)


def compute_corroboration(
    current: NewsItem,
    recent_articles: Sequence[NewsItem],
    independent_sources_only: bool = True,
) -> float:
    """
    Rewards similar reporting from separate sources, while limiting the
    contribution of copied/syndicated versions.
    """
    sources = set()
    supporting = 0
    for article in recent_articles:
        if article.article_id == current.article_id:
            continue
        similarity = token_jaccard(current.text, article.text)
        if similarity >= 0.22:
            if not independent_sources_only or article.source != current.source:
                sources.add(article.source)
                supporting += 1

    return clamp(1.0 - exp(-0.45 * min(supporting, len(sources))))


def compute_contradiction_penalty(
    direction: float,
    related_feature_history: Sequence[ArticleFeatures],
) -> float:
    """
    Penalises cases in which credible recent reports point in the opposite
    direction. A production implementation should use semantic entailment.
    """
    opposite_weight = 0.0
    total_weight = 0.0

    for prior in related_feature_history:
        weight = (
            prior.source_credibility
            * prior.materiality
            * prior.entity_relevance
            * prior.age_decay
        )
        total_weight += weight
        if np.sign(prior.event_direction) != np.sign(direction):
            opposite_weight += weight

    if total_weight == 0:
        return 0.0
    return clamp(opposite_weight / total_weight)


def source_credibility(source_class: str) -> float:
    return SOURCE_PRIORS.get(source_class, SOURCE_PRIORS["unknown"])


# ---------------------------------------------------------------------------
# Distinctive component: Event Surprise Graph
# ---------------------------------------------------------------------------

EVENT_CARRYOVER = {
    # Approximate persistence by event class. Learn these values from data.
    "earnings": 1.00,
    "regulatory": 1.10,
    "contract_order": 0.85,
    "merger_acquisition": 1.15,
    "product_technology": 0.70,
    "capital_structure": 0.90,
    "leadership": 0.55,
    "legal_risk": 1.05,
    "other": 0.40,
}


def event_surprise_graph_score(f: ArticleFeatures) -> float:
    """
    The article contribution is multiplicative rather than a plain weighted
    sentiment average. Weakness in any causal link suppresses the signal.

    This is designed to answer:
       "Is this credible, material, new and surprising information about the
        named company, rather than merely positive language?"
    """
    positive_direction = max(0.0, f.event_direction)
    information_quality = (
        max(0.02, f.source_credibility)
        * max(0.02, f.entity_relevance)
        * max(0.02, f.materiality)
        * max(0.02, f.novelty)
    ) ** 0.25

    surprise_layer = 0.35 + 0.65 * f.expectation_surprise
    validation_layer = 0.65 + 0.35 * f.corroboration
    penalty_layer = (
        (1.0 - 0.85 * f.contradiction_penalty)
        * (1.0 - 0.90 * f.manipulation_risk)
    )
    persistence = EVENT_CARRYOVER.get(f.event_type, 0.40)

    return (
        positive_direction
        * information_quality
        * surprise_layer
        * validation_layer
        * penalty_layer
        * f.age_decay
        * persistence
    )


# ---------------------------------------------------------------------------
# Price confirmation, regime and liquidity
# ---------------------------------------------------------------------------

def abnormal_return_30m(m: MarketSnapshot) -> float:
    benchmark = 0.6 * m.market_return_30m + 0.4 * m.sector_return_30m
    return m.return_30m - benchmark


def price_confirmation_score(m: MarketSnapshot) -> float:
    """
    Avoids chasing news with no market confirmation, while also penalising
    moves that may already be exhausted.
    """
    abnormal = abnormal_return_30m(m)

    direction_confirmation = sigmoid(80.0 * abnormal)
    volume_confirmation = sigmoid(1.8 * (m.volume_ratio_30m - 1.0))

    # Exhaustion penalty: large immediate moves have poorer entry asymmetry.
    exhaustion = sigmoid(25.0 * (abs(m.return_30m) - 0.055))

    spread_quality = 1.0 - clamp(m.spread_bps / 80.0)
    return clamp(
        0.40 * direction_confirmation
        + 0.25 * volume_confirmation
        + 0.20 * spread_quality
        + 0.15 * (1.0 - exhaustion)
    )


def market_regime_score(
    market_breadth: float,
    volatility_percentile: float,
    index_trend: float,
) -> float:
    """
    Gives higher scores to constructive regimes. This should be sector- and
    horizon-specific in a production model.
    """
    breadth_component = clamp((market_breadth + 1.0) / 2.0)
    vol_component = 1.0 - clamp(volatility_percentile)
    trend_component = sigmoid(30.0 * index_trend)
    return clamp(
        0.40 * breadth_component
        + 0.30 * vol_component
        + 0.30 * trend_component
    )


def liquidity_score(m: MarketSnapshot) -> float:
    adv_component = sigmoid((log(max(m.average_daily_value_gbp, 1.0)) - log(2_000_000)) / 1.2)
    spread_component = 1.0 - clamp(m.spread_bps / 100.0)
    return clamp(0.65 * adv_component + 0.35 * spread_component)


def estimated_round_trip_cost(m: MarketSnapshot) -> float:
    spread_cost = m.spread_bps / 10_000.0
    volatility_slippage = 0.04 * m.realised_volatility_20d
    liquidity_penalty = 0.0015 * (1.0 - liquidity_score(m))
    return spread_cost + volatility_slippage + liquidity_penalty


# ---------------------------------------------------------------------------
# Probability model and calibration
# ---------------------------------------------------------------------------

@dataclass
class ProbabilityWeights:
    intercept: float = -2.15
    event_alpha: float = 3.30
    price_confirmation: float = 1.25
    regime: float = 0.75
    liquidity: float = 0.70
    volume: float = 0.35
    short_squeeze: float = 0.18
    volatility_penalty: float = -0.55
    exhaustion_penalty: float = -0.80


class NewsAlphaProbabilityModel:
    """
    Transparent baseline. Replace coefficients with walk-forward trained
    logistic regression, gradient boosting or a calibrated neural model.
    """

    def __init__(self, weights: Optional[ProbabilityWeights] = None) -> None:
        self.w = weights or ProbabilityWeights()

    def raw_probability(
        self,
        event_alpha: float,
        market: MarketSnapshot,
        regime: float,
    ) -> float:
        confirmation = price_confirmation_score(market)
        liquidity = liquidity_score(market)
        volume = clamp((market.volume_ratio_30m - 0.5) / 3.0)
        short_squeeze = sigmoid(market.short_interest_zscore) - 0.5
        vol = clamp(market.realised_volatility_20d / 1.20)
        exhaustion = clamp(abs(market.return_30m) / 0.10)

        logit = (
            self.w.intercept
            + self.w.event_alpha * event_alpha
            + self.w.price_confirmation * confirmation
            + self.w.regime * regime
            + self.w.liquidity * liquidity
            + self.w.volume * volume
            + self.w.short_squeeze * short_squeeze
            + self.w.volatility_penalty * vol
            + self.w.exhaustion_penalty * exhaustion
        )
        return sigmoid(logit)


class BetaBinCalibrator:
    """
    Simple online probability calibration by score bucket.

    Production alternatives:
      - isotonic regression;
      - Platt scaling;
      - beta calibration;
      - rolling calibration by market, sector, event type and horizon.
    """

    def __init__(self, buckets: int = 20, prior_strength: float = 20.0) -> None:
        self.buckets = buckets
        self.alpha = np.ones(buckets) * (0.5 * prior_strength)
        self.beta = np.ones(buckets) * (0.5 * prior_strength)

    def _bucket(self, probability: float) -> int:
        return min(self.buckets - 1, int(clamp(probability) * self.buckets))

    def update(self, raw_probability: float, outcome_up: bool) -> None:
        idx = self._bucket(raw_probability)
        self.alpha[idx] += float(outcome_up)
        self.beta[idx] += float(not outcome_up)

    def calibrate(self, raw_probability: float) -> Tuple[float, float]:
        idx = self._bucket(raw_probability)
        a, b = self.alpha[idx], self.beta[idx]
        posterior_mean = a / (a + b)
        observations = a + b
        confidence = clamp(1.0 - exp(-observations / 75.0))
        # Blend raw score with empirical posterior while evidence accumulates.
        calibrated = confidence * posterior_mean + (1.0 - confidence) * raw_probability
        return float(calibrated), confidence


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GlobalNewsAlphaEngine:
    def __init__(
        self,
        text_model: TextModel,
        probability_model: Optional[NewsAlphaProbabilityModel] = None,
        calibrator: Optional[BetaBinCalibrator] = None,
    ) -> None:
        self.text_model = text_model
        self.probability_model = probability_model or NewsAlphaProbabilityModel()
        self.calibrator = calibrator or BetaBinCalibrator()
        self.article_history: List[NewsItem] = []
        self.feature_history: List[ArticleFeatures] = []
        self.seen_hashes: set[str] = set()

    def analyse_article(
        self,
        article: NewsItem,
        ticker: str,
        source_class: str,
        now: Optional[datetime] = None,
    ) -> Optional[ArticleFeatures]:
        now = now or datetime.now(timezone.utc)
        article_hash = stable_text_hash(article.text)

        if article_hash in self.seen_hashes:
            return None

        nlp = self.text_model.analyse(article.text, ticker)
        related_articles = [
            a for a in self.article_history
            if ticker in a.mentioned_tickers
        ]
        related_features = [
            f for f in self.feature_history
            if f.ticker == ticker
        ]

        feature = ArticleFeatures(
            ticker=ticker,
            article_id=article.article_id,
            event_type=str(nlp["event_type"]),
            event_direction=float(nlp["event_direction"]),
            sentiment=float(nlp["sentiment"]),
            materiality=float(nlp["materiality"]),
            novelty=compute_novelty(article, related_articles),
            source_credibility=source_credibility(source_class),
            entity_relevance=float(nlp["entity_relevance"]),
            expectation_surprise=float(nlp["expectation_surprise"]),
            corroboration=compute_corroboration(article, related_articles),
            contradiction_penalty=compute_contradiction_penalty(
                float(nlp["event_direction"]), related_features
            ),
            age_decay=exponential_age_decay(article.published_at, now),
            manipulation_risk=float(nlp["manipulation_risk"]),
            explanation=list(nlp.get("explanation", [])),
        )

        self.seen_hashes.add(article_hash)
        self.article_history.append(article)
        self.feature_history.append(feature)
        return feature

    def aggregate_event_alpha(
        self,
        ticker: str,
        horizon_hours: float = 12.0,
        now: Optional[datetime] = None,
    ) -> Tuple[float, List[str]]:
        now = now or datetime.now(timezone.utc)
        relevant = []
        for feature in self.feature_history:
            if feature.ticker != ticker:
                continue
            age_hours = (
                now - next(
                    a.published_at
                    for a in self.article_history
                    if a.article_id == feature.article_id
                )
            ).total_seconds() / 3600
            if age_hours <= horizon_hours:
                relevant.append(feature)

        if not relevant:
            return 0.0, ["No qualifying recent event"]

        contributions = [event_surprise_graph_score(f) for f in relevant]

        # Noisy-OR aggregation: several independent moderate reports can build
        # conviction without allowing repeated copies to increase linearly.
        event_alpha = 1.0 - np.prod([1.0 - clamp(c) for c in contributions])

        top = sorted(
            zip(relevant, contributions),
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        reasons: List[str] = []
        for feature, contribution in top:
            reasons.append(
                f"{feature.event_type}: contribution={contribution:.3f}, "
                f"novelty={feature.novelty:.2f}, "
                f"credibility={feature.source_credibility:.2f}, "
                f"materiality={feature.materiality:.2f}"
            )
        return float(event_alpha), reasons

    def score_stock(
        self,
        ticker: str,
        market: MarketSnapshot,
        market_breadth: float,
        volatility_percentile: float,
        index_trend: float,
        now: Optional[datetime] = None,
    ) -> StockSignal:
        now = now or datetime.now(timezone.utc)
        event_alpha, reasons = self.aggregate_event_alpha(ticker, now=now)
        regime = market_regime_score(
            market_breadth=market_breadth,
            volatility_percentile=volatility_percentile,
            index_trend=index_trend,
        )

        raw_probability = self.probability_model.raw_probability(
            event_alpha=event_alpha,
            market=market,
            regime=regime,
        )
        probability_up, calibration_confidence = self.calibrator.calibrate(
            raw_probability
        )

        # Illustrative expected-return mapping. Learn this from historical
        # abnormal returns conditional on probability, event and market state.
        expected_abnormal_return = max(
            -0.03,
            min(
                0.08,
                (probability_up - 0.50)
                * (0.035 + 0.30 * market.realised_volatility_20d)
                * (0.60 + 0.40 * event_alpha),
            ),
        )
        cost = estimated_round_trip_cost(market)
        net_expected_return = expected_abnormal_return - cost

        # Signal score includes probability, edge, confidence and tradability.
        signal_score = (
            max(0.0, probability_up - 0.50)
            * max(0.0, net_expected_return)
            * (0.40 + 0.60 * calibration_confidence)
            * liquidity_score(market)
            * 10_000
        )

        # Conservative research thresholds.
        if (
            probability_up >= 0.68
            and net_expected_return >= 0.004
            and liquidity_score(market) >= 0.55
            and event_alpha >= 0.25
        ):
            action = "HIGH-CONVICTION WATCH"
        elif probability_up >= 0.58 and net_expected_return > 0:
            action = "WATCH"
        else:
            action = "IGNORE"

        reasons.extend([
            f"Probability up={probability_up:.1%}",
            f"Event alpha={event_alpha:.3f}",
            f"Price confirmation={price_confirmation_score(market):.2f}",
            f"Regime={regime:.2f}",
            f"Liquidity={liquidity_score(market):.2f}",
            f"Estimated cost={cost:.2%}",
        ])

        return StockSignal(
            ticker=ticker,
            timestamp=now,
            probability_up=probability_up,
            calibrated_confidence=calibration_confidence,
            expected_abnormal_return=expected_abnormal_return,
            estimated_cost=cost,
            net_expected_return=net_expected_return,
            signal_score=float(signal_score),
            action=action,
            reasons=reasons,
        )

    def rank_watchlist(
        self,
        market_snapshots: Sequence[MarketSnapshot],
        market_breadth: float,
        volatility_percentile: float,
        index_trend: float,
    ) -> pd.DataFrame:
        signals = [
            self.score_stock(
                ticker=m.ticker,
                market=m,
                market_breadth=market_breadth,
                volatility_percentile=volatility_percentile,
                index_trend=index_trend,
            )
            for m in market_snapshots
        ]

        rows = [
            {
                "ticker": s.ticker,
                "probability_up": s.probability_up,
                "confidence": s.calibrated_confidence,
                "expected_abnormal_return": s.expected_abnormal_return,
                "estimated_cost": s.estimated_cost,
                "net_expected_return": s.net_expected_return,
                "signal_score": s.signal_score,
                "action": s.action,
                "reasons": " | ".join(s.reasons),
            }
            for s in signals
        ]
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["signal_score", "probability_up"],
                ascending=False,
            )
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

def example() -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    engine = GlobalNewsAlphaEngine(RuleBasedFinanceTextModel())

    article = NewsItem(
        article_id="demo-001",
        published_at=now,
        source="Example Regulatory News",
        headline="ABC raises guidance after revenue beats expectations",
        body=(
            "ABC reported record revenue, beat expectations and raised its "
            "full-year guidance following stronger customer demand."
        ),
        mentioned_tickers=("ABC",),
    )

    engine.analyse_article(
        article=article,
        ticker="ABC",
        source_class="company_rns",
        now=now,
    )

    market = MarketSnapshot(
        ticker="ABC",
        timestamp=now,
        last_price=42.50,
        return_5m=0.008,
        return_30m=0.021,
        return_1d=0.027,
        market_return_30m=0.002,
        sector_return_30m=0.004,
        volume_ratio_30m=2.6,
        spread_bps=12,
        realised_volatility_20d=0.34,
        average_daily_value_gbp=18_000_000,
        short_interest_zscore=0.8,
        options_implied_move=0.045,
    )

    return engine.rank_watchlist(
        [market],
        market_breadth=0.35,
        volatility_percentile=0.42,
        index_trend=0.006,
    )


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", 120)
    print(example().to_string(index=False))
